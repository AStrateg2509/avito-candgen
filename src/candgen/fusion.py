"""Слияние источников и приоров: линейный скор по ВСЕМУ корпусу на GPU.

score(q, d) = Σ_src w_src · cos_src(q, d)
            + w_loc · log(P(loc_d | loc_q) + eps_loc)
            + w_mc  · log(P_mc(mc_d | q) + eps_mc)
            + w_filter · bonus_filter(q, d)

Почему по всему корпусу, а не по top-300 текстового поиска: приоры сильно
переставляют объявления. Нужное может быть 2 000-м по тексту, но единственным
в городе запроса. Поэтому батч запросов (512) скорится со всеми 189 тыс.
объявлений: разреженная матрица корпуса × плотный батч запросов на GPU (~0,1 с),
к результату прибавляются приоры, затем берётся top-k. В памяти в каждый момент
только B × N скоров (~390 МБ), а сохраняются лишь индексы и скоры top-k.

Вариант (b) из ТЗ — поиск внутри «подкорпуса локаций» — реализован как маска:
объявления с P_loc ниже порога получают штраф −1e4 (loc_mask_min_p).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from candgen.torch_utils import dense_batch_T, to_torch_csr

MASK_PENALTY = 1e4


@dataclass
class Components:
    """Всё, что нужно скореру для одного набора запросов.

    text     — {источник: (корпус N × V_q CSR, запросы n_q × V_q CSR)};
    loc_P    — P(item_loc | search_loc) [n_uq × n_ui]; q_loc_idx — строка для
               каждого запроса; item_loc_idx — столбец для каждого объявления;
    mc_P     — P_mc [n_q × n_mc]; item_mc_idx — столбец для каждого объявления;
    filt_W, filt_F — бонус фильтра = filt_W @ filt_F  ([n_q × n_p] @ [n_p × N]);
    timings  — время подготовки частей (для отчёта).
    """
    query_ids: list[str]
    item_ids: np.ndarray
    text: dict[str, tuple[sp.csr_matrix, sp.csr_matrix]]
    loc_P: np.ndarray
    q_loc_idx: np.ndarray
    item_loc_idx: np.ndarray
    mc_P: np.ndarray
    item_mc_idx: np.ndarray
    filt_W: np.ndarray
    filt_F: np.ndarray
    timings: dict = field(default_factory=dict)


class LinearScorer:
    """Держит компоненты на устройстве и считает top-k для любых весов.

    Компоненты загружаются на GPU один раз, поэтому перебор весов (абляция,
    Optuna) стоит только пересчёта скоров: ~7 с на 6,5 тыс. запросов.
    """

    def __init__(self, comp: Components, device: torch.device, loc_eps: float, mc_eps: float):
        """Вход: компоненты; устройство; eps для логарифмов приоров."""
        self.comp = comp
        self.device = device
        self.n_items = len(comp.item_ids)
        self.text = {name: (to_torch_csr(X, device), Q) for name, (X, Q) in comp.text.items()}
        self.loc_P = torch.from_numpy(comp.loc_P).to(device)
        self.q_loc = torch.from_numpy(comp.q_loc_idx).long().to(device)
        self.i_loc = torch.from_numpy(comp.item_loc_idx).long().to(device)
        self.mc_P = torch.from_numpy(comp.mc_P).to(device)
        self.i_mc = torch.from_numpy(comp.item_mc_idx).long().to(device)
        self.filt_W = torch.from_numpy(comp.filt_W).to(device)
        self.filt_F = torch.from_numpy(comp.filt_F.astype(np.float32)).to(device)
        self.set_eps(loc_eps, mc_eps)

    def set_eps(self, loc_eps: float, mc_eps: float) -> None:
        """Пересчитывает логарифмы приоров с новыми eps (нужно при подборе в Optuna).

        eps задаёт «пол» для нулевой вероятности: чем он меньше, тем сильнее
        штрафуется объявление из локации или подкатегории, куда запрос не кликал.
        """
        self.loc_logP = torch.log(self.loc_P + loc_eps)
        self.mc_logP = torch.log(self.mc_P + mc_eps)

    @torch.no_grad()
    def batch_scores(self, rows: slice, weights: dict, loc_mask_min_p: float | None = None) -> torch.Tensor:
        """Полная матрица скоров B × N для среза запросов rows."""
        b = rows.stop - rows.start
        S = torch.zeros((b, self.n_items), device=self.device)
        for name, (X, Q) in self.text.items():
            w = weights.get(name, 0.0)
            if w:
                S.add_(torch.sparse.mm(X, dense_batch_T(Q, rows, self.device)).T, alpha=w)
        q_loc = self.q_loc[rows]
        if weights.get("loc", 0.0):
            S.add_(self.loc_logP[q_loc][:, self.i_loc], alpha=weights["loc"])
        if loc_mask_min_p is not None:
            S.sub_((self.loc_P[q_loc][:, self.i_loc] < loc_mask_min_p).float(), alpha=MASK_PENALTY)
        if weights.get("mc", 0.0):
            S.add_(self.mc_logP[rows][:, self.i_mc], alpha=weights["mc"])
        if weights.get("filter", 0.0):
            S.add_(self.filt_W[rows] @ self.filt_F, alpha=weights["filter"])
        return S

    @torch.no_grad()
    def topk(self, weights: dict, k: int = 50, batch_size: int = 512,
             loc_mask_min_p: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """top-k объявлений для всех запросов.

        Вход: веса {char, word, loc, mc, filter}; k; размер батча; порог маски (b).
        Выход: (индексы объявлений int64 [n_q × k], скоры float32 [n_q × k]),
               по убыванию скора.
        """
        n_q = len(self.comp.query_ids)
        idx = np.empty((n_q, k), dtype=np.int64)
        val = np.empty((n_q, k), dtype=np.float32)
        for start in range(0, n_q, batch_size):
            rows = slice(start, min(start + batch_size, n_q))
            v, i = torch.topk(self.batch_scores(rows, weights, loc_mask_min_p), k, dim=1)
            idx[rows], val[rows] = i.cpu().numpy(), v.cpu().numpy()
        return idx, val

    def candidates(self, weights: dict, k: int = 50, batch_size: int = 512,
                   loc_mask_min_p: float | None = None) -> dict[str, list[str]]:
        """То же, что topk, но в формате {query_id: [item_id, ...]} для evaluate/сабмита."""
        idx, _ = self.topk(weights, k, batch_size, loc_mask_min_p)
        ids = self.comp.item_ids
        return {qid: ids[row].tolist() for qid, row in zip(self.comp.query_ids, idx)}
