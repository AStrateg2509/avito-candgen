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
    timings  — время подготовки частей (для отчёта);
    dense    — {модель: (эмбеддинги корпуса [N × d], эмбеддинги запросов [n_q × d])}, fp16;
    sparse   — {источник: разреженная матрица скоров [n_q × N]} (например, память train);
    loc_geo  — данные для гео-сглаживания P_loc (LocationPrior.geo_matrices: C, N, D, w);
    q_is_agg — запрос из локации-агрегата (региона), а не города [n_q] (фишка 6).
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
    dense: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    sparse: dict[str, sp.csr_matrix] = field(default_factory=dict)
    loc_geo: dict = field(default_factory=dict)
    q_is_agg: np.ndarray | None = None


# Ключи, которые задают не веса слагаемых, а параметры приоров (LinearScorer.set_params).
PARAM_KEYS = ("loc_eps", "mc_eps", "loc_sigma", "loc_lambda")


class LinearScorer:
    """Держит компоненты на устройстве и считает top-k для любых весов.

    Компоненты загружаются на GPU один раз, поэтому перебор весов (абляция,
    Optuna) стоит только пересчёта скоров: ~7 с на 6,5 тыс. запросов.
    """

    def __init__(self, comp: Components, device: torch.device, loc_eps: float, mc_eps: float,
                 loc_sigma: float = 30.0, loc_lambda: float = 0.0):
        """Вход: компоненты; устройство; eps для логарифмов приоров; параметры
        гео-сглаживания P_loc (σ в км, λ — псевдосчётчик; λ = 0 — без сглаживания)."""
        self.comp = comp
        self.device = device
        self.n_items = len(comp.item_ids)
        self.text = {name: (to_torch_csr(X, device), Q) for name, (X, Q) in comp.text.items()}
        self.loc_P_raw = torch.from_numpy(comp.loc_P).to(device)
        self.geo = {k: torch.from_numpy(v).to(device) for k, v in comp.loc_geo.items()
                    if k in ("C", "N", "D", "w")}
        is_agg = comp.q_is_agg if comp.q_is_agg is not None else np.zeros(len(comp.query_ids), bool)
        self.q_is_agg = torch.from_numpy(is_agg).to(device)
        self.q_loc = torch.from_numpy(comp.q_loc_idx).long().to(device)
        self.i_loc = torch.from_numpy(comp.item_loc_idx).long().to(device)
        self.mc_P = torch.from_numpy(comp.mc_P).to(device)
        self.i_mc = torch.from_numpy(comp.item_mc_idx).long().to(device)
        self.filt_W = torch.from_numpy(comp.filt_W).to(device)
        self.filt_F = torch.from_numpy(comp.filt_F.astype(np.float32)).to(device)
        self.dense = {name: (torch.from_numpy(E).to(device), torch.from_numpy(Q).to(device))
                      for name, (E, Q) in comp.dense.items()}
        self.sparse = comp.sparse  # строки батча уплотняются на лету
        self.params = {"loc_eps": loc_eps, "mc_eps": mc_eps, "loc_sigma": loc_sigma, "loc_lambda": loc_lambda}
        self.set_params()

    def set_params(self, **params) -> None:
        """Пересчитывает приоры с новыми параметрами (абляция, Optuna) на GPU.

        loc_eps / mc_eps — «пол» для нулевой вероятности в логарифме: чем он
        меньше, тем сильнее штраф объявлению из локации или подкатегории, куда
        запрос не кликал. loc_lambda > 0 включает гео-сглаживание (фишка 2):
        P = (C + λ·K) / (N + λ), K ∝ n_items(i)·exp(−dist/σ), строка K суммируется в 1.
        У строк без кликов (N = 0) это даёт P = K — «соседи по карте».
        """
        self.params.update({k: v for k, v in params.items() if v is not None})
        p = self.params
        if p["loc_lambda"] > 0 and self.geo:
            g = self.geo
            K = g["w"][None, :] * torch.exp(-g["D"] / p["loc_sigma"])
            # нет координат у локации поиска -> ядро пропорционально числу объявлений
            K = torch.where(torch.isnan(K), g["w"][None, :].expand_as(K), K)
            K = K / K.sum(dim=1, keepdim=True).clamp(min=1e-12)
            self.loc_P = (g["C"] + p["loc_lambda"] * K) / (g["N"][:, None] + p["loc_lambda"])
        else:
            self.loc_P = self.loc_P_raw
        self.loc_logP = torch.log(self.loc_P + p["loc_eps"])
        self.mc_logP = torch.log(self.mc_P + p["mc_eps"])

    def set_eps(self, loc_eps: float, mc_eps: float) -> None:
        """Совместимость: то же, что set_params(loc_eps=..., mc_eps=...)."""
        self.set_params(loc_eps=loc_eps, mc_eps=mc_eps)

    def _weight(self, weights: dict, name: str, rows: slice) -> torch.Tensor | None:
        """Вес слагаемого для строк батча: столбец [B × 1] или None, если он нулевой.

        Фишка 6: если задан ключ «<имя>_agg», запросы из регионов-агрегатов
        получают этот вес, а запросы из городов — обычный «<имя>».
        """
        w_city = weights.get(name, 0.0)
        w_agg = weights.get(f"{name}_agg", w_city)
        if not (w_city or w_agg):
            return None
        return torch.where(self.q_is_agg[rows], torch.tensor(float(w_agg), device=self.device),
                           torch.tensor(float(w_city), device=self.device))[:, None]

    @torch.no_grad()
    def batch_scores(self, rows: slice, weights: dict, loc_mask_min_p: float | None = None) -> torch.Tensor:
        """Полная матрица скоров B × N для среза запросов rows."""
        b = rows.stop - rows.start
        S = torch.zeros((b, self.n_items), device=self.device)
        for name, (X, Q) in self.text.items():
            w = self._weight(weights, name, rows)
            if w is not None:
                S.add_(torch.sparse.mm(X, dense_batch_T(Q, rows, self.device)).T * w)
        for name, (E, Q) in self.dense.items():
            w = self._weight(weights, name, rows)
            if w is not None:
                S.add_((Q[rows] @ E.T).float() * w)  # косинус: векторы L2-нормированы
        for name, M in self.sparse.items():
            w = self._weight(weights, name, rows)
            if w is not None:
                S.add_(torch.from_numpy(M[rows].toarray()).to(self.device) * w)
        q_loc = self.q_loc[rows]
        w = self._weight(weights, "loc", rows)
        if w is not None:
            S.add_(self.loc_logP[q_loc][:, self.i_loc] * w)
        if loc_mask_min_p is not None:
            S.sub_((self.loc_P[q_loc][:, self.i_loc] < loc_mask_min_p).float(), alpha=MASK_PENALTY)
        w = self._weight(weights, "mc", rows)
        if w is not None:
            S.add_(self.mc_logP[rows][:, self.i_mc] * w)
        w = self._weight(weights, "filter", rows)
        if w is not None:
            S.add_((self.filt_W[rows] @ self.filt_F) * w)
        return S

    @torch.no_grad()
    def explain(self, rows: slice, idx: np.ndarray, weights: dict) -> dict[str, np.ndarray]:
        """Вклад каждого слагаемого скора для заданных объявлений (для демо-страницы).

        Вход: срез запросов; индексы объявлений [B × m] (например, их top-m);
              веса слагаемых (как в batch_scores).
        Выход: {слагаемое: вклад с учётом веса [B × m]}; сумма вкладов = скор.
        """
        take = torch.from_numpy(idx).to(self.device)
        parts = {}
        for name in list(self.text) + list(self.dense) + list(self.sparse) + ["loc", "mc", "filter"]:
            w = weights.get(name, 0.0)
            if not w:
                continue
            S = self.batch_scores(rows, {name: w})  # скор только этого слагаемого
            parts[name] = torch.gather(S, 1, take).cpu().numpy()
        return parts

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
        return self.to_candidates(idx)

    def to_candidates(self, idx: np.ndarray) -> dict[str, list[str]]:
        """Матрица индексов объявлений [n_q × k] -> {query_id: [item_id, ...]}."""
        ids = self.comp.item_ids
        return {qid: ids[row].tolist() for qid, row in zip(self.comp.query_ids, idx)}


# --------------------------------------------------------------------------
# Слияние по каналам: RRF и квоты (альтернативы линейной сумме)
# --------------------------------------------------------------------------
# Канал — это набор весов для LinearScorer (например, «лексика + приоры»
# или «dense + приоры»). Каждый канал даёт свой ранжированный top-depth,
# потом списки объединяются. Приоры входят в каждый канал.

def rrf_fuse(scorer: LinearScorer, channels: dict, k: int = 50, depth: int = 200,
             k_rrf: float = 60.0, batch_size: int = 512) -> np.ndarray:
    """Reciprocal Rank Fusion с весами каналов.

    score(d) = Σ_c w_c / (k_rrf + rank_c(d)), rank с 1; документ вне top-depth
    канала вклада от него не получает.

    Вход: скорер; {канал: {"weights": {...}, "w": вес канала}}; итоговый k;
          глубина списков каналов; константа RRF.
    Выход: индексы объявлений [n_q × k].
    """
    lists, ws = [], []
    for ch in channels.values():
        idx, _ = scorer.topk(ch["weights"], depth, batch_size)
        lists.append(idx)
        ws.append(ch.get("w", 1.0))
    contrib = np.concatenate([np.broadcast_to(w / (k_rrf + np.arange(1, depth + 1)), (lists[0].shape[0], depth))
                              for w in ws], axis=1)
    allidx = np.concatenate(lists, axis=1)
    out = np.empty((allidx.shape[0], k), dtype=np.int64)
    for qi in range(allidx.shape[0]):
        uniq, inv = np.unique(allidx[qi], return_inverse=True)
        score = np.bincount(inv, weights=contrib[qi])
        # сортировка по убыванию скора; при равенстве — по индексу (детерминированно)
        out[qi] = uniq[np.lexsort((uniq, -score))[:k]]
    return out


def quota_fuse(scorer: LinearScorer, channels: dict, k: int = 50, batch_size: int = 512) -> np.ndarray:
    """Квоты: из каждого канала по порядку берутся первые n_c новых кандидатов,
    затем список добивается первым каналом до k.

    Вход: скорер; {канал: {"weights": {...}, "n": квота}} (порядок важен);
          итоговый k.
    Выход: индексы объявлений [n_q × k].
    """
    tops = [scorer.topk(ch["weights"], k, batch_size)[0] for ch in channels.values()]
    quotas = [ch["n"] for ch in channels.values()]
    out = np.empty((tops[0].shape[0], k), dtype=np.int64)
    for qi in range(out.shape[0]):
        chosen, seen = [], set()
        for top, n in zip(tops, quotas):
            taken = 0
            for d in top[qi]:
                if taken >= n:
                    break
                if d not in seen:
                    chosen.append(d)
                    seen.add(d)
                    taken += 1
        for d in tops[0][qi]:  # добивка основным каналом
            if len(chosen) >= k:
                break
            if d not in seen:
                chosen.append(d)
                seen.add(d)
        out[qi] = chosen[:k]
    return out
