"""Источники из истории train: «память» запросов и doc expansion объявлений.

Оба строятся только по переданной части train: на валидации это остаток
(без отложенных строк), на сабмите — весь train.

1) Память (memory_matrix). Если нормализованный текст запроса встречался
   в train, выбранные тогда объявления корпуса получают скор
       log(1 + клики по тексту) + same_loc_bonus · log(1 + клики по тексту в той же локации).
   Сигнал есть только у seen-запросов: в бенчмарке у ~40% из них.

2) Doc expansion (expansion_texts). Объявлению корпуса, которое выбирали
   в train, приписывается «документ» из текстов запросов, которые к нему
   привели (самые частые — первыми). Он индексируется отдельным char TF-IDF
   источником: новый запрос, похожий на старые запросы к объявлению, находит
   его, даже если в самом объявлении нужных слов нет.

Ограничение: оба источника знают только объявления, выбиравшиеся в train
(в корпусе их ~18 тыс. из 189 тыс.). Их пользу на валидации нужно сверять
по item-срезу метрики (item_in_rest / item_new): новым объявлениям они не помогают.

3) Graph doc expansion (graph_expansion_texts) снимает это
   ограничение: запросы переносятся с похожих объявлений train (dense-kNN
   внутри подкатегории) на любое объявление корпуса, в том числе новое.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def memory_matrix(queries: pd.DataFrame, train: pd.DataFrame, item_ids: np.ndarray,
                  same_loc_bonus: float) -> sp.csr_matrix:
    """Матрица скоров памяти [n_q × N] (float32, разреженная).

    Вход: запросы (query_norm, search_location_id); train-часть (query_norm,
          search_location_id, item_id); id корпуса в порядке скорера; бонус за
          клики в той же локации.
    Выход: CSR [n_q × N]; ненулевые только у пар (запрос, выбранное объявление).
    """
    ipos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    tr = train[train["item_id"].isin(ipos.index)]
    by_text = tr.groupby(["query_norm", "item_id"]).size().rename("n").reset_index()
    by_loc = (tr.groupby(["query_norm", "search_location_id", "item_id"]).size()
              .rename("n_loc").reset_index())
    q = pd.DataFrame({"qi": np.arange(len(queries)), "query_norm": queries["query_norm"].to_numpy(),
                      "search_location_id": queries["search_location_id"].to_numpy()})
    m = q.merge(by_text, on="query_norm").merge(by_loc, on=["query_norm", "search_location_id", "item_id"],
                                                how="left")
    vals = np.log1p(m["n"].to_numpy()) + same_loc_bonus * np.log1p(m["n_loc"].fillna(0).to_numpy())
    return sp.csr_matrix((vals.astype(np.float32), (m["qi"].to_numpy(), ipos.loc[m["item_id"]].to_numpy())),
                         shape=(len(queries), len(item_ids)))


def graph_expansion_texts(corpus_emb: np.ndarray, corpus_mc: np.ndarray, corpus_ids: np.ndarray,
                          train_items: pd.DataFrame, train_emb: np.ndarray, train: pd.DataFrame,
                          gcfg: dict, device: torch.device, batch_size: int = 2048) -> tuple[list[str], dict]:
    """Перенос запросов train на похожие объявления корпуса (graph doc expansion).

    Для каждого объявления корпуса ищутся k ближайших по dense-эмбеддингу
    объявлений train (из переданной части train) той же подкатегории, с
    косинусом ≥ min_sim, исключая само объявление. Его «запросный» текст —
    склейка самых частых запросов этих соседей (по queries_per_neighbor на
    соседа, соседи по убыванию сходства). Новые объявления, которые никто не
    выбирал, так получают запросный текст от похожих старых.

    Вход: эмбеддинги, подкатегории и id корпуса; таблица уникальных объявлений
          train (item_id, item_microcat_id) и их эмбеддинги; train-часть
          (query_norm, item_id); секция graph_expansion конфига; устройство.
    Выход: (тексты длины N_корпуса, статистика: доля объявлений с соседями и т.п.).
    """
    # Доступны только объявления, которые есть в переданной части train.
    avail = train_items["item_id"].isin(pd.Index(train["item_id"].unique())).to_numpy()
    t_ids = train_items["item_id"].to_numpy()[avail]
    t_mc = train_items["item_microcat_id"].to_numpy()[avail]

    # Для каждого объявления train — строка из его самых частых запросов.
    cnt = (train[train["query_norm"] != ""].groupby(["item_id", "query_norm"]).size().rename("n").reset_index()
           .sort_values(["item_id", "n", "query_norm"], ascending=[True, False, True], kind="mergesort"))
    top_q = cnt.groupby("item_id").head(gcfg["queries_per_neighbor"]).groupby("item_id")["query_norm"].agg(" ".join)
    t_text = top_q.reindex(t_ids).fillna("").to_numpy()

    T = torch.from_numpy(train_emb[avail]).to(device)
    Tmc = torch.from_numpy(t_mc.astype(np.int64)).to(device)
    # Совпадение id корпуса и train — через позицию объявления корпуса среди t_ids.
    tpos = pd.Series(np.arange(len(t_ids)), index=t_ids)
    self_pos = torch.from_numpy(tpos.reindex(corpus_ids).fillna(-1).to_numpy().astype(np.int64)).to(device)
    k, min_sim = gcfg["k"], gcfg["min_sim"]
    out, n_with = [""] * len(corpus_ids), 0
    for start in range(0, len(corpus_ids), batch_size):
        rows = slice(start, min(start + batch_size, len(corpus_ids)))
        C = torch.from_numpy(corpus_emb[rows]).to(device)
        sims = (C @ T.T).float()                                         # B × n_train
        if gcfg["same_microcat"]:
            cmc = torch.from_numpy(corpus_mc[rows].astype(np.int64)).to(device)
            sims.masked_fill_(cmc[:, None] != Tmc[None, :], -1.0)
        sp_ = self_pos[rows]
        has_self = sp_ >= 0
        sims[has_self.nonzero(as_tuple=True)[0], sp_[has_self]] = -1.0  # само себя не берём
        vals, idx = torch.topk(sims, k, dim=1)
        vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
        for bi in range(vals.shape[0]):
            parts = [t_text[j] for v, j in zip(vals[bi], idx[bi]) if v >= min_sim and t_text[j]]
            if parts:
                out[start + bi] = " ".join(parts)
                n_with += 1
    return out, {"items_with_neighbors": n_with, "train_items_available": int(avail.sum())}


def expansion_texts(train: pd.DataFrame, item_ids: np.ndarray, max_queries: int) -> list[str]:
    """Тексты doc expansion для каждого объявления корпуса.

    Вход: train-часть (query_norm, item_id); id корпуса; сколько самых частых
          разных запросов брать на объявление.
    Выход: список строк длины N; у объявлений, не выбиравшихся в train, — "".
    """
    ipos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    tr = train[train["item_id"].isin(ipos.index) & (train["query_norm"] != "")]
    cnt = tr.groupby(["item_id", "query_norm"]).size().rename("n").reset_index()
    cnt = cnt.sort_values(["item_id", "n", "query_norm"], ascending=[True, False, True], kind="mergesort")
    cnt = cnt.groupby("item_id").head(max_queries)
    joined = cnt.groupby("item_id")["query_norm"].agg(" ".join)
    out = [""] * len(item_ids)
    for item, text in joined.items():
        out[ipos[item]] = text
    return out
