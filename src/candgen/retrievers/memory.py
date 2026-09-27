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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


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
