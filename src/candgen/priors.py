"""Приоры, не зависящие от текста объявления: локация, подкатегория, фильтр.

Все приоры считаются только по переданной части train: на валидации это
остаток (без отложенных строк), на сабмите — весь train.

  * LocationPrior — матрица переходов P(item_loc | search_loc) по кликам.
    В корпусе нет объявлений в 17% локаций поиска (регионы-агрегаты), поэтому
    сравнивать локации на равенство нельзя, нужна именно матрица переходов;
  * MicrocatPrior — P(подкатегория | запрос) через kNN по похожим текстам
    запросов train (char 2–4 TF-IDF): 86% кликов запроса приходятся на его
    главную подкатегорию;
  * FilterPrior — бонус, если значение «Вид услуги» / «Тип услуги» из фильтра
    поиска есть в параметрах объявления (у выбранных — в 98% случаев).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from candgen.text import compile_filter_keys, parse_filters
from candgen.torch_utils import dense_batch_T, to_torch_csr


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Расстояние по дуге большого круга в км (векторно, с broadcasting)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class LocationPrior:
    """P(item_loc | search_loc) по частоте кликов в train (+ данные для фишек 2 и 6).

    Для локации поиска, которой нет в train, берётся запасной вариант из
    конфига: 'self' — вся масса на ту же локацию (если в корпусе она есть),
    'uniform' — равномерно по локациям корпуса.

    Иерархия локаций из данных: каждой локации приписываются
    координаты — медиана координат объявлений в ней, а для локации поиска без
    своих объявлений (регион) — центр её кликов. Скорер может сгладить
    P(i|s) = (C[s,i] + λ·K[s,i]) / (N_s + λ), K[s,i] ∝ n_items(i)·exp(−dist(s,i)/σ):
    соседние населённые пункты получают ненулевую вероятность, даже если из
    города туда ещё не кликали.
    Город или агрегат: доля кликов локации поиска в саму себя.
    У городов она > 0.6, у регионов-агрегатов ≈ 0 (в данных распределение бимодально).
    """

    def __init__(self, loc_cfg: dict):
        """Вход: секция priors.loc конфига."""
        self.fallback = loc_cfg["fallback"]
        self.counts: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "LocationPrior":
        """Считает число кликов для каждой пары (search_location_id, item_location_id)."""
        self.counts = train.groupby(["search_location_id", "item_location_id"]).size()
        return self

    def geo_matrices(self, uq: np.ndarray, ui: np.ndarray, items: pd.DataFrame) -> dict:
        """Данные для сглаживания и маршрутизации по типу локации.

        Вход: уникальные локации запросов (uq) и корпуса (ui) — в том же порядке,
              что у matrix(); корпус с item_location_id, item_latitude, item_longitude.
        Выход: {C: клики [n_uq × n_ui], N: всего кликов строки [n_uq],
                D: расстояния в км [n_uq × n_ui] (NaN, если координат нет),
                w: число объявлений корпуса в локации [n_ui], self_share: [n_uq]}.
        """
        cen = items.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
        w = items["item_location_id"].value_counts().reindex(ui).fillna(0).to_numpy(np.float32)
        df = self.counts.rename("n").reset_index()
        N = df.groupby("search_location_id")["n"].sum().reindex(uq).fillna(0).to_numpy(np.float32)
        qpos = pd.Series(np.arange(len(uq)), index=uq)
        ipos = pd.Series(np.arange(len(ui)), index=ui)
        C = np.zeros((len(uq), len(ui)), dtype=np.float32)
        sub = df[df["search_location_id"].isin(qpos.index) & df["item_location_id"].isin(ipos.index)]
        C[qpos.loc[sub["search_location_id"]].to_numpy(), ipos.loc[sub["item_location_id"]].to_numpy()] = sub["n"]

        # Координаты локации поиска: своя, если в ней есть объявления корпуса;
        # иначе — центр её кликов, взвешенный числом кликов (для регионов).
        # copy=True: в pandas 3 (Copy-on-Write) to_numpy() отдаёт массив только для чтения
        qlat = cen["item_latitude"].reindex(uq).to_numpy(dtype=float, copy=True)
        qlon = cen["item_longitude"].reindex(uq).to_numpy(dtype=float, copy=True)
        ilat = cen["item_latitude"].reindex(ui).to_numpy()
        ilon = cen["item_longitude"].reindex(ui).to_numpy()
        clicks = C.sum(axis=1)
        no_own = np.isnan(qlat) & (clicks > 0)
        qlat[no_own] = (C[no_own] @ np.nan_to_num(ilat)) / clicks[no_own]
        qlon[no_own] = (C[no_own] @ np.nan_to_num(ilon)) / clicks[no_own]
        D = haversine_km(qlat[:, None], qlon[:, None], ilat[None, :], ilon[None, :]).astype(np.float32)

        self_share = np.array([C[qpos[s], ipos[s]] / N[qpos[s]] if s in ipos.index and N[qpos[s]] > 0 else 0.0
                               for s in uq], dtype=np.float32)
        return {"C": C, "N": N, "D": D, "w": w, "self_share": self_share}

    def matrix(self, query_locs: np.ndarray, item_locs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Плотная матрица P для локаций запросов (строки) и локаций корпуса (столбцы).

        Строка нормирована по ВСЕМ локациям объявлений в train, поэтому масса,
        ушедшая в локации без объявлений корпуса, теряется честно, без перенормировки.

        Вход: локации запросов и объявлений корпуса (с повторами).
        Выход: (P float32 [n_uq × n_ui], уникальные локации запросов,
                уникальные локации корпуса, число локаций с запасным вариантом).
        """
        uq, ui = np.unique(query_locs), np.unique(item_locs)
        qpos = pd.Series(np.arange(len(uq)), index=uq)
        ipos = pd.Series(np.arange(len(ui)), index=ui)
        P = np.zeros((len(uq), len(ui)), dtype=np.float32)

        df = self.counts.rename("n").reset_index()
        totals = df.groupby("search_location_id")["n"].sum()
        df = df[df["search_location_id"].isin(qpos.index) & df["item_location_id"].isin(ipos.index)]
        P[qpos.loc[df["search_location_id"]].to_numpy(), ipos.loc[df["item_location_id"]].to_numpy()] = (
            df["n"].to_numpy() / totals.loc[df["search_location_id"]].to_numpy())

        missing = [loc for loc in uq if loc not in totals.index]
        for loc in missing:
            if self.fallback == "self" and loc in ipos.index:
                P[qpos[loc], ipos[loc]] = 1.0
            else:
                P[qpos[loc], :] = 1.0 / len(ui)
        return P, uq, ui, len(missing)


# Запас кандидатов для детерминированного top-k и точность округления скоров.
KNN_MARGIN = 16
KNN_DECIMALS = 5


def stable_topk_dim0(sims: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Детерминированный top-k по столбцам (dim=0).

    torch.topk при равных и почти равных значениях выбирает соседей произвольно,
    а sparse.mm на GPU в float32 ещё и «дрожит» в последних знаках: из-за этого
    P_mc, а за ним пул кандидатов и ответ, различались между одинаковыми
    запусками. Берём запас k + KNN_MARGIN кандидатов, округляем скоры до
    KNN_DECIMALS знаков и сортируем по (−скор, индекс) двумя стабильными сортировками.

    Вход: матрица скоров [n × B], k.
    Выход: (округлённые скоры [k × B], индексы строк [k × B]).
    """
    m = min(k + KNN_MARGIN, sims.shape[0])
    vals, idx = torch.topk(sims, m, dim=0)
    vals = torch.round(vals, decimals=KNN_DECIMALS)
    o = torch.argsort(idx, dim=0, stable=True)                  # вторичный ключ: индекс
    vals, idx = vals.gather(0, o), idx.gather(0, o)
    o = torch.argsort(-vals, dim=0, stable=True)[:k]            # главный ключ: −скор
    return vals.gather(0, o), idx.gather(0, o)


class MicrocatPrior:
    """P(подкатегория | запрос): взвешенная смесь распределений k ближайших текстов train.

    Тексты train (нормализованные, уникальные) представлены char 2–4 TF-IDF.
    Для каждого запроса ищутся k самых похожих по косинусу, их распределения
    кликов по подкатегориям смешиваются с весами sim**power. Seen-запрос
    находит сам себя (sim = 1) и получает в основном собственную историю.
    """

    def __init__(self, mc_cfg: dict, device: torch.device):
        """Вход: секция priors.mc конфига и устройство для kNN."""
        self.cfg = mc_cfg
        self.device = device

    def fit(self, train: pd.DataFrame, mc_index: np.ndarray) -> "MicrocatPrior":
        """Строит распределения подкатегорий по текстам и TF-IDF текстов.

        Вход: train с query_norm и item_microcat_id; подкатегории корпуса
              (столбцы результата). Клики в подкатегории вне корпуса не учитываются.
        """
        self.mc_index = mc_index
        mpos = pd.Series(np.arange(len(mc_index)), index=mc_index)
        tr = train[train["item_microcat_id"].isin(mpos.index) & (train["query_norm"] != "")]
        cnt = tr.groupby(["query_norm", "item_microcat_id"]).size().rename("n").reset_index()
        self.texts = np.sort(cnt["query_norm"].unique())
        tpos = pd.Series(np.arange(len(self.texts)), index=self.texts)
        D = np.zeros((len(self.texts), len(mc_index)), dtype=np.float32)
        np.add.at(D, (tpos.loc[cnt["query_norm"]].to_numpy(), mpos.loc[cnt["item_microcat_id"]].to_numpy()),
                  cnt["n"].to_numpy(dtype=np.float32))
        self.D = D / D.sum(axis=1, keepdims=True)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(self.cfg["ngram_range"]),
                                          sublinear_tf=True, dtype=np.float32)
        self.T = self.vectorizer.fit_transform(self.texts)
        return self

    def predict(self, query_texts: list[str], batch_size: int = 512) -> np.ndarray:
        """Распределение подкатегорий для каждого запроса.

        Вход: нормализованные тексты запросов.
        Выход: float32 [n_q × n_mc], строки суммируются в 1. Если у запроса нет
               ни одного похожего текста (все sim = 0), строка равномерная.
        """
        k, power = self.cfg["k"], self.cfg["power"]
        Q = self.vectorizer.transform(query_texts)
        T = to_torch_csr(self.T, self.device, torch.float64)  # float64 — воспроизводимость
        D = torch.from_numpy(self.D).to(self.device)
        out = np.empty((len(query_texts), D.shape[1]), dtype=np.float32)
        for start in range(0, len(query_texts), batch_size):
            rows = slice(start, min(start + batch_size, len(query_texts)))
            sims = torch.sparse.mm(T, dense_batch_T(Q, rows, self.device, T.dtype)).float()  # n_texts × B
            vals, idx = stable_topk_dim0(sims, k)                           # k × B
            w = vals.clamp(min=0) ** power
            P = torch.einsum("kb,kbm->bm", w, D[idx])                      # B × n_mc
            s = w.sum(dim=0)
            P = torch.where(s[:, None] > 0, P / s.clamp(min=1e-12)[:, None],
                            torch.full_like(P, 1.0 / D.shape[1]))
            out[rows] = P.cpu().numpy()
        return out


class FilterPrior:
    """Бонус за совпадение фильтра поиска с параметрами объявления.

    Для каждого главного ключа (Вид услуги, Тип услуги) с непустым значением
    объявление получает вес ключа, если в его параметрах есть подстрока
    «ключ значение» (match: pair) или просто «значение» (match: value).
    Итоговый бонус запроса = Σ вес_ключа · [совпало] ∈ [0, 1].
    """

    # Сколько символов после «ключ » сохраняем для поиска значения (значения
    # «Вид/Тип услуги» короче 100 символов).
    TAIL_WIDTH = 150
    SEP = "\x00"

    def __init__(self, filters_cfg: dict, filter_cfg: dict):
        """Вход: секция filters (список ключей) и priors.filter конфига."""
        self.pattern = compile_filter_keys(filters_cfg["keys"])
        self.key_weights = filter_cfg["key_weights"]
        self.match = filter_cfg["match"]
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._tails: dict[str, pa.Array] = {}

    def _key_tails(self, key: str, item_params: list[str]) -> pa.Array:
        """Для каждого объявления — склейка фрагментов, идущих сразу после «ключ ».

        Каждый фрагмент начинается с разделителя SEP, поэтому проверка
        «SEP + значение ⊂ хвосты» равносильна «в параметрах есть «ключ значение»»,
        но ищем в ~20 МБ хвостов, а не в ~180 МБ параметров: в разы быстрее.
        """
        if key not in self._tails:
            prefix, out = key + " ", []
            for p in item_params:
                parts, start = [], p.find(prefix)
                while start != -1:
                    s = start + len(prefix)
                    parts.append(self.SEP + p[s:s + self.TAIL_WIDTH])
                    start = p.find(prefix, s)
                out.append("".join(parts))
            self._tails[key] = pa.array(out)
        return self._tails[key]

    def build(self, filter_texts: list[str], item_params: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Матрица весов «запрос × пара» и матрица совпадений «пара × объявление».

        Бонус запросов = W @ F. Поиск подстроки идёт векторно в arrow
        (pc.match_substring), результаты кэшируются по паре (ключ, значение).

        Вход: тексты фильтров запросов; параметры объявлений корпуса (список строк).
        Выход: (W float32 [n_q × n_pairs], F bool [n_pairs × N], список пар (ключ, значение)).
        """
        pairs: dict[tuple[str, str], int] = {}
        entries = []  # (строка запроса, номер пары, вес)
        for qi, text in enumerate(filter_texts):
            for key, value in parse_filters(text, self.pattern):
                if key in self.key_weights and value:
                    pid = pairs.setdefault((key, value), len(pairs))
                    entries.append((qi, pid, self.key_weights[key]))
        W = np.zeros((len(filter_texts), max(len(pairs), 1)), dtype=np.float32)
        for qi, pid, w in entries:
            W[qi, pid] = w  # повтор того же ключа не суммируется
        F = np.zeros((max(len(pairs), 1), len(item_params)), dtype=bool)
        full = None
        for (key, value), pid in pairs.items():
            if (key, value) not in self._cache:
                if self.match == "pair":
                    hay, needle = self._key_tails(key, item_params), self.SEP + value
                else:
                    full = full if full is not None else pa.array(item_params)
                    hay, needle = full, value
                self._cache[(key, value)] = pc.match_substring(hay, needle).to_numpy(zero_copy_only=False)
            F[pid] = self._cache[(key, value)]
        return W, F, list(pairs)
