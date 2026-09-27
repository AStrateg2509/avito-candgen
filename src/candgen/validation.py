"""Валидация: отложенные срезы из train и метрика Recall@K.

Устройство (решения зафиксированы в CLAUDE.md и PLAN.md, этап 1):
  * «группа» — строки train с одинаковым ключом `validation.group_keys`,
    по умолчанию это полный контекст поиска (нормализованный текст, локация,
    фильтр, доставка, категория). Аналог одного query_id бенчмарка;
  * unseen-срез: `n_unseen` текстов. Для каждого берётся случайная группа
    с объявлениями корпуса, а ВСЕ строки этих текстов удаляются из остатка train;
  * seen-срез: `n_seen` других текстов, у которых ≥2 групп. Откладывается одна
    случайная группа с объявлениями корпуса, остальные группы текста остаются
    в остатке (их объявления могут быть любыми — как в бенчмарке, где «память»
    в корпусе есть не у всех seen-запросов);
  * relevant = выбранные в отложенной группе item_id, которые есть в корпусе;
  * контрольный вариант «ql» строится из тех же текстов и групп, но relevant
    берутся по всей группе (запрос, локация), как в исходном baseline.
    Сравнение с «ctx» парное, без шума выборки.

Весь пайплайн кандидатогенерации — функция (остаток train, корпус, запросы).
На валидации ему подаются остаток и val-запросы (та же схема, что у
benchmark_queries), на сабмите — весь train и бенчмарк.

Главная метрика — bench_adj: recall по 4 ячейкам (seen/unseen × фильтр
есть/нет), взвешенный долями этих ячеек в бенчмарке.

CLI:
  python -m candgen.validation --build      # построить и сохранить срезы
  python -m candgen.validation --selftest   # проверки evaluate и детерминизма
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from candgen import io
from candgen.text import has_filter, normalize_series

# Поля поиска, из которых состоит запрос (как в benchmark_queries).
SEARCH_COLS = ["search_query", "search_location_id", "search_is_delivery_search",
               "search_infm_params_text", "search_category"]
# Ячейки постстратификации: (срез, есть ли фильтр).
CELLS = [("seen", True), ("seen", False), ("unseen", True), ("unseen", False)]
VARIANTS = ("ctx", "ql")
QID_PREFIX = {"unseen": "vu", "seen": "vs"}


def cell_name(slice_name: str, with_filter: bool) -> str:
    """Имя ячейки весов: 'seen_filter', 'unseen_nofilter' и т.п."""
    return f"{slice_name}_{'filter' if with_filter else 'nofilter'}"


def _to_py(x):
    """Рекурсивно превращает numpy-типы в обычные питоновские (для json)."""
    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


# --------------------------------------------------------------------------
# Веса бенчмарка
# --------------------------------------------------------------------------

def bench_strata(queries: pd.DataFrame, train: pd.DataFrame, top_n: int) -> dict:
    """Считает доли бенчмарка по ячейкам (seen/unseen × фильтр) и сопутствующие доли.

    seen — нормализованный текст запроса встречается в train (то же понятие,
    что и на валидации). Дословная доля нужна только для сверки с фактом из ТЗ
    (907 / 2452). «Память» — у seen-запроса в train есть выбранные объявления
    из корпуса.

    Вход: benchmark_queries; train с колонками search_query, query_norm,
          in_corpus; размер списка топ-локаций.
    Выход: словарь с весами ячеек `cell_weights`, долями и топ-локациями.
    """
    train_norm = pd.Index(train["query_norm"].unique())
    train_exact = pd.Index(train["search_query"].unique())
    mem_texts = pd.Index(train.loc[train["in_corpus"], "query_norm"].unique())

    qn = normalize_series(queries["search_query"])
    seen = qn.isin(train_norm).to_numpy()
    filt = has_filter(queries["search_infm_params_text"]).to_numpy()
    memory = qn.isin(mem_texts).to_numpy()

    weights = {cell_name(s, f): float(np.mean((seen == (s == "seen")) & (filt == f)))
               for s, f in CELLS}
    # Топ-локации: по числу запросов, при равенстве — по id (детерминированно).
    loc_counts = (queries["search_location_id"].value_counts().rename("n").reset_index()
                  .sort_values(["n", "search_location_id"], ascending=[False, True]))
    return {
        "n_queries": int(len(queries)),
        "seen_share_norm": float(seen.mean()),
        "seen_share_exact": float(queries["search_query"].isin(train_exact).mean()),
        "filter_share": float(filt.mean()),
        "filter_share_seen": float(filt[seen].mean()),
        "filter_share_unseen": float(filt[~seen].mean()),
        "seen_memory_share": float(memory[seen].mean()),
        "cell_weights": weights,
        "top_locations": [int(x) for x in loc_counts["search_location_id"].head(top_n)],
    }


# --------------------------------------------------------------------------
# Построение срезов
# --------------------------------------------------------------------------

def load_train_for_split(cfg: dict) -> pd.DataFrame:
    """Загружает колонки train, нужные для срезов, и добавляет вычисляемые поля.

    Вход: конфиг.
    Выход: DataFrame в исходном порядке строк train.parquet (важно для масок)
           с полями поиска, item_id, query_norm и in_corpus (объявление есть в корпусе).
    """
    tr = io.load_train(cfg, SEARCH_COLS + ["item_id"])
    tr["query_norm"] = normalize_series(tr["search_query"])
    corpus_ids = pd.Index(io.load_items(cfg, ["item_id"])["item_id"])
    tr["in_corpus"] = tr["item_id"].isin(corpus_ids)
    return tr


def _joint_mode(rows: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """Самая частая комбинация значений `cols` внутри каждой группы.

    Нужна, чтобы у val-запроса был ровно один набор полей поиска, как у
    query_id бенчмарка. Для ключа ctx различается только сырой текст запроса
    (регистр, пунктуация), для ql — ещё и фильтр. При равенстве частот
    берётся лексикографически меньшая комбинация, так что результат детерминирован.

    Вход: строки train с колонкой group_col; список колонок.
    Выход: DataFrame (group_col + cols), по одной строке на группу.
    """
    cnt = rows.groupby([group_col] + cols, dropna=False).size().rename("_n").reset_index()
    cnt = cnt.sort_values([group_col, "_n"] + cols,
                          ascending=[True, False] + [True] * len(cols), kind="mergesort")
    return cnt.drop_duplicates(group_col)[[group_col] + cols].reset_index(drop=True)


def _materialize(tr: pd.DataFrame, gid: np.ndarray, chosen: pd.DataFrame,
                 rest: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит таблицу val-запросов и таблицу relevant для выбранных групп.

    Вход: train (load_train_for_split); номер группы каждой строки; выбранные
          группы (text, slice, gid, query_id); маска остатка train.
    Выход: (queries, rel).
      queries: query_id, slice, cell, has_filter, has_memory, query_norm + SEARCH_COLS;
      rel: query_id, item_id, n_clicks (сколько раз выбрано в группе),
           in_rest (объявление встречается в остатке train — для item-среза).
    """
    sel = np.isin(gid, chosen["gid"].to_numpy())
    rows = tr.loc[sel, SEARCH_COLS + ["item_id", "in_corpus"]].copy()
    rows["gid"] = gid[sel]

    q = chosen.merge(_joint_mode(rows, "gid", SEARCH_COLS), on="gid", how="left")
    q["has_filter"] = has_filter(q["search_infm_params_text"])
    q["cell"] = [cell_name(s, f) for s, f in zip(q["slice"], q["has_filter"])]
    # «Память»: у текста в остатке train есть выбранные объявления корпуса
    # (у unseen её нет по построению).
    mem_texts = pd.Index(tr.loc[rest & tr["in_corpus"].to_numpy(), "query_norm"].unique())
    q["has_memory"] = q["text"].isin(mem_texts)

    rel = (rows[rows["in_corpus"]].groupby(["gid", "item_id"]).size()
           .rename("n_clicks").reset_index())
    rel = rel.merge(q[["gid", "query_id"]], on="gid")[["query_id", "item_id", "n_clicks"]]
    items_in_rest = pd.Index(tr.loc[rest, "item_id"].unique())
    rel["in_rest"] = rel["item_id"].isin(items_in_rest)

    q = q.rename(columns={"text": "query_norm"})
    cols = ["query_id", "slice", "cell", "has_filter", "has_memory", "query_norm"] + SEARCH_COLS
    q = q[cols].sort_values("query_id").reset_index(drop=True)
    rel = rel.sort_values(["query_id", "item_id"]).reset_index(drop=True)
    return q, rel


def split_hash(queries: pd.DataFrame, rel: pd.DataFrame, rest: np.ndarray) -> str:
    """Короткий хэш содержимого среза для проверки детерминизма."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(queries, index=False).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(rel, index=False).to_numpy().tobytes())
    h.update(np.packbits(rest).tobytes())
    return h.hexdigest()[:16]


def _variant_stats(q: pd.DataFrame, rel: pd.DataFrame, rest: np.ndarray, k: int) -> dict:
    """Сводка по варианту срезов: размеры, ячейки, |rel|, доля «памяти»."""
    n_rel = rel.groupby("query_id").size()
    return _to_py({
        "n_queries": len(q),
        "n_by_slice": q["slice"].value_counts().to_dict(),
        "n_by_cell": q["cell"].value_counts().to_dict(),
        "filter_share_by_slice": q.groupby("slice")["has_filter"].mean().to_dict(),
        "seen_memory_share": q.loc[q["slice"] == "seen", "has_memory"].mean(),
        "rel_per_query": {"mean": n_rel.mean(), "median": n_rel.median(),
                          "p90": n_rel.quantile(0.9), "max": n_rel.max(),
                          "share_eq_1": (n_rel == 1).mean(),
                          f"share_gt_{k}": (n_rel > k).mean()},
        "rel_items_in_rest_share": rel["in_rest"].mean(),
        "train_rest_rows": int(rest.sum()),
        "hash": split_hash(q, rel, rest),
    })


def build_splits(cfg: dict, tr: pd.DataFrame | None = None) -> dict:
    """Строит оба варианта срезов (ctx и ql) и метаданные с весами бенчмарка.

    Все случайности идут из одного `default_rng(seed)` в фиксированном порядке:
    приоритеты групп → выбор unseen-текстов → выбор seen-текстов. Поэтому
    повторный запуск даёт побитово тот же результат (проверяется в selftest).

    Вход: конфиг; train (если уже загружен, иначе загружается).
    Выход: {"ctx": {...}, "ql": {...}, "meta": {...}}; у варианта ключи
           queries, rel, rest (bool-маска строк train, входящих в остаток).
    """
    vcfg = cfg["validation"]
    ctx_keys, ql_keys = vcfg["group_keys"], vcfg["alt_group_keys"]
    if not set(ql_keys) <= set(ctx_keys):
        raise ValueError("alt_group_keys должен быть подмножеством group_keys")
    if tr is None:
        tr = load_train_for_split(cfg)
    rng = np.random.default_rng(cfg["seed"])

    # 1. Номер группы каждой строки для обоих ключей.
    gid_ctx = tr.groupby(ctx_keys, sort=True, dropna=False).ngroup().to_numpy()
    gid_ql = tr.groupby(ql_keys, sort=True, dropna=False).ngroup().to_numpy()

    # 2. Статистика групп: текст и число строк с объявлениями корпуса.
    g = (pd.DataFrame({"gid": gid_ctx, "text": tr["query_norm"].to_numpy(),
                       "corp": tr["in_corpus"].to_numpy()})
         .groupby("gid").agg(text=("text", "first"), n_corp=("corp", "sum")))
    # Случайный приоритет каждой группы. Внутри текста берётся группа
    # с объявлениями корпуса и максимальным приоритетом — это равномерный
    # случайный выбор одной группы на текст.
    g["prio"] = rng.random(len(g))
    g_corp = g[g["n_corp"] > 0]
    chosen_gid = g_corp.groupby("text")["prio"].idxmax()  # текст -> gid

    t = g.groupby("text").size().rename("n_groups").to_frame().sort_index()
    t["n_corp_groups"] = g_corp.groupby("text").size().reindex(t.index, fill_value=0)
    t = t[t.index != ""]  # запросы только из знаков препинания не берём

    # 3. Выбор текстов (списки кандидатов отсортированы -> детерминизм).
    elig_unseen = t.index[t["n_corp_groups"] >= 1].to_numpy()
    if len(elig_unseen) < vcfg["n_unseen"]:
        raise ValueError(f"кандидатов в unseen {len(elig_unseen)} < n_unseen")
    unseen_texts = np.sort(rng.choice(elig_unseen, size=vcfg["n_unseen"], replace=False))
    elig_seen = t.index[(t["n_groups"] >= 2) & (t["n_corp_groups"] >= 1)
                        & ~t.index.isin(unseen_texts)].to_numpy()
    if len(elig_seen) < vcfg["n_seen"]:
        raise ValueError(f"кандидатов в seen {len(elig_seen)} < n_seen")
    seen_texts = np.sort(rng.choice(elig_seen, size=vcfg["n_seen"], replace=False))

    chosen = pd.concat([pd.DataFrame({"text": unseen_texts, "slice": "unseen"}),
                        pd.DataFrame({"text": seen_texts, "slice": "seen"})],
                       ignore_index=True)
    chosen["gid"] = chosen_gid.loc[chosen["text"]].to_numpy()
    order = chosen.groupby("slice").cumcount()
    chosen["query_id"] = [f"{QID_PREFIX[s]}_{i:05d}" for s, i in zip(chosen["slice"], order)]

    # 4. Остаток train для ctx: без всех строк unseen-текстов и без
    #    отложенных seen-групп.
    drop_unseen = tr["query_norm"].isin(unseen_texts).to_numpy()
    is_seen = (chosen["slice"] == "seen").to_numpy()
    rest_ctx = ~(drop_unseen | np.isin(gid_ctx, chosen.loc[is_seen, "gid"].to_numpy()))

    # 5. Вариант ql: те же тексты и группы, но группа расширяется до
    #    (запрос, локация), и из остатка удаляется вся она.
    ctx_to_ql = (pd.DataFrame({"c": gid_ctx, "q": gid_ql}).drop_duplicates("c")
                 .set_index("c")["q"])
    chosen_ql = chosen.copy()
    chosen_ql["gid"] = ctx_to_ql.loc[chosen["gid"]].to_numpy()
    rest_ql = ~(drop_unseen | np.isin(gid_ql, chosen_ql.loc[is_seen, "gid"].to_numpy()))
    # seen-текст, у которого в остатке не осталось строк, перестал быть seen —
    # в ql-варианте его нет (для парного сравнения используем общие query_id).
    texts_left = pd.Index(tr.loc[rest_ql, "query_norm"].unique())
    keep = (chosen_ql["slice"] == "unseen") | chosen_ql["text"].isin(texts_left)
    chosen_ql = chosen_ql[keep.to_numpy()]

    splits = {}
    for name, gid, ch, rest in (("ctx", gid_ctx, chosen, rest_ctx),
                                ("ql", gid_ql, chosen_ql, rest_ql)):
        q, rel = _materialize(tr, gid, ch, rest)
        splits[name] = {"queries": q, "rel": rel, "rest": rest}

    bench = bench_strata(io.load_queries(cfg), tr, vcfg["top_locations_n"])
    splits["meta"] = _to_py({
        "seed": cfg["seed"],
        "validation_cfg": vcfg,
        "train_rows": len(tr),
        # Сколько текстов вообще подходило под срезы и какую долю строк train
        # «съели» unseen-тексты: это цена валидации для приоров остатка.
        "n_texts": len(t),
        "n_eligible_unseen": len(elig_unseen),
        "n_eligible_seen": len(elig_seen),
        "rows_share_removed_unseen": float(drop_unseen.mean()),
        "bench": bench,
        "variants": {v: _variant_stats(splits[v]["queries"], splits[v]["rel"],
                                       splits[v]["rest"], vcfg["k"]) for v in VARIANTS},
    })
    return splits


def save_splits(cfg: dict, splits: dict) -> None:
    """Сохраняет срезы в artifacts/: val_{queries,rel,train_mask}_<вариант>.parquet
    и общий val_meta.json. Маска: строка i соответствует строке i train.parquet."""
    art = Path(cfg["paths"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        s = splits[v]
        s["queries"].to_parquet(art / f"val_queries_{v}.parquet", index=False)
        s["rel"].to_parquet(art / f"val_rel_{v}.parquet", index=False)
        pd.DataFrame({"row_idx": np.arange(len(s["rest"]), dtype=np.int32),
                      "in_rest": s["rest"]}).to_parquet(art / f"val_train_mask_{v}.parquet",
                                                        index=False)
    with open(art / "val_meta.json", "w", encoding="utf-8") as f:
        json.dump(splits["meta"], f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Загрузка срезов и метрика
# --------------------------------------------------------------------------

@dataclass
class ValSet:
    """Загруженный вариант валидации.

    queries — val-запросы (схема benchmark_queries + slice/cell/has_filter/...);
    rel — query_id -> множество релевантных item_id;
    rel_in_rest — query_id -> те релевантные, что встречаются в остатке train;
    meta — содержимое val_meta.json (веса бенчмарка, топ-локации, статистика).
    """
    variant: str
    queries: pd.DataFrame
    rel: dict[str, frozenset]
    rel_in_rest: dict[str, frozenset]
    meta: dict
    k: int
    min_cell_n: int


def load_val(cfg: dict, variant: str = "ctx") -> ValSet:
    """Читает сохранённый вариант срезов (сначала нужен --build).

    Вход: конфиг, вариант ('ctx' — основной, 'ql' — контрольный).
    Выход: ValSet.
    """
    art = Path(cfg["paths"]["artifacts_dir"])
    q = pd.read_parquet(art / f"val_queries_{variant}.parquet")
    q["query_id"] = q["query_id"].astype(str)
    rel = pd.read_parquet(art / f"val_rel_{variant}.parquet")
    rel["query_id"] = rel["query_id"].astype(str)
    rel["item_id"] = rel["item_id"].astype(str)
    with open(art / "val_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    rel_sets = rel.groupby("query_id")["item_id"].agg(frozenset).to_dict()
    rest_sets = rel[rel["in_rest"]].groupby("query_id")["item_id"].agg(frozenset).to_dict()
    rest_sets = {qid: rest_sets.get(qid, frozenset()) for qid in rel_sets}
    vcfg = cfg["validation"]
    return ValSet(variant, q, rel_sets, rest_sets, meta, vcfg["k"], vcfg["min_cell_n"])


def subset_val(val: ValSet, query_ids: set[str]) -> ValSet:
    """Подмножество валидации (например, половина для честной проверки подбора весов).

    Вход: ValSet и множество query_id. Выход: новый ValSet только с этими запросами.
    """
    q = val.queries[val.queries["query_id"].isin(query_ids)].reset_index(drop=True)
    keep = set(q["query_id"])
    return ValSet(val.variant, q, {k: v for k, v in val.rel.items() if k in keep},
                  {k: v for k, v in val.rel_in_rest.items() if k in keep}, val.meta, val.k, val.min_cell_n)


def split_halves(val: ValSet, seed: int) -> tuple[set[str], set[str]]:
    """Делит запросы валидации на две половины, стратифицируя по ячейкам.

    Вход: ValSet, seed. Выход: (query_id половины A, query_id половины B).
    """
    rng = np.random.default_rng(seed)
    a, b = set(), set()
    for _, grp in val.queries.groupby("cell", sort=True):
        ids = rng.permutation(np.sort(grp["query_id"].to_numpy()))
        a.update(ids[: len(ids) // 2])
        b.update(ids[len(ids) // 2:])
    return a, b


def load_train_rest_mask(cfg: dict, variant: str = "ctx") -> np.ndarray:
    """Булева маска «строка train входит в обучающий остаток» для варианта.

    Остаток — единственная часть train, по которой на валидации можно считать
    приоры, память и т.п. Для сабмита используется весь train.
    """
    path = Path(cfg["paths"]["artifacts_dir"]) / f"val_train_mask_{variant}.parquet"
    return pd.read_parquet(path)["in_rest"].to_numpy()


def _stat(x: np.ndarray) -> dict:
    """Среднее, стандартная ошибка среднего и размер для массива recall."""
    n = len(x)
    mean = float(x.mean()) if n else math.nan
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {"recall": mean, "se": se, "n": n}


def evaluate(candidates: dict[str, list[str]], val: ValSet | None = None,
             k: int | None = None, cfg: dict | None = None) -> dict:
    """Считает Recall@K по всем срезам валидации.

    Правила: из списка кандидатов берутся первые k уникальных id (порядок
    сохраняется). Запрос без кандидатов получает recall 0 и предупреждение.
    Неизвестный query_id — ошибка: это почти наверняка баг пайплайна.

    Вход: {query_id: [item_id, ...]}; ValSet (по умолчанию основной 'ctx');
          k (по умолчанию из конфига); конфиг (если val не передан).
    Выход: словарь метрик:
      bench_adj (+ _se) — главная: Σ вес_ячейки·recall_ячейки;
      weighted (+ _se) — (1−s)·unseen + s·seen, s — доля seen в бенчмарке;
      unseen, seen, unseen_fadj, seen_fadj — срезы и они же, перевзвешенные
      по доле фильтра бенчмарка; filter_adj — то же для всей валидации;
      slices — recall/se/n по срезам; cells — по 4 ячейкам с весами;
      item_in_rest / item_new — micro-recall по релевантным объявлениям,
      которые были / не были в остатке train;
      mean_len, share_short, n_missing, warnings.
    """
    if val is None:
        val = load_val(cfg or io.load_config(), "ctx")
    k = k or val.k
    q = val.queries
    qids = q["query_id"].tolist()
    known = set(qids)
    extra = [x for x in candidates if x not in known]
    if extra:
        raise ValueError(f"{len(extra)} неизвестных query_id в кандидатах, например {extra[:3]}")

    n = len(qids)
    recall = np.zeros(n)
    lens = np.zeros(n, dtype=int)
    n_missing = 0
    hit_rest = tot_rest = hit_new = tot_new = 0
    for i, qid in enumerate(qids):
        cand = candidates.get(qid)
        if cand is None:
            n_missing += 1
            cand = ()
        # dict.fromkeys убирает повторы, сохраняя порядок; затем обрезка до k.
        top = set(list(dict.fromkeys(cand))[:k])
        lens[i] = len(top)
        rel, in_rest = val.rel[qid], val.rel_in_rest[qid]
        hit = rel & top
        recall[i] = len(hit) / len(rel)
        h_rest = len(hit & in_rest)
        hit_rest += h_rest
        tot_rest += len(in_rest)
        hit_new += len(hit) - h_rest
        tot_new += len(rel) - len(in_rest)

    bench = val.meta["bench"]
    sl = q["slice"].to_numpy()
    hf = q["has_filter"].to_numpy(dtype=bool)
    top_loc = q["search_location_id"].isin(bench["top_locations"]).to_numpy()
    slices = {
        "all": _stat(recall),
        "unseen": _stat(recall[sl == "unseen"]),
        "seen": _stat(recall[sl == "seen"]),
        "filter": _stat(recall[hf]),
        "no_filter": _stat(recall[~hf]),
        "top_loc": _stat(recall[top_loc]),
        "other_loc": _stat(recall[~top_loc]),
    }
    cells = {}
    for s, f in CELLS:
        name = cell_name(s, f)
        cells[name] = {**_stat(recall[(sl == s) & (hf == f)]),
                       "weight": bench["cell_weights"][name]}

    warnings = []
    for name, c in cells.items():
        if c["n"] < val.min_cell_n:
            warnings.append(f"в ячейке {name} всего {c['n']} запросов (< {val.min_cell_n})")
    if n_missing:
        warnings.append(f"{n_missing} запросов без кандидатов (засчитан recall 0)")

    bench_adj = sum(c["weight"] * c["recall"] for c in cells.values())
    bench_adj_se = math.sqrt(sum((c["weight"] * c["se"]) ** 2 for c in cells.values()))
    s_share = bench["seen_share_norm"]
    weighted = (1 - s_share) * slices["unseen"]["recall"] + s_share * slices["seen"]["recall"]
    weighted_se = math.sqrt(((1 - s_share) * slices["unseen"]["se"]) ** 2
                            + (s_share * slices["seen"]["se"]) ** 2)
    f_u, f_s, f_all = (bench["filter_share_unseen"], bench["filter_share_seen"],
                       bench["filter_share"])

    def fadj(a: float, b: float, w: float) -> float:
        """Смешивает recall «с фильтром» (a) и «без» (b) с долей фильтра w из бенчмарка."""
        return w * a + (1 - w) * b

    return {
        "variant": val.variant,
        "k": k,
        "bench_adj": bench_adj,
        "bench_adj_se": bench_adj_se,
        "weighted": weighted,
        "weighted_se": weighted_se,
        "unseen": slices["unseen"]["recall"],
        "seen": slices["seen"]["recall"],
        "unseen_fadj": fadj(cells["unseen_filter"]["recall"], cells["unseen_nofilter"]["recall"], f_u),
        "seen_fadj": fadj(cells["seen_filter"]["recall"], cells["seen_nofilter"]["recall"], f_s),
        "filter_adj": fadj(slices["filter"]["recall"], slices["no_filter"]["recall"], f_all),
        "slices": slices,
        "cells": cells,
        "item_in_rest": hit_rest / tot_rest if tot_rest else math.nan,
        "item_new": hit_new / tot_new if tot_new else math.nan,
        "n_item_in_rest": tot_rest,
        "n_item_new": tot_new,
        "mean_len": float(lens.mean()),
        "share_short": float((lens < k).mean()),
        "n_missing": n_missing,
        "warnings": warnings,
    }


def format_report(res: dict) -> str:
    """Markdown-таблицы метрик для отчётов этапов и EXPERIMENTS.md.

    Вход: результат evaluate. Выход: многострочная строка markdown.
    """
    def f(x: float) -> str:
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

    s, c = res["slices"], res["cells"]
    lines = [
        f"**Recall@{res['k']}, вариант `{res['variant']}`**",
        "",
        "| метрика | recall | ±SE | n |",
        "|---|---|---|---|",
        f"| **bench_adj** | **{f(res['bench_adj'])}** | {f(res['bench_adj_se'])} | {s['all']['n']} |",
        f"| weighted | {f(res['weighted'])} | {f(res['weighted_se'])} | {s['all']['n']} |",
        f"| unseen | {f(s['unseen']['recall'])} | {f(s['unseen']['se'])} | {s['unseen']['n']} |",
        f"| seen | {f(s['seen']['recall'])} | {f(s['seen']['se'])} | {s['seen']['n']} |",
        f"| unseen, перевзв. по фильтру | {f(res['unseen_fadj'])} | | |",
        f"| seen, перевзв. по фильтру | {f(res['seen_fadj'])} | | |",
        f"| всё, перевзв. по фильтру | {f(res['filter_adj'])} | | |",
        f"| с фильтром | {f(s['filter']['recall'])} | {f(s['filter']['se'])} | {s['filter']['n']} |",
        f"| без фильтра | {f(s['no_filter']['recall'])} | {f(s['no_filter']['se'])} | {s['no_filter']['n']} |",
        f"| топ-локации | {f(s['top_loc']['recall'])} | {f(s['top_loc']['se'])} | {s['top_loc']['n']} |",
        f"| остальные локации | {f(s['other_loc']['recall'])} | {f(s['other_loc']['se'])} | {s['other_loc']['n']} |",
        f"| объявления из остатка (micro) | {f(res['item_in_rest'])} | | {res['n_item_in_rest']} |",
        f"| новые объявления (micro) | {f(res['item_new'])} | | {res['n_item_new']} |",
        "",
        "| ячейка | вес бенчмарка | recall | ±SE | n |",
        "|---|---|---|---|---|",
    ]
    for name, cell in c.items():
        lines.append(f"| {name} | {cell['weight']:.3f} | {f(cell['recall'])} | "
                     f"{f(cell['se'])} | {cell['n']} |")
    lines += ["", f"Средняя длина списка: {res['mean_len']:.1f}; "
                  f"доля списков короче {res['k']}: {res['share_short']:.1%}"]
    lines += [f"⚠ {w}" for w in res["warnings"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI: сборка и самопроверка
# --------------------------------------------------------------------------

def _print_build_summary(meta: dict) -> None:
    """Печатает сводку после --build: веса бенчмарка и размеры срезов."""
    b = meta["bench"]
    print(f"Текстов train: {meta['n_texts']}; подходят в unseen: {meta['n_eligible_unseen']}, "
          f"в seen: {meta['n_eligible_seen']}; unseen-тексты занимают "
          f"{meta['rows_share_removed_unseen']:.1%} строк train")
    print("Бенчмарк: seen(norm)={:.3f} seen(exact)={:.3f} фильтр={:.3f} "
          "(seen {:.3f} / unseen {:.3f}), память у seen={:.3f}".format(
              b["seen_share_norm"], b["seen_share_exact"], b["filter_share"],
              b["filter_share_seen"], b["filter_share_unseen"], b["seen_memory_share"]))
    print("Веса ячеек:", {k: round(v, 3) for k, v in b["cell_weights"].items()})
    for v, st in meta["variants"].items():
        print(f"\n[{v}] запросов={st['n_queries']} по ячейкам={st['n_by_cell']}")
        print(f"  фильтр по срезам={ {k: round(x, 3) for k, x in st['filter_share_by_slice'].items()} }"
              f" память у seen={st['seen_memory_share']:.3f}")
        print(f"  |rel|: {st['rel_per_query']}  rel-объявлений в остатке={st['rel_items_in_rest_share']:.3f}")
        print(f"  строк остатка train={st['train_rest_rows']} hash={st['hash']}")
        small = {c: n for c, n in st["n_by_cell"].items() if n < meta["validation_cfg"]["min_cell_n"]}
        if small:
            print(f"  ⚠ ячейки меньше min_cell_n: {small}")


def selftest(cfg: dict) -> int:
    """Проверки evaluate на тривиальных кандидатах и детерминизма сборки.

    1) все релевантные (k = max|rel|) -> ровно 1.0 во всех срезах;
    2) пустые списки -> 0.0; 3) половина релевантных -> строго между 0 и 1;
    4) случайные 50 объявлений корпуса -> ≈0; 5) лишний query_id -> ошибка;
    6) повторная сборка с тем же seed -> те же хэши срезов.

    Вход: конфиг. Выход: код возврата (0 — всё прошло).
    """
    failures = []

    def check(name: str, cond: bool) -> None:
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    corpus = io.load_items(cfg, ["item_id"])["item_id"].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    for v in VARIANTS:
        val = load_val(cfg, v)
        max_rel = max(len(r) for r in val.rel.values())
        perfect = {qid: sorted(r) for qid, r in val.rel.items()}
        r = evaluate(perfect, val, k=max_rel)
        main = [r["bench_adj"], r["weighted"], r["unseen"], r["seen"], r["filter_adj"],
                r["item_in_rest"], r["item_new"]] + [x["recall"] for x in r["slices"].values()]
        check(f"[{v}] все релевантные -> 1.0", all(abs(x - 1) < 1e-12 for x in main))
        ceiling = evaluate(perfect, val)["bench_adj"]
        print(f"      потолок bench_adj@{val.k} из-за |rel|>{val.k}: {ceiling:.4f}")

        r = evaluate({qid: [] for qid in val.rel}, val)
        check(f"[{v}] пустые списки -> 0.0", r["bench_adj"] == 0 and r["slices"]["all"]["recall"] == 0)

        half = {qid: sorted(rel)[: len(rel) // 2] for qid, rel in val.rel.items()}
        r = evaluate(half, val, k=max_rel)
        check(f"[{v}] половина релевантных -> (0, 1)", 0 < r["bench_adj"] < 1)

        rand = {qid: list(rng.choice(corpus, size=50, replace=False)) for qid in val.rel}
        r = evaluate(rand, val)
        check(f"[{v}] случайные 50 -> < 0.01 (получено {r['bench_adj']:.5f})", r["bench_adj"] < 0.01)

        try:
            evaluate({"не_существует": []}, val)
            check(f"[{v}] лишний query_id -> ошибка", False)
        except ValueError:
            check(f"[{v}] лишний query_id -> ошибка", True)

    print("Пересборка срезов для проверки детерминизма…")
    saved = load_val(cfg, "ctx").meta["variants"]
    rebuilt = build_splits(cfg)["meta"]["variants"]
    for v in VARIANTS:
        check(f"[{v}] детерминизм: hash {saved[v]['hash']} == {rebuilt[v]['hash']}",
              saved[v]["hash"] == rebuilt[v]["hash"])

    print("\nИТОГ:", "OK" if not failures else f"{len(failures)} проверок не прошли")
    return 0 if not failures else 1


def main() -> None:
    """CLI: --build (построить и сохранить срезы), --selftest (самопроверка)."""
    parser = argparse.ArgumentParser(description="Валидационные срезы и Recall@K")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--build", action="store_true", help="построить и сохранить срезы")
    parser.add_argument("--selftest", action="store_true", help="проверки evaluate и детерминизма")
    args = parser.parse_args()
    cfg = io.load_config(args.config)
    if args.build:
        splits = build_splits(cfg)
        save_splits(cfg, splits)
        _print_build_summary(splits["meta"])
    if args.selftest:
        sys.exit(selftest(cfg))
    if not (args.build or args.selftest):
        parser.print_help()


if __name__ == "__main__":
    main()
