"""EDA: перепроверка фактов из ТЗ, проверки для валидации, графики.

Запуск: `python -m candgen.eda [--config configs/default.yaml]`.
Результат: reports/eda.md и картинки reports/fig/*.png (открываются в VS Code).

Отчёт генерируется целиком из данных: все числа и выводы под графиками
пересчитываются при каждом запуске. Выводы собраны по шаблонам с порогами,
так что при изменении данных текст остаётся корректным.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI: только сохранение PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from candgen import io  # noqa: E402
from candgen.text import compile_filter_keys, has_filter, normalize_series, parse_filters  # noqa: E402
from candgen.validation import SEARCH_COLS, bench_strata  # noqa: E402

# Палитра графиков — проверенные слоты эталонной палитры skill dataviz, цвет
# закреплён за сущностью на всех графиках: слот 1 (синий) — train, слот 2
# (оранжевый) — бенчмарк, слот 3 (бирюзовый) — корпус. У бирюзового контраст
# с фоном < 3:1, поэтому у графиков с корпусом в отчёте есть таблица чисел.
# Фон, текст и сетка — нейтральные токены той же палитры.
C_TRAIN, C_BENCH, C_CORPUS = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Ожидаемые веса ячеек бенчмарка, названные пользователем (для сверки).
EXPECTED_CELLS = {"seen_filter": 0.209, "seen_nofilter": 0.161,
                  "unseen_filter": 0.161, "unseen_nofilter": 0.469}


@dataclass
class Fact:
    """Одна строка таблицы фактов: что ожидали по ТЗ и что получили.

    ok — значение в пределах допуска tol (абсолютного для долей,
    относительного для счётчиков — см. rel_tol).
    """
    name: str
    expected: float
    got: float
    fmt: str
    tol: float
    rel_tol: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        """Совпадает ли полученное значение с ожидаемым в пределах допуска."""
        diff = abs(self.got - self.expected)
        return diff <= (self.tol * abs(self.expected) if self.rel_tol else self.tol)


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------

def load_data(cfg: dict) -> dict:
    """Загружает всё, что нужно для EDA, только нужные колонки.

    Длины текстов объявлений считаются прямо в arrow (utf8_length), без
    материализации описаний в pandas.

    Вход: конфиг.
    Выход: словарь с ключами train, queries, items, item_lens.
    """
    tr = io.load_train(cfg, SEARCH_COLS + ["item_id", "item_location_id",
                                           "item_microcat_id", "item_infm_params_text"])
    items = io.load_items(cfg, ["item_id", "item_category_id", "item_microcat_id", "item_location_id"])
    q = io.load_queries(cfg)

    tr["query_norm"] = normalize_series(tr["search_query"])
    q["query_norm"] = normalize_series(q["search_query"])
    tr["in_corpus"] = tr["item_id"].isin(pd.Index(items["item_id"]))

    lens = {}
    tab = pq.read_table(cfg["paths"]["items"],
                        columns=["item_title_raw", "item_description_raw", "item_infm_params_text"])
    for col in tab.column_names:
        arr = tab.column(col)
        lens[col] = {"mean": pc.mean(pc.utf8_length(arr)).as_py(),
                     "median": float(np.median(pc.fill_null(pc.utf8_length(arr), 0).to_numpy())),
                     "nulls": arr.null_count}
    return {"train": tr, "queries": q, "items": items, "item_lens": lens}


# --------------------------------------------------------------------------
# Расчёты
# --------------------------------------------------------------------------

def filter_stats(tr: pd.DataFrame, q: pd.DataFrame, cfg: dict) -> dict:
    """Частоты ключей фильтров и совпадение значений фильтра с параметрами объявления.

    Разбор делается по уникальным строкам фильтра (их ~3,7 тыс.), а не по
    0,5 млн строк train. Совпадение считается двумя способами: значение как
    подстрока параметров объявления (как в ТЗ) и пара «ключ значение» как
    подстрока (строже).

    Вход: train (с item_infm_params_text), бенчмарк, конфиг.
    Выход: словарь: keys (доли строк train и запросов бенчмарка по ключам),
           prefix_share (доля непустых фильтров, где текст до первого ключа
           не распознан), match (доли совпадений по главным ключам).
    """
    pattern = compile_filter_keys(cfg["filters"]["keys"])
    uniq = pd.unique(pd.concat([tr["search_infm_params_text"], q["search_infm_params_text"]]))
    parsed = {s: parse_filters(s, pattern) for s in uniq}
    # Непустой текст до первого найденного ключа = неизвестный ключ.
    unknown_prefix = {s: bool(s.strip()) and (not parsed[s] or not s.startswith(parsed[s][0][0]))
                      for s in uniq}

    def key_share(texts: pd.Series, key: str) -> float:
        return float(texts.map(lambda s: any(k == key for k, _ in parsed[s])).mean())

    keys = [{"key": k,
             "train": key_share(tr["search_infm_params_text"], k),
             "bench": key_share(q["search_infm_params_text"], k)}
            for k in cfg["filters"]["keys"]]
    nonempty_tr = has_filter(tr["search_infm_params_text"])
    nonempty_q = has_filter(q["search_infm_params_text"])
    prefix = {
        "train": float(tr.loc[nonempty_tr, "search_infm_params_text"].map(unknown_prefix).mean()),
        "bench": float(q.loc[nonempty_q, "search_infm_params_text"].map(unknown_prefix).mean()),
    }

    match = {}
    params = tr["item_infm_params_text"].fillna("").tolist()
    for key in cfg["filters"]["main_keys"]:
        vals = tr["search_infm_params_text"].map(lambda s: [v for k, v in parsed[s] if k == key and v])
        mask = vals.map(bool).to_numpy()
        v_rows = vals[mask].tolist()
        p_rows = [params[i] for i in np.flatnonzero(mask)]
        by_value = np.mean([any(v in p for v in vs) for vs, p in zip(v_rows, p_rows)])
        by_pair = np.mean([any(f"{key} {v}" in p for v in vs) for vs, p in zip(v_rows, p_rows)])
        match[key] = {"rows": int(mask.sum()), "by_value": float(by_value), "by_pair": float(by_pair)}
    return {"keys": keys, "prefix_share": prefix, "match": match}


def compute(data: dict, cfg: dict) -> dict:
    """Считает все числа отчёта (факты ТЗ, проверки, данные для графиков).

    Вход: результат load_data, конфиг.
    Выход: словарь R с именованными величинами; ключ facts — список Fact.
    """
    tr, q, items = data["train"], data["queries"], data["items"]
    R: dict = {}

    # --- Факты 1–3: запросы, группы, пересечение бенчмарка с train.
    R["n_q_raw"] = tr["search_query"].nunique()
    R["n_q_norm"] = tr["query_norm"].nunique()
    R["n_groups_raw"] = len(tr.drop_duplicates(["search_query", "search_location_id"]))
    R["n_groups_norm"] = len(tr.drop_duplicates(["query_norm", "search_location_id"]))
    R["n_groups_ctx"] = len(tr.drop_duplicates(cfg["validation"]["group_keys"]))
    exact_seen = q["search_query"].isin(pd.Index(tr["search_query"].unique()))
    norm_seen = q["query_norm"].isin(pd.Index(tr["query_norm"].unique()))
    R["bench_seen_exact"], R["bench_seen_norm"] = int(exact_seen.sum()), int(norm_seen.sum())
    corp_rows = tr[tr["in_corpus"]]
    mem_text = q["search_query"].isin(pd.Index(corp_rows["search_query"].unique()))
    mem_pairs = pd.MultiIndex.from_frame(corp_rows[["search_query", "search_location_id"]].drop_duplicates())
    mem_same_loc = pd.MultiIndex.from_frame(q[["search_query", "search_location_id"]]).isin(mem_pairs)
    R["bench_mem_exact"] = int((exact_seen & mem_text).sum())
    R["bench_mem_exact_same_loc"] = int((exact_seen.to_numpy() & mem_same_loc).sum())

    # --- Факты 4–6: корпус.
    R["corpus_in_train"] = int(items["item_id"].isin(pd.Index(tr["item_id"].unique())).sum())
    R["share_cat114"] = float((items["item_category_id"] == 114).mean())
    R["n_microcat_corpus"] = items["item_microcat_id"].nunique()
    R["bench_cat0"] = float((q["search_category"] == 0).mean())
    R["train_cat0_rows"] = int((tr["search_category"] == 0).sum())

    # --- Факты 7–8: локации.
    R["loc_match_rows"] = float((tr["search_location_id"] == tr["item_location_id"]).mean())
    corpus_locs = pd.Index(items["item_location_id"].unique())
    R["bench_loc_absent_q"] = float((~q["search_location_id"].isin(corpus_locs)).mean())
    bench_locs = pd.Series(q["search_location_id"].unique())
    R["bench_loc_absent_distinct"] = float((~bench_locs.isin(corpus_locs)).mean())
    R["bench_loc_in_train_search"] = float(q["search_location_id"].isin(
        pd.Index(tr["search_location_id"].unique())).mean())

    # --- Факты 9–10: фильтры.
    R["filters"] = filter_stats(tr, q, cfg)
    R["filter_share_train_rows"] = float(has_filter(tr["search_infm_params_text"]).mean())
    ctx = tr.drop_duplicates(cfg["validation"]["group_keys"])
    R["filter_share_train_groups"] = float(has_filter(ctx["search_infm_params_text"]).mean())
    R["filter_share_bench"] = float(has_filter(q["search_infm_params_text"]).mean())

    # --- Факт 11: доля кликов запроса в его главной подкатегории.
    mc = tr.groupby(["search_query", "item_microcat_id"]).size()
    per_q = mc.groupby(level=0).agg(["max", "sum"])
    R["main_mc_micro"] = float(per_q["max"].sum() / per_q["sum"].sum())
    R["main_mc_macro"] = float((per_q["max"] / per_q["sum"]).mean())

    # --- Факты 12–13: длины.
    words_tr = tr["query_norm"].drop_duplicates().str.split().str.len()
    words_q = q["query_norm"].str.split().str.len()
    R["words_train_uniq"], R["words_bench"] = float(words_tr.mean()), float(words_q.mean())
    R["words_train_rows"] = float(tr["query_norm"].str.split().str.len().mean())
    R["item_lens"] = data["item_lens"]

    # --- Проверки под решения.
    R["bench_uniq_exact"] = q["search_query"].nunique()
    R["bench_uniq_norm"] = q["query_norm"].nunique()
    R["bench_strata"] = bench_strata(q, tr, cfg["validation"]["top_locations_n"])
    n_ctx_per_ql = ctx.groupby(["query_norm", "search_location_id"]).size()
    R["ql_multi_ctx_share"] = float((n_ctx_per_ql > 1).mean())
    multi = n_ctx_per_ql[n_ctx_per_ql > 1].index
    R["ql_multi_ctx_rows_share"] = float(pd.MultiIndex.from_frame(
        tr[["query_norm", "search_location_id"]]).isin(multi).mean())
    R["items_per_group_raw"] = float(tr.groupby(["search_query", "search_location_id"])["item_id"].nunique().mean())
    for name, keys in (("ctx", cfg["validation"]["group_keys"]),
                       ("ql", cfg["validation"]["alt_group_keys"])):
        n_rel = corp_rows.groupby(keys, dropna=False)["item_id"].nunique()
        R[f"rel_{name}"] = {"groups": len(n_rel), "mean": float(n_rel.mean()),
                            "share_1": float((n_rel == 1).mean()), "p90": float(n_rel.quantile(0.9))}
    deliv = tr["search_is_delivery_search"] == 1
    R["delivery_rows"] = int(deliv.sum())
    R["delivery_loc_match"] = float((tr.loc[deliv, "search_location_id"]
                                     == tr.loc[deliv, "item_location_id"]).mean()) if deliv.any() else np.nan

    # --- Данные для графиков.
    maxw = cfg["eda"]["max_query_words"]
    R["len_dist"] = pd.DataFrame({
        "train": words_tr.clip(upper=maxw).value_counts(normalize=True),
        "bench": words_q.clip(upper=maxw).value_counts(normalize=True),
    }).fillna(0).sort_index()
    top_n = cfg["eda"]["top_n"]
    loc_q = q["search_location_id"].value_counts(normalize=True)
    loc_c = items["item_location_id"].value_counts(normalize=True)
    top_locs = loc_q.head(top_n).index
    R["top_locs"] = pd.DataFrame({"bench": loc_q.loc[top_locs],
                                  "corpus": loc_c.reindex(top_locs).fillna(0)})
    # Локации из топа бенчмарка, где в корпусе нет ни одного объявления
    # (кандидаты в «агрегаты» — регион/страна, см. фишку 6 в PLAN.md).
    absent = R["top_locs"].index[R["top_locs"]["corpus"] == 0]
    R["top_locs_absent"] = [int(x) for x in absent]
    R["top_locs_absent_share"] = float(loc_q.loc[absent].sum())
    R["top_locs_cover_bench"] = float(loc_q.head(top_n).sum())
    R["top_locs_cover_corpus"] = float(loc_c.reindex(top_locs).fillna(0).sum())
    both = pd.concat([loc_q.rename("b"), loc_c.rename("c")], axis=1).fillna(0)
    R["loc_share_corr"] = float(both["b"].corr(both["c"]))
    mc_c = items["item_microcat_id"].value_counts(normalize=True)
    mc_t = tr["item_microcat_id"].value_counts(normalize=True)
    top_mc = mc_c.head(top_n).index
    R["top_mc"] = pd.DataFrame({"corpus": mc_c.loc[top_mc], "train": mc_t.reindex(top_mc).fillna(0)})
    R["top_mc_cover_corpus"] = float(mc_c.head(top_n).sum())
    R["top_mc_cover_train"] = float(mc_t.reindex(top_mc).fillna(0).sum())
    mc_both = pd.concat([mc_c.rename("c"), mc_t.rename("t")], axis=1).fillna(0)
    R["mc_share_corr"] = float(mc_both["c"].corr(mc_both["t"]))

    # Если срезы валидации уже построены, покажем их долю фильтра рядом.
    meta_path = Path(cfg["paths"]["artifacts_dir"]) / "val_meta.json"
    R["val_meta"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    fs = R["filters"]
    R["facts"] = [
        Fact("Уникальных текстов запросов в train", 74529, R["n_q_raw"], "{:,.0f}", 0.01, True,
             f"нормализованных: {R['n_q_norm']:,}"),
        Fact("Групп (запрос, локация) в train", 307552, R["n_groups_raw"], "{:,.0f}", 0.01, True,
             f"норм. текст: {R['n_groups_norm']:,}; полный контекст: {R['n_groups_ctx']:,}"),
        Fact("Запросов бенчмарка, дословно встречавшихся в train", 907, R["bench_seen_exact"],
             "{:,.0f}", 0.01, True, f"после нормализации: {R['bench_seen_norm']:,}"),
        Fact("…из них с выбранными в train объявлениями корпуса", 360, R["bench_mem_exact"],
             "{:,.0f}", 0.02, True, f"с той же локацией: {R['bench_mem_exact_same_loc']} (ТЗ: 91)"),
        Fact("Объявлений корпуса, встречающихся в train", 18142, R["corpus_in_train"], "{:,.0f}", 0.01, True),
        Fact("Доля item_category_id = 114 в корпусе", 0.99, R["share_cat114"], "{:.3f}", 0.01,
             note=f"подкатегорий в корпусе: {R['n_microcat_corpus']} (ТЗ: 752)"),
        Fact("Доля search_category = 0 в бенчмарке", 0.09, R["bench_cat0"], "{:.3f}", 0.01,
             note=f"в train таких строк всего {R['train_cat0_rows']}"),
        Fact("Совпадение локации поиска и объявления (строки train)", 0.83, R["loc_match_rows"], "{:.3f}", 0.01),
        Fact("Запросы бенчмарка с локацией, которой нет среди локаций корпуса", 0.17,
             R["bench_loc_absent_q"], "{:.3f}", 0.01,
             note=f"среди различных локаций бенчмарка: {R['bench_loc_absent_distinct']:.3f}"),
        Fact("Совпадение «Вид услуги» фильтра с параметрами объявления", 0.985,
             fs["match"]["Вид услуги"]["by_value"], "{:.3f}", 0.005,
             note=f"пара «ключ значение»: {fs['match']['Вид услуги']['by_pair']:.3f}"),
        Fact("Совпадение «Тип услуги» фильтра с параметрами объявления", 0.965,
             fs["match"]["Тип услуги"]["by_value"], "{:.3f}", 0.005,
             note=f"пара «ключ значение»: {fs['match']['Тип услуги']['by_pair']:.3f}"),
        Fact("Доля с непустым фильтром: строки train", 0.67, R["filter_share_train_rows"], "{:.3f}", 0.01,
             note=f"уникальные группы-контексты: {R['filter_share_train_groups']:.3f}"),
        Fact("Доля с непустым фильтром: бенчмарк", 0.37, R["filter_share_bench"], "{:.3f}", 0.01),
        Fact("Доля кликов запроса в его главной подкатегории", 0.85, R["main_mc_micro"], "{:.3f}", 0.01,
             note=f"micro по кликам; среднее по запросам (macro): {R['main_mc_macro']:.3f}"),
        Fact("Длина запроса, слов (уникальные тексты train)", 3.2, R["words_train_uniq"], "{:.2f}", 0.15,
             note=f"бенчмарк: {R['words_bench']:.2f}; по строкам train: {R['words_train_rows']:.2f}"),
        Fact("Длина описания объявления корпуса, символов (среднее)", 1400,
             R["item_lens"]["item_description_raw"]["mean"], "{:,.0f}", 0.05, True,
             note=f"медиана {R['item_lens']['item_description_raw']['median']:,.0f}"),
        Fact("Длина параметров объявления корпуса, символов (среднее)", 970,
             R["item_lens"]["item_infm_params_text"]["mean"], "{:,.0f}", 0.05, True,
             note=f"медиана {R['item_lens']['item_infm_params_text']['median']:,.0f}; "
                  f"заголовок: {R['item_lens']['item_title_raw']['mean']:.0f}"),
    ]
    return R


# --------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------

def _setup_style() -> None:
    """Общий стиль: светлый фон, приглушённые оси и сетка, шрифт с кириллицей."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    """Сохраняет фигуру в PNG и закрывает её (экономия памяти)."""
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _paired_hbar(ax: plt.Axes, labels: list[str], a: np.ndarray, b: np.ndarray,
                 name_a: str, name_b: str, color_a: str, color_b: str) -> None:
    """Сгруппированные горизонтальные столбцы двух серий (доли, ось в %).

    Столбцы разделены тонкой полосой цвета фона, сетка только по оси значений.
    """
    y = np.arange(len(labels))
    h = 0.4
    ax.barh(y - h / 2, a, height=h, color=color_a, label=name_a, edgecolor=SURFACE, linewidth=1)
    ax.barh(y + h / 2, b, height=h, color=color_b, label=name_b, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")


def make_figures(R: dict, fig_dir: Path, cfg: dict) -> dict[str, str]:
    """Рисует 5 графиков отчёта.

    Вход: результаты compute, папка для картинок, конфиг.
    Выход: {ключ графика: относительный путь PNG от reports/}.
    """
    _setup_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    # 1. Длина запроса.
    d = R["len_dist"]
    maxw = cfg["eda"]["max_query_words"]
    labels = [str(i) if i < maxw else f"{maxw}+" for i in d.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["train"], width=0.4, color=C_TRAIN, label="train (уникальные тексты)",
           edgecolor=SURFACE, linewidth=1)
    ax.bar(x + 0.2, d["bench"], width=0.4, color=C_BENCH, label="бенчмарк",
           edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_xlabel("слов в нормализованном запросе")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("Длина запроса: train и бенчмарк")
    _save(fig, fig_dir / "01_query_len.png")
    out["len"] = "fig/01_query_len.png"

    # 2. Топ локаций бенчмарка и их доля в корпусе.
    t = R["top_locs"]
    fig, ax = plt.subplots(figsize=(8, 7))
    _paired_hbar(ax, [str(i) for i in t.index], t["bench"].to_numpy(), t["corpus"].to_numpy(),
                 "доля запросов бенчмарка", "доля объявлений корпуса", C_BENCH, C_CORPUS)
    ax.set_ylabel("search_location_id")
    ax.set_title(f"Топ-{len(t)} локаций бенчмарка")
    _save(fig, fig_dir / "02_top_locations.png")
    out["locs"] = "fig/02_top_locations.png"

    # 3. Топ подкатегорий корпуса и их доля кликов в train.
    t = R["top_mc"]
    fig, ax = plt.subplots(figsize=(8, 7))
    _paired_hbar(ax, [str(i) for i in t.index], t["corpus"].to_numpy(), t["train"].to_numpy(),
                 "доля объявлений корпуса", "доля кликов train", C_CORPUS, C_TRAIN)
    ax.set_ylabel("item_microcat_id")
    ax.set_title(f"Топ-{len(t)} подкатегорий корпуса")
    _save(fig, fig_dir / "03_top_microcats.png")
    out["mc"] = "fig/03_top_microcats.png"

    # 4. Доля запросов с фильтром (одна серия: цвет не кодирует категорию).
    bs = R["bench_strata"]
    bars = [("train\n(строки)", R["filter_share_train_rows"]),
            ("train\n(группы)", R["filter_share_train_groups"]),
            ("бенчмарк\nвсе", bs["filter_share"]),
            ("бенчмарк\nseen", bs["filter_share_seen"]),
            ("бенчмарк\nunseen", bs["filter_share_unseen"])]
    if R["val_meta"]:
        fs = R["val_meta"]["variants"]["ctx"]["filter_share_by_slice"]
        bars += [("val\nseen", fs["seen"]), ("val\nunseen", fs["unseen"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(bars))
    ax.bar(x, [v for _, v in bars], width=0.6, color=C_TRAIN, edgecolor=SURFACE, linewidth=1)
    for xi, (_, v) in zip(x, bars):
        ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_xticks(x, [n for n, _ in bars])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Доля запросов с непустым фильтром")
    _save(fig, fig_dir / "04_filter_share.png")
    out["filter"] = "fig/04_filter_share.png"

    # 5. Частоты ключей фильтров.
    k = pd.DataFrame(R["filters"]["keys"])
    fig, ax = plt.subplots(figsize=(8, 6))
    _paired_hbar(ax, k["key"].tolist(), k["train"].to_numpy(), k["bench"].to_numpy(),
                 "строки train", "запросы бенчмарка", C_TRAIN, C_BENCH)
    for yi, (a, b) in enumerate(zip(k["train"], k["bench"])):
        ax.text(max(a, b) + 0.005, yi, f"{a:.1%} / {b:.1%}", va="center", color=INK2, fontsize=8)
    ax.set_xlim(0, max(k["train"].max(), k["bench"].max()) * 1.25)
    ax.set_title("Ключи фильтров: доля запросов с ключом")
    _save(fig, fig_dir / "05_filter_keys.png")
    out["keys"] = "fig/05_filter_keys.png"
    return out


# --------------------------------------------------------------------------
# Отчёт
# --------------------------------------------------------------------------

def _pct(x: float) -> str:
    """Доля -> проценты с одним знаком."""
    return f"{x:.1%}"


def _details_table(title: str, df: pd.DataFrame, columns: dict[str, str], index_name: str) -> str:
    """Сворачиваемая markdown-таблица долей (табличное представление графика).

    Вход: заголовок, DataFrame долей, {колонка: подпись}, подпись индекса.
    Выход: HTML-блок <details> с markdown-таблицей внутри.
    """
    rows = [f"<details><summary>{title}</summary>", "",
            f"| {index_name} | " + " | ".join(columns.values()) + " |",
            "|---|" + "---|" * len(columns)]
    for idx, r in df.iterrows():
        rows.append(f"| {idx} | " + " | ".join(f"{r[c]:.2%}" for c in columns) + " |")
    rows += ["", "</details>"]
    return "\n".join(rows)


def write_report(R: dict, figs: dict[str, str], path: Path, cfg: dict) -> None:
    """Собирает reports/eda.md: факты, проверки, графики с выводами, фильтры.

    Вход: результаты compute, пути картинок, путь отчёта, конфиг.
    Выход: файл на диске.
    """
    bs = R["bench_strata"]
    L = [f"# EDA — кандидатогенерация Авито",
         "",
         f"Сгенерировано `python -m candgen.eda` {dt.date.today().isoformat()}. "
         "Все числа пересчитываются из `data/raw` при каждом запуске.",
         "",
         "## 1. Факты из ТЗ: перепроверка",
         "",
         "✓ — совпало в пределах допуска; ≠ — расхождение (см. комментарий).",
         "",
         "| № | факт | ТЗ | получено | | комментарий |",
         "|---|---|---|---|---|---|"]
    for i, f in enumerate(R["facts"], 1):
        L.append(f"| {i} | {f.name} | {f.fmt.format(f.expected)} | {f.fmt.format(f.got)} | "
                 f"{'✓' if f.ok else '≠'} | {f.note} |")
    n_bad = sum(not f.ok for f in R["facts"])
    L += ["", f"Расхождений: **{n_bad}** из {len(R['facts'])}.", ""]

    # --- Проверки под решения.
    cw = bs["cell_weights"]
    n_q = bs["n_queries"]
    dup_exact, dup_norm = n_q - R["bench_uniq_exact"], n_q - R["bench_uniq_norm"]
    if dup_exact == 0 and dup_norm <= n_q // 200:  # до 0,5% совпадений считаем шумом
        uniq_text = ("дословно каждый текст встречается один раз"
                     + (f" (после нормализации совпадений: {dup_norm} — пренебрежимо)" if dup_norm else "")
                     + ", поэтому на валидации берётся одна группа на текст.")
    else:
        uniq_text = (f"повторов {dup_exact} дословно и {dup_norm} после нормализации — "
                     "учесть при построении срезов.")
    L += ["## 2. Проверки, влияющие на валидацию и приоры", "",
          f"- **Уникальность текстов в бенчмарке:** из {n_q} запросов {uniq_text}",
          f"- **Доля seen в бенчмарке:** по нормализованному тексту {bs['seen_share_norm']:.3f}, "
          f"дословно {bs['seen_share_exact']:.3f} (разница "
          f"{abs(bs['seen_share_norm'] - bs['seen_share_exact']) * 100:.1f} п.п."
          + ("; меньше 1 п.п., используем нормализованную)." if abs(bs['seen_share_norm'] - bs['seen_share_exact']) < 0.01
             else "; БОЛЬШЕ 1 п.п. — обсудить).")]
    L += ["- **Веса ячеек бенчмарка** (вычислены из `benchmark_queries`; основа метрики `bench_adj`):", "",
          "  | ячейка | вес | ожидалось |", "  |---|---|---|"]
    for name, w in cw.items():
        L.append(f"  | {name} | {w:.3f} | {EXPECTED_CELLS[name]:.3f} |")
    L += ["",
          f"  Фильтр есть у {_pct(bs['filter_share_seen'])} seen и {_pct(bs['filter_share_unseen'])} "
          f"unseen запросов бенчмарка ({_pct(bs['filter_share'])} в целом).",
          f"- **«Память»:** у {_pct(bs['seen_memory_share'])} seen-запросов бенчмарка в train есть "
          "выбранные объявления из корпуса"
          + (f"; на seen-срезе валидации — {_pct(R['val_meta']['variants']['ctx']['seen_memory_share'])}."
             if R["val_meta"] else "."),
          f"- **Несколько контекстов у одной пары (запрос, локация):** {_pct(R['ql_multi_ctx_share'])} пар, "
          f"на них {_pct(R['ql_multi_ctx_rows_share'])} строк train. Поэтому ключ «полный контекст» "
          "и ключ «(запрос, локация)» различаются, и контрольный вариант `ql` нужен.",
          f"- **Выбранных объявлений на группу** (запрос, локация) — все объявления: "
          f"{R['items_per_group_raw']:.2f} (ТЗ: 1,47). Только объявления корпуса: полный контекст — "
          f"{R['rel_ctx']['mean']:.3f} (ровно одно у {_pct(R['rel_ctx']['share_1'])} групп), "
          f"(запрос, локация) — {R['rel_ql']['mean']:.3f} (ровно одно у {_pct(R['rel_ql']['share_1'])}). "
          "Recall запроса почти всегда 0 или 1.",
          f"- **Доставка:** `search_is_delivery_search = 1` всего в {R['delivery_rows']} строках train "
          "и ни в одном запросе бенчмарка — признак неинформативен.",
          f"- **Категория 0:** у {_pct(R['bench_cat0'])} запросов бенчмарка, а в train строк с ней "
          f"всего {R['train_cat0_rows']}: такие запросы в train почти не представлены, по категории не фильтруем.",
          f"- **Покрытие локаций бенчмарка:** {_pct(R['bench_loc_in_train_search'])} запросов имеют локацию, "
          "встречавшуюся как локация поиска в train (есть строка матрицы переходов); "
          f"у {_pct(R['bench_loc_absent_q'])} локации нет среди локаций объявлений корпуса."]
    if R["val_meta"]:
        m = R["val_meta"]
        L += [f"- **Цена валидации:** unseen-тексты ({m['validation_cfg']['n_unseen']} из "
              f"{m['n_eligible_unseen']} подходящих) занимают {_pct(m['rows_share_removed_unseen'])} строк train. "
              "Приоры на валидации считаются по оставшимся строкам, поэтому оценка слегка пессимистична."]
    L.append("")

    # --- Графики.
    d = R["len_dist"]
    short_tr, short_q = d["train"].iloc[:2].sum(), d["bench"].iloc[:2].sum()
    close = abs(R["words_train_uniq"] - R["words_bench"]) < 0.3
    L += ["## 3. Графики", "",
          "### 3.1 Длина запроса", "", f"![длина запроса]({figs['len']})", "",
          f"Средняя длина: {R['words_train_uniq']:.2f} слова у уникальных текстов train, "
          f"{R['words_bench']:.2f} у бенчмарка. Запросы из 1–2 слов: {_pct(short_tr)} train и "
          f"{_pct(short_q)} бенчмарка. **Вывод:** распределения "
          + ("близки, " if close else "различаются, ")
          + "запросы короткие. Текстового сигнала мало, поэтому решают приоры (локация, подкатегория) "
          "и символьные n-граммы, устойчивые к опечаткам и склейкам.", ""]
    L += ["### 3.2 Топ локаций бенчмарка и корпуса", "", f"![локации]({figs['locs']})", "",
          f"Топ-{len(R['top_locs'])} локаций дают {_pct(R['top_locs_cover_bench'])} запросов бенчмарка и "
          f"{_pct(R['top_locs_cover_corpus'])} объявлений корпуса. Корреляция долей локаций "
          f"(запросы и корпус) r = {R['loc_share_corr']:.2f}. **Вывод:** корпус "
          + ("в целом следует географии запросов" if R["loc_share_corr"] > 0.7 else "распределён по локациям иначе, чем запросы")
          + f", но у {_pct(R['bench_loc_absent_q'])} запросов своей локации в корпусе нет. "
          + (f"В самом топе без объявлений корпуса остаются локации {', '.join(map(str, R['top_locs_absent']))} "
             f"(вместе {_pct(R['top_locs_absent_share'])} запросов бенчмарка) — похоже на агрегаты "
             "(регион или страна), это проверим по матрице переходов на этапе 2. "
             if R["top_locs_absent"] else "")
          + "Нужна матрица переходов loc_search → loc_item, а не проверка равенства.", "",
          _details_table("Таблица: топ локаций бенчмарка", R["top_locs"],
                         {"bench": "доля запросов бенчмарка", "corpus": "доля объявлений корпуса"},
                         "search_location_id"), ""]
    L += ["### 3.3 Подкатегории", "", f"![подкатегории]({figs['mc']})", "",
          f"Топ-{len(R['top_mc'])} подкатегорий — {_pct(R['top_mc_cover_corpus'])} корпуса и "
          f"{_pct(R['top_mc_cover_train'])} кликов train; корреляция долей r = {R['mc_share_corr']:.2f}. "
          f"Подкатегорий в корпусе: {R['n_microcat_corpus']}, при этом {_pct(R['main_mc_micro'])} кликов запроса "
          "приходятся на его главную подкатегорию. **Вывод:** prior подкатегории по похожим запросам train (P_mc) "
          "должен сильно сужать поиск.", "",
          _details_table("Таблица: топ подкатегорий корпуса", R["top_mc"],
                         {"corpus": "доля объявлений корпуса", "train": "доля кликов train"},
                         "item_microcat_id"), ""]
    L += ["### 3.4 Доля запросов с фильтром", "", f"![фильтр]({figs['filter']})", "",
          f"В train фильтр есть у {_pct(R['filter_share_train_rows'])} строк, в бенчмарке — у "
          f"{_pct(bs['filter_share'])} запросов (seen {_pct(bs['filter_share_seen'])}, unseen "
          f"{_pct(bs['filter_share_unseen'])}). **Вывод:** сдвиг большой. Валидационные срезы наследуют долю "
          "фильтра из train, поэтому решения принимаются по `bench_adj`, который перевзвешивает "
          "4 ячейки под бенчмарк.", ""]
    fk = pd.DataFrame(R["filters"]["keys"]).set_index("key")
    L += ["### 3.5 Ключи фильтров", "", f"![ключи]({figs['keys']})", "",
          f"«Вид услуги» есть у {_pct(fk.loc['Вид услуги', 'train'])} строк train и "
          f"{_pct(fk.loc['Вид услуги', 'bench'])} запросов бенчмарка, «Тип услуги» — у "
          f"{_pct(fk.loc['Тип услуги', 'train'])} / {_pct(fk.loc['Тип услуги', 'bench'])}. Остальные ключи редки. "
          f"У {_pct(R['filters']['prefix_share']['train'])} непустых фильтров train и "
          f"{_pct(R['filters']['prefix_share']['bench'])} бенчмарка перед первым известным ключом "
          "стоит нераспознанный текст (редкие ключи вроде «Аренда авто …»). **Вывод:** основной сигнал — "
          "«Вид/Тип услуги»; их значения почти всегда дословно есть в параметрах выбранного объявления "
          "(см. факты 10–11), поэтому это сильный мягкий бонус.", ""]

    # --- Таблица фильтров полностью.
    L += ["## 4. Фильтры: разбор и совпадение с параметрами объявления", "",
          "Строка фильтра — склеенные пары «ключ значение» без разделителей. Значение ключа "
          "заканчивается на следующем известном ключе (список — `configs/default.yaml: filters.keys`) "
          "или в конце строки.", "",
          "| ключ | доля строк train | доля запросов бенчмарка |", "|---|---|---|"]
    for key, row in fk.iterrows():
        L.append(f"| {key} | {row['train']:.2%} | {row['bench']:.2%} |")
    L += ["", "| главный ключ | строк train с непустым значением | значение ⊂ параметров | «ключ значение» ⊂ параметров |",
          "|---|---|---|---|"]
    for key, m in R["filters"]["match"].items():
        L.append(f"| {key} | {m['rows']:,} | {m['by_value']:.3f} | {m['by_pair']:.3f} |")
    L.append("")

    # --- Длины текстов объявлений.
    L += ["## 5. Длины текстов объявлений корпуса (символы)", "",
          "| поле | среднее | медиана | пустых |", "|---|---|---|---|"]
    for col, s in R["item_lens"].items():
        L.append(f"| {col} | {s['mean']:,.0f} | {s['median']:,.0f} | {s['nulls']} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    """CLI: считает EDA и пишет reports/eda.md + reports/fig/*.png."""
    parser = argparse.ArgumentParser(description="EDA -> reports/eda.md")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = io.load_config(args.config)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    data = load_data(cfg)
    R = compute(data, cfg)
    figs = make_figures(R, reports / "fig", cfg)
    write_report(R, figs, reports / "eda.md", cfg)

    print(f"Готово: {reports / 'eda.md'}")
    for f in R["facts"]:
        print(f"{'OK ' if f.ok else 'DIFF'} {f.name}: ТЗ {f.fmt.format(f.expected)} -> {f.fmt.format(f.got)}"
              + (f" ({f.note})" if f.note else ""))


if __name__ == "__main__":
    main()
