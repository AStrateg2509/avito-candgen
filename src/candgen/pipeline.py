"""Пайплайн кандидатогенерации: компоненты для набора запросов + запуск на валидации.

Один и тот же код работает в двух режимах:
  * val   — запросы val-среза (вариант ctx или ql), train = остаток по маске среза;
  * bench — benchmark_queries, train = весь train (используется в submit.py).

CLI (валидация):
  python -m candgen.pipeline                      # веса из fusion.weights, полный отчёт
  python -m candgen.pipeline --ablation           # все наборы из experiments.ablation
  python -m candgen.pipeline --variant ql         # контрольный вариант срезов
  python -m candgen.pipeline --set fusion.weights.word=0.3 --set docs.clean_params=true
"""

from __future__ import annotations

import argparse
import copy
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from candgen import io
from candgen.fusion import Components, LinearScorer, quota_fuse, rrf_fuse
from candgen.priors import FilterPrior, LocationPrior, MicrocatPrior
from candgen.retrievers.dense import DenseModel, item_embeddings, train_item_embeddings
from candgen.retrievers.lexical import HashedTfidf, build_or_load_corpus_index, restrict_to_queries
from candgen.retrievers.memory import expansion_texts, graph_expansion_texts, memory_matrix
from candgen.text import Lemmatizer, build_doc_texts, normalize_series
from candgen.torch_utils import get_device
from candgen.validation import evaluate, format_report, load_train_rest_mask, load_val

ITEM_COLS = ["item_id", "item_location_id", "item_microcat_id", "item_title_raw",
             "item_infm_params_text", "item_description_raw", "item_latitude", "item_longitude"]
TRAIN_COLS = ["search_query", "search_location_id", "item_id", "item_location_id", "item_microcat_id"]


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Возвращает копию конфига с правками вида 'a.b.c=значение' (значение — YAML).

    Числовой сегмент пути — индекс в списке: 'docs.fields.1.max_chars=800'.
    Нужна для экспериментов без правки configs/default.yaml; итоговые
    значения всё равно фиксируются в конфиге после выбора.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        path, value = item.split("=", 1)
        node = cfg
        keys = [int(k) if k.isdigit() else k for k in path.split(".")]
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = yaml.safe_load(value)
    return cfg


def load_query_set(cfg: dict, mode: str, variant: str = "ctx") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Запросы и часть train для режима.

    Вход: конфиг; 'val' или 'bench'; вариант срезов для val.
    Выход: (запросы со схемой benchmark_queries + query_norm; train-часть
            с query_norm). Для val train — строго остаток по маске среза.
    """
    train = io.load_train(cfg, TRAIN_COLS)
    train["query_norm"] = normalize_series(train["search_query"])
    if mode == "val":
        q = pd.read_parquet(Path(cfg["paths"]["artifacts_dir"]) / f"val_queries_{variant}.parquet")
        q["query_id"] = q["query_id"].astype(str)
        train = train[load_train_rest_mask(cfg, variant)]
    elif mode == "bench":
        q = io.load_queries(cfg)
        q["query_norm"] = normalize_series(q["search_query"])
    else:
        raise ValueError(f"неизвестный режим {mode}")
    return q.reset_index(drop=True), train.reset_index(drop=True)


class Pipeline:
    """Корпус и его индексы (кэшируются) + подготовка компонентов для запросов."""

    def __init__(self, cfg: dict):
        """Загружает корпус; индексы строятся лениво при первом обращении."""
        self.cfg = cfg
        self.device = get_device(cfg)
        self.items = io.load_items(cfg, ITEM_COLS)
        self.item_ids = self.items["item_id"].to_numpy()
        self.lemmatizer = Lemmatizer(Path(cfg["paths"]["artifacts_dir"]) / "lemmas.parquet")
        self.filter_prior = FilterPrior(cfg["filters"], cfg["priors"]["filter"])
        self.item_params = self.items["item_infm_params_text"].fillna("").astype(str).tolist()
        self._docs: dict[tuple[str, bool], list[str]] = {}

    def doc_texts(self, src_cfg: dict) -> list[str]:
        """Тексты документов для источника, с кэшем в памяти.

        Поля берутся из src_cfg["fields"], если они заданы (например, только
        заголовок), иначе из docs.fields; очистка параметров общая (docs.clean_params).
        При src_cfg["lemmatize"] слова заменяются леммами.
        """
        docs_cfg = {**self.cfg["docs"], **({"fields": src_cfg["fields"]} if "fields" in src_cfg else {})}
        fkey = repr(docs_cfg["fields"])
        if (fkey, False) not in self._docs:
            self._docs[(fkey, False)] = build_doc_texts(self.items, docs_cfg)
        if src_cfg["lemmatize"] and (fkey, True) not in self._docs:
            self._docs[(fkey, True)] = self.lemmatizer.lemmatize_texts(self._docs[(fkey, False)])
        return self._docs[(fkey, src_cfg["lemmatize"])]

    def train_item_table(self) -> pd.DataFrame:
        """Уникальные объявления ВСЕГО train (id, подкатегория, поля dense-документа), по id.

        Эмбеддинги считаются по всем объявлениям train один раз и кэшируются.
        Какие из них реально доступны (остаток на валидации), решает
        graph_expansion_texts по переданной части train.
        """
        cols = ["item_id", "item_microcat_id"] + [f["col"] for f in self.cfg["dense"]["doc_fields"]]
        tr = io.load_train(self.cfg, list(dict.fromkeys(cols)))
        return tr.drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)

    def index(self, name: str) -> tuple:
        """(источник, матрица корпуса CSC) из кэша artifacts/lex или построенные заново.

        Полная матрица в памяти не держится: вызывающий вырезает нужные столбцы
        и отпускает её (char-матрица по длинным описаниям — несколько ГБ).
        """
        return build_or_load_corpus_index(self.cfg, name, self.doc_texts)

    def prepare(self, queries: pd.DataFrame, train: pd.DataFrame) -> Components:
        """Считает все компоненты скора для набора запросов.

        Вход: запросы (с query_norm) и часть train, по которой считаются приоры.
        Выход: Components с временем подготовки каждой части.
        """
        cfg, items, t = self.cfg, self.items, {}
        qn = queries["query_norm"].tolist()

        text = {}
        for name, src in cfg["lexical"]["sources"].items():
            t0 = time.time()
            model, X = self.index(name)
            q_texts = self.lemmatizer.lemmatize_texts(qn) if src["lemmatize"] else qn
            text[name] = restrict_to_queries(X, model.transform(q_texts))
            del X  # полная матрица корпуса больше не нужна
            t[f"text_{name}"] = time.time() - t0
        self._docs.clear()  # тексты документов тоже (нужны только при построении индексов)

        t0 = time.time()
        loc_prior = LocationPrior(cfg["priors"]["loc"]).fit(train)
        P, uq, ui, n_fallback = loc_prior.matrix(queries["search_location_id"].to_numpy(),
                                                 items["item_location_id"].to_numpy())
        q_loc_idx = np.searchsorted(uq, queries["search_location_id"].to_numpy())
        item_loc_idx = np.searchsorted(ui, items["item_location_id"].to_numpy())
        # Фишки 2 и 6: гео-данные для сглаживания и тип локации запроса.
        loc_geo = loc_prior.geo_matrices(uq, ui, items)
        q_is_agg = loc_geo["self_share"][q_loc_idx] < cfg["priors"]["loc"]["agg_self_share_max"]
        t["loc"] = time.time() - t0
        t["loc_fallback_locations"] = n_fallback
        t["agg_queries"] = int(q_is_agg.sum())

        t0 = time.time()
        mc_index = np.unique(items["item_microcat_id"].to_numpy())
        mc_P = MicrocatPrior(cfg["priors"]["mc"], self.device).fit(train, mc_index).predict(qn)
        item_mc_idx = np.searchsorted(mc_index, items["item_microcat_id"].to_numpy())
        t["mc"] = time.time() - t0

        t0 = time.time()
        W, F, needles = self.filter_prior.build(
            queries["search_infm_params_text"].fillna("").astype(str).tolist(), self.item_params)
        t["filter"] = time.time() - t0
        t["filter_pairs"] = len(needles)

        # История train: память по тексту запроса и doc expansion (строятся по
        # переданной части train, поэтому не кэшируются на диск).
        t0 = time.time()
        mcfg = cfg["memory"]
        sparse = {"memory": memory_matrix(queries, train, items["item_id"].to_numpy(), mcfg["same_loc_bonus"])}
        exp_model = HashedTfidf("expansion", mcfg["expansion_source"], cfg["lexical"]["sublinear_tf"])
        X_exp = exp_model.fit_transform(expansion_texts(train, items["item_id"].to_numpy(),
                                                        mcfg["expansion_max_queries"]),
                                        cfg["n_jobs"], cfg["lexical"]["chunk_size"]).tocsc()
        text["expansion"] = restrict_to_queries(X_exp, exp_model.transform(qn))
        del X_exp
        t["memory+expansion"] = time.time() - t0
        t["memory_queries"] = int((sparse["memory"].getnnz(axis=1) > 0).sum())

        # Dense-модели: активные источники + модель для фишки 4 (поиск соседей).
        dense, dcfg, gcfg = {}, cfg["dense"], cfg["graph_expansion"]
        needed = list(dict.fromkeys(dcfg["active"] + ([gcfg["model"]] if gcfg["enabled"] else [])))
        for name in needed:
            t0 = time.time()
            model = DenseModel(name, dcfg["models"][name], self.device, dcfg["fp16"], dcfg["batch_size"])
            E = item_embeddings(cfg, name, model, items)
            if name in dcfg["active"]:
                dense[name] = (E, model.encode_queries(queries["search_query"].fillna("").astype(str).tolist()))
            if gcfg["enabled"] and name == gcfg["model"]:
                titems = self.train_item_table()
                E_tr = train_item_embeddings(cfg, name, model, titems)
                g_texts, g_stats = graph_expansion_texts(
                    E, items["item_microcat_id"].to_numpy(), items["item_id"].to_numpy(),
                    titems, E_tr, train, gcfg, self.device)
                g_model = HashedTfidf("gexp", gcfg["source"], cfg["lexical"]["sublinear_tf"])
                X_g = g_model.fit_transform(g_texts, cfg["n_jobs"], cfg["lexical"]["chunk_size"]).tocsc()
                text["gexp"] = restrict_to_queries(X_g, g_model.transform(qn))
                del X_g, g_texts
                t.update({f"gexp_{k}": v for k, v in g_stats.items()})
            model.close()
            t[f"dense_{name}"] = time.time() - t0

        return Components(queries["query_id"].tolist(), self.item_ids, text, P, q_loc_idx,
                          item_loc_idx, mc_P, item_mc_idx, W, F, t, dense, sparse, loc_geo, q_is_agg)

    def scorer(self, comp: Components) -> LinearScorer:
        """LinearScorer с параметрами приоров из конфига."""
        return LinearScorer(comp, self.device, **default_params(self.cfg))


def default_params(cfg: dict) -> dict:
    """Параметры приоров из конфига в формате LinearScorer.set_params."""
    loc, mc = cfg["priors"]["loc"], cfg["priors"]["mc"]
    return {"loc_eps": loc["eps"], "mc_eps": mc["eps"],
            "loc_sigma": loc["sigma_km"], "loc_lambda": loc["smooth_lambda"]}


def summary_line(name: str, res: dict) -> str:
    """Одна строка таблицы абляции: главные метрики и 4 ячейки."""
    c = res["cells"]
    return (f"| {name} | **{res['bench_adj']:.4f}** | {res['weighted']:.4f} | {res['unseen']:.4f} | "
            f"{res['seen']:.4f} | {c['seen_filter']['recall']:.4f} | {c['seen_nofilter']['recall']:.4f} | "
            f"{c['unseen_filter']['recall']:.4f} | {c['unseen_nofilter']['recall']:.4f} | "
            f"{res['item_in_rest']:.4f} | {res['item_new']:.4f} |")


# item_rest / item_new — micro-recall по релевантным объявлениям, которые
# встречались / не встречались в остатке train (история помогает только первым).
SUMMARY_HEADER = ("| конфиг | bench_adj | weighted | unseen | seen | seen_f | seen_nf | unseen_f | unseen_nf "
                  "| item_rest | item_new |\n|---|---|---|---|---|---|---|---|---|---|---|")


def main() -> None:
    """CLI валидации: отчёт по текущим весам или абляция по experiments.ablation."""
    parser = argparse.ArgumentParser(description="Пайплайн на валидации")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", default="ctx", choices=["ctx", "ql"])
    parser.add_argument("--ablation", nargs="?", const="ablation", default=None, metavar="NAME",
                        help="прогнать набор experiments.<NAME> (по умолчанию ablation)")
    parser.add_argument("--set", action="append", default=[], help="правка конфига a.b=значение")
    args = parser.parse_args()
    cfg = apply_overrides(io.load_config(args.config), args.set)

    t0 = time.time()
    pipe = Pipeline(cfg)
    queries, train = load_query_set(cfg, "val", args.variant)
    comp = pipe.prepare(queries, train)
    scorer = pipe.scorer(comp)
    val = load_val(cfg, args.variant)
    print(f"Подготовка: {time.time() - t0:.1f} с; части: "
          + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in comp.timings.items()))

    fcfg = cfg["fusion"]
    runs = (cfg["experiments"][args.ablation] if args.ablation
            else [{"name": "config", "weights": fcfg["weights"]}])
    print(f"\nВариант срезов: {args.variant}; набор: {args.ablation or 'config'}; "
          f"правки: {args.set or 'нет'}\n\n{SUMMARY_HEADER}")
    res = None
    agg_by_qid = dict(zip(comp.query_ids, comp.q_is_agg))
    is_agg = np.array([agg_by_qid[q] for q in val.queries["query_id"]])
    print(f"Запросов из регионов-агрегатов: {is_agg.sum()} из {len(is_agg)}")
    for run in runs:
        t1 = time.time()
        # параметры приоров: из конфига, поверх — правки конкретного запуска
        scorer.set_params(**{**default_params(cfg), **run.get("params", {})})
        method = run.get("method", "linear")
        if method == "linear":
            cands = scorer.candidates(run["weights"], fcfg["k"], fcfg["batch_size"],
                                      run.get("loc_mask_min_p", fcfg["loc_mask_min_p"]))
        elif method == "rrf":
            cands = scorer.to_candidates(rrf_fuse(scorer, run["channels"], fcfg["k"], run.get("depth", 200),
                                                  run.get("k_rrf", 60.0), fcfg["batch_size"]))
        elif method == "quota":
            cands = scorer.to_candidates(quota_fuse(scorer, run["channels"], fcfg["k"], fcfg["batch_size"]))
        else:
            raise ValueError(f"неизвестный метод слияния {method}")
        res = evaluate(cands, val)
        pq_ = res["per_query"]
        print(summary_line(run["name"], res)
              + f" город {pq_[~is_agg].mean():.4f} / агрегат {pq_[is_agg].mean():.4f}"
              + f"  ({time.time() - t1:.1f} с)")
    print("\n" + format_report(res))
    # Пиковая память основного процесса (ru_maxrss в Linux — в КБ).
    print(f"\nПик RSS процесса: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.1f} ГБ")


if __name__ == "__main__":
    main()
