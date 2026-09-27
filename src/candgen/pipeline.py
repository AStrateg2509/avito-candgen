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
from candgen.fusion import Components, LinearScorer
from candgen.priors import FilterPrior, LocationPrior, MicrocatPrior
from candgen.retrievers.lexical import build_or_load_corpus_index, restrict_to_queries
from candgen.text import Lemmatizer, build_doc_texts, normalize_series
from candgen.torch_utils import get_device
from candgen.validation import evaluate, format_report, load_train_rest_mask, load_val

ITEM_COLS = ["item_id", "item_location_id", "item_microcat_id", "item_title_raw",
             "item_infm_params_text", "item_description_raw"]
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
        P, uq, ui, n_fallback = LocationPrior(cfg["priors"]["loc"]).fit(train).matrix(
            queries["search_location_id"].to_numpy(), items["item_location_id"].to_numpy())
        q_loc_idx = np.searchsorted(uq, queries["search_location_id"].to_numpy())
        item_loc_idx = np.searchsorted(ui, items["item_location_id"].to_numpy())
        t["loc"] = time.time() - t0
        t["loc_fallback_locations"] = n_fallback

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

        return Components(queries["query_id"].tolist(), self.item_ids, text, P, q_loc_idx,
                          item_loc_idx, mc_P, item_mc_idx, W, F, t)

    def scorer(self, comp: Components) -> LinearScorer:
        """LinearScorer с eps из конфига."""
        pc_ = self.cfg["priors"]
        return LinearScorer(comp, self.device, pc_["loc"]["eps"], pc_["mc"]["eps"])


def summary_line(name: str, res: dict) -> str:
    """Одна строка таблицы абляции: главные метрики и 4 ячейки."""
    c = res["cells"]
    return (f"| {name} | **{res['bench_adj']:.4f}** | {res['weighted']:.4f} | {res['unseen']:.4f} | "
            f"{res['seen']:.4f} | {c['seen_filter']['recall']:.4f} | {c['seen_nofilter']['recall']:.4f} | "
            f"{c['unseen_filter']['recall']:.4f} | {c['unseen_nofilter']['recall']:.4f} |")


SUMMARY_HEADER = ("| конфиг | bench_adj | weighted | unseen | seen | seen_f | seen_nf | unseen_f | unseen_nf |\n"
                  "|---|---|---|---|---|---|---|---|---|")


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
    for run in runs:
        t1 = time.time()
        cands = scorer.candidates(run["weights"], fcfg["k"], fcfg["batch_size"],
                                  run.get("loc_mask_min_p", fcfg["loc_mask_min_p"]))
        res = evaluate(cands, val)
        print(summary_line(run["name"], res) + f"  ({time.time() - t1:.1f} с)")
    print("\n" + format_report(res))
    # Пиковая память основного процесса (ru_maxrss в Linux — в КБ).
    print(f"\nПик RSS процесса: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.1f} ГБ")


if __name__ == "__main__":
    main()
