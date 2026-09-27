"""Фишка 9: интерактивная демо-страница результатов на валидации -> reports/demo.html.

Для выборки val-запросов (попадания и промахи в каждой из 4 ячеек, города и
регионы) страница показывает:
  * запрос, локацию (город / регион-агрегат), фильтр, recall@50;
  * релевантные объявления: найдены ли и на каком месте полного ранжирования;
  * 50 кандидатов с разложением скора по слагаемым (char, word, dense,
    expansion, локация, подкатегория, фильтр) и главным текстовым источником.
Страница — один HTML-файл без внешних зависимостей (данные встроены JSON).

  python -m candgen.demo [--per-bucket 25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from candgen import io
from candgen.pipeline import Pipeline, apply_overrides, load_query_set
from candgen.validation import evaluate, load_val

TEMPLATE = Path(__file__).with_name("demo_template.html")
# Подписи слагаемых на странице; порядок = порядок цветов палитры.
SOURCE_LABELS = {"char": "char n-граммы", "word": "слова (леммы)", "e5_small_ft": "dense (e5-small ft)",
                 "user_base": "dense (USER-base)", "expansion": "doc expansion", "loc": "локация",
                 "mc": "подкатегория", "filter": "фильтр"}
TEXT_SOURCES = ("char", "word", "e5_small_ft", "user_base", "expansion")


def pick_queries(val, recall: np.ndarray, per_bucket: int, seed: int) -> np.ndarray:
    """Индексы запросов для демо: по per_bucket попаданий и промахов в каждой ячейке."""
    rng = np.random.default_rng(seed)
    chosen = []
    cells = val.queries["cell"].to_numpy()
    for cell in sorted(set(cells)):
        for hit in (True, False):
            pool = np.flatnonzero((cells == cell) & ((recall > 0) == hit))
            chosen.extend(rng.choice(pool, size=min(per_bucket, len(pool)), replace=False))
    return np.sort(np.array(chosen))


def build_payload(cfg: dict, per_bucket: int) -> dict:
    """Считает всё для страницы: top-50, вклады слагаемых, ранги пропущенных релевантных.

    Вход: конфиг; сколько запросов брать на корзину (ячейка × попадание/промах).
    Выход: словарь, который встраивается в HTML как JSON.
    """
    pipe = Pipeline(cfg)
    queries, train = load_query_set(cfg, "val", "ctx")
    comp = pipe.prepare(queries, train)
    scorer = pipe.scorer(comp)
    val = load_val(cfg, "ctx")
    weights, k = cfg["fusion"]["weights"], cfg["fusion"]["k"]
    idx, scores = scorer.topk(weights, k, cfg["fusion"]["batch_size"])
    res = evaluate(scorer.to_candidates(idx), val)
    # порядок запросов в скорере = порядок val.queries (оба из val_queries_ctx.parquet)
    assert list(comp.query_ids) == val.queries["query_id"].tolist()

    items = pipe.items.set_index("item_id")
    ids = comp.item_ids
    id_pos = {x: i for i, x in enumerate(ids)}

    def item_info(i: int) -> dict:
        row = items.loc[ids[i]]
        return {"id": ids[i], "title": str(row["item_title_raw"]), "loc": int(row["item_location_id"]),
                "mc": int(row["item_microcat_id"])}

    out = []
    for qi in pick_queries(val, res["per_query"], per_bucket, cfg["seed"]):
        rows = slice(int(qi), int(qi) + 1)
        q = val.queries.iloc[qi]
        rel = val.rel[q["query_id"]]
        parts = scorer.explain(rows, idx[qi:qi + 1], weights)
        full = scorer.batch_scores(rows, weights)[0]  # для рангов пропущенных релевантных
        rel_list = []
        for item in sorted(rel):
            pos = id_pos[item]
            rank = int((full > full[pos]).sum().item()) + 1
            rel_list.append({**item_info(pos), "rank": rank, "found": rank <= k})
        cands = []
        for j, pos in enumerate(idx[qi]):
            contrib = {src: round(float(v[0, j]), 4) for src, v in parts.items()}
            text_parts = {s: contrib[s] for s in TEXT_SOURCES if s in contrib}
            cands.append({**item_info(int(pos)), "score": round(float(scores[qi, j]), 4),
                          "parts": contrib, "main": max(text_parts, key=text_parts.get) if text_parts else None,
                          "relevant": ids[pos] in rel})
        out.append({"qid": q["query_id"], "query": q["search_query"], "loc": int(q["search_location_id"]),
                    "is_agg": bool(comp.q_is_agg[qi]), "filter": q["search_infm_params_text"] or "",
                    "slice": q["slice"], "cell": q["cell"], "recall": float(res["per_query"][qi]),
                    "rel": rel_list, "cands": cands})

    active = [s for s in SOURCE_LABELS if weights.get(s, 0.0) and (s in comp.text or s in comp.dense
                                                                    or s in ("loc", "mc", "filter"))]
    return {"metrics": {m: round(res[m], 4) for m in ("bench_adj", "weighted", "unseen", "seen")},
            "cells": {c: round(v["recall"], 4) for c, v in res["cells"].items()},
            "weights": {s: weights[s] for s in active},
            "sources": [{"key": s, "label": SOURCE_LABELS[s]} for s in active],
            "k": k, "queries": out}


def main() -> None:
    """CLI: собирает reports/demo.html из шаблона и данных валидации."""
    parser = argparse.ArgumentParser(description="Демо-страница результатов (фишка 9)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--per-bucket", type=int, default=25)
    parser.add_argument("--set", action="append", default=[], help="правка конфига a.b=значение")
    args = parser.parse_args()
    cfg = apply_overrides(io.load_config(args.config), args.set)
    payload = build_payload(cfg, args.per_bucket)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    out = Path(cfg["paths"]["reports_dir"]) / "demo.html"
    out.write_text(html, encoding="utf-8")
    print(f"[demo] {out}: запросов {len(payload['queries'])}, {out.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
