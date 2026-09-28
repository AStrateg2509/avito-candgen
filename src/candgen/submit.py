"""Сборка answer.csv для бенчмарка и его обязательная проверка перед сабмитом.

  python -m candgen.submit                          # собрать answer.csv по конфигу и проверить
  python -m candgen.submit --validate answer.csv    # только проверить готовый файл

Формат: колонки ровно query_id,answer; answer — item_id через один пробел;
по строке на каждый query_id бенчмарка; ≤50 уникальных id из корпуса.
Всегда добиваем до 50: пустой слот — это гарантированно потерянный шанс.
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd

from candgen import io
from candgen.pipeline import Pipeline, apply_overrides, load_query_set

K_MAX = 50


def validate(path: str, cfg: dict) -> dict:
    """Проверяет answer.csv по правилам формата ответа; при ошибке — ValueError.

    Проверки: колонки ровно [query_id, answer]; множество query_id совпадает с
    бенчмарком, без дублей, длина 16; в строке 1..50 токенов через одиночный
    пробел без повторов; каждый токен — hex16 и есть в корпусе.

    Вход: путь к файлу, конфиг. Выход: статистика (среднее число кандидатов,
    доля строк короче 50). Файл читается как строки (dtype=str), без NA.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["query_id", "answer"]:
        raise ValueError(f"колонки должны быть ровно ['query_id', 'answer'], а не {list(df.columns)}")
    bench = set(io.load_queries(cfg, ["query_id"])["query_id"])
    got = df["query_id"].tolist()
    if len(got) != len(set(got)):
        raise ValueError(f"повторяющиеся query_id: {len(got) - len(set(got))}")
    if set(got) != bench:
        raise ValueError(f"query_id не совпадают с бенчмарком: лишних {len(set(got) - bench)}, "
                         f"не хватает {len(bench - set(got))}")
    bad_len = [q for q in got if len(q) != 16]
    if bad_len:
        raise ValueError(f"query_id не длины 16: {bad_len[:3]}")

    corpus = set(io.load_items(cfg, ["item_id"])["item_id"])
    hex16 = re.compile(r"^[0-9a-f]{16}$")
    counts = []
    for qid, answer in zip(df["query_id"], df["answer"]):
        tokens = answer.split(" ")  # двойной пробел даст пустой токен и будет пойман ниже
        if not 1 <= len(tokens) <= K_MAX:
            raise ValueError(f"{qid}: {len(tokens)} токенов, нужно от 1 до {K_MAX}")
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"{qid}: повторяющиеся item_id")
        for tok in tokens:
            if not hex16.match(tok):
                raise ValueError(f"{qid}: токен {tok!r} не соответствует ^[0-9a-f]{{16}}$")
            if tok not in corpus:
                raise ValueError(f"{qid}: item_id {tok} нет в корпусе")
        counts.append(len(tokens))
    counts = np.array(counts)
    stats = {"rows": len(df), "mean_candidates": float(counts.mean()),
             "share_short": float((counts < K_MAX).mean())}
    print(f"[validate] {path}: OK, строк {stats['rows']}, среднее кандидатов {stats['mean_candidates']:.2f}, "
          f"доля строк < {K_MAX}: {stats['share_short']:.2%}")
    return stats


def backfill(cands: list[str], pools: list[np.ndarray], k: int = K_MAX) -> list[str]:
    """Добивает список кандидатов до k из пулов по порядку, без повторов.

    Вход: исходные кандидаты; пулы item_id по убыванию приоритета; k.
    Выход: список длины min(k, число доступных уникальных id).
    """
    out, seen = list(cands), set(cands)
    for pool in pools:
        for item in pool:
            if len(out) >= k:
                return out
            if item not in seen:
                out.append(item)
                seen.add(item)
    return out


def make_answer(cfg: dict) -> str:
    """Строит answer.csv: весь train как база приоров, скор по всему корпусу, top-50.

    Добивка (если источники дали меньше 50): популярные в (главная подкатегория
    P_mc, главная локация P_loc), затем популярные в главной локации, затем
    популярные в корпусе. Популярность = число кликов в train.
    При полном переборе корпуса добивка срабатывать не должна, но страхует.

    Вход: конфиг. Выход: путь к записанному и проверенному файлу.
    """
    t0 = time.time()
    cfg = apply_overrides(cfg, cfg["submit"]["bench_overrides"])  # модели, обученные на всём train
    pipe = Pipeline(cfg)
    queries, train = load_query_set(cfg, "bench")
    comp = pipe.prepare(queries, train)
    scorer = pipe.scorer(comp)
    fcfg = cfg["fusion"]
    idx, _ = scorer.topk(fcfg["weights"], fcfg["k"], fcfg["batch_size"], fcfg["loc_mask_min_p"])
    print(f"[submit] подготовка и скоринг: {time.time() - t0:.1f} с")

    items = pipe.items
    pop = items["item_id"].map(train["item_id"].value_counts()).fillna(0).to_numpy()
    order = np.argsort(-pop, kind="stable")  # популярные первыми, при равенстве — порядок корпуса
    ids, locs, mcs = items["item_id"].to_numpy(), items["item_location_id"].to_numpy(), items["item_microcat_id"].to_numpy()
    mc_index = np.unique(mcs)
    item_locs = np.unique(locs)

    rows, n_filled = [], 0
    for qi, qid in enumerate(comp.query_ids):
        cands = list(dict.fromkeys(ids[idx[qi]].tolist()))
        if len(cands) < K_MAX:
            n_filled += 1
            loc1 = item_locs[np.argmax(comp.loc_P[comp.q_loc_idx[qi]])]
            mc1 = mc_index[np.argmax(comp.mc_P[qi])]
            pools = [ids[order][(locs[order] == loc1) & (mcs[order] == mc1)],
                     ids[order][locs[order] == loc1], ids[order]]
            cands = backfill(cands, pools)
        rows.append((qid, " ".join(cands[:K_MAX])))

    path = cfg["paths"]["answer"]
    pd.DataFrame(rows, columns=["query_id", "answer"]).to_csv(path, index=False, encoding="utf-8",
                                                               lineterminator="\n")
    print(f"[submit] записан {path}; запросов с добивкой: {n_filled}; всего {time.time() - t0:.1f} с")
    validate(path, cfg)
    return path


def main() -> None:
    """CLI: сборка answer.csv или проверка готового файла."""
    parser = argparse.ArgumentParser(description="answer.csv: сборка и проверка")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--validate", metavar="PATH", help="только проверить файл")
    parser.add_argument("--set", action="append", default=[], help="правка конфига a.b=значение")
    args = parser.parse_args()
    cfg = apply_overrides(io.load_config(args.config), args.set)
    if args.validate:
        validate(args.validate, cfg)
    else:
        make_answer(cfg)


if __name__ == "__main__":
    main()
