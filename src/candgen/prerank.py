"""Этап 5: LightGBM-предранкер поверх пула кандидатов всех источников.

Схема:
  1) пул кандидатов запроса — объединение top-pool_depth каналов
     (prerank.channels: итоговый линейный скор и каналы «источник + приоры»);
  2) признаки пары (запрос, кандидат): сырые значения каждого источника, ранги
     в каналах, P_loc и P_mc, совпадения фильтра по ключам, расстояние,
     свойства объявления и запроса (см. FEATURE_DOC);
  3) LGBMRanker (lambdarank), группа = запрос, метка = объявление выбрано;
  4) ответ — top-50 пула по предсказанию.

Почему обучение на val-запросах (cross-fitting), а не на запросах остатка train:
у val-запросов все признаки честные — отложенные группы не входят ни в
статистики (P_mc, память, expansion), ни в обучение дообученного e5. Запросы
остатка дали бы утечку: их клики уже внутри этих признаков. К тому же текстов
с выбранными объявлениями корпуса в train всего ~12 тыс., 6,5 тыс. из них
уже в валидации. Оценка честная: 2 фолда по запросам; каждый запрос
предсказывает модель, обученная на другой половине.

  python -m candgen.prerank                # cross-fitting на валидации, сравнение с линейным
  python -m candgen.prerank --mode bench   # ранкер на всех val-запросах -> answer.csv
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from candgen import io
from candgen.fusion import LinearScorer
from candgen.pipeline import Pipeline, apply_overrides, load_query_set
from candgen.submit import K_MAX, validate
from candgen.validation import ValSet, evaluate, format_report, load_val, split_halves

ITEM_EXTRA_COLS = ["item_id", "item_rating", "item_rating_reviews_count", "item_price",
                   "item_is_phone_hidden", "item_is_message_forbidden", "item_title_raw",
                   "item_description_raw"]
FEATURE_DOC = {
    "final": "итоговый линейный скор (этап 4)", "final_gap": "final минус лучший final в пуле",
    "char/word/e5_small_ft/expansion/memory": "сырое значение источника (косинус / скор памяти)",
    "loc_logp / loc_p": "log(P_loc + eps) и сама P_loc", "mc_logp / mc_p": "то же для подкатегории",
    "filter, filter_<ключ>": "бонус фильтра и совпадение по каждому ключу",
    "rank_<канал>": "место в канале (pool_depth + 1, если не попал)",
    "same_loc, dist_km": "та же локация; расстояние от центра локации запроса",
    "item_*": "рейтинг, отзывы, log цены, длины текстов, флаги, популярность в train",
    "q_*": "длина запроса, фильтр, регион, seen, острота P_mc, число объявлений в памяти",
}


def channel_weights(cfg: dict) -> dict[str, dict]:
    """Веса каналов пула ("final" — текущие fusion.weights)."""
    return {name: (cfg["fusion"]["weights"] if w == "final" else w)
            for name, w in cfg["prerank"]["channels"].items()}


def build_pool(scorer: LinearScorer, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Пул кандидатов и ранги в каналах.

    Вход: скорер, конфиг.
    Выход: (пул [n_q × P] индексов объявлений с дополнением 0, маска валидных
            позиций [n_q × P], {канал: ранг [n_q × P] (depth+1 — не попал)}).
    """
    depth, bs = cfg["prerank"]["pool_depth"], cfg["prerank"]["batch_size"]
    tops = {name: scorer.topk(w, depth, bs)[0] for name, w in channel_weights(cfg).items()}
    n_q = next(iter(tops.values())).shape[0]
    pools = [np.unique(np.concatenate([t[qi] for t in tops.values()])) for qi in range(n_q)]
    P = max(len(p) for p in pools)
    pool = np.zeros((n_q, P), dtype=np.int64)
    valid = np.zeros((n_q, P), dtype=bool)
    ranks = {name: np.full((n_q, P), depth + 1, dtype=np.float32) for name in tops}
    for qi, p in enumerate(pools):
        pool[qi, :len(p)], valid[qi, :len(p)] = p, True
        for name, t in tops.items():
            # пул отсортирован, top канала ⊂ пул -> позиции через searchsorted
            ranks[name][qi, np.searchsorted(p, t[qi])] = np.arange(1, depth + 1)
    return pool, valid, ranks


def item_static(pipe: Pipeline, train: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Признаки объявлений корпуса (векторы длины N в порядке корпуса)."""
    it = io.load_items(cfg, ITEM_EXTRA_COLS).set_index("item_id").reindex(pipe.item_ids)
    # Клики на миллион строк train: на валидации train — остаток (~70% строк),
    # на бенчмарке — весь train; в абсолютных числах шкала признака уплыла бы.
    clicks = pd.Series(pipe.item_ids).map(train["item_id"].value_counts()).fillna(0).to_numpy() * 1e6 / len(train)
    return {
        "item_rating": it["item_rating"].fillna(-1).to_numpy(np.float32),
        "item_reviews": np.log1p(it["item_rating_reviews_count"].fillna(0).to_numpy(np.float32)),
        "item_log_price": np.log1p(it["item_price"].fillna(0).clip(lower=0).to_numpy(np.float32)),
        "item_title_len": it["item_title_raw"].fillna("").str.len().to_numpy(np.float32),
        "item_desc_len": np.log1p(it["item_description_raw"].fillna("").str.len().to_numpy(np.float32)),
        "item_phone_hidden": it["item_is_phone_hidden"].astype(float).to_numpy(np.float32),
        "item_msg_forbidden": it["item_is_message_forbidden"].astype(float).to_numpy(np.float32),
        "item_pop": np.log1p(clicks).astype(np.float32),
    }


@torch.no_grad()
def pool_features(pipe: Pipeline, comp, scorer: LinearScorer, queries: pd.DataFrame, train: pd.DataFrame,
                  cfg: dict, rel: dict | None = None) -> pd.DataFrame:
    """Таблица признаков всех пар (запрос, кандидат пула) и, если дан rel, метки.

    Вход: пайплайн, компоненты, скорер, запросы, train-часть, конфиг,
          {query_id: множество релевантных} (для val) или None (для бенчмарка).
    Выход: DataFrame, отсортированный по запросу: qi, item (индекс в корпусе),
           признаки, label (если rel задан).
    """
    pool, valid, ranks = build_pool(scorer, cfg)
    n_q, P = pool.shape
    device, bs = scorer.device, cfg["prerank"]["batch_size"]
    comps = {src: {src: 1.0} for src in list(scorer.text) + list(scorer.dense) + list(scorer.sparse)}
    comps.update({"loc_logp": {"loc": 1.0}, "mc_logp": {"mc": 1.0}, "filter": {"filter": 1.0},
                  "final": cfg["fusion"]["weights"]})
    keys = list(cfg["priors"]["filter"]["key_weights"])
    n_pairs = scorer.filt_W.shape[1]
    key_mask = {k: torch.tensor([i < len(comp.filt_keys) and comp.filt_keys[i] == k for i in range(n_pairs)],
                                device=device, dtype=torch.float32) for k in keys}
    F = {name: np.zeros((n_q, P), np.float32) for name in list(comps) + ["loc_p"] + [f"filter_{k}" for k in keys]}
    for start in range(0, n_q, bs):
        rows = slice(start, min(start + bs, n_q))
        take = torch.from_numpy(pool[rows]).to(device)
        for name, w in comps.items():
            F[name][rows] = torch.gather(scorer.batch_scores(rows, w), 1, take).cpu().numpy()
        F["loc_p"][rows] = torch.gather(scorer.loc_P[scorer.q_loc[rows]][:, scorer.i_loc], 1, take).cpu().numpy()
        for k in keys:
            M = (scorer.filt_W[rows] * key_mask[k][None, :]) @ scorer.filt_F
            F[f"filter_{k}"][rows] = (torch.gather(M, 1, take) > 0).float().cpu().numpy()

    F["final_gap"] = F["final"] - np.where(valid, F["final"], -np.inf).max(axis=1, keepdims=True)
    F["mc_p"] = np.take_along_axis(comp.mc_P, comp.item_mc_idx[pool], axis=1)
    for name, r in ranks.items():
        F[f"rank_{name}"] = r
    item_locs = pipe.items["item_location_id"].to_numpy()
    F["same_loc"] = (item_locs[pool] == queries["search_location_id"].to_numpy()[:, None]).astype(np.float32)
    D = np.nan_to_num(comp.loc_geo["D"], nan=5000.0)
    F["dist_km"] = np.log1p(D[comp.q_loc_idx[:, None], comp.item_loc_idx[pool]]).astype(np.float32)
    for name, vec in item_static(pipe, train, cfg).items():
        F[name] = vec[pool]

    # признаки запроса (одинаковы для всех кандидатов запроса)
    mc_P = comp.mc_P
    q = {
        "q_words": queries["query_norm"].str.split().str.len().fillna(0).to_numpy(np.float32),
        "q_has_filter": (queries["search_infm_params_text"].fillna("").str.strip() != "").to_numpy(np.float32),
        "q_is_agg": comp.q_is_agg.astype(np.float32),
        "q_seen": queries["query_norm"].isin(pd.Index(train["query_norm"].unique())).to_numpy(np.float32),
        "q_mc_max": mc_P.max(axis=1),
        "q_mc_entropy": -(mc_P * np.log(mc_P + 1e-12)).sum(axis=1),
        "q_memory_items": (comp.sparse["memory"].getnnz(axis=1)).astype(np.float32),
        "q_loc_self_share": comp.loc_geo["self_share"][comp.q_loc_idx],
        "q_pool_size": valid.sum(axis=1).astype(np.float32),
    }
    for name, v in q.items():
        F[name] = np.broadcast_to(v[:, None], (n_q, P))

    qi_idx, pos = np.nonzero(valid)
    df = pd.DataFrame({"qi": qi_idx, "item": pool[qi_idx, pos]})
    for name, arr in F.items():
        df[name] = arr[qi_idx, pos]
    if rel is not None:
        ids = comp.item_ids
        qids = np.array(comp.query_ids)
        df["label"] = [ids[i] in rel[qids[qi]] for qi, i in zip(df["qi"].to_numpy(), df["item"].to_numpy())]
        df["label"] = df["label"].astype(np.int8)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Все столбцы-признаки (без служебных qi, item, label)."""
    return [c for c in df.columns if c not in ("qi", "item", "label")]


def fit_ranker(df: pd.DataFrame, cfg: dict) -> lgb.LGBMRanker:
    """Обучает LGBMRanker с early stopping на отложенной части обучающих запросов.

    Запросы без положительных в пуле не дают сигнала lambdarank и отбрасываются.
    """
    pcfg = cfg["prerank"]
    feats = feature_columns(df)
    df = df[df.groupby("qi")["label"].transform("max") > 0]
    qids = np.sort(df["qi"].unique())
    rng = np.random.default_rng(cfg["seed"])
    va_q = set(rng.choice(qids, size=int(len(qids) * pcfg["valid_share"]), replace=False))
    is_va = df["qi"].isin(va_q).to_numpy()
    tr, va = df[~is_va], df[is_va]
    model = lgb.LGBMRanker(**pcfg["lgbm"], random_state=cfg["seed"], n_jobs=cfg["n_jobs"])
    model.fit(tr[feats], tr["label"], group=tr.groupby("qi", sort=True).size().to_numpy(),
              eval_set=[(va[feats], va["label"])], eval_group=[va.groupby("qi", sort=True).size().to_numpy()],
              eval_at=[pcfg["eval_at"]], callbacks=[lgb.early_stopping(pcfg["early_stopping_rounds"], verbose=False)])
    return model


def top_by_model(model: lgb.LGBMRanker, df: pd.DataFrame, k: int) -> dict[int, np.ndarray]:
    """top-k индексов объявлений каждого запроса по предсказанию модели."""
    df = df.assign(pred=model.predict(df[feature_columns(df.drop(columns=["label"], errors="ignore"))]))
    df = df.sort_values(["qi", "pred"], ascending=[True, False], kind="mergesort")
    return {qi: g["item"].to_numpy()[:k] for qi, g in df.groupby("qi", sort=False)}


def to_cands(top: dict[int, np.ndarray], comp, fallback: np.ndarray, k: int) -> dict[str, list[str]]:
    """{query_id: [item_id]}; если в пуле меньше k, добивка из линейного top-k."""
    ids, out = comp.item_ids, {}
    for qi, qid in enumerate(comp.query_ids):
        items = list(dict.fromkeys(list(top.get(qi, [])) + list(fallback[qi])))[:k]
        out[qid] = ids[items].tolist()
    return out


def run_val(cfg: dict) -> None:
    """Cross-fitting на валидации: 2 фолда по запросам, сравнение с линейным слиянием."""
    t0 = time.time()
    pipe = Pipeline(cfg)
    queries, train = load_query_set(cfg, "val", "ctx")
    comp = pipe.prepare(queries, train)
    scorer = pipe.scorer(comp)
    val = load_val(cfg, "ctx")
    k = cfg["fusion"]["k"]
    lin_idx, _ = scorer.topk(cfg["fusion"]["weights"], k, cfg["fusion"]["batch_size"])
    df = pool_features(pipe, comp, scorer, queries, train, cfg, val.rel)
    print(f"[prerank] признаки: {len(df):,} пар, {len(feature_columns(df))} признаков, "
          f"{time.time() - t0:.0f} с; средний пул {df.groupby('qi').size().mean():.0f}")
    pool_hit = df.groupby("qi")["label"].max()
    print(f"[prerank] доля запросов с релевантным в пуле: {pool_hit.mean():.4f}")

    half_a, half_b = split_halves(val, cfg["seed"])
    qpos = {q: i for i, q in enumerate(comp.query_ids)}
    top, importances = {}, []
    for train_ids, test_ids in ((half_a, half_b), (half_b, half_a)):
        tr_qi = {qpos[q] for q in train_ids}
        model = fit_ranker(df[df["qi"].isin(tr_qi)], cfg)
        print(f"[prerank] фолд: лучшая итерация {model.best_iteration_}")
        top.update(top_by_model(model, df[~df["qi"].isin(tr_qi)], k))
        importances.append(pd.Series(model.booster_.feature_importance("gain"),
                                     index=feature_columns(df)))
    res_rank = evaluate(to_cands(top, comp, lin_idx, k), val)
    res_lin = evaluate(scorer.to_candidates(lin_idx), val)
    print(f"\nЛинейное слияние: bench_adj {res_lin['bench_adj']:.4f}")
    print(f"LightGBM-предранкер (cross-fitting): bench_adj {res_rank['bench_adj']:.4f} "
          f"(Δ {100 * (res_rank['bench_adj'] - res_lin['bench_adj']):+.2f} п.п.)\n")
    print(format_report(res_rank))
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    print("\nВажность признаков (gain, среднее по фолдам), топ-15:")
    print((imp / imp.sum()).head(15).round(4).to_string())


def run_bench(cfg: dict) -> None:
    """Ранкер на всех val-запросах -> пул и признаки бенчмарка -> answer.csv (+ проверка)."""
    t0 = time.time()
    pipe = Pipeline(cfg)
    queries, train = load_query_set(cfg, "val", "ctx")
    comp = pipe.prepare(queries, train)
    scorer = pipe.scorer(comp)
    val = load_val(cfg, "ctx")
    df_val = pool_features(pipe, comp, scorer, queries, train, cfg, val.rel)
    model = fit_ranker(df_val, cfg)
    print(f"[prerank] ранкер обучен на всех val-запросах: итерация {model.best_iteration_}")
    del scorer, comp, df_val
    torch.cuda.empty_cache()

    bcfg = apply_overrides(cfg, cfg["submit"]["bench_overrides"])
    pipe_b = Pipeline(bcfg)
    queries_b, train_b = load_query_set(bcfg, "bench")
    comp_b = pipe_b.prepare(queries_b, train_b)
    scorer_b = pipe_b.scorer(comp_b)
    k = cfg["fusion"]["k"]
    lin_idx, _ = scorer_b.topk(cfg["fusion"]["weights"], k, cfg["fusion"]["batch_size"])
    df_b = pool_features(pipe_b, comp_b, scorer_b, queries_b, train_b, bcfg)
    cands = to_cands(top_by_model(model, df_b, k), comp_b, lin_idx, k)
    path = cfg["paths"]["answer"]
    pd.DataFrame([(q, " ".join(c)) for q, c in cands.items()], columns=["query_id", "answer"]).to_csv(
        path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"[prerank] записан {path} за {time.time() - t0:.0f} с")
    validate(path, cfg)


def main() -> None:
    """CLI: --mode val (оценка cross-fitting) или --mode bench (answer.csv)."""
    parser = argparse.ArgumentParser(description="LightGBM-предранкер (этап 5)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["val", "bench"], default="val")
    parser.add_argument("--set", action="append", default=[], help="правка конфига a.b=значение")
    args = parser.parse_args()
    cfg = apply_overrides(io.load_config(args.config), args.set)
    run_val(cfg) if args.mode == "val" else run_bench(cfg)


if __name__ == "__main__":
    main()
