"""Подбор весов слияния через Optuna (TPE, seed из конфига) по метрике bench_adj.

Схема с честной оценкой переобучения:
  1) подбор на половине A валидации (стратифицирована по ячейкам). Лучший
     набор проверяется на половине B и сравнивается с исходными весами конфига
     на той же B. Прирост на B — несмещённая оценка пользы подбора;
  2) подбор на всей валидации -> финальные веса. Они печатаются и сохраняются
     в reports/tune_<вариант>.json, в configs/default.yaml переносятся вручную.

Параметры: веса источников и приоров (char фиксирован = 1 как масштаб) и
eps логарифмов приоров. Пространство поиска — секция `tune` конфига.

  python -m candgen.tune [--variant ctx] [--trials 50]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optuna

from candgen import io
from candgen.fusion import LinearScorer
from candgen.pipeline import Pipeline, apply_overrides, load_query_set
from candgen.validation import ValSet, evaluate, load_val, split_halves, subset_val


def suggest(trial: optuna.Trial, space: dict) -> dict:
    """Параметры trial из пространства {имя: [низ, верх, 'log'|'uniform']}."""
    return {name: trial.suggest_float(name, lo, hi, log=(kind == "log"))
            for name, (lo, hi, kind) in space.items()}


def score_params(scorer: LinearScorer, params: dict, base: dict, fcfg: dict, val: ValSet) -> dict:
    """Метрики валидации (подмножества) для набора параметров.

    Вход: скорер; параметры (веса и eps; чего нет — берётся из base);
          базовые параметры; секция fusion; ValSet (возможно, подмножество).
    Выход: результат evaluate.
    """
    p = {**base, **params}
    scorer.set_eps(p["loc_eps"], p["mc_eps"])
    weights = {k: v for k, v in p.items() if k not in ("loc_eps", "mc_eps")}
    cands = scorer.candidates(weights, fcfg["k"], fcfg["batch_size"], fcfg["loc_mask_min_p"])
    keep = set(val.queries["query_id"])
    return evaluate({q: c for q, c in cands.items() if q in keep}, val)


def run_study(scorer: LinearScorer, val: ValSet, base: dict, cfg: dict, n_trials: int, tag: str) -> optuna.Study:
    """Optuna-исследование, максимизирующее bench_adj на переданной валидации.

    Первым trial ставятся базовые параметры, чтобы подбор не мог оказаться хуже исходной точки.
    """
    space = cfg["tune"]["space"]
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.enqueue_trial({k: base[k] for k in space})

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        res = score_params(scorer, suggest(trial, space), base, cfg["fusion"], val)
        print(f"[{tag}] trial {trial.number:3d}: bench_adj={res['bench_adj']:.4f} "
              f"(лучший {max(res['bench_adj'], study.best_value if trial.number else 0):.4f}) "
              f"{time.time() - t0:.1f} с", flush=True)
        return res["bench_adj"]

    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    """CLI: подбор на половине A с проверкой на B, затем подбор на всей валидации."""
    parser = argparse.ArgumentParser(description="Подбор весов слияния (Optuna)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", default="ctx", choices=["ctx", "ql"])
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], help="правка конфига a.b=значение")
    args = parser.parse_args()
    cfg = apply_overrides(io.load_config(args.config), args.set)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = args.trials or cfg["tune"]["n_trials"]

    pipe = Pipeline(cfg)
    queries, train = load_query_set(cfg, "val", args.variant)
    scorer = pipe.scorer(pipe.prepare(queries, train))
    val = load_val(cfg, args.variant)
    base = {**cfg["fusion"]["weights"], "loc_eps": cfg["priors"]["loc"]["eps"],
            "mc_eps": cfg["priors"]["mc"]["eps"]}

    ids_a, ids_b = split_halves(val, cfg["seed"])
    val_a, val_b = subset_val(val, ids_a), subset_val(val, ids_b)
    study_a = run_study(scorer, val_a, base, cfg, n_trials, "A")
    best_a = {**base, **study_a.best_params}
    b_base = score_params(scorer, {}, base, cfg["fusion"], val_b)["bench_adj"]
    b_best = score_params(scorer, best_a, base, cfg["fusion"], val_b)["bench_adj"]
    print(f"\nПоловина B: исходные {b_base:.4f} -> подобранные на A {b_best:.4f} "
          f"(Δ {100 * (b_best - b_base):+.2f} п.п.); на A: {study_a.best_value:.4f}")

    study = run_study(scorer, val, base, cfg, n_trials, "full")
    best = {**base, **study.best_params}
    res_base = score_params(scorer, {}, base, cfg["fusion"], val)
    res_best = score_params(scorer, best, base, cfg["fusion"], val)
    print(f"\nВся валидация: исходные {res_base['bench_adj']:.4f} -> {res_best['bench_adj']:.4f}")
    print("Лучшие параметры:", {k: round(v, 5) for k, v in best.items()})

    out = Path(cfg["paths"]["reports_dir"]) / f"tune_{args.variant}.json"
    out.write_text(json.dumps({
        "space": cfg["tune"]["space"], "n_trials": n_trials, "base": base,
        "half_a_best": best_a, "half_a_value": study_a.best_value,
        "half_b_base": b_base, "half_b_tuned_on_a": b_best,
        "full_best": best, "full_base_bench_adj": res_base["bench_adj"],
        "full_best_bench_adj": res_best["bench_adj"],
        "trials_full": [{"params": t.params, "value": t.value} for t in study.trials],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
