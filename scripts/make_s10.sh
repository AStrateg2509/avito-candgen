#!/usr/bin/env bash
# Кандидат s10: рецепт s9 (ранкер на ctx_cold, без expansion, без истории и
# «качества» объявления) с dense на документе с описанием (e5_small_ftd).
# Холодная валидация: 0.9235 против 0.9154 у s9. Нужны модели
# e5_small_ftd_val_cold (признаки ранкера) и e5_small_ftd_full (бенчмарк).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

python -m candgen.prerank --mode bench --variant ctx_cold \
    --set "dense.active=[e5_small_ftd]" \
    --set fusion.weights.e5_small_ft=0 --set fusion.weights.e5_small_ftd=0.5 \
    --set fusion.weights.expansion=0 --set prerank.channels.expansion=skip \
    --set "prerank.channels.dense={e5_small_ftd: 1.0, loc: 0.067, mc: 0.012, filter: 0.31}" \
    --set "prerank.drop_features=[item_pop,expansion,rank_expansion,q_memory_items,item_reviews,item_rating,item_desc_len,item_title_len]" \
    --set prerank.final_n_estimators=150 \
    --set paths.answer=artifacts/answers/answer_s10_ranker_ftd.csv
