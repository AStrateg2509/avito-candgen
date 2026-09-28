#!/usr/bin/env bash
# Ответы для бенчмарка от «чистых» ранкеров: обучены на холодной валидации
# (ctx_cold), без expansion (в базовом скоре, в пуле и в признаках) и без
# признаков истории объявления; второй вариант — ещё и без «качества».
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

COLD_MODEL="dense.models.e5_small_ft.hf_id=models/finetuned/e5_small_ft_val_cold"
NO_EXP=(--set fusion.weights.expansion=0 --set prerank.channels.expansion=skip)
HIST="item_pop,expansion,rank_expansion,q_memory_items"
QUAL="item_reviews,item_rating,item_desc_len,item_title_len"

python -m candgen.prerank --mode bench --variant ctx_cold --set "$COLD_MODEL" "${NO_EXP[@]}" \
    --set "prerank.drop_features=[$HIST]" --set paths.answer=artifacts/answers/answer_s8_ranker_nohist.csv
python -m candgen.prerank --mode bench --variant ctx_cold --set "$COLD_MODEL" "${NO_EXP[@]}" \
    --set "prerank.drop_features=[$HIST,$QUAL]" --set paths.answer=artifacts/answers/answer_s9_ranker_nohist_noqual.csv
