#!/usr/bin/env bash
# Калибровка холодной валидации по лидерборду (s3 линейное: LB 0.8801; s6 ранкер: LB 0.8912)
# и ранкер, независимый от истории объявлений.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

COLD_MODEL="dense.models.e5_small_ft.hf_id=models/finetuned/e5_small_ft_val_cold"
NO_EXP=(--set fusion.weights.expansion=0 --set prerank.channels.expansion=skip)

# 1) рецепт сабмита №6: ранкер учится на тёплой валидации, оценивается на холодной
python -m candgen.prerank --mode transfer --variant ctx --eval-variant ctx_cold --eval-set "$COLD_MODEL"
# 2) ранкер без истории объявления (без expansion нигде), обучен и оценён на холодной
python -m candgen.prerank --variant ctx_cold --set "$COLD_MODEL" "${NO_EXP[@]}" \
    --set "prerank.drop_features=[item_pop,expansion,rank_expansion,q_memory_items]"
# 3) то же без признаков «качества» объявления
python -m candgen.prerank --variant ctx_cold --set "$COLD_MODEL" "${NO_EXP[@]}" \
    --set "prerank.drop_features=[item_pop,expansion,rank_expansion,q_memory_items,item_reviews,item_rating,item_desc_len,item_title_len]"
