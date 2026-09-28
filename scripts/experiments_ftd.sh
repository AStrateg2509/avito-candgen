#!/usr/bin/env bash
# Исследование после лидерборда: dense на документе с описанием (e5_small_ftd).
# Нужна модель models/finetuned/e5_small_ftd_val_cold
# (finetune --mode val_cold --set finetune.target_model=e5_small_ftd --set finetune.batch_size=64).
# База: линейное без expansion 0.9031, ранкер s9 0.9154 (холодная валидация).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

COLD_FT="dense.models.e5_small_ft.hf_id=models/finetuned/e5_small_ft_val_cold"
python -m candgen.pipeline --ablation ftd --variant ctx_cold --set "$COLD_FT" \
    --set "dense.active=[e5_small_ft,e5_small_ftd]"

# рецепт s9 с новым dense: пул, признак и итоговый линейный скор — на e5_small_ftd
python -m candgen.prerank --variant ctx_cold \
    --set "dense.active=[e5_small_ftd]" \
    --set fusion.weights.e5_small_ft=0 --set fusion.weights.e5_small_ftd=0.5 \
    --set fusion.weights.expansion=0 --set prerank.channels.expansion=skip \
    --set "prerank.channels.dense={e5_small_ftd: 1.0, loc: 0.067, mc: 0.012, filter: 0.31}" \
    --set "prerank.drop_features=[item_pop,expansion,rank_expansion,q_memory_items,item_reviews,item_rating,item_desc_len,item_title_len]"
