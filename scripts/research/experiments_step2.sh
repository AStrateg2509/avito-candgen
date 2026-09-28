#!/usr/bin/env bash
# Шаг 2 после лидерборда: рецепт s9 + признаки cross-encoder (ce_score, rank_ce, ce_gap)
# на холодной валидации. Нужна модель models/finetuned/crossenc_val_cold
# (python -m candgen.crossenc --mode val_cold). База: s9 = 0.9154.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

S9=(--variant ctx_cold
    --set dense.models.e5_small_ft.hf_id=models/finetuned/e5_small_ft_val_cold
    --set fusion.weights.expansion=0 --set prerank.channels.expansion=skip
    --set "prerank.drop_features=[item_pop,expansion,rank_expansion,q_memory_items,item_reviews,item_rating,item_desc_len,item_title_len]")

python -m candgen.prerank "${S9[@]}" --set crossenc.enabled=true
