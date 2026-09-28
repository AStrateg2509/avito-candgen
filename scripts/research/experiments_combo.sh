#!/usr/bin/env bash
# Связка исследования на холодной валидации: рецепт s9 + dense с описанием
# (e5_small_ftd, вес 0.8) + пул глубиной 200 + канал «ftd без приоров».
# Порог для сабмита: ≥ 0.9254 (s9 0.9154 + 1.0 п.п.).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

python -m candgen.prerank --variant ctx_cold \
    --set "dense.active=[e5_small_ftd]" \
    --set fusion.weights.e5_small_ft=0 --set fusion.weights.e5_small_ftd=0.8 \
    --set fusion.weights.expansion=0 --set prerank.channels.expansion=skip \
    --set "prerank.channels.dense={e5_small_ftd: 1.0, loc: 0.067, mc: 0.012, filter: 0.31}" \
    --set "prerank.channels.e5only={e5_small_ftd: 1.0}" --set prerank.pool_depth=200 \
    --set "prerank.drop_features=[item_pop,expansion,rank_expansion,q_memory_items,item_reviews,item_rating,item_desc_len,item_title_len]"
