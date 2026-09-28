#!/usr/bin/env bash
# Шаг 1 после лидерборда: дешёвые улучшения пула и ранкера поверх рецепта s9
# (ранкер обучен на ctx_cold, без expansion, без истории и «качества» объявления).
# База для сравнения: s9 на холодной валидации = 0.9154, потолок пула 0.9612.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

S9=(--variant ctx_cold
    --set dense.models.e5_small_ft.hf_id=models/finetuned/e5_small_ft_val_cold
    --set fusion.weights.expansion=0 --set prerank.channels.expansion=skip
    --set "prerank.drop_features=[item_pop,expansion,rank_expansion,q_memory_items,item_reviews,item_rating,item_desc_len,item_title_len]")

# 1) пул глубже + канал «dense без приоров»
python -m candgen.prerank "${S9[@]}" --set prerank.pool_depth=200 --set "prerank.channels.e5only={e5_small_ft: 1.0}"
# 2) бинарная цель вместо lambdarank
python -m candgen.prerank "${S9[@]}" --set prerank.lgbm.objective=binary
