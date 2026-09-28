#!/usr/bin/env bash
# Разрыв «валидация 0.938 -> лидерборд 0.891»: одни и те же конфиги на тёплой
# (ctx) и холодной (ctx_cold) валидации, затем предранкер на холодной.
# Нужна модель models/finetuned/e5_small_ft_val_cold (finetune --mode val_cold).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

COLD_MODEL="dense.models.e5_small_ft.hf_id=models/finetuned/e5_small_ft_val_cold"
NO_HISTORY="prerank.drop_features=[item_pop,expansion,rank_expansion,q_memory_items]"

python -m candgen.pipeline --ablation cold
python -m candgen.pipeline --ablation cold --variant ctx_cold --set "$COLD_MODEL"
python -m candgen.prerank --variant ctx_cold --set "$COLD_MODEL"
python -m candgen.prerank --variant ctx_cold --set "$COLD_MODEL" --set "$NO_HISTORY"
