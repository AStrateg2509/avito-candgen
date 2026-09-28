#!/usr/bin/env bash
# Эксперименты этапа 2, серия c: длина описания в тексте документа
# (в серии b описание 600 символов вместо 300 дало +1.45 п.п. bench_adj).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

python -m candgen.pipeline --ablation base --set docs.fields.2.max_chars=1000
python -m candgen.pipeline --ablation base --set docs.fields.2.max_chars=2000
python -m candgen.pipeline --ablation base --set docs.fields.2.max_chars=null
python -m candgen.pipeline --ablation base --set docs.fields.2.max_chars=1000 --set docs.fields.1.max_chars=800
