#!/usr/bin/env bash
# Эксперименты этапа 2, серия d (после перехода на float32 TF-IDF «на месте»):
#   1) desc[:2000] — должен повторить 0.8669 (проверка эквивалентности реализации);
#   2) описание целиком (в серии c не досчитано из-за нехватки памяти хоста).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

python -m candgen.pipeline --ablation base --set docs.fields.2.max_chars=2000
python -m candgen.pipeline --ablation base --set docs.fields.2.max_chars=null
