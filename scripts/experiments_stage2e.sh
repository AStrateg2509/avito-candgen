#!/usr/bin/env bash
# Эксперименты этапа 2, серия e: длина параметров при описании целиком.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

python -m candgen.pipeline --ablation base --set docs.fields.1.max_chars=800
python -m candgen.pipeline --ablation base --set docs.fields.1.max_chars=null
