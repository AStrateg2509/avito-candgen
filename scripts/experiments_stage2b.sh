#!/usr/bin/env bash
# Эксперименты этапа 2, серия b (вариант ctx, очищенные параметры):
#   1) отдельный источник по заголовку поверх baseline;
#   2) состав документа: длиннее параметры / описание, без описания, заголовок ×3;
#   3) kNN подкатегорий: число соседей и степень веса.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

python -m candgen.pipeline --ablation title
python -m candgen.pipeline --ablation base --set docs.fields.1.max_chars=800
python -m candgen.pipeline --ablation base --set docs.fields.2.max_chars=600
python -m candgen.pipeline --ablation base --set docs.fields.2.repeat=0
python -m candgen.pipeline --ablation base --set docs.fields.0.repeat=3
python -m candgen.pipeline --ablation base --set priors.mc.k=10
python -m candgen.pipeline --ablation base --set priors.mc.k=50
python -m candgen.pipeline --ablation base --set priors.mc.power=2
