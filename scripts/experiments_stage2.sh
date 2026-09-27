#!/usr/bin/env bash
# Эксперименты этапа 2 на валидации (вариант ctx). Каждый блок печатает
# таблицу «конфиг | bench_adj | weighted | ...»; итоги — в EXPERIMENTS.md.
#   1) вклад word-источника (леммы pymorphy3) поверх baseline;
#   2) вариант (a) весь корпус против (b) маски по P_loc;
#   3) очистка параметров объявлений от служебного шума (перестраивает индексы).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

python -m candgen.pipeline --ablation word
python -m candgen.pipeline --ablation mask
python -m candgen.pipeline --ablation word --set docs.clean_params=true
