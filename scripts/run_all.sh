#!/usr/bin/env bash
# Полный воспроизводимый прогон итогового рецепта (s10, LB 0.8963) -> answer.csv.
# Офлайн, одной командой.
#
# Предварительно (единственные шаги с сетью, один раз):
#   bash scripts/install.sh          # .venv и пинованные зависимости
#   bash scripts/download_models.sh  # open-source модели в models/hf
# Затем:
#   bash scripts/run_all.sh          # переиспользует кэши и модели, если они есть
#   bash scripts/run_all.sh --clean  # итоговый рецепт с нуля
#
# Шаги:
#   данные -> срезы валидации (в т.ч. ctx_cold)
#   -> дообучение e5-small на тексте с описанием объявления: на холодном
#      остатке train (для признаков ранкера) и на всём train (для бенчмарка)
#   -> индексы, эмбеддинги, приоры -> LightGBM-ранкер (обучен на ctx_cold)
#   -> answer.csv + проверка формата.
# Все настройки рецепта — в configs/default.yaml. У каждого шага печатается время.
# --clean удаляет кэши и модели итогового рецепта; модели исследований
# (models/finetuned/*) и копии сабмитов (artifacts/answers) не трогает.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh   # venv, PYTHONPATH=src, PYTHONHASHSEED=42, HF offline
mkdir -p logs

T0=$(date +%s)
step() {
    # step <имя> <команда...>: выполнить команду и напечатать время шага
    local name="$1"; shift
    local t=$(date +%s)
    echo "=== [$name] $*"
    "$@"
    echo "=== [$name] готово за $(( $(date +%s) - t )) с"
}

DENSE=e5_small_ftd   # finetune.target_model в конфиге
if [ "${1:-}" = "--clean" ]; then
    echo "=== [clean] удаляю кэши и модели итогового рецепта"
    rm -rf artifacts/lex artifacts/dense artifacts/lemmas.parquet artifacts/val_* artifacts/ce_cache \
           "models/finetuned/${DENSE}_val_cold" "models/finetuned/${DENSE}_full"
fi

if [ ! -d models/hf/hub ]; then
    echo "Нет моделей в models/hf: сначала bash scripts/download_models.sh" >&2
    exit 1
fi

step data bash scripts/setup_data.sh
step splits python -m candgen.validation --build
if [ ! -d "models/finetuned/${DENSE}_val_cold" ]; then
    step finetune_val_cold python -m candgen.finetune --mode val_cold
fi
if [ ! -d "models/finetuned/${DENSE}_full" ]; then
    step finetune_full python -m candgen.finetune --mode full
fi
step answer python -m candgen.prerank --mode bench

echo "=== Итого: $(( ($(date +%s) - T0) / 60 )) мин $(( ($(date +%s) - T0) % 60 )) с -> answer.csv"
