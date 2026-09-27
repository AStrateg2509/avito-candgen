#!/usr/bin/env bash
# Полный воспроизводимый прогон -> answer.csv (офлайн, одной командой).
#
# Предварительно (единственные шаги с сетью, один раз):
#   bash scripts/install.sh          # .venv и пинованные зависимости
#   bash scripts/download_models.sh  # open-source модели в models/hf
# Затем:
#   bash scripts/run_all.sh          # переиспользует кэши, если они есть
#   bash scripts/run_all.sh --clean  # всё с нуля: индексы, эмбеддинги, срезы, дообучение
#
# Шаги: данные -> срезы валидации -> дообучение e5-small на остатке train
# (для признаков ранкера) и на всём train (для сабмита) -> индексы, эмбеддинги,
# приоры -> LightGBM-предранкер -> answer.csv + проверка формата.
# У каждого шага печатается время. --clean не трогает artifacts/answers
# (сохранённые копии сабмитов) и reports/.
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

if [ "${1:-}" = "--clean" ]; then
    echo "=== [clean] удаляю кэши и дообученные модели"
    rm -rf artifacts/lex artifacts/dense artifacts/lemmas.parquet artifacts/val_* models/finetuned
fi

if [ ! -d models/hf/hub ]; then
    echo "Нет моделей в models/hf: сначала bash scripts/download_models.sh" >&2
    exit 1
fi

step data bash scripts/setup_data.sh
step splits python -m candgen.validation --build
if [ ! -d models/finetuned/e5_small_ft_val ]; then
    step finetune_val python -m candgen.finetune --mode val
fi
if [ ! -d models/finetuned/e5_small_ft_full ]; then
    step finetune_full python -m candgen.finetune --mode full
fi
step answer python -m candgen.prerank --mode bench

echo "=== Итого: $(( ($(date +%s) - T0) / 60 )) мин $(( ($(date +%s) - T0) % 60 )) с -> answer.csv"
