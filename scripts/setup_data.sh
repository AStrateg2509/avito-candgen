#!/usr/bin/env bash
# Подготовка сырых данных в data/raw (идемпотентно).
#
#   - все три parquet уже на месте -> распаковку пропускаем;
#   - иначе распаковываем data/raw/dataset.zip;
#   - если zip в data/raw нет, но задан DATASET_ZIP=/путь/к/dataset.zip,
#     один раз копируем его в data/raw.
# В конце печатаем число строк, схему и базовые проверки id
# (python -m candgen.io --describe).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

RAW=data/raw
FILES=(train.parquet benchmark_queries.parquet benchmark_items.parquet)
mkdir -p "$RAW"

missing=0
for f in "${FILES[@]}"; do
    [ -f "$RAW/$f" ] || missing=1
done

if [ "$missing" -eq 0 ]; then
    echo "[setup_data] все parquet уже в $RAW — распаковку пропускаю"
else
    if [ ! -f "$RAW/dataset.zip" ]; then
        if [ -n "${DATASET_ZIP:-}" ] && [ -f "$DATASET_ZIP" ]; then
            echo "[setup_data] копирую $DATASET_ZIP -> $RAW/dataset.zip"
            cp "$DATASET_ZIP" "$RAW/dataset.zip"
        else
            echo "[setup_data] ОШИБКА: нет ни parquet, ни $RAW/dataset.zip." >&2
            echo "  Положите dataset.zip в $RAW/ или запустите с DATASET_ZIP=/путь/к/dataset.zip" >&2
            exit 1
        fi
    fi
    echo "[setup_data] распаковываю $RAW/dataset.zip"
    unzip -o "$RAW/dataset.zip" -d "$RAW"
fi

python -m candgen.io --describe --config configs/default.yaml
