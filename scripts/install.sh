#!/usr/bin/env bash
# Создание виртуального окружения .venv и установка зависимостей (без sudo).
#
# Два режима:
#   1) есть requirements.txt -> ставим ровно пинованные версии (воспроизводимо,
#      так делает проверяющий на чистом Linux);
#   2) requirements.txt нет -> ставим актуальные версии из списка PACKAGES
#      и фиксируем прямые зависимости в requirements.txt.
#
# torch ставится с индекса PyTorch с CUDA-сборкой. cu128 подходит для любых
# драйверов с CUDA >= 12.8 (у нас 13.x); при ошибке пробуем cu126.
# Индекс можно переопределить: TORCH_INDEX=... bash scripts/install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_INDEX_FALLBACK="https://download.pytorch.org/whl/cu126"
PACKAGES=(pandas pyarrow duckdb polars numpy scipy scikit-learn pyyaml tqdm
          pymorphy3 snowballstemmer rapidfuzz lightgbm optuna matplotlib
          sentence-transformers bm25s)
# Имена в формате pip freeze, которые фиксируем в requirements.txt
# (прямые зависимости + transformers/huggingface-hub, от которых зависит
# поведение sentence-transformers, + словари pymorphy3).
PIN_REGEX='^(torch|pandas|pyarrow|duckdb|polars|numpy|scipy|scikit-learn|PyYAML|tqdm|pymorphy3|pymorphy3-dicts-ru|snowballstemmer|rapidfuzz|lightgbm|optuna|matplotlib|sentence-transformers|transformers|huggingface-hub|bm25s)=='

if [ ! -d .venv ]; then
    echo "[install] создаю .venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

if [ -f requirements.txt ]; then
    echo "[install] ставлю пинованные версии из requirements.txt"
    pip install -r requirements.txt
else
    echo "[install] ставлю torch с $TORCH_INDEX"
    USED_INDEX="$TORCH_INDEX"
    if ! pip install torch --index-url "$TORCH_INDEX"; then
        echo "[install] не вышло, пробую $TORCH_INDEX_FALLBACK"
        USED_INDEX="$TORCH_INDEX_FALLBACK"
        pip install torch --index-url "$TORCH_INDEX_FALLBACK"
    fi
    echo "[install] ставлю остальные пакеты: ${PACKAGES[*]}"
    pip install "${PACKAGES[@]}"

    echo "[install] фиксирую версии в requirements.txt"
    {
        echo "# Пинованные прямые зависимости (сгенерировано scripts/install.sh)."
        echo "# torch берётся с индекса PyTorch (CUDA-сборка), остальное с PyPI."
        echo "--extra-index-url $USED_INDEX"
        pip freeze | grep -iE "$PIN_REGEX" | sort -f
    } > requirements.txt
fi

echo "[install] проверка импорта и CUDA"
python - <<'EOF'
import torch
import pymorphy3

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
# Создание анализатора проверяет, что словари pymorphy3-dicts-ru на месте.
pymorphy3.MorphAnalyzer()
print("pymorphy3 ok")
EOF
echo "[install] готово"
