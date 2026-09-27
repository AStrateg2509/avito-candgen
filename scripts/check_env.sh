#!/usr/bin/env bash
# Отчёт о среде выполнения: GPU, память, CPU, Python, данные и (если venv
# уже создан) версии ключевых пакетов + проверка CUDA через torch.
# Ничего не меняет, можно запускать сколько угодно раз.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

echo "=== GPU ===";      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi недоступен"
echo "=== RAM (ГБ) ==="; free -g
echo "=== CPU ===";      echo "nproc: $(nproc)"
echo "=== Python ===";   python3 --version
echo "=== data/raw ==="; ls -la data/raw

if [ -x .venv/bin/python ]; then
    echo "=== venv: пакеты и CUDA ==="
    python - <<'EOF'
import importlib

# Версии ключевых библиотек: печатаем, чтобы сверять с requirements.txt.
for name in ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "polars", "duckdb",
             "lightgbm", "optuna", "pymorphy3", "sentence_transformers", "bm25s"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name:22s} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # отчёт не должен падать из-за одного пакета
        print(f"{name:22s} НЕ ИМПОРТИРУЕТСЯ: {exc}")

import torch

print(f"{'torch':22s} {torch.__version__} (CUDA {torch.version.cuda})")
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    # Маленькое умножение матриц на GPU: проверяем, что вычисления реально идут.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    s = (a @ a).float().abs().mean().item()
    print("gpu:", torch.cuda.get_device_name(0), "| matmul ok, mean|a@a| =", round(s, 3))
EOF
else
    echo "=== venv ещё не создан (bash scripts/install.sh) ==="
fi
