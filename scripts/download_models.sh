#!/usr/bin/env bash
# Единственное место проекта, где используется сеть: разовое скачивание
# open-source моделей из HuggingFace Hub в models/hf (HF_HOME из scripts/env.sh).
# Список моделей берётся из configs/default.yaml (dense.models).
# Скачиваются только веса safetensors, токенизатор и конфиги: ONNX/OpenVINO/TF
# и дублирующие .bin-веса не нужны. После этого все запуски идут офлайн.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE   # сеть разрешена только здесь
mkdir -p models/hf

python - <<'EOF'
import yaml
from huggingface_hub import HfApi, snapshot_download

cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
api = HfApi()
for name, m in cfg["dense"]["models"].items():
    files = api.list_repo_files(m["hf_id"])
    ignore = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot"]
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")  # есть safetensors — .bin-копия весов не нужна
    path = snapshot_download(m["hf_id"], ignore_patterns=ignore)
    print(f"[download_models] {name}: {m['hf_id']} -> {path}")
EOF
du -sh models/hf
