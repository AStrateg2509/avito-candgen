# Общее окружение проекта. Подключается через `source scripts/env.sh`
# из w.cmd (запуск с Windows) и из scripts/*.sh (запуск на чистом Linux).
# Ничего не печатает и не меняет текущую папку.

# Корень проекта вычисляется от расположения этого файла, чтобы скрипт
# работал из любой текущей папки.
_CANDGEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# venv может ещё не существовать (этап установки) — тогда просто пропускаем.
if [ -f "$_CANDGEN_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_CANDGEN_ROOT/.venv/bin/activate"
fi

# Пакет candgen лежит в src/ и импортируется без установки через pip.
export PYTHONPATH="$_CANDGEN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Фиксированный хэш строк: иначе порядок обхода set/dict строк меняется
# между запусками и результаты перестают быть воспроизводимыми.
export PYTHONHASHSEED=42
# Небуферизованный вывод, чтобы логи фоновых задач обновлялись сразу.
export PYTHONUNBUFFERED=1
# Модели HuggingFace лежат локально в models/hf и никогда не качаются
# на инференсе. Сеть разрешена только в scripts/download_models.sh.
export HF_HOME="$_CANDGEN_ROOT/models/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
