#!/usr/bin/env bash
# Фоновый запуск долгой задачи внутри WSL/Linux.
#
# Использование:  bash scripts/bg.sh <имя> <команда...>
# Пример:         bash scripts/bg.sh eda python -m candgen.eda
#
# Вывод задачи пишется в logs/<имя>.log, PID — в logs/<имя>.pid, PID
# печатается в stdout. Составные команды (&&, |, >) передавать не надо:
# их нужно оформить отдельным scripts/*.sh.
#
# Зачем setsid + nohup + </dev/null: процесс отвязывается от сессии wsl.exe
# и её терминала. wsl.exe сразу возвращает управление, а задача переживает
# закрытие сессии.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: bash scripts/bg.sh <имя> <команда...>" >&2
    exit 2
fi

NAME="$1"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/$NAME.log"
PIDFILE="$ROOT/logs/$NAME.pid"

cd "$ROOT"
# $* склеивает аргументы в одну строку команды; внутри задачи заново
# подключаем окружение проекта (venv, PYTHONPATH, seed).
setsid nohup bash -c "source scripts/env.sh; $*" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"

# Пауза обязательна. Если сессия wsl.exe закроется раньше, чем дочерний
# процесс успеет выполнить setsid, WSL убьёт его вместе с сессией
# (проверено: без паузы задача умирала). Заодно ловим мгновенное падение.
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
    # Процесс — наш прямой потомок, поэтому wait вернёт его код выхода.
    RC=0
    wait "$PID" || RC=$?
    if [ "$RC" -ne 0 ]; then
        echo "Задача '$NAME' упала сразу после старта (код $RC). Хвост лога:" >&2
        tail -n 20 "$LOG" >&2
        exit "$RC"
    fi
    echo "Задача '$NAME' уже завершилась успешно (быстрее 1 с), лог: $LOG" >&2
fi
echo "$PID"
