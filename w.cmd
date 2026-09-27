@echo off
rem Мост Windows в WSL: выполняет ОДНУ строку-команду в bash внутри Ubuntu
rem от пользователя dockhub, в папке проекта, с окружением из scripts/env.sh
rem (venv, PYTHONPATH=src, PYTHONHASHSEED=42, offline-флаги HuggingFace).
rem
rem PowerShell:  .\w.cmd 'python -m candgen.eda'
rem              одинарные кавычки, чтобы $ не раскрывался на стороне Windows
rem Git Bash:    MSYS_NO_PATHCONV=1 ./w.cmd 'echo $HOME'
rem              без MSYS_NO_PATHCONV Git Bash портит Linux-пути
rem
rem Внутри команды НЕ использовать двойные кавычки: PowerShell 5.1 их портит.
rem Сложные конструкции выносить в scripts/*.sh или python -m.
rem --exec нужен, чтобы wsl.exe не пропускал строку через второй shell
rem (иначе переменные раскрываются дважды). %%~1 снимает внешние кавычки аргумента.
wsl.exe -d Ubuntu -u dockhub --cd /home/dockhub/avito-candgen --exec bash -lc "source scripts/env.sh 2>/dev/null; %~1"
