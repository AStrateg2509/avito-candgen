#!/usr/bin/env bash
# Сборка файлов для выгрузки в облако -> dist/ (в git не идёт).
#
# Использование:  bash scripts/pack_release.sh
# Код берётся из последнего коммита (git archive HEAD), поэтому сначала коммит.
#
# Что получается в dist/:
#   answer.csv                            финальный ответ (копия answer.csv из корня)
#   avito-candgen_code.zip                код, конфиги и документация из HEAD
#   avito-candgen_models_s10.tar.gz       дообученные модели, на которых собран финальный сабмит s10
#   avito-candgen_models_rerun.tar.gz     те же модели, заново обученные run_all.sh --clean (альтернатива s10)
#   avito-candgen_models_research.tar.gz  модели исследований (e5_small_ft_*, cross-encoder) + кэш скоров cross-encoder
#   avito-candgen_models_hf.tar.gz        базовые модели HuggingFace (кэш models/hf, без него нужен download_models.sh)
#   SHA256SUMS, README.md                 контрольные суммы и пояснения
# Архивы моделей распаковываются из корня проекта: tar -xzf <архив>.
# Архивы s10 и rerun кладут модели по одним и тем же путям — распаковывать один из двух.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT=dist
FT=models/finetuned
# Копия моделей s10, сделанная перед прогоном run_all.sh --clean (он переобучил модели в models/finetuned).
S10=models/backup_s10
mkdir -p "$OUT"

pack() {
    # pack <имя архива> <правило --transform или ""> <пути...>: tar.gz из корня проекта.
    # gzip -1: веса float32 почти не сжимаются, более сильное сжатие лишь тратит время.
    local name="$1" transform="$2"; shift 2
    local opts=() paths=()
    [ -n "$transform" ] && opts+=(--transform="$transform")  # путь внутри архива != путь на диске
    for p in "$@"; do
        if [ -e "$p" ]; then paths+=("$p"); else echo "!!! нет $p — пропускаю" >&2; fi
    done
    local t=$(date +%s)
    tar --use-compress-program='gzip -1' "${opts[@]}" -cf "$OUT/$name" "${paths[@]}"
    echo "=== $name: $(du -h "$OUT/$name" | cut -f1), $(( $(date +%s) - t )) с"
}

cp answer.csv "$OUT/answer.csv"
git archive --format=zip --prefix=avito-candgen/ -o "$OUT/avito-candgen_code.zip" HEAD
echo "=== avito-candgen_code.zip: коммит $(git rev-parse --short HEAD)"

pack avito-candgen_models_s10.tar.gz "s,^$S10/,$FT/," \
    "$S10/e5_small_ftd_val_cold" "$S10/e5_small_ftd_full"
pack avito-candgen_models_rerun.tar.gz "" \
    "$FT/e5_small_ftd_val_cold" "$FT/e5_small_ftd_full"
pack avito-candgen_models_research.tar.gz "s,^$S10/ce_cache,artifacts/ce_cache," \
    "$FT/e5_small_ft_val" "$FT/e5_small_ft_val_cold" "$FT/e5_small_ft_full" "$FT/crossenc_val_cold" \
    "$S10/ce_cache"
# Только hub/: рядом huggingface_hub кладёт служебные файлы (логи xet, реестр агентов), моделям они не нужны.
pack avito-candgen_models_hf.tar.gz "" models/hf/hub

cat > "$OUT/README.md" <<'EOF'
# avito-candgen: файлы для выгрузки

Финальный сабмит — s10, лидерборд Recall@50 = 0,8963.

| файл | что внутри | куда |
|---|---|---|
| `answer.csv` | финальный ответ s10 (ровно тот, что загружен) | — |
| `avito-candgen_code.zip` | код, конфиги, README, журнал экспериментов | распаковать в любую папку |
| `avito-candgen_models_s10.tar.gz` | две дообученные e5-small (`e5_small_ftd_val_cold`, `e5_small_ftd_full`), на которых собран s10 | `tar -xzf` из корня проекта → `models/finetuned/` |
| `avito-candgen_models_rerun.tar.gz` | те же модели, заново обученные `run_all.sh --clean`; качество то же, ответ совпадает с s10 на 95% | вместо s10: те же пути |
| `avito-candgen_models_research.tar.gz` | модели исследований: e5-small этапа 4 (s3–s9), cross-encoder; кэш скоров cross-encoder | `tar -xzf` из корня проекта |
| `avito-candgen_models_hf.tar.gz` | базовые модели HuggingFace (e5-small, USER-base, e5-base) | `tar -xzf` из корня проекта → `models/hf/` |

Порядок: распаковать код → `bash scripts/install.sh` → распаковать `models_hf` (или `bash scripts/download_models.sh`) и `models_s10` → положить `dataset.zip` в `data/raw/` → `bash scripts/run_all.sh` (без `--clean` — модели не переобучаются, ~5 мин).

Проверка целостности: `sha256sum -c SHA256SUMS`.
EOF

(cd "$OUT" && sha256sum answer.csv *.zip *.tar.gz > SHA256SUMS)
echo "=== готово:"
ls -lh "$OUT"
