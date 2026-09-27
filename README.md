# Авито: кандидатогенерация для поиска услуг (Recall@50)

Для каждого запроса бенчмарка (`benchmark_queries.parquet`, 2 452 шт.) нужно вернуть до 50 уникальных объявлений из корпуса (`benchmark_items.parquet`, 189 212 шт.). Метрика — Recall@50: доля релевантных объявлений запроса, попавших в top-50, усреднённая по запросам. Обучающие данные — `train.parquet` (497 673 пары «запрос → выбранное объявление»).

> Полная схема пайплайна и итоговая таблица ablation будут в финальной версии; журнал экспериментов — `EXPERIMENTS.md`.

## Решение (состояние после этапа 3: лексика + dense + doc expansion + приоры)

Для каждого запроса скорится **весь корпус** (189 тыс. объявлений), а не top-N текстового поиска. Приоры сильно переставляют объявления: нужное может быть 2 000-м по тексту, но единственным в городе запроса.

```
score(q, d) = cos_char(q, d) + 0.61·cos_word(q, d)
            + 0.22·cos_USER(q, d)                 # dense: deepvk/USER-base
            + 0.39·cos_expansion(q, d)            # запросы train, приводившие к объявлению
            + 0.067·log(P(loc_d | loc_q) + 3e-6)
            + 0.012·log(P_mc(mc_d | q) + 8.6e-4)
            + 0.31 ·bonus_filter(q, d)
```

- **cos_USER** — косинус эмбеддингов deepvk/USER-base (запрос с префиксом «query: », объявление «passage: » + заголовок + параметры[:300], до 128 токенов, fp16). Эмбеддинги корпуса кэшируются, кодирование занимает ~4 мин.
- **cos_expansion** — doc expansion: объявлению, которое выбирали в train, приписан текст приводивших к нему запросов (до 30 самых частых), char TF-IDF. Помогает только объявлениям, встречавшимся в train; на новых объявлениях recall немного падает (см. `EXPERIMENTS.md`, item-срезы).
- Проверено и не вошло в итог: e5-base поверх USER-base (+0,12 п.п.), «память» по тексту запроса (+0,01), RRF и квоты вместо линейной суммы (хуже).

- **Текст объявления:** заголовок ×2 + параметры[:400] (без служебного шума: график работы, тип стоимости, адрес) + описание целиком, нормализованный (нижний регистр, ё→е, только буквы и цифры).
- **cos_char:** TF-IDF по char_wb 3–5-граммам, устойчив к опечаткам и склейкам. **cos_word** — TF-IDF по словам и биграммам лемм (pymorphy3). Оба источника построены на HashingVectorizer, IDF считается по корпусу.
- **P(loc_d | loc_q)** — матрица переходов «локация поиска → локация объявления» по кликам train. В корпусе нет объявлений в 17% локаций поиска (регионы-агрегаты), поэтому сравнивать локации на равенство нельзя.
- **P_mc** — распределение подкатегорий по 20 ближайшим текстам запросов train (char 2–4 TF-IDF, kNN на GPU).
- **bonus_filter** — по 0,5 за «Вид услуги» и «Тип услуги» из фильтра поиска, если пара «ключ значение» есть в параметрах объявления.
- **Скоринг:** батч из 512 запросов × весь корпус на GPU. Разреженная матрица корпуса (только признаки, встречающиеся в запросах) умножается на плотный батч запросов, прибавляются приоры, берётся `torch.topk`. На 2 452 запроса бенчмарка уходит ~25 с вместе с подготовкой.
- **Веса и eps** подобраны Optuna (TPE, seed 42) по метрике `bench_adj`. Честный прирост подбора на отложенной половине валидации: +1,47 п.п. на этапе 2, +1,85 п.п. на этапе 3.

Валидация: 5 000 unseen- и 1 500 seen-запросов, отложенных из train (`src/candgen/validation.py`, отчёт — `reports/eda.md`). Метрика решений `bench_adj` — recall@50 по 4 ячейкам (seen/unseen × фильтр есть/нет) с весами бенчмарка.

## Как запустить (чистый Linux, Python 3.12, NVIDIA GPU)

```bash
bash scripts/install.sh      # .venv + пинованные зависимости из requirements.txt
bash scripts/setup_data.sh   # распаковка data/raw/dataset.zip (идемпотентно) + проверки данных
bash scripts/check_env.sh    # отчёт о среде: GPU, RAM, версии, CUDA
```

Анализ данных и валидация (из корня, после `source scripts/env.sh`):

```bash
python -m candgen.eda                        # reports/eda.md + reports/fig/*.png
python -m candgen.validation --build         # срезы валидации -> artifacts/val_*
python -m candgen.validation --selftest      # проверка метрики и детерминизма срезов
python -m candgen.pipeline                   # recall@50 на валидации для весов из конфига
python -m candgen.pipeline --ablation        # лестница baseline (наборы — experiments.* в конфиге)
python -m candgen.tune                       # подбор весов Optuna -> reports/tune_ctx.json
python -m candgen.submit                     # answer.csv по всему train + обязательная проверка формата
```

Серии экспериментов этапа 2 воспроизводятся скриптами `scripts/experiments_stage2*.sh`. Для построения индексов с полным описанием нужно ~12 ГБ RAM.

Все скрипты сами подключают `scripts/env.sh`: venv, `PYTHONPATH=src`, `PYTHONHASHSEED=42`, offline-режим HuggingFace. Модули запускаются как `python -m candgen.<модуль> --config configs/default.yaml`.

Разработка велась из Windows: Claude Code/VS Code в Windows, вычисления в WSL2 Ubuntu. Команды проходят через мост `w.cmd` (см. `CLAUDE.md`). Для запуска на Linux он не нужен.

## Структура

```
configs/default.yaml   все пути и гиперпараметры
scripts/               install / setup_data / check_env / bg (фоновые задачи) / env
src/candgen/           код пакета (io, text, validation, priors, retrievers, fusion, submit)
reports/               отчёты EDA и экспериментов
artifacts/, logs/      кэши и логи (не в git)
```

## Использованные компоненты

| Компонент | Версия | Лицензия | Где используется |
|---|---|---|---|
| Python | 3.12.3 | PSF | вся кодовая база |
| pandas | 3.0.6 | BSD-3-Clause | таблицы, агрегаты |
| pyarrow | 25.0.1 | Apache-2.0 | чтение parquet, векторные строковые операции |
| numpy | 2.5.3 | BSD-3-Clause | численные операции, top-K |
| PyYAML | 6.0.3 | MIT | конфиги |
| PyTorch | 2.11.0+cu128 | BSD-3-Clause | скоринг всего корпуса на GPU: sparse CSR × dense батч, top-k; kNN подкатегорий |
| scipy | 1.18.1 | BSD-3-Clause | разреженные TF-IDF матрицы (CSR/CSC) |
| scikit-learn | 1.9.1 | BSD-3-Clause | HashingVectorizer + TfidfTransformer (char_wb 3–5, word 1–2), TfidfVectorizer char 2–4 для kNN подкатегорий |
| joblib | (зависимость sklearn) | BSD-3-Clause | параллельная векторизация корпуса, сохранение моделей |
| polars | 1.44.2 | MIT | установлен, используется по необходимости |
| duckdb | 1.5.5 | MIT | установлен, используется по необходимости |
| pymorphy3 (+ pymorphy3-dicts-ru) | 2.0.6 | MIT (словари OpenCorpora — CC BY-SA) | лемматизация для word-источника (кэш по уникальным словам) |
| snowballstemmer | 3.1.1 | BSD-3-Clause | установлен, не понадобился (лемматизации достаточно) |
| RapidFuzz | 3.14.6 | MIT | нечёткое сравнение строк (по необходимости) |
| bm25s | 0.3.11 | MIT | установлен; опциональное сравнение с BM25 на этапе 2 не проводилось |
| sentence-transformers | 6.1.0 | Apache-2.0 | dense-эмбеддинги запросов и объявлений (этап 3) |
| intfloat/multilingual-e5-base (модель) | snapshot d1287505 | MIT | dense-поиск, префиксы «query: » / «passage: » (Wang et al., 2024, arXiv:2402.05672) |
| deepvk/USER-base (модель) | snapshot e8446472 | Apache-2.0 | dense-поиск, сравнение с e5 (русскоязычный энкодер) |
| intfloat/multilingual-e5-small (модель) | snapshot 614241f6 | MIT | скачана как облегчённая альтернатива |
| huggingface-hub | (зависимость) | Apache-2.0 | разовое скачивание моделей (`scripts/download_models.sh`) |
| transformers | 5.17.0 | Apache-2.0 | зависимость sentence-transformers |
| LightGBM | 4.7.0 | MIT | предранкер (этап 5, опционально) |
| Optuna | 5.0.0 | MIT | подбор весов слияния, TPE (этап 2) |
| matplotlib | 3.11.2 | Matplotlib License (BSD-совместимая) | графики EDA |
| tqdm | 4.70.1 | MPL-2.0 / MIT | прогресс-бары |
