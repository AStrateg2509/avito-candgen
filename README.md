# Авито: кандидатогенерация для поиска услуг (Recall@50)

Для каждого запроса бенчмарка (`benchmark_queries.parquet`, 2 452 шт.) нужно вернуть до 50 уникальных объявлений из корпуса (`benchmark_items.parquet`, 189 212 шт.). Метрика — Recall@50: доля релевантных объявлений запроса, попавших в top-50, усреднённая по запросам. Обучающие данные — `train.parquet` (497 673 пары «запрос → выбранное объявление»).

> Раздел описания решения, схема пайплайна и таблица ablation будут заполнены по ходу этапов (журнал — `EXPERIMENTS.md`).

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
```

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
| PyTorch | 2.11.0+cu128 | BSD-3-Clause | dense-поиск на GPU (этап 3) |
| scipy | 1.18.1 | BSD-3-Clause | разреженные матрицы (этап 2) |
| scikit-learn | 1.9.1 | BSD-3-Clause | TF-IDF, kNN (этап 2) |
| polars | 1.44.2 | MIT | установлен, используется по необходимости |
| duckdb | 1.5.5 | MIT | установлен, используется по необходимости |
| pymorphy3 (+ pymorphy3-dicts-ru) | 2.0.6 | MIT (словари OpenCorpora — CC BY-SA) | лемматизация (этап 2) |
| snowballstemmer | 3.1.1 | BSD-3-Clause | стемминг (этап 2, сравнение) |
| RapidFuzz | 3.14.6 | MIT | нечёткое сравнение строк (по необходимости) |
| bm25s | 0.3.11 | MIT | BM25 (этап 2, сравнение) |
| sentence-transformers | 6.1.0 | Apache-2.0 | эмбеддинги (этап 3) |
| transformers | 5.17.0 | Apache-2.0 | зависимость sentence-transformers |
| LightGBM | 4.7.0 | MIT | предранкер (этап 5, опционально) |
| Optuna | 5.0.0 | MIT | подбор весов (этап 2) |
| matplotlib | 3.11.2 | Matplotlib License (BSD-совместимая) | графики EDA |
| tqdm | 4.70.1 | MPL-2.0 / MIT | прогресс-бары |
