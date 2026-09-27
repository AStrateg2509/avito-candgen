"""Загрузка конфига и сырых таблиц проекта.

Главные гарантии модуля:
  * `query_id` и `item_id` всегда строки (никаких преобразований в число);
  * читаются только запрошенные колонки: train весит ~0.5 ГБ в parquet
    и несколько ГБ в памяти, если загружать все тексты.

Запуск как скрипта: `python -m candgen.io --describe` печатает размеры,
схемы таблиц и базовые проверки id (используется в scripts/setup_data.sh).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

# Колонки-идентификаторы, которые всегда приводятся к str.
ID_COLUMNS = ("query_id", "item_id")
# Форматы id (проверено на данных): item_id — 16 hex-символов в нижнем
# регистре, query_id — 16 символов base62 (регистр значим, не менять!).
ITEM_ID_REGEX = r"^[0-9a-f]{16}$"
QUERY_ID_REGEX = r"^[0-9A-Za-z]{16}$"
# Логические имена таблиц -> ключи в cfg["paths"].
TABLES = {"train": "train", "queries": "queries", "items": "items"}


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Читает YAML-конфиг проекта.

    Вход: путь к yaml (по умолчанию configs/default.yaml от корня проекта).
    Выход: словарь с настройками (пути, seed, гиперпараметры).
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Читает parquet в pandas и приводит id-колонки к строкам.

    Зачем отдельная функция: единая точка, где гарантируется `dtype=str`
    для query_id/item_id, и где можно ограничить набор колонок.
    Цена и координаты лежат в parquet как decimal128. В pandas они стали бы
    медленными объектами Decimal, поэтому сразу приводятся к float64
    (точности float64 для цены и координат с запасом хватает).

    Вход: путь к parquet, список колонок (None — все).
    Выход: DataFrame; колонки из ID_COLUMNS имеют строковый тип.
    """
    table = pq.read_table(path, columns=columns)
    for i, field in enumerate(table.schema):
        if pa.types.is_decimal(field.type):
            table = table.set_column(i, field.name, pc.cast(table.column(i), pa.float64()))
    df = table.to_pandas()
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def load_train(cfg: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """train.parquet: пары «запрос -> выбранное объявление» (см. read_table)."""
    return read_table(cfg["paths"]["train"], columns)


def load_queries(cfg: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """benchmark_queries.parquet: запросы, для которых строим ответ."""
    return read_table(cfg["paths"]["queries"], columns)


def load_items(cfg: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """benchmark_items.parquet: корпус объявлений, из которого выбираем top-50."""
    return read_table(cfg["paths"]["items"], columns)


def _logical_type(arrow_type: pa.DataType) -> str:
    """Имя типа без различия string/large_string (у них одинаковый смысл,
    отличается только размер смещений внутри arrow)."""
    return "string" if pa.types.is_large_string(arrow_type) else str(arrow_type)


def _check_ids(path: str, col: str, regex: str, must_be_unique: bool) -> list[str]:
    """Проверяет колонку-идентификатор прямо в arrow, без pandas.

    Вход: путь к parquet, имя колонки, регулярка формата, нужна ли уникальность.
    Выход: список текстов ошибок (пустой — всё в порядке).
    """
    errors = []
    arr = pq.read_table(path, columns=[col]).column(col)
    if not str(arr.type).startswith(("string", "large_string")):
        errors.append(f"{col}: тип {arr.type}, ожидалась строка")
        arr = pc.cast(arr, "string")
    n_null = arr.null_count
    if n_null:
        errors.append(f"{col}: {n_null} пустых значений")
    n_bad = len(arr) - n_null - pc.sum(pc.match_substring_regex(arr, regex)).as_py()
    if n_bad:
        errors.append(f"{col}: {n_bad} значений не соответствуют {regex}")
    if must_be_unique:
        n_unique = pc.count_distinct(arr).as_py()
        if n_unique != len(arr):
            errors.append(f"{col}: {len(arr) - n_unique} дублей")
    return errors


def describe(cfg: dict) -> int:
    """Печатает размеры и схемы трёх таблиц и проверяет данные.

    Проверки: число строк совпадает с cfg["data_check"]["expected_rows"];
    item_id в корпусе и query_id в бенчмарке уникальны; item_id — hex16,
    query_id — base62 длины 16;
    одноимённые колонки и колонки location_id/microcat_id имеют одинаковый
    тип во всех таблицах (иначе join и сравнения будут молча ломаться).

    Вход: конфиг. Выход: код возврата (0 — всё в порядке, 1 — есть ошибки).
    """
    expected = cfg.get("data_check", {}).get("expected_rows", {})
    errors: list[str] = []
    types: dict[str, dict[str, str]] = {}  # имя колонки -> {таблица: тип}

    for name, key in TABLES.items():
        path = cfg["paths"][key]
        pf = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows
        status = ""
        if name in expected:
            ok = n_rows == expected[name]
            status = "OK" if ok else f"ОЖИДАЛОСЬ {expected[name]}"
            if not ok:
                errors.append(f"{name}: {n_rows} строк, ожидалось {expected[name]}")
        print(f"\n=== {name}: {path} ===")
        print(f"shape = ({n_rows}, {len(pf.schema_arrow)})  {status}  row_groups={pf.num_row_groups}")

        # dtypes pandas берём по маленькому батчу, чтобы не грузить таблицу целиком.
        sample = next(pf.iter_batches(batch_size=5)).to_pandas()
        for field in pf.schema_arrow:
            print(f"  {field.name:32s} {str(field.type):14s} -> {sample[field.name].dtype}")
            types.setdefault(field.name, {})[name] = _logical_type(field.type)

    # Проверки идентификаторов.
    errors += _check_ids(cfg["paths"]["items"], "item_id", ITEM_ID_REGEX, must_be_unique=True)
    errors += _check_ids(cfg["paths"]["queries"], "query_id", QUERY_ID_REGEX, must_be_unique=True)
    errors += _check_ids(cfg["paths"]["train"], "item_id", ITEM_ID_REGEX, must_be_unique=False)

    # Одинаковые колонки в разных таблицах должны иметь один тип.
    for col, by_table in types.items():
        if len(set(by_table.values())) > 1:
            errors.append(f"колонка {col} имеет разные типы: {by_table}")
    # location_id поиска и объявления сравниваются между собой — тип должен совпадать.
    loc_types = {c: t for c, d in types.items() for t in d.values() if c.endswith("location_id")}
    if len(set(loc_types.values())) > 1:
        errors.append(f"location_id разных типов: {loc_types}")

    print("\n=== Проверки ===")
    if errors:
        for e in errors:
            print("ОШИБКА:", e)
        return 1
    print("OK: число строк, формат и уникальность id, согласованность типов")
    return 0


def main() -> None:
    """CLI: `python -m candgen.io --describe [--config path]`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--describe", action="store_true", help="схемы таблиц и проверки данных")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.describe:
        sys.exit(describe(cfg))
    parser.print_help()


if __name__ == "__main__":
    main()
