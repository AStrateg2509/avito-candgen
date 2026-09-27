"""Нормализация текстов и разбор фильтров поиска.

Этап 1 использует отсюда две вещи:
  * `normalize_query` / `normalize_series` — единое понятие «тот же текст
    запроса» для срезов seen/unseen (иначе «Ремонт холодильников» и
    «ремонт холодильников » считались бы разными запросами и давали утечку);
  * `parse_filters` — разбор `search_infm_params_text` на пары (ключ, значение).
На этапе 2 сюда добавится сборка текстов документов и лемматизация.
"""

from __future__ import annotations

import re

import pandas as pd

# Всё, что не латиница, не кириллица и не цифра, заменяется пробелом.
# «ё» к этому моменту уже заменена на «е», поэтому диапазона а-я достаточно.
_NON_ALNUM = re.compile(r"[^0-9a-zа-я]+")


def normalize_query(text: str) -> str:
    """Нормализует текст запроса: нижний регистр, ё→е, только буквы и цифры.

    Вход: исходная строка (может быть None).
    Выход: строка из слов, разделённых одиночными пробелами, без краевых пробелов.
    Пример: «Ремонт  Холодильников-Samsung!» → «ремонт холодильников samsung».
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("ё", "е")
    return _NON_ALNUM.sub(" ", text).strip()


def normalize_series(texts: pd.Series) -> pd.Series:
    """Векторная нормализация: считаем только по уникальным значениям.

    В train ~0.5 млн строк, но всего ~75 тыс. уникальных запросов, поэтому
    нормализуем словарь уникальных текстов и отображаем его на строки.

    Вход: Series строк. Выход: Series нормализованных строк с тем же индексом.
    """
    uniq = pd.unique(texts)
    mapping = {t: normalize_query(t) for t in uniq}
    return texts.map(mapping).astype(str)


def has_filter(params: pd.Series) -> pd.Series:
    """Признак «у запроса есть фильтр»: непустой search_infm_params_text.

    Строка вида «Вид услуги» без значения тоже считается фильтром: так
    считаются доли 67% / 37% из ТЗ, и так определены ячейки весов.

    Вход: Series текстов фильтров. Выход: булев Series.
    """
    return params.fillna("").astype(str).str.strip() != ""


def compile_filter_keys(keys: list[str]) -> re.Pattern:
    """Собирает регулярку, которая находит ключи фильтров в тексте.

    Ключи сортируются по убыванию длины, иначе «Тип услуги» перехватывал бы
    начало «Тип услуги автосервиса». Ключ ищется только как отдельная фраза:
    слева начало строки или пробел, справа пробел или конец строки.

    Вход: список ключей (из configs: filters.keys).
    Выход: скомпилированный re.Pattern с одной группой — найденным ключом.
    """
    alternation = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    return re.compile(rf"(?:^|(?<= ))({alternation})(?= |$)")


def parse_filters(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Разбирает строку фильтров на пары (ключ, значение).

    Значение ключа — текст до следующего известного ключа или до конца строки.
    Текст до первого ключа (редкие неизвестные ключи вроде «Аренда авто …»)
    отбрасывается. Ключ может повторяться (например, «Кто оказывает услуги
    Частный исполнитель Кто оказывает услуги Компания»), тогда пар несколько.

    Вход: строка search_infm_params_text, регулярка из compile_filter_keys.
    Выход: список пар в порядке появления; значение может быть пустым.
    Пример: «Тип услуги Сантехника Вид услуги Ремонт и отделка» →
            [("Тип услуги", "Сантехника"), ("Вид услуги", "Ремонт и отделка")].
    """
    if not isinstance(text, str) or not text:
        return []
    matches = list(pattern.finditer(text))
    pairs = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pairs.append((m.group(1), text[m.end():end].strip()))
    return pairs
