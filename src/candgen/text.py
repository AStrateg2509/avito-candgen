"""Нормализация текстов, разбор фильтров поиска, тексты документов, леммы.

Что здесь есть:
  * `normalize_query` / `normalize_series` — единое понятие «тот же текст»
    (срезы seen/unseen) и единая нормализация запросов и документов;
  * `parse_filters` — разбор `search_infm_params_text` на пары (ключ, значение);
  * `build_doc_texts` — текст объявления для лексического поиска
    (заголовок ×2 + параметры + описание, с обрезкой и опциональной очисткой);
  * `Lemmatizer` — лемматизация pymorphy3 с кэшем по уникальным словам.
"""

from __future__ import annotations

import re
from pathlib import Path

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


# --------------------------------------------------------------------------
# Тексты документов
# --------------------------------------------------------------------------

def build_doc_texts(items: pd.DataFrame, docs_cfg: dict) -> list[str]:
    """Собирает нормализованный текст каждого объявления для лексического поиска.

    Порядок действий для каждого поля из docs_cfg["fields"]:
    очистка шума (только параметры и только при clean_params) → обрезка до
    max_chars → повтор repeat раз. Поля склеиваются через пробел, результат
    проходит normalize_query. Обрезка идёт после очистки: служебный шум
    («График работы от 25200…») не должен съедать лимит символов.

    Вход: DataFrame корпуса с нужными колонками; секция конфига `docs`.
    Выход: список строк в порядке строк items.
    """
    noise = ([re.compile(p) for p in docs_cfg["params_noise"]]
             if docs_cfg.get("clean_params") else [])
    parts = []
    for field in docs_cfg["fields"]:
        col = items[field["col"]].fillna("").astype(str)
        if noise and field["col"] == "item_infm_params_text":
            col = col.map(lambda s: _remove_noise(s, noise))
        if field.get("max_chars"):
            col = col.str.slice(0, field["max_chars"])
        for _ in range(field.get("repeat", 1)):
            parts.append(col)
    joined = parts[0]
    for p in parts[1:]:
        joined = joined + " " + p
    return [normalize_query(t) for t in joined]


def build_raw_doc_texts(items: pd.DataFrame, fields: list[dict]) -> list[str]:
    """Сырой текст объявления для dense-модели: поля через «. », без нормализации.

    Трансформеры сами работают с регистром и пунктуацией, а нормализация
    (удаление знаков, ё→е) только уводила бы текст от того, на чём модель училась.

    Вход: корпус; список полей {col, max_chars} (секция dense.doc_fields).
    Выход: список строк в порядке строк items.
    """
    parts = []
    for field in fields:
        col = items[field["col"]].fillna("").astype(str)
        if field.get("max_chars"):
            col = col.str.slice(0, field["max_chars"])
        parts.append(col)
    joined = parts[0]
    for p in parts[1:]:
        joined = joined + ". " + p
    return joined.str.strip(" .").tolist()


def _remove_noise(text: str, patterns: list[re.Pattern]) -> str:
    """Вырезает из строки параметров все фрагменты, подходящие под регулярки шума."""
    for p in patterns:
        text = p.sub(" ", text)
    return text


# --------------------------------------------------------------------------
# Лемматизация
# --------------------------------------------------------------------------

_CYRILLIC_WORD = re.compile(r"^[а-я]+$")


class Lemmatizer:
    """Лемматизация pymorphy3 с кэшем «слово -> лемма».

    Зачем кэш: в корпусе ~0,5 млн уникальных словоформ на ~25 млн словоупотреблений.
    Разбирать каждое слово один раз в десятки раз быстрее, чем разбирать тексты.
    Кэш сохраняется в parquet и переиспользуется между запусками.
    Разбираются только чисто кириллические слова; числа, латиница и смешанные
    токены остаются как есть. «ё» в лемме заменяется на «е» (как в normalize_query).
    """

    def __init__(self, cache_path: str | Path | None = None):
        """Вход: путь к parquet-кэшу (None — без сохранения на диск)."""
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            self.cache = dict(zip(df["word"], df["lemma"]))
        self._morph = None

    def _lemma(self, word: str) -> str:
        """Лемма одного слова (без кэша)."""
        if not _CYRILLIC_WORD.match(word):
            return word
        if self._morph is None:
            import pymorphy3  # импорт здесь: грузит словари ~1 с, нужен не всегда
            self._morph = pymorphy3.MorphAnalyzer()
        return self._morph.parse(word)[0].normal_form.replace("ё", "е")

    def lemmatize_texts(self, texts: list[str]) -> list[str]:
        """Лемматизирует нормализованные тексты (слова через пробел).

        Вход: список строк после normalize_query.
        Выход: список строк той же длины, каждое слово заменено леммой.
        Новые слова добавляются в кэш; кэш сохраняется, если что-то добавилось.
        """
        vocab = set()
        for t in texts:
            vocab.update(t.split())
        new_words = [w for w in vocab if w not in self.cache]
        for w in new_words:
            self.cache[w] = self._lemma(w)
        if new_words and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"word": list(self.cache), "lemma": list(self.cache.values())}).to_parquet(
                self.cache_path, index=False)
        c = self.cache
        return [" ".join(c[w] for w in t.split()) for t in texts]
