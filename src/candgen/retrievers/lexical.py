"""Лексические источники: TF-IDF поверх HashingVectorizer.

Почему хэширование, а не обычный TfidfVectorizer:
  * char n-граммы 3–5 по 189 тыс. документов дают миллионы признаков. Словарь
    в памяти Python медленный и тяжёлый, а HashingVectorizer его не хранит;
  * он не имеет состояния, поэтому корпус векторизуется параллельно по кускам
    (joblib). На 14 процессах ~10 с вместо ~80 с;
  * 2**22 корзин при ~5 млн различных n-грамм дают редкие коллизии, на качестве
    поиска это не сказывается.
IDF считается по корпусу объявлений по формуле sklearn (smooth idf):
idf = ln((1 + n) / (1 + df)) + 1; tf -> 1 + ln(tf) (sublinear), строки
L2-нормируются. Тогда косинус «запрос–документ» — это просто скалярное
произведение строк. Всё считается на месте в float32: TfidfTransformer
делал бы копии в float64, а на 300+ млн ненулей это лишние гигабайты памяти.

Матрица корпуса кэшируется в artifacts/lex/<источник>_<хэш настроек>/ в формате
CSC: для конкретного набора запросов из неё быстро вырезаются только столбцы,
встречающиеся в запросах (`restrict_to_queries`). Остальные n-граммы на скор
не влияют, а матрица становится в разы меньше и помещается на GPU.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

# Версия реализации: входит в ключ кэша, чтобы старые индексы не подхватывались
# после изменения способа расчёта.
IMPL_VERSION = 2


def _make_hasher(src_cfg: dict) -> HashingVectorizer:
    """Создаёт HashingVectorizer по настройкам источника из конфига.

    Тексты уже нормализованы (normalize_query), поэтому lowercase не нужен.
    Для word-анализатора токен — любая последовательность без пробелов
    (стандартный шаблон sklearn выбросил бы однобуквенные слова и цифры вроде «5»).
    """
    kw = dict(analyzer=src_cfg["analyzer"], ngram_range=tuple(src_cfg["ngram_range"]),
              n_features=src_cfg["n_features"], alternate_sign=False, norm=None,
              dtype=np.float32, lowercase=False)
    if src_cfg["analyzer"] == "word":
        kw["token_pattern"] = r"\S+"
    return HashingVectorizer(**kw)


def _hash_chunk(hasher: HashingVectorizer, texts: list[str]) -> sp.csr_matrix:
    """Векторизует кусок текстов (функция верхнего уровня — нужна joblib)."""
    return hasher.transform(texts)


class HashedTfidf:
    """TF-IDF-источник: хэширование n-грамм + IDF, выученный на корпусе.

    Атрибуты: name — имя источника (char, word, title); src_cfg — его настройки;
    hasher — HashingVectorizer; idf — вектор IDF float32 (после fit).
    """

    def __init__(self, name: str, src_cfg: dict, sublinear_tf: bool):
        """Вход: имя источника, его секция конфига, флаг sublinear tf (1+log tf)."""
        self.name = name
        self.src_cfg = src_cfg
        self.sublinear_tf = sublinear_tf
        self.hasher = _make_hasher(src_cfg)
        self.idf: np.ndarray | None = None

    def _weight_inplace(self, C: sp.csr_matrix) -> sp.csr_matrix:
        """Частоты -> TF-IDF с L2-нормой строк, на месте (без копий данных)."""
        if self.sublinear_tf:
            np.log(C.data, out=C.data)
            C.data += 1.0
        C.data *= self.idf[C.indices]
        return normalize(C, norm="l2", copy=False)

    def counts(self, texts: list[str], n_jobs: int = 1, chunk_size: int = 4000) -> sp.csr_matrix:
        """Матрица частот n-грамм (строки — тексты); параллельно по кускам, если текстов много."""
        if n_jobs <= 1 or len(texts) <= chunk_size:
            return self.hasher.transform(texts)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        parts = Parallel(n_jobs=n_jobs)(delayed(_hash_chunk)(self.hasher, c) for c in chunks)
        return sp.vstack(parts, format="csr")

    def fit_transform(self, texts: list[str], n_jobs: int, chunk_size: int) -> sp.csr_matrix:
        """Учит IDF на корпусе и возвращает L2-нормированную TF-IDF матрицу (float32).

        df — число документов с признаком: у CSR после суммирования дублей
        каждый признак встречается в строке не более одного раза.
        """
        C = self.counts(texts, n_jobs, chunk_size)
        C.sum_duplicates()
        n_docs = C.shape[0]
        df = np.bincount(C.indices, minlength=C.shape[1])
        self.idf = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)
        return self._weight_inplace(C)

    def transform(self, texts: list[str], n_jobs: int = 1, chunk_size: int = 4000) -> sp.csr_matrix:
        """TF-IDF запросов с IDF корпуса (L2-норма, float32)."""
        return self._weight_inplace(self.counts(texts, n_jobs, chunk_size))


def index_key(docs_cfg: dict, src_cfg: dict, sublinear_tf: bool) -> str:
    """Короткий хэш настроек, от которых зависит матрица корпуса (имя папки кэша)."""
    payload = json.dumps([docs_cfg, src_cfg, sublinear_tf, IMPL_VERSION], sort_keys=True,
                         ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def build_or_load_corpus_index(cfg: dict, name: str,
                               get_doc_texts: Callable[[dict], list[str]]) -> tuple[HashedTfidf, sp.csc_matrix]:
    """Возвращает источник и TF-IDF матрицу корпуса (CSC), строит при отсутствии кэша.

    Вход: конфиг; имя источника (ключ lexical.sources); функция, которая по
          настройкам источника возвращает тексты документов в порядке корпуса.
    Выход: (HashedTfidf с выученным IDF, матрица N_items × n_features в CSC).
    """
    lcfg = cfg["lexical"]
    src = lcfg["sources"][name]
    # Ключ кэша — от реально используемых полей: у источника со своими fields
    # смена docs.fields не должна пересобирать индекс.
    docs_eff = {**cfg["docs"], **({"fields": src["fields"]} if "fields" in src else {})}
    cache = (Path(cfg["paths"]["artifacts_dir"]) / "lex"
             / f"{name}_{index_key(docs_eff, src, lcfg['sublinear_tf'])}")
    if (cache / "X_csc.npz").exists():
        return joblib.load(cache / "model.joblib"), sp.load_npz(cache / "X_csc.npz")

    t0 = time.time()
    texts = get_doc_texts(src)
    model = HashedTfidf(name, src, lcfg["sublinear_tf"])
    X = model.fit_transform(texts, cfg["n_jobs"], lcfg["chunk_size"])
    del texts
    X = X.tocsc()  # CSR освобождается сразу после конвертации
    cache.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cache / "model.joblib")
    sp.save_npz(cache / "X_csc.npz", X, compressed=False)
    print(f"[lexical] индекс {name}: {X.shape[0]} док., nnz={X.nnz:,}, "
          f"{time.time() - t0:.0f} с -> {cache}")
    return model, X


def restrict_to_queries(X_csc: sp.csc_matrix, Q: sp.csr_matrix) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Оставляет только признаки, которые есть хотя бы в одном запросе.

    Скалярное произведение от этого не меняется: у остальных признаков вес
    запроса нулевой.

    Вход: матрица корпуса (CSC), матрица запросов (CSR) в одном пространстве признаков.
    Выход: (корпус N × V_q в CSR, запросы n_q × V_q в CSR).
    """
    cols = np.unique(Q.indices)
    return X_csc[:, cols].tocsr(), Q[:, cols].tocsr()
