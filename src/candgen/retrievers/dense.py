"""Dense-поиск: эмбеддинги sentence-transformers (E5 / USER) на GPU.

Модели заранее скачаны scripts/download_models.sh в models/hf и грузятся
офлайн. Запрос и документ кодируются со своими префиксами («query: » /
«passage: » у E5), эмбеддинги L2-нормированы, поэтому косинус — это
скалярное произведение. Скор «запрос × весь корпус» считает LinearScorer
(плотное умножение B × d на d × N в fp16).

Эмбеддинги корпуса кэшируются в artifacts/dense/emb_items_<модель>_<хэш>.npy
(float16, 189 тыс. × 768 ≈ 290 МБ): кодирование корпуса делается один раз.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from candgen.text import build_raw_doc_texts


class DenseModel:
    """Обёртка над SentenceTransformer: загрузка офлайн, fp16, кодирование.

    Атрибуты: name — имя модели из конфига; mcfg — её настройки (hf_id,
    префиксы, max_len); model — загруженный SentenceTransformer.
    """

    def __init__(self, name: str, mcfg: dict, device: torch.device, fp16: bool, batch_size: int):
        """Загружает модель из локального кэша HF (сеть не нужна)."""
        from sentence_transformers import SentenceTransformer  # тяжёлый импорт — только по требованию

        self.name, self.mcfg, self.batch_size = name, mcfg, batch_size
        try:
            self.model = SentenceTransformer(mcfg["hf_id"], device=str(device))
        except TypeError:
            # Репозитории в старом формате (например, deepvk/USER-base: модуль
            # Normalize с path "") sentence-transformers 6.x читает неверно.
            # Собираем Transformer + Pooling вручную; нормировку делает encode().
            self.model = self._build_manually(mcfg, device)
        self.model.max_seq_length = mcfg["max_len"]
        if fp16 and device.type == "cuda":
            self.model.half()

    @staticmethod
    def _build_manually(mcfg: dict, device: torch.device):
        """SentenceTransformer из модулей Transformer + Pooling (конфиг пулинга из 1_Pooling)."""
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer, models

        path = snapshot_download(mcfg["hf_id"])  # офлайн: путь к локальному кэшу
        word = models.Transformer(path, max_seq_length=mcfg["max_len"])
        pool = models.Pooling.load(str(Path(path) / "1_Pooling"))
        return SentenceTransformer(modules=[word, pool], device=str(device))

    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        """Эмбеддинги текстов с префиксом, L2-нормированные, float16 [n × d]."""
        emb = self.model.encode([prefix + t for t in texts], batch_size=self.batch_size,
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)
        return emb.astype(np.float16)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Эмбеддинги запросов (префикс query_prefix)."""
        return self.encode(texts, self.mcfg["query_prefix"])

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        """Эмбеддинги документов (префикс doc_prefix)."""
        return self.encode(texts, self.mcfg["doc_prefix"])

    def close(self) -> None:
        """Освобождает память GPU, занятую моделью (перед загрузкой скорера)."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _local_model_stamp(mcfg: dict) -> str:
    """Для локальной (дообученной) модели — время изменения её весов, иначе "".

    Входит в ключ кэша эмбеддингов: переобучили модель по тому же пути —
    эмбеддинги корпуса пересчитаются, а не возьмутся устаревшие.
    """
    path = Path(mcfg["hf_id"])
    if not path.is_dir():
        return ""
    weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin"))
    return str(max(w.stat().st_mtime_ns for w in weights)) if weights else ""


def train_item_embeddings(cfg: dict, name: str, model: DenseModel, train_items: pd.DataFrame) -> np.ndarray:
    """Эмбеддинги уникальных объявлений train (для фишки 4), с кэшем на диске.

    Текст документа — тот же, что у корпуса (dense.doc_fields), поэтому
    объявления train и корпуса лежат в одном пространстве. Ключ кэша зависит
    и от списка объявлений (train фиксирован, но так надёжнее).

    Вход: конфиг, имя модели, загруженная модель, таблица уникальных
          объявлений train (item_id + поля документа) в фиксированном порядке.
    Выход: float16 [n_train_items × d] в порядке строк train_items.
    """
    dcfg = cfg["dense"]
    ids_hash = hashlib.sha1("".join(train_items["item_id"]).encode()).hexdigest()[:8]
    payload = json.dumps([dcfg["models"][name], dcfg["doc_fields"], ids_hash], sort_keys=True,
                         ensure_ascii=False)
    key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    path = Path(cfg["paths"]["artifacts_dir"]) / "dense" / f"emb_train_items_{name}_{key}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    emb = model.encode_docs(build_raw_doc_texts(train_items, dcfg["doc_fields"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    print(f"[dense] {name}: объявления train ({len(train_items)}) закодированы за {time.time() - t0:.0f} с -> {path}")
    return emb


def item_embeddings(cfg: dict, name: str, model: DenseModel, items: pd.DataFrame) -> np.ndarray:
    """Эмбеддинги корпуса для модели: из кэша или кодированием (с печатью скорости).

    Ключ кэша — хэш настроек модели и полей документа: поменяли поля или
    max_len — эмбеддинги пересчитаются.

    Вход: конфиг, имя модели, загруженная модель, корпус.
    Выход: float16 [N × d] в порядке строк items.
    """
    dcfg = cfg["dense"]
    stamp = _local_model_stamp(dcfg["models"][name])
    # метка добавляется только локальным моделям: ключи кэша моделей с HF не меняются
    payload = json.dumps([dcfg["models"][name], dcfg["doc_fields"]] + ([stamp] if stamp else []),
                         sort_keys=True, ensure_ascii=False)
    key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    path = Path(cfg["paths"]["artifacts_dir"]) / "dense" / f"emb_items_{name}_{key}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    emb = model.encode_docs(build_raw_doc_texts(items, dcfg["doc_fields"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    dt = time.time() - t0
    print(f"[dense] {name}: корпус {len(items)} док. закодирован за {dt:.0f} с "
          f"({len(items) / dt:.0f} док/с) -> {path}")
    return emb
