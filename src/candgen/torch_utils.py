"""Общие помощники для вычислений на torch (GPU, если есть)."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import torch

# Разреженные CSR-тензоры torch помечены как beta и предупреждают об этом при
# каждом создании; для нас это шум в логах (корректность проверяется метрикой).
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")


def get_device(cfg: dict) -> torch.device:
    """Устройство из конфига: 'auto' -> cuda при наличии, иначе cpu."""
    name = cfg.get("device", "auto")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def to_torch_csr(X: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Переносит scipy CSR в torch sparse CSR на устройство.

    Индексы int32 вдвое экономят память по сравнению с int64 по умолчанию;
    для наших размеров (nnz < 2**31) этого достаточно.
    """
    X = X.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(X.indptr.astype(np.int32)),
        torch.from_numpy(X.indices.astype(np.int32)),
        torch.from_numpy(X.data.astype(np.float32)),
        size=X.shape,
    ).to(device)


def dense_batch_T(Q: sp.csr_matrix, rows: slice, device: torch.device) -> torch.Tensor:
    """Плотный транспонированный кусок строк разреженной матрицы: (n_features × B).

    Нужен для torch.sparse.mm(корпус_CSR, запросы) -> (N × B).
    """
    return torch.from_numpy(np.ascontiguousarray(Q[rows].toarray().T)).to(device)
