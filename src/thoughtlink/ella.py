from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linear import BinaryLogReg, SoftmaxReg, train_binary_logreg, train_softmax_reg


@dataclass(frozen=True)
class EllaBasis:
    """Shared low-rank basis for per-task linear models: w ~= L @ s.

    This is an ELLA-style representation (shared basis + task-specific coefficients).
    We keep it intentionally simple and dependency-free (numpy only).
    """

    L: np.ndarray  # (d, k)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project features into the latent space: z = x @ L (n, k)."""
        return (x @ self.L).astype(np.float32, copy=False)

    def reconstruct(self, s: np.ndarray) -> np.ndarray:
        """Reconstruct full-space weights from latent coefficients: w = L @ s (d,)."""
        return (self.L @ s.reshape(-1)).astype(np.float32, copy=False)


def svd_basis(weight_vectors: np.ndarray, *, k: int, center: bool = False) -> EllaBasis:
    """Learn a shared basis from a collection of weight vectors.

    weight_vectors: (n_tasks, d) matrix of per-task weights in a common feature space.
    k: target basis rank (clipped to <= min(n_tasks, d)).
    center: if True, subtract mean weight before SVD (ELLA-style implementations
            often rely on a quadratic approximation and do not require centering;
            we default to False to keep calibration simple).
    """

    w = np.asarray(weight_vectors, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight_vectors must be 2D (n_tasks, d), got {w.shape}")
    n_tasks, d = w.shape
    r = min(int(n_tasks), int(d))
    k_eff = max(1, min(int(k), r))
    w0 = w
    if bool(center):
        w0 = w0 - w0.mean(axis=0, keepdims=True)
    # Right singular vectors span the weight space.
    _u, _s, vt = np.linalg.svd(w0, full_matrices=False)
    L = vt[:k_eff].T.astype(np.float32, copy=False)  # (d, k)
    return EllaBasis(L=L)


def train_task_binary_via_basis(
    *,
    x: np.ndarray,
    y: np.ndarray,
    basis: EllaBasis,
    lr: float = 0.1,
    epochs: int = 50,
    l2: float = 1e-3,
    w_pos: float = 1.0,
    w_neg: float = 1.0,
    seed: int = 0,
) -> tuple[BinaryLogReg, np.ndarray]:
    """Train a task-specific binary classifier constrained to w = L @ s.

    Returns (full_space_model, latent_s).
    """

    z = basis.project(x)
    latent = train_binary_logreg(
        z,
        y,
        lr=lr,
        epochs=epochs,
        l2=l2,
        w_pos=w_pos,
        w_neg=w_neg,
        seed=seed,
    )
    s = latent.w.astype(np.float32, copy=False)  # (k,)
    w_full = basis.reconstruct(s)
    return BinaryLogReg(w=w_full, b=float(latent.b)), s


def train_task_softmax_via_basis(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    basis: EllaBasis,
    lr: float = 0.1,
    epochs: int = 50,
    l2: float = 1e-3,
    seed: int = 0,
) -> tuple[SoftmaxReg, np.ndarray]:
    """Train a task-specific softmax classifier constrained to W = S @ L^T.

    Returns (full_space_model, latent_S) where latent_S is (n_classes, k).
    """

    z = basis.project(x)
    latent = train_softmax_reg(
        z,
        y,
        n_classes=int(n_classes),
        lr=lr,
        epochs=epochs,
        l2=l2,
        seed=seed,
    )
    S = latent.w.astype(np.float32, copy=False)  # (n_classes, k)
    W_full = (S @ basis.L.T).astype(np.float32, copy=False)  # (n_classes, d)
    return SoftmaxReg(w=W_full, b=latent.b.astype(np.float32, copy=False)), S

