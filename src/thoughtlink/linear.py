from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -30.0, 30.0))
    return ex / np.sum(ex, axis=1, keepdims=True)


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray  # (d,)
    std: np.ndarray  # (d,)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def fit_scaler(x: np.ndarray, *, eps: float = 1e-6) -> StandardScaler:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < eps, eps, std)
    return StandardScaler(mean=mean, std=std)


@dataclass(frozen=True)
class BinaryLogReg:
    w: np.ndarray  # (d,)
    b: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w + self.b
        return _sigmoid(z.astype(np.float32))

    def predict(self, x: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(np.int64)


def train_binary_logreg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    epochs: int = 50,
    l2: float = 1e-3,
    w_pos: float = 1.0,
    w_neg: float = 1.0,
    seed: int = 0,
) -> BinaryLogReg:
    rng = np.random.default_rng(seed)
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False).reshape(-1)
    n, d = x.shape
    if y.shape[0] != n:
        raise ValueError("x and y length mismatch")

    w = (0.01 * rng.standard_normal(d)).astype(np.float32)
    b = np.float32(0.0)

    for _ in range(int(epochs)):
        z = x @ w + b
        p = _sigmoid(z)
        # grad of weighted mean BCE + L2
        weights = np.where(y > 0.5, float(w_pos), float(w_neg)).astype(np.float32)
        denom = float(weights.sum()) if float(weights.sum()) > 0 else float(n)
        diff = (p - y) * weights
        grad_w = (x.T @ diff) / denom + (l2 * w)
        grad_b = float(diff.sum()) / denom
        w = w - (lr * grad_w).astype(np.float32)
        b = np.float32(b - lr * grad_b)

    return BinaryLogReg(w=w, b=float(b))


@dataclass(frozen=True)
class SoftmaxReg:
    w: np.ndarray  # (k, d)
    b: np.ndarray  # (k,)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.w.T + self.b
        return _softmax(logits.astype(np.float32))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1).astype(np.int64)


def train_softmax_reg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    lr: float = 0.1,
    epochs: int = 50,
    l2: float = 1e-3,
    seed: int = 0,
) -> SoftmaxReg:
    rng = np.random.default_rng(seed)
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False).reshape(-1)
    n, d = x.shape
    if y.shape[0] != n:
        raise ValueError("x and y length mismatch")

    k = int(n_classes)
    w = (0.01 * rng.standard_normal((k, d))).astype(np.float32)
    b = np.zeros((k,), dtype=np.float32)

    # One-hot on the fly (k is small).
    for _ in range(int(epochs)):
        logits = x @ w.T + b
        p = _softmax(logits)
        y_onehot = np.zeros((n, k), dtype=np.float32)
        y_onehot[np.arange(n), y] = 1.0
        diff = (p - y_onehot).astype(np.float32)
        grad_w = (diff.T @ x) / n + (l2 * w)
        grad_b = diff.mean(axis=0)
        w = w - (lr * grad_w).astype(np.float32)
        b = b - (lr * grad_b).astype(np.float32)

    return SoftmaxReg(w=w, b=b)
