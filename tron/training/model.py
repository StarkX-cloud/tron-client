"""A small, dependency-free (pure numpy) neural net — deliberately not
PyTorch/JAX. The point of Phase 3 is to demonstrate the *training
algorithm* (local SGD + weight-space merging) honestly at a scale that
runs on a laptop CPU in seconds, not to demonstrate a framework
integration. See benchmark.py for why this scale is the honest choice,
and ROADMAP.md for what scaling this up to a real model size would need.

Architecture: one hidden layer, ReLU, softmax + cross-entropy. Small
enough that its gradient is hand-verifiable (see
tests/test_training_model.py's numerical gradient check) and its full
parameter vector is a few hundred floats — cheap enough to log every
sync's bytes exactly, which is the whole point of the communication
benchmark.
"""
from __future__ import annotations

import numpy as np


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class TinyMLP:
    """input_dim -> hidden_dim (ReLU) -> num_classes (softmax)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.w1 = _xavier_init(input_dim, hidden_dim, rng)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = _xavier_init(hidden_dim, num_classes, rng)
        self.b2 = np.zeros(num_classes)

    # -- weight-space plumbing --------------------------------------
    # Everything downstream (local-SGD outer sync, weight merging) treats
    # a model as one flat vector — this is what makes "average two
    # models' weights" or "compute a weight delta" a one-liner instead of
    # bespoke per-layer logic.

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()])

    def set_flat_params(self, flat: np.ndarray) -> None:
        i = 0
        n = self.w1.size
        self.w1 = flat[i:i + n].reshape(self.w1.shape); i += n
        n = self.b1.size
        self.b1 = flat[i:i + n].reshape(self.b1.shape); i += n
        n = self.w2.size
        self.w2 = flat[i:i + n].reshape(self.w2.shape); i += n
        n = self.b2.size
        self.b2 = flat[i:i + n].reshape(self.b2.shape); i += n

    def num_params(self) -> int:
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def clone(self) -> "TinyMLP":
        other = TinyMLP(self.input_dim, self.hidden_dim, self.num_classes)
        other.set_flat_params(self.get_flat_params().copy())
        return other

    # -- forward / backward ------------------------------------------

    def forward(self, x: np.ndarray) -> dict:
        """Returns the intermediate activations backward() needs — kept
        explicit (a dict, not instance state) so forward/backward stay
        pure functions of their inputs, which is what makes the
        numerical gradient check in tests/test_training_model.py valid."""
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0.0, z1)  # ReLU
        z2 = a1 @ self.w2 + self.b2
        # softmax, numerically stable
        z2_shifted = z2 - z2.max(axis=1, keepdims=True)
        exp_z2 = np.exp(z2_shifted)
        probs = exp_z2 / exp_z2.sum(axis=1, keepdims=True)
        return {"x": x, "z1": z1, "a1": a1, "z2": z2, "probs": probs}

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
        n = probs.shape[0]
        eps = 1e-12
        return float(-np.mean(np.log(probs[np.arange(n), y] + eps)))

    def backward(self, cache: dict, y: np.ndarray) -> dict:
        """Returns gradients w.r.t. w1, b1, w2, b2, matching forward()'s
        cache. Standard softmax-cross-entropy + ReLU backprop."""
        x, a1, probs = cache["x"], cache["a1"], cache["probs"]
        n = x.shape[0]

        dz2 = probs.copy()
        dz2[np.arange(n), y] -= 1.0
        dz2 /= n

        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * (cache["z1"] > 0)  # ReLU derivative

        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}

    def apply_gradients(self, grads: dict, lr: float) -> None:
        self.w1 -= lr * grads["w1"]
        self.b1 -= lr * grads["b1"]
        self.w2 -= lr * grads["w2"]
        self.b2 -= lr * grads["b2"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x)["probs"], axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        """One SGD step on a batch; returns the loss before the step."""
        cache = self.forward(x)
        loss = self.cross_entropy_loss(cache["probs"], y)
        grads = self.backward(cache, y)
        self.apply_gradients(grads, lr)
        return loss
