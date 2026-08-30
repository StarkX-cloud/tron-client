"""Tests for the Phase 3 TinyMLP. The numerical gradient check is the one
that matters most here: it independently verifies the hand-written
backward() math against finite differences, which is the standard way to
catch a backprop bug that would otherwise silently produce a model that
"trains" but not correctly.
"""
import numpy as np
import pytest

from tron.training.model import TinyMLP


def test_flat_params_roundtrip():
    model = TinyMLP(input_dim=4, hidden_dim=5, num_classes=3, seed=1)
    original = model.get_flat_params().copy()
    model.set_flat_params(original)
    np.testing.assert_array_equal(model.get_flat_params(), original)


def test_num_params_matches_flat_vector_length():
    model = TinyMLP(input_dim=4, hidden_dim=5, num_classes=3, seed=1)
    assert model.num_params() == len(model.get_flat_params())


def test_clone_is_independent():
    model = TinyMLP(input_dim=4, hidden_dim=5, num_classes=3, seed=1)
    clone = model.clone()
    clone.w1[0, 0] += 100.0
    assert model.w1[0, 0] != clone.w1[0, 0]


def test_forward_produces_valid_probability_distribution():
    model = TinyMLP(input_dim=4, hidden_dim=5, num_classes=3, seed=1)
    x = np.random.default_rng(0).normal(size=(10, 4))
    probs = model.forward(x)["probs"]
    assert probs.shape == (10, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)
    assert (probs >= 0).all()


def test_loss_decreases_after_training_steps():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(32, 4))
    y = rng.integers(0, 3, size=32)

    model = TinyMLP(input_dim=4, hidden_dim=8, num_classes=3, seed=1)
    first_loss = model.train_step(x, y, lr=0.1)
    for _ in range(50):
        loss = model.train_step(x, y, lr=0.1)
    assert loss < first_loss


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_matches_numerical_gradient(seed):
    """The correctness test: perturb each parameter by epsilon, measure
    the resulting change in loss, and confirm it matches the analytic
    gradient from backward() to within numerical tolerance. This is
    checked for a representative sample of parameters, not all of
    them (checking all ~100 would be redundant given they go through the
    same backprop code paths, and slow).
    """
    rng = np.random.default_rng(seed)
    model = TinyMLP(input_dim=3, hidden_dim=4, num_classes=2, seed=seed)
    x = rng.normal(size=(6, 3))
    y = rng.integers(0, 2, size=6)

    cache = model.forward(x)
    analytic_grads = model.backward(cache, y)

    epsilon = 1e-5
    param_arrays = {"w1": model.w1, "b1": model.b1, "w2": model.w2, "b2": model.b2}

    def loss_at_current_params():
        return model.cross_entropy_loss(model.forward(x)["probs"], y)

    checked = 0
    for name, arr in param_arrays.items():
        flat = arr.ravel()
        # Sample a handful of indices per parameter array rather than all
        # of them — sufficient to catch a sign error or wrong-shape bug in
        # backward(), which would show up consistently across indices.
        indices = rng.choice(len(flat), size=min(3, len(flat)), replace=False)
        for idx in indices:
            original_value = flat[idx]

            flat[idx] = original_value + epsilon
            loss_plus = loss_at_current_params()

            flat[idx] = original_value - epsilon
            loss_minus = loss_at_current_params()

            flat[idx] = original_value  # restore

            numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
            analytic_grad = analytic_grads[name].ravel()[idx]

            assert numerical_grad == pytest.approx(analytic_grad, abs=1e-3), (
                f"{name}[{idx}]: numerical={numerical_grad}, analytic={analytic_grad}"
            )
            checked += 1

    assert checked > 0
