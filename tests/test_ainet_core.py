import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from wrappers.AiNetCore import AiNetCore


def make_data(n_windows=80, n_features=20, n_models=5, horizon=45, seed=0):
    rng = np.random.RandomState(seed)
    features = rng.randn(n_windows, n_features)
    preds_stack = rng.rand(n_windows, n_models, horizon) * 100 + 50
    y_true = rng.rand(n_windows, horizon) * 100 + 50
    return features, preds_stack, y_true


def test_fit_produces_correct_shapes():
    features, preds, y = make_data()
    ainet = AiNetCore(n_antibodies=5, max_iter=3, random_state=0)
    ainet.fit(features, preds, y)
    assert ainet.centroids_.shape == (5, 20), f"Got {ainet.centroids_.shape}"
    assert ainet.weights_.shape == (5, 5), f"Got {ainet.weights_.shape}"


def test_weights_are_simplex():
    """Cada anticuerpo debe tener pesos que sumen 1 y sean >= 0."""
    features, preds, y = make_data()
    ainet = AiNetCore(n_antibodies=8, max_iter=3, random_state=1)
    ainet.fit(features, preds, y)
    sums = ainet.weights_.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), f"Weights don't sum to 1: {sums}"
    assert (ainet.weights_ >= 0).all(), "Weights must be non-negative"


def test_get_adaptive_weights_shape():
    features, preds, y = make_data(n_windows=80)
    ainet = AiNetCore(n_antibodies=5, max_iter=3, random_state=0)
    ainet.fit(features, preds, y)

    X_new = np.random.randn(15, 20)
    w = ainet.get_adaptive_weights(X_new)
    assert w.shape == (15, 5), f"Got {w.shape}"


def test_get_adaptive_weights_simplex():
    features, preds, y = make_data()
    ainet = AiNetCore(n_antibodies=5, max_iter=3, random_state=0)
    ainet.fit(features, preds, y)
    w = ainet.get_adaptive_weights(features[:10])
    sums = w.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), "Adaptive weights must sum to 1"
    assert (w >= 0).all()


def test_with_subject_memory():
    features, preds, y = make_data(n_windows=80)
    K = 5
    ainet = AiNetCore(n_antibodies=K, max_iter=3, random_state=0)
    ainet.fit(features, preds, y)

    memory = np.ones((10, K)) / K  # uniform memory
    w = ainet.get_adaptive_weights(features[:10], subject_memory=memory)
    assert w.shape == (10, 5)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)


def test_raises_before_fit():
    ainet = AiNetCore(n_antibodies=5)
    with pytest.raises(AssertionError):
        ainet.get_adaptive_weights(np.random.randn(5, 20))
