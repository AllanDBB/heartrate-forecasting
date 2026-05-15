import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from wrappers.FeatureExtractor import FeatureExtractor


def make_windows(n=50, length=200, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, length).astype(np.float32)


def test_output_shape():
    X = make_windows(50, 200)
    fe = FeatureExtractor()
    out = fe.fit_transform(X)
    assert out.shape == (50, 20), f"Expected (50, 20), got {out.shape}"


def test_normalized_mean_std():
    """Features extraídas sobre fit deben tener media≈0, std≈1 por columna."""
    X = make_windows(200, 200)
    fe = FeatureExtractor()
    out = fe.fit_transform(X)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-6), "Mean should be ~0"
    assert np.allclose(out.std(axis=0), 1, atol=1e-6), "Std should be ~1"


def test_transform_uses_fit_params():
    """transform() debe usar los params de fit(), no recalcular."""
    X_fit = make_windows(100, 200, seed=0)
    X_new = make_windows(10, 200, seed=42)
    fe = FeatureExtractor()
    fe.fit(X_fit)
    out1 = fe.transform(X_new)
    out2 = fe.transform(X_new)
    assert np.allclose(out1, out2), "transform() must be deterministic"


def test_raises_before_fit():
    X = make_windows(10, 200)
    fe = FeatureExtractor()
    with pytest.raises(AssertionError):
        fe.transform(X)


def test_short_window():
    """Debe funcionar con ventanas de longitud 45 también."""
    X = make_windows(30, 45)
    fe = FeatureExtractor()
    out = fe.fit_transform(X)
    assert out.shape == (30, 20)
