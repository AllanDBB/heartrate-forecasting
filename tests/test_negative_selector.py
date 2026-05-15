import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from wrappers.NegativeSelector import NegativeSelector


def test_self_samples_mostly_not_flagged():
    """La mayoría de las ventanas de ens_fit deben ser clasificadas como self."""
    rng = np.random.RandomState(0)
    X_self = rng.randn(200, 20)
    ns = NegativeSelector(threshold_percentile=95)
    ns.fit(X_self)
    is_nonself = ns.predict(X_self)
    assert is_nonself.mean() < 0.10, f"Too many self flagged: {is_nonself.mean():.2%}"


def test_outliers_flagged_as_nonself():
    """Puntos muy lejos del self space deben ser detectados como nonself."""
    rng = np.random.RandomState(1)
    X_self = rng.randn(200, 20)
    X_outliers = rng.randn(20, 20) + 100
    ns = NegativeSelector(threshold_percentile=95)
    ns.fit(X_self)
    is_nonself = ns.predict(X_outliers)
    assert is_nonself.all(), "All outliers must be flagged as nonself"


def test_output_is_boolean():
    rng = np.random.RandomState(2)
    X = rng.randn(50, 20)
    ns = NegativeSelector()
    ns.fit(X)
    result = ns.predict(X[:10])
    assert result.dtype == bool, f"Expected bool, got {result.dtype}"
    assert result.shape == (10,)


def test_raises_before_fit():
    ns = NegativeSelector()
    with pytest.raises(AssertionError):
        ns.predict(np.random.randn(5, 20))


def test_nonself_ratio_property():
    rng = np.random.RandomState(3)
    X_self = rng.randn(100, 20)
    X_mixed = np.vstack([rng.randn(50, 20), rng.randn(50, 20) + 50])
    ns = NegativeSelector(threshold_percentile=95)
    ns.fit(X_self)
    is_nonself = ns.predict(X_mixed)
    assert is_nonself[50:].mean() > 0.8
