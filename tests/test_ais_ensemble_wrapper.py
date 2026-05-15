import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from wrappers.AISEnsembleWrapper import AISEnsembleWrapper


def make_scenario(n_fit=80, n_eval=40, n_models=4, horizon=45, input_size=45, seed=0):
    rng = np.random.RandomState(seed)
    X_fit = rng.randn(n_fit, input_size).astype(np.float32)
    X_eval = rng.randn(n_eval, input_size).astype(np.float32)
    y_fit = rng.rand(n_fit, horizon) * 100 + 60
    y_eval = rng.rand(n_eval, horizon) * 100 + 60
    ids_fit = np.array([f'S{i % 3}' for i in range(n_fit)])
    ids_eval = np.array([f'S{i % 3}' for i in range(n_eval)])
    model_names = [f'M{i}' for i in range(n_models)]
    preds_fit = {m: rng.rand(n_fit, horizon) * 100 + 60 for m in model_names}
    preds_eval = {m: rng.rand(n_eval, horizon) * 100 + 60 for m in model_names}
    return X_fit, X_eval, y_fit, y_eval, ids_fit, ids_eval, model_names, preds_fit, preds_eval


def build_fitted_wrapper(seed=0):
    X_fit, X_eval, y_fit, y_eval, ids_fit, ids_eval, model_names, preds_fit, preds_eval \
        = make_scenario(seed=seed)

    ais = AISEnsembleWrapper(n_antibodies=5, max_iter=2, random_state=seed)
    for m in model_names:
        ais.add_model(m, preds_fit[m], split='fit')
        ais.add_model(m, preds_eval[m], split='eval')
    ais.fit(X_fit, y_fit, ids_fit)
    return ais, X_eval, y_eval, ids_eval, preds_eval


def test_predict_output_shape():
    ais, X_eval, y_eval, ids_eval, _ = build_fitted_wrapper()
    y_pred = ais.predict(X_eval, ids_eval)
    assert y_pred.shape == (40, 45), f"Got {y_pred.shape}"


def test_predict_is_finite():
    ais, X_eval, y_eval, ids_eval, _ = build_fitted_wrapper()
    y_pred = ais.predict(X_eval, ids_eval)
    assert np.isfinite(y_pred).all(), "Predictions contain NaN or Inf"


def test_add_model_interface():
    """add_model() debe aceptar split='fit' y split='eval' sin error."""
    ais = AISEnsembleWrapper(n_antibodies=3, max_iter=2)
    X_fit, X_eval, y_fit, y_eval, ids_fit, ids_eval, model_names, preds_fit, preds_eval \
        = make_scenario()
    for m in model_names:
        ais.add_model(m, preds_fit[m], split='fit')
        ais.add_model(m, preds_eval[m], split='eval')
    assert set(ais._model_names) == set(model_names)


def test_diagnostics_keys():
    ais, X_eval, y_eval, ids_eval, _ = build_fitted_wrapper()
    diag = ais.get_diagnostics(X_eval, ids_eval)
    assert 'nonself_ratio' in diag
    assert 'nonself_by_subject' in diag
    assert 'mean_adaptive_weights' in diag
    assert 'weight_std' in diag


def test_raises_predict_before_fit():
    ais = AISEnsembleWrapper(n_antibodies=3, max_iter=2)
    with pytest.raises(AssertionError):
        ais.predict(np.random.randn(5, 45), np.array(['A'] * 5))
