import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from wrappers.AiNetCore import AiNetCore
from wrappers.SubjectMemory import SubjectMemory


def make_trained_ainet(K=5, n_models=4, seed=0):
    rng = np.random.RandomState(seed)
    ainet = AiNetCore(n_antibodies=K, max_iter=2, random_state=seed)
    features = rng.randn(60, 20)
    preds = rng.rand(60, n_models, 45) * 100 + 60
    y = rng.rand(60, 45) * 100 + 60
    ainet.fit(features, preds, y)
    return ainet


def test_memory_shape_and_sums():
    """M_i debe tener K valores que sumen 1 por sujeto."""
    rng = np.random.RandomState(0)
    K, n_models = 5, 4
    ainet = make_trained_ainet(K=K, n_models=n_models)

    n_windows = 60
    features = rng.randn(n_windows, 20)
    preds = rng.rand(n_windows, n_models, 45) * 100 + 60
    y = rng.rand(n_windows, 45) * 100 + 60
    ids = np.array(['A'] * 20 + ['B'] * 20 + ['C'] * 20)

    sm = SubjectMemory()
    sm.fit(ainet, features, preds, y, ids)

    mat = sm.get_memory_matrix(ids)
    assert mat.shape == (n_windows, K), f"Expected ({n_windows}, {K}), got {mat.shape}"
    assert np.allclose(mat.sum(axis=1), 1.0, atol=1e-6), "Memory rows must sum to 1"
    assert (mat >= 0).all()


def test_unknown_subject_gets_uniform():
    """Sujeto no visto en fit() debe recibir memoria uniforme."""
    rng = np.random.RandomState(1)
    K, n_models = 4, 3
    ainet = make_trained_ainet(K=K, n_models=n_models)

    features = rng.randn(30, 20)
    preds = rng.rand(30, n_models, 45) * 100 + 60
    y = rng.rand(30, 45) * 100 + 60
    ids_fit = np.array(['A'] * 30)

    sm = SubjectMemory()
    sm.fit(ainet, features, preds, y, ids_fit)

    ids_new = np.array(['Z'] * 5)
    mat = sm.get_memory_matrix(ids_new)
    expected = np.ones((5, K)) / K
    assert np.allclose(mat, expected, atol=1e-6), "Unknown subject must get uniform memory"


def test_different_subjects_get_different_memory():
    """
    Sujetos con distintas predicciones deben tener memorias distintas.
    Usamos un mock de AiNetCore con pesos pre-definidos para garantizar
    que la diferencia sea determinista:
      - Anticuerpo 0: todo el peso en modelo 0
      - Anticuerpo 1: todo el peso en modelo 1
    Sujeto A: modelo 0 predice perfecto → M_A concentrado en anticuerpo 0
    Sujeto B: modelo 1 predice perfecto → M_B concentrado en anticuerpo 1
    """
    rng = np.random.RandomState(3)
    horizon, n_windows = 45, 20

    # Mock mínimo de AiNetCore — SubjectMemory solo necesita .K y .weights_
    class MockAiNet:
        K = 2
        weights_ = np.array([
            [1.0, 0.0],   # anticuerpo 0: todo en modelo 0
            [0.0, 1.0],   # anticuerpo 1: todo en modelo 1
        ])

    ainet = MockAiNet()

    y_A = rng.rand(n_windows, horizon) * 100 + 60
    y_B = rng.rand(n_windows, horizon) * 100 + 60

    # Sujeto A: modelo 0 = predicción perfecta, modelo 1 = ruido
    preds_A = np.stack([y_A, rng.rand(n_windows, horizon) * 200], axis=1)
    # Sujeto B: modelo 0 = ruido, modelo 1 = predicción perfecta
    preds_B = np.stack([rng.rand(n_windows, horizon) * 200, y_B], axis=1)

    features = rng.randn(n_windows * 2, 20)
    preds = np.concatenate([preds_A, preds_B], axis=0)
    y = np.concatenate([y_A, y_B], axis=0)
    ids = np.array(['A'] * n_windows + ['B'] * n_windows)

    sm = SubjectMemory()
    sm.fit(ainet, features, preds, y, ids)

    mat = sm.get_memory_matrix(ids)
    mem_A = mat[:n_windows].mean(axis=0)   # esperado: concentrado en anticuerpo 0
    mem_B = mat[n_windows:].mean(axis=0)   # esperado: concentrado en anticuerpo 1

    # A debe preferir anticuerpo 0, B el anticuerpo 1
    assert mem_A[0] > mem_A[1], f"Sujeto A debe preferir anticuerpo 0: {mem_A}"
    assert mem_B[1] > mem_B[0], f"Sujeto B debe preferir anticuerpo 1: {mem_B}"
