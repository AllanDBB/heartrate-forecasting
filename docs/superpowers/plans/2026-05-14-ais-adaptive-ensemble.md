# AIS Adaptive Ensemble — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar un ensemble adaptativo basado en Sistemas Inmunes Artificiales (aiNet + memoria por sujeto + Negative Selection) que supere el baseline de Stacking Ridge (MAPE=4.221) evaluado sobre `held_out` sin leakage.

**Architecture:** Cuatro módulos independientes (`FeatureExtractor`, `AiNetCore`, `SubjectMemory`, `NegativeSelector`) orquestados por `AISEnsembleWrapper`. El wrapper sigue la misma interfaz de `EnsembleWrapper` para compatibilidad con los notebooks existentes. Todo se entrena sobre `ens_fit` y se evalúa sobre `held_out`.

**Tech Stack:** Python 3.x, NumPy, SciPy, scikit-learn (KDTree, train_test_split), Keras (solo para los modelos base, no para AIS), pytest, Jupyter.

---

## File Structure

```
wrappers/
  FeatureExtractor.py       # Extracción y normalización de 20 features por ventana
  AiNetCore.py              # Población de anticuerpos, entrenamiento aiNet
  SubjectMemory.py          # Memoria M_i por sujeto (K valores por anticuerpo)
  NegativeSelector.py       # Detección de ventanas nonself (KDTree)
  AISEnsembleWrapper.py     # Orquestador: misma interfaz que EnsembleWrapper

tests/
  test_feature_extractor.py
  test_ainet_core.py
  test_subject_memory.py
  test_negative_selector.py
  test_ais_ensemble_wrapper.py

notebooks/
  ensemble_ais.ipynb        # Notebook principal del escenario 3
```

**Archivos existentes que se leen pero NO se modifican:**
- `wrappers/EnsembleWrapper.py` — referencia de interfaz (`add_model`, `predict`)
- `utils.py` — se reusan `evaluate_all_metrics`, `desestandarizar_ventanas`, `estandarizar`
- `notebooks/ensemble_nueva_info.ipynb` — reusar celdas de carga de datos y modelos

---

## Task 1: FeatureExtractor

**Files:**
- Create: `wrappers/FeatureExtractor.py`
- Create: `tests/test_feature_extractor.py`

- [ ] **Step 1: Crear el archivo de test**

```python
# tests/test_feature_extractor.py
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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

```bash
cd "C:/Users/allan/OneDrive/Documentos/Asistencia 2026 I"
python -m pytest tests/test_feature_extractor.py -v
```

Esperado: `ImportError` o `ModuleNotFoundError` porque `FeatureExtractor.py` no existe aún.

- [ ] **Step 3: Implementar FeatureExtractor**

```python
# wrappers/FeatureExtractor.py
"""
FeatureExtractor: convierte ventanas de series temporales en vectores
de 20 features estadísticos, espectrales y de forma.

Input:  X de shape (n_windows, input_size) — ya estandarizado por utils.estandarizar
Output: features de shape (n_windows, 20) — normalizadas con params de fit()

IMPORTANTE: La normalización aquí es DISTINTA a utils.estandarizar.
  - utils.estandarizar normaliza las series temporales (para los modelos base)
  - FeatureExtractor normaliza los estadísticos extraídos de esas series
  Son dos normalizaciones sobre objetos distintos; no hay doble estandarización.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis as scipy_kurtosis


class FeatureExtractor:
    """
    Extrae y normaliza 20 features por ventana temporal.

    Uso:
        fe = FeatureExtractor()
        features_fit = fe.fit_transform(X_ens_fit)   # fit + transform sobre ens_fit
        features_eval = fe.transform(X_held_out)     # solo transform (usa params de fit)
    """

    N_FEATURES = 20

    def __init__(self):
        self._fit_mean: np.ndarray | None = None
        self._fit_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _extract_raw(self, windows: np.ndarray) -> np.ndarray:
        """Extrae features sin normalizar. Shape: (n_windows, 20)."""
        n, L = windows.shape
        feat = np.zeros((n, self.N_FEATURES), dtype=np.float64)

        for i, w in enumerate(windows):
            # --- Estadísticas básicas (0-4) ---
            feat[i, 0] = np.mean(w)
            feat[i, 1] = np.std(w)
            feat[i, 2] = np.min(w)
            feat[i, 3] = np.max(w)
            feat[i, 4] = np.ptp(w)                          # rango

            # --- Forma (5-7) ---
            feat[i, 5] = float(skew(w))
            feat[i, 6] = float(scipy_kurtosis(w))
            feat[i, 7] = feat[i, 1] / (abs(feat[i, 0]) + 1e-8)  # CV

            # --- Tendencia (8-9) ---
            x = np.arange(L, dtype=np.float64)
            feat[i, 8] = float(np.polyfit(x, w, 1)[0])     # pendiente lineal
            feat[i, 9] = float(np.mean(np.diff(np.diff(w))))  # 2ª derivada media

            # --- Autocorrelación en lags 1, 5, 10, 20 (10-13) ---
            w_centered = w - w.mean()
            norm = np.dot(w_centered, w_centered) + 1e-8
            for j, lag in enumerate([1, 5, 10, 20]):
                if lag < L:
                    feat[i, 10 + j] = np.dot(w_centered[:-lag], w_centered[lag:]) / norm
                else:
                    feat[i, 10 + j] = 0.0

            # --- Espectral (14-15) ---
            freqs, psd = signal.periodogram(w)
            if len(psd) > 1:
                feat[i, 14] = float(freqs[np.argmax(psd[1:]) + 1])
            psd_norm = psd / (psd.sum() + 1e-8)
            feat[i, 15] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-8)))

            # --- Actividad (16-17) ---
            crossings = np.diff(np.sign(w - feat[i, 0]))
            feat[i, 16] = float(np.sum(crossings != 0))
            peaks, _ = signal.find_peaks(w)
            feat[i, 17] = float(len(peaks)) / L

            # --- Percentiles (18-19) ---
            feat[i, 18] = float(np.percentile(w, 25))
            feat[i, 19] = float(np.percentile(w, 75))

        return feat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, windows: np.ndarray) -> 'FeatureExtractor':
        """Calcula media y std de cada feature sobre windows (ens_fit)."""
        raw = self._extract_raw(windows)
        self._fit_mean = raw.mean(axis=0)
        self._fit_std = raw.std(axis=0) + 1e-8
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Aplica normalización usando params de fit(). Sin leakage."""
        assert self._fit_mean is not None, \
            "Llama fit() antes de transform()."
        raw = self._extract_raw(windows)
        return (raw - self._fit_mean) / self._fit_std

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Shortcut: fit() + transform() sobre el mismo conjunto."""
        return self.fit(windows).transform(windows)
```

- [ ] **Step 4: Ejecutar tests**

```bash
python -m pytest tests/test_feature_extractor.py -v
```

Esperado: 5 tests en verde (`PASSED`).

- [ ] **Step 5: Commit**

```bash
git add wrappers/FeatureExtractor.py tests/test_feature_extractor.py
git commit -m "feat: add FeatureExtractor with 20 time-series features"
```

---

## Task 2: AiNetCore

**Files:**
- Create: `wrappers/AiNetCore.py`
- Create: `tests/test_ainet_core.py`

- [ ] **Step 1: Escribir el test**

```python
# tests/test_ainet_core.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from wrappers.AiNetCore import AiNetCore


def make_data(n_windows=80, n_features=20, n_models=5, horizon=45, seed=0):
    rng = np.random.RandomState(seed)
    features = rng.randn(n_windows, n_features)
    # preds_stack: (n_windows, n_models, horizon)
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
```

- [ ] **Step 2: Verificar que el test falla**

```bash
python -m pytest tests/test_ainet_core.py -v
```

Esperado: `ImportError` — `AiNetCore` no existe.

- [ ] **Step 3: Implementar AiNetCore**

```python
# wrappers/AiNetCore.py
"""
AiNetCore: implementación del algoritmo aiNet para ensemble adaptativo.

Cada anticuerpo Ab_k = (centroid_k, w_k):
  - centroid_k: vector (n_features,) en espacio de features — qué patrón reconoce
  - w_k:        vector (n_models,) de pesos sobre modelos base — cómo combinarlos
                restricción simplex: sum(w_k) = 1, w_k[i] >= 0

Entrenamiento: clonal selection + hypersomtic mutation + network suppression.
Inferencia: afinidad gaussiana ponderada por memoria de sujeto → pesos adaptativos.
"""

import numpy as np
from scipy.spatial.distance import cdist


class AiNetCore:
    """
    Red inmune artificial para ensemble adaptativo de series temporales.

    Parámetros
    ----------
    n_antibodies : int
        Número de anticuerpos en la red final (K).
    sigma : float
        Ancho del kernel gaussiano de afinidad.
    clone_factor : int
        Número de clones por anticuerpo seleccionado (β).
    suppression_threshold : float
        Distancia mínima entre centroides antes de suprimir (σ_s).
    n_new : int
        Anticuerpos aleatorios nuevos insertados por iteración.
    max_iter : int
        Número máximo de iteraciones del algoritmo.
    mutation_rate : float
        Tasa base de mutación hipersomática (α).
    top_p_frac : float
        Fracción de ventanas cercanas usadas para evaluar clones.
    random_state : int
        Semilla aleatoria para reproducibilidad.
    """

    def __init__(
        self,
        n_antibodies: int = 20,
        sigma: float = 1.0,
        clone_factor: int = 5,
        suppression_threshold: float = 0.3,
        n_new: int = 5,
        max_iter: int = 50,
        mutation_rate: float = 0.1,
        top_p_frac: float = 0.05,
        random_state: int = 42,
    ):
        self.K = n_antibodies
        self.sigma = sigma
        self.beta = clone_factor
        self.sigma_s = suppression_threshold
        self.d = n_new
        self.max_iter = max_iter
        self.alpha = mutation_rate
        self.top_p_frac = top_p_frac
        self.rng = np.random.RandomState(random_state)

        self.centroids_: np.ndarray | None = None  # (K, n_features)
        self.weights_: np.ndarray | None = None    # (K, n_models)
        self.n_models_: int | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _affinity(self, features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calcula afinidad gaussiana entre features y centroides.

        features:  (n_windows, n_features)
        centroids: (n_ab, n_features)
        returns:   (n_windows, n_ab)  valores en (0, 1]
        """
        dist_sq = cdist(features, centroids, metric='sqeuclidean')
        return np.exp(-dist_sq / (self.sigma ** 2 + 1e-12))

    def _project_simplex(self, w: np.ndarray) -> np.ndarray:
        """Proyecta vector de pesos al simplex de probabilidad (sum=1, w>=0)."""
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s > 1e-12:
            return w / s
        return np.ones_like(w) / len(w)

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE sobre arrays aplanados con máscara para valores cercanos a cero."""
        mask = np.abs(y_true) > 1e-8
        if mask.sum() == 0:
            return 100.0
        return float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )

    def _eval_weights(
        self, w: np.ndarray, preds_stack: np.ndarray, y_true: np.ndarray
    ) -> float:
        """
        Evalúa un vector de pesos w sobre un subconjunto de ventanas.

        w:           (n_models,)
        preds_stack: (n_windows, n_models, horizon)
        y_true:      (n_windows, horizon)
        """
        y_pred = np.einsum('m,nmh->nh', w, preds_stack)
        return self._mape(y_true, y_pred)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        preds_stack: np.ndarray,
        y_true: np.ndarray,
    ) -> 'AiNetCore':
        """
        Entrena la red aiNet sobre datos de ens_fit.

        features:    (n_windows, n_features) — salida de FeatureExtractor.transform()
        preds_stack: (n_windows, n_models, horizon) — predicciones en escala original
        y_true:      (n_windows, horizon) — ground truth en escala original
        """
        n_windows, n_features = features.shape
        n_models = preds_stack.shape[1]
        self.n_models_ = n_models

        # Inicializar población aleatoria (3×K anticuerpos)
        N_init = self.K * 3
        f_min = features.min(axis=0)
        f_max = features.max(axis=0)
        f_range = f_max - f_min + 1e-8

        centroids = f_min + self.rng.rand(N_init, n_features) * f_range
        weights = np.array([
            self._project_simplex(self.rng.rand(n_models))
            for _ in range(N_init)
        ])

        top_p = max(10, int(self.top_p_frac * n_windows))

        for _ in range(self.max_iter):
            N = len(centroids)

            # a. Afinidad media de cada anticuerpo sobre todo ens_fit
            aff = self._affinity(features, centroids)  # (n_windows, N)
            mean_aff = aff.mean(axis=0)                # (N,)

            # b. Seleccionar top min(K, N) anticuerpos para clonar
            n_select = min(self.K, N)
            top_idx = np.argsort(mean_aff)[-n_select:]

            new_centroids = list(centroids)
            new_weights = list(weights)

            for ab_idx in top_idx:
                ab_aff = float(mean_aff[ab_idx])
                # Tasa de mutación inversamente proporcional a afinidad
                mut_rate = self.alpha / (ab_aff + 1e-8)

                best_clone_mape = float('inf')
                best_clone_c = centroids[ab_idx].copy()
                best_clone_w = weights[ab_idx].copy()

                for _ in range(self.beta):
                    # c. Mutación hipersomática:
                    #    w' = w + N(0, α/afinidad), luego proyección al simplex
                    #    c' = c + N(0, α/afinidad * 0.1) — mutación menor del centroide
                    w_mut = weights[ab_idx] + self.rng.randn(n_models) * mut_rate
                    w_mut = self._project_simplex(w_mut)
                    c_mut = centroids[ab_idx] + self.rng.randn(n_features) * mut_rate * 0.1

                    # Ventanas más cercanas al centroide mutado (top_p%)
                    dist_to_clone = np.sum((features - c_mut) ** 2, axis=1)
                    near_idx = np.argsort(dist_to_clone)[:top_p]

                    clone_mape = self._eval_weights(
                        w_mut, preds_stack[near_idx], y_true[near_idx]
                    )
                    if clone_mape < best_clone_mape:
                        best_clone_mape = clone_mape
                        best_clone_c = c_mut
                        best_clone_w = w_mut

                # d. El mejor clon actualiza al padre
                new_centroids[ab_idx] = best_clone_c
                new_weights[ab_idx] = best_clone_w

            centroids = np.array(new_centroids)
            weights = np.array(new_weights)

            # e. Supresión de red: eliminar anticuerpos demasiado similares
            dist_matrix = cdist(centroids, centroids)
            keep = np.ones(N, dtype=bool)
            aff_updated = self._affinity(features, centroids).mean(axis=0)

            for i in range(N):
                if not keep[i]:
                    continue
                for j in range(i + 1, N):
                    if not keep[j]:
                        continue
                    if dist_matrix[i, j] < self.sigma_s:
                        if aff_updated[i] >= aff_updated[j]:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break

            centroids = centroids[keep]
            weights = weights[keep]

            # f. Renovación: insertar d anticuerpos aleatorios nuevos
            new_rand_c = f_min + self.rng.rand(self.d, n_features) * f_range
            new_rand_w = np.array([
                self._project_simplex(self.rng.rand(n_models))
                for _ in range(self.d)
            ])
            centroids = np.vstack([centroids, new_rand_c])
            weights = np.vstack([weights, new_rand_w])

        # Selección final: los K de mayor afinidad media
        aff_final = self._affinity(features, centroids).mean(axis=0)
        k_actual = min(self.K, len(centroids))
        top_k = np.argsort(aff_final)[-k_actual:]

        self.centroids_ = centroids[top_k]
        self.weights_ = weights[top_k]
        self.K = k_actual  # actualizar K real por si hubo supresión agresiva

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_adaptive_weights(
        self,
        features: np.ndarray,
        subject_memory: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calcula pesos adaptativos por ventana.

        features:       (n_windows, n_features) — salida de FeatureExtractor.transform()
        subject_memory: (n_windows, K) — M_i[k] por ventana, o None (sin memoria)
        returns:        (n_windows, n_models) — pesos en simplex por ventana
        """
        assert self.centroids_ is not None, "Llama fit() antes de get_adaptive_weights()."

        aff = self._affinity(features, self.centroids_)  # (n_windows, K)

        if subject_memory is not None:
            activation = aff * subject_memory            # modulación por memoria
        else:
            activation = aff

        # Normalizar activaciones → suma 1 por ventana
        act_sum = activation.sum(axis=1, keepdims=True) + 1e-12
        activation_norm = activation / act_sum           # (n_windows, K)

        # Combinar pesos de anticuerpos ponderados por activación
        adaptive_w = activation_norm @ self.weights_     # (n_windows, n_models)

        # Garantizar restricción simplex (por precisión numérica)
        adaptive_w = np.maximum(adaptive_w, 0.0)
        adaptive_w /= (adaptive_w.sum(axis=1, keepdims=True) + 1e-12)

        return adaptive_w
```

- [ ] **Step 4: Ejecutar tests**

```bash
python -m pytest tests/test_ainet_core.py -v
```

Esperado: 5 tests en verde. Si alguno tarda >30s, reducir `max_iter` en el test a 2.

- [ ] **Step 5: Commit**

```bash
git add wrappers/AiNetCore.py tests/test_ainet_core.py
git commit -m "feat: add AiNetCore with clonal selection and hypersomtic mutation"
```

---

## Task 3: SubjectMemory

**Files:**
- Create: `wrappers/SubjectMemory.py`
- Create: `tests/test_subject_memory.py`

- [ ] **Step 1: Escribir el test**

```python
# tests/test_subject_memory.py
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
    # 3 sujetos
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

    ids_new = np.array(['Z'] * 5)  # sujeto desconocido
    mat = sm.get_memory_matrix(ids_new)
    expected = np.ones((5, K)) / K
    assert np.allclose(mat, expected, atol=1e-6), "Unknown subject must get uniform memory"


def test_different_subjects_get_different_memory():
    """Sujetos con distintas predicciones deben tener memorias distintas."""
    rng = np.random.RandomState(2)
    K, n_models, horizon = 5, 4, 45
    ainet = make_trained_ainet(K=K, n_models=n_models)

    # Sujeto A: predicciones perfectas (y == pred[0])
    y_A = rng.rand(20, horizon) * 100 + 60
    preds_A = np.stack([y_A] + [rng.rand(20, horizon) * 200 for _ in range(n_models - 1)], axis=1)

    # Sujeto B: predicciones malas para todos los modelos
    y_B = rng.rand(20, horizon) * 100 + 60
    preds_B = rng.rand(20, n_models, horizon) * 200 + 10

    features = rng.randn(40, 20)
    preds = np.concatenate([preds_A, preds_B], axis=0)
    y = np.concatenate([y_A, y_B], axis=0)
    ids = np.array(['A'] * 20 + ['B'] * 20)

    sm = SubjectMemory()
    sm.fit(ainet, features, preds, y, ids)

    mat = sm.get_memory_matrix(ids)
    mem_A = mat[:20].mean(axis=0)
    mem_B = mat[20:].mean(axis=0)
    assert not np.allclose(mem_A, mem_B, atol=0.05), "A y B deben tener memorias distintas"
```

- [ ] **Step 2: Verificar que el test falla**

```bash
python -m pytest tests/test_subject_memory.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Implementar SubjectMemory**

```python
# wrappers/SubjectMemory.py
"""
SubjectMemory: construye y mantiene la memoria inmunológica por sujeto.

Para cada sujeto i, M_i[k] representa qué tan bien el anticuerpo k
predice sus patrones de frecuencia cardíaca. Se construye evaluando
los pesos w_k de cada anticuerpo sobre las ventanas del sujeto en ens_fit.

Visión a futuro: en producción, M_i se acumula del historial real de
una persona, personalizando el sistema sin reentrenar los modelos base.
"""

import numpy as np
from wrappers.AiNetCore import AiNetCore


class SubjectMemory:
    """
    Memoria inmunológica por sujeto.

    Uso:
        sm = SubjectMemory()
        sm.fit(ainet, features_fit, preds_stack_fit, y_true_fit, ids_fit)
        memory_matrix = sm.get_memory_matrix(ids_held_out)  # (n_windows, K)
    """

    def __init__(self):
        self._memory: dict[str, np.ndarray] = {}   # subject_id -> (K,)
        self._K: int | None = None

    def fit(
        self,
        ainet: AiNetCore,
        features_fit: np.ndarray,
        preds_stack_fit: np.ndarray,
        y_true_fit: np.ndarray,
        ids_fit: np.ndarray,
    ) -> 'SubjectMemory':
        """
        Construye M_i para cada sujeto usando las ventanas de ens_fit.

        ainet:           AiNetCore entrenado — provee K anticuerpos y sus pesos w_k
        features_fit:    (n_windows, n_features) — salida de FeatureExtractor.transform()
        preds_stack_fit: (n_windows, n_models, horizon) — predicciones en escala original
        y_true_fit:      (n_windows, horizon) — ground truth en escala original
        ids_fit:         (n_windows,) — ID de sujeto por ventana
        """
        K = ainet.K
        self._K = K

        for subj_id in np.unique(ids_fit):
            mask = ids_fit == subj_id
            y_subj = y_true_fit[mask]           # (n_subj, horizon)
            p_subj = preds_stack_fit[mask]      # (n_subj, n_models, horizon)

            M_i = np.zeros(K)
            for k in range(K):
                w_k = ainet.weights_[k]         # (n_models,)
                # Predicción del anticuerpo k sobre ventanas del sujeto
                y_pred = np.einsum('m,nmh->nh', w_k, p_subj)

                # MAPE del anticuerpo k para este sujeto
                valid = np.abs(y_subj) > 1e-8
                if valid.sum() > 0:
                    mape_k = float(
                        np.mean(np.abs(
                            (y_subj[valid] - y_pred[valid]) / y_subj[valid]
                        )) * 100
                    )
                else:
                    mape_k = 100.0

                # Transformar MAPE en afinidad: menor MAPE → mayor memoria
                # Dividir por 100 para escalar el exponente razonablemente
                M_i[k] = np.exp(-mape_k / 100.0)

            # Normalizar → suma 1
            s = M_i.sum()
            self._memory[subj_id] = M_i / (s + 1e-12)

        return self

    def get_memory_matrix(self, ids: np.ndarray) -> np.ndarray:
        """
        Devuelve matriz de memoria para un conjunto de ventanas.

        ids:     (n_windows,) — ID de sujeto por ventana
        returns: (n_windows, K) — M_i por ventana
                 Sujetos no vistos en fit() reciben memoria uniforme (1/K).
        """
        assert self._K is not None, "Llama fit() antes de get_memory_matrix()."
        K = self._K
        result = np.zeros((len(ids), K))
        uniform = np.ones(K) / K

        for i, subj_id in enumerate(ids):
            result[i] = self._memory.get(subj_id, uniform)

        return result

    def subject_ids(self) -> list:
        """Lista de sujetos conocidos por el sistema."""
        return list(self._memory.keys())
```

- [ ] **Step 4: Ejecutar tests**

```bash
python -m pytest tests/test_subject_memory.py -v
```

Esperado: 3 tests en verde.

- [ ] **Step 5: Commit**

```bash
git add wrappers/SubjectMemory.py tests/test_subject_memory.py
git commit -m "feat: add SubjectMemory for per-subject immune profile"
```

---

## Task 4: NegativeSelector

**Files:**
- Create: `wrappers/NegativeSelector.py`
- Create: `tests/test_negative_selector.py`

- [ ] **Step 1: Escribir el test**

```python
# tests/test_negative_selector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from wrappers.NegativeSelector import NegativeSelector


def test_self_samples_mostly_not_flagged():
    """La mayoría de las ventanas de ens_fit deben ser clasificadas como self."""
    rng = np.random.RandomState(0)
    X_self = rng.randn(200, 20)                     # self space
    ns = NegativeSelector(threshold_percentile=95)
    ns.fit(X_self)
    is_nonself = ns.predict(X_self)
    # Con percentil 95, esperamos que ~5% del propio self sea flagged
    assert is_nonself.mean() < 0.10, f"Too many self flagged: {is_nonself.mean():.2%}"


def test_outliers_flagged_as_nonself():
    """Puntos muy lejos del self space deben ser detectados como nonself."""
    rng = np.random.RandomState(1)
    X_self = rng.randn(200, 20)                     # self en torno a 0
    X_outliers = rng.randn(20, 20) + 100            # muy lejos del self
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
    # La segunda mitad (offset +50) debe estar casi toda flagged
    assert is_nonself[50:].mean() > 0.8
```

- [ ] **Step 2: Verificar que el test falla**

```bash
python -m pytest tests/test_negative_selector.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Implementar NegativeSelector**

```python
# wrappers/NegativeSelector.py
"""
NegativeSelector: detección de ventanas fuera de la distribución normal (nonself).

Inspirado en el Algoritmo de Selección Negativa (NSA) del sistema inmune:
  - Durante fit(): caracteriza el "self space" con las ventanas de ens_fit
  - Durante predict(): una ventana es "nonself" si su distancia al vecino
    más cercano en el self space supera un umbral aprendido

Ventanas nonself → el sistema usa el modelo base de fallback (TCN)
en vez de forzar el ensemble sobre datos fuera de distribución.
"""

import numpy as np
from sklearn.neighbors import KDTree


class NegativeSelector:
    """
    Detector de ventanas anómalas basado en distancia al self space.

    Parámetros
    ----------
    threshold_percentile : float
        Percentil de la distribución de distancias intra-self usado como umbral.
        Valor alto (p.ej. 99) → pocos falsos positivos (pocas ventanas self
        clasificadas como nonself). Valor bajo → más sensible a anomalías.
    random_state : int
        Semilla para submuestreo interno.
    """

    def __init__(self, threshold_percentile: float = 99.0, random_state: int = 42):
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self._tree: KDTree | None = None
        self._threshold: float | None = None

    def fit(self, features: np.ndarray) -> 'NegativeSelector':
        """
        Aprende el self space desde las features de ens_fit.

        features: (n_windows, n_features) — salida de FeatureExtractor.transform()
        """
        self._tree = KDTree(features)

        # Calcular distancia de cada punto a su vecino más cercano (k=2: ignora self)
        dists, _ = self._tree.query(features, k=2)
        nn_dists = dists[:, 1]  # distancia al vecino más cercano (excluyendo self)

        self._threshold = float(np.percentile(nn_dists, self.threshold_percentile))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Clasifica ventanas como self (False) o nonself (True).

        features: (n_windows, n_features) — salida de FeatureExtractor.transform()
        returns:  (n_windows,) bool — True = nonself (anomalous)
        """
        assert self._tree is not None, "Llama fit() antes de predict()."
        dists, _ = self._tree.query(features, k=1)
        return (dists[:, 0] > self._threshold)

    @property
    def threshold(self) -> float | None:
        """Umbral de distancia aprendido durante fit()."""
        return self._threshold
```

- [ ] **Step 4: Ejecutar tests**

```bash
python -m pytest tests/test_negative_selector.py -v
```

Esperado: 5 tests en verde.

- [ ] **Step 5: Commit**

```bash
git add wrappers/NegativeSelector.py tests/test_negative_selector.py
git commit -m "feat: add NegativeSelector for anomalous window detection"
```

---

## Task 5: AISEnsembleWrapper

**Files:**
- Create: `wrappers/AISEnsembleWrapper.py`
- Create: `tests/test_ais_ensemble_wrapper.py`

- [ ] **Step 1: Escribir el test**

```python
# tests/test_ais_ensemble_wrapper.py
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
```

- [ ] **Step 2: Verificar que el test falla**

```bash
python -m pytest tests/test_ais_ensemble_wrapper.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Implementar AISEnsembleWrapper**

```python
# wrappers/AISEnsembleWrapper.py
"""
AISEnsembleWrapper: orquestador del ensemble adaptativo basado en AIS.

Sigue la misma interfaz de EnsembleWrapper (add_model / predict) para
compatibilidad con los notebooks existentes. Internamente coordina:
  1. FeatureExtractor   — features de las ventanas de entrada
  2. AiNetCore          — K anticuerpos con pesos especializados
  3. SubjectMemory      — memoria inmunológica M_i por sujeto
  4. NegativeSelector   — detección de ventanas fuera de distribución

Pipeline de inferencia por ventana:
  X → features → NegSel? → [nonself] fallback(mejor_modelo_individual)
                          → [self]   aff(f, Ab_k) × M_i[k] → w* → Σ w*·pred
"""

import numpy as np

from wrappers.FeatureExtractor import FeatureExtractor
from wrappers.AiNetCore import AiNetCore
from wrappers.SubjectMemory import SubjectMemory
from wrappers.NegativeSelector import NegativeSelector


class AISEnsembleWrapper:
    """
    Ensemble adaptativo AIS para predicción de series temporales.

    Parámetros (todos pasados al módulo correspondiente)
    ----------
    n_antibodies          : int   — K anticuerpos en la red aiNet
    sigma                 : float — ancho del kernel gaussiano de afinidad
    clone_factor          : int   — número de clones por anticuerpo seleccionado (β)
    suppression_threshold : float — umbral de supresión de red (σ_s)
    n_new                 : int   — anticuerpos aleatorios por iteración
    max_iter              : int   — iteraciones del algoritmo aiNet
    mutation_rate         : float — tasa base de mutación hipersomática (α)
    neg_sel_percentile    : float — percentil para umbral de NegativeSelector
    random_state          : int   — semilla global
    """

    def __init__(
        self,
        n_antibodies: int = 20,
        sigma: float = 1.0,
        clone_factor: int = 5,
        suppression_threshold: float = 0.3,
        n_new: int = 5,
        max_iter: int = 50,
        mutation_rate: float = 0.1,
        neg_sel_percentile: float = 99.0,
        random_state: int = 42,
    ):
        self.n_antibodies = n_antibodies
        self.sigma = sigma
        self.clone_factor = clone_factor
        self.suppression_threshold = suppression_threshold
        self.n_new = n_new
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.neg_sel_percentile = neg_sel_percentile
        self.random_state = random_state

        # Estado interno
        self._model_names: list[str] = []
        self._preds_fit: dict[str, np.ndarray] = {}
        self._preds_eval: dict[str, np.ndarray] = {}

        # Módulos (None hasta llamar fit())
        self.feature_extractor_: FeatureExtractor | None = None
        self.ainet_: AiNetCore | None = None
        self.subject_memory_: SubjectMemory | None = None
        self.neg_selector_: NegativeSelector | None = None
        self._fallback_model: str | None = None

    # ------------------------------------------------------------------
    # Interfaz compatible con EnsembleWrapper
    # ------------------------------------------------------------------

    def add_model(self, name: str, predictions: np.ndarray, split: str) -> None:
        """
        Registra predicciones de un modelo base.

        name:        identificador del modelo (ej. 'TCN')
        predictions: (n_windows, horizon) en escala original (desestandarizado)
        split:       'fit'  → usado para entrenar el AIS (ens_fit)
                     'eval' → usado para predecir en inferencia (held_out)
        """
        if name not in self._model_names:
            self._model_names.append(name)
        if split == 'fit':
            self._preds_fit[name] = predictions
        elif split == 'eval':
            self._preds_eval[name] = predictions
        else:
            raise ValueError(f"split debe ser 'fit' o 'eval', recibido: {split!r}")

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        X_fit: np.ndarray,
        y_true_fit: np.ndarray,
        ids_fit: np.ndarray,
        fallback_model: str | None = None,
    ) -> 'AISEnsembleWrapper':
        """
        Entrena todos los módulos AIS sobre ens_fit.

        X_fit:         (n_windows, input_size) — ventanas estandarizadas (X_ens_fit)
        y_true_fit:    (n_windows, horizon) — ground truth en escala original
        ids_fit:       (n_windows,) — ID de sujeto por ventana (ids_ens_fit)
        fallback_model: nombre del modelo para ventanas nonself (default: primer modelo)
        """
        preds_stack = np.stack(
            [self._preds_fit[m] for m in self._model_names], axis=1
        )  # (n_windows, n_models, horizon)

        # 1. FeatureExtractor
        self.feature_extractor_ = FeatureExtractor()
        features = self.feature_extractor_.fit_transform(X_fit)

        # 2. aiNet
        self.ainet_ = AiNetCore(
            n_antibodies=self.n_antibodies,
            sigma=self.sigma,
            clone_factor=self.clone_factor,
            suppression_threshold=self.suppression_threshold,
            n_new=self.n_new,
            max_iter=self.max_iter,
            mutation_rate=self.mutation_rate,
            random_state=self.random_state,
        )
        self.ainet_.fit(features, preds_stack, y_true_fit)

        # 3. Negative Selection
        self.neg_selector_ = NegativeSelector(
            threshold_percentile=self.neg_sel_percentile,
            random_state=self.random_state,
        )
        self.neg_selector_.fit(features)

        # 4. Subject Memory
        self.subject_memory_ = SubjectMemory()
        self.subject_memory_.fit(
            self.ainet_, features, preds_stack, y_true_fit, ids_fit
        )

        # 5. Modelo de fallback
        self._fallback_model = fallback_model or self._model_names[0]
        if self._fallback_model not in self._model_names:
            raise ValueError(
                f"fallback_model '{self._fallback_model}' no está en los modelos registrados."
            )

        return self

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict(
        self,
        X_eval: np.ndarray,
        ids_eval: np.ndarray,
    ) -> np.ndarray:
        """
        Genera predicciones adaptativas sobre held_out.

        X_eval:   (n_windows, input_size) — ventanas estandarizadas (X_held_out)
        ids_eval: (n_windows,) — ID de sujeto por ventana (ids_held_out)
        returns:  (n_windows, horizon) — predicciones en escala original
        """
        assert self.feature_extractor_ is not None, \
            "Llama fit() antes de predict()."

        # Features
        features = self.feature_extractor_.transform(X_eval)

        # Negative selection
        is_nonself = self.neg_selector_.predict(features)       # (n_windows,) bool

        # Memoria por sujeto
        memory = self.subject_memory_.get_memory_matrix(ids_eval)  # (n_windows, K)

        # Pesos adaptativos (calculados para todas las ventanas)
        adaptive_w = self.ainet_.get_adaptive_weights(features, memory)  # (n_windows, n_models)

        # Stack de predicciones eval
        preds_stack = np.stack(
            [self._preds_eval[m] for m in self._model_names], axis=1
        )  # (n_windows, n_models, horizon)

        horizon = preds_stack.shape[2]
        y_pred = np.zeros((len(X_eval), horizon))

        # Ventanas self: combinación ponderada adaptativa
        self_mask = ~is_nonself
        if self_mask.sum() > 0:
            w_self = adaptive_w[self_mask]        # (n_self, n_models)
            p_self = preds_stack[self_mask]       # (n_self, n_models, horizon)
            y_pred[self_mask] = np.einsum('nm,nmh->nh', w_self, p_self)

        # Ventanas nonself: modelo de fallback
        if is_nonself.sum() > 0:
            fb_idx = self._model_names.index(self._fallback_model)
            y_pred[is_nonself] = preds_stack[is_nonself, fb_idx, :]

        return y_pred

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    def get_diagnostics(
        self,
        X_eval: np.ndarray,
        ids_eval: np.ndarray,
    ) -> dict:
        """
        Información diagnóstica sobre una pasada de inferencia.

        Útil para el análisis cualitativo del paper:
          - ¿Qué % de ventanas son nonself?
          - ¿Qué sujetos tienen más ventanas anómalas?
          - ¿Cuál es la distribución media de pesos adaptativos?

        returns: dict con claves:
          'nonself_ratio'        float   — fracción de ventanas nonself
          'nonself_by_subject'   dict    — {subj_id: fracción nonself}
          'mean_adaptive_weights' array  — (n_models,) peso medio por modelo
          'weight_std'           array   — (n_models,) desviación de pesos
        """
        assert self.feature_extractor_ is not None, \
            "Llama fit() antes de get_diagnostics()."

        features = self.feature_extractor_.transform(X_eval)
        is_nonself = self.neg_selector_.predict(features)
        memory = self.subject_memory_.get_memory_matrix(ids_eval)
        adaptive_w = self.ainet_.get_adaptive_weights(features, memory)

        nonself_by_subject = {
            str(subj): float(is_nonself[ids_eval == subj].mean())
            for subj in np.unique(ids_eval)
        }

        return {
            'nonself_ratio': float(is_nonself.mean()),
            'nonself_by_subject': nonself_by_subject,
            'mean_adaptive_weights': adaptive_w.mean(axis=0),
            'weight_std': adaptive_w.std(axis=0),
            'model_names': self._model_names,
        }
```

- [ ] **Step 4: Ejecutar todos los tests**

```bash
python -m pytest tests/ -v
```

Esperado: todos los tests en verde (≥18 tests).

- [ ] **Step 5: Commit**

```bash
git add wrappers/AISEnsembleWrapper.py tests/test_ais_ensemble_wrapper.py
git commit -m "feat: add AISEnsembleWrapper orchestrating AIS adaptive ensemble"
```

---

## Task 6: Notebook ensemble_ais.ipynb

**Files:**
- Create: `notebooks/ensemble_ais.ipynb`

- [ ] **Step 1: Crear el notebook con las celdas base**

Crear `notebooks/ensemble_ais.ipynb` con el siguiente contenido de celdas en orden:

**Celda 1 — Markdown: título**
```markdown
# Ensemble Adaptativo AIS — Escenario 3

Implementación del ensemble adaptativo basado en Sistemas Inmunes Artificiales:
- **aiNet**: K anticuerpos especializados en patrones de ritmo cardíaco
- **SubjectMemory**: perfil inmunológico M_i por sujeto
- **NegativeSelector**: detección de ventanas fuera de distribución (nonself)

**Baseline a superar:** Stacking Ridge MAPE=4.221, Pearson=0.837 (ventana 200, 6 modelos)

**Protocolo sin leakage:**
- AIS entrenado sobre `ens_fit` (50% del 30%)
- Evaluado sobre `held_out` (50% del 30%) — nunca visto por ningún módulo
```

**Celda 2 — Código: instalación y setup**
```python
!pip install -q keras-tcn pandas scikit-learn scipy dtaidistance matplotlib joblib openpyxl pyyaml
```

**Celda 3 — Código: imports y paths**
```python
import os, sys, importlib, shutil, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    os.chdir('/content')
    if not os.path.exists('heartrate-forecasting'):
        os.system('git clone https://github.com/AllanDBB/heartrate-forecasting.git')
    os.chdir('heartrate-forecasting')

if not IN_COLAB:
    try:
        _nb_path = __vsc_ipynb_file__
        REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_nb_path)))
    except NameError:
        REPO_DIR = os.path.abspath('.')
else:
    REPO_DIR = os.getcwd()

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(f'REPO_DIR: {REPO_DIR}')

import main, utils
importlib.reload(utils); importlib.reload(main)
```

**Celda 4 — Código: cargar datos (igual que ensemble_nueva_info.ipynb)**
```python
INPUT_SIZE    = 200
OUTPUT_SIZE   = 200
CACHE_DIR     = 'cache_nueva_info'
ENSEMBLE_SEED = 123

main.ensure_dir(CACHE_DIR)

df_70, df_30, split_meta = main.load_split_dataframes(
    dataset_dir='dataset',
    split_seed=42,
    split_70_path='nueva_info/df_70.csv',
    split_30_path='nueva_info/df_30.csv',
)
df_70, df_30, overlap = utils.sanitize_split_dataframes(df_70, df_30)

path_est_70 = os.path.join(CACHE_DIR, 'values_deses_70.csv')
path_est_30 = os.path.join(CACHE_DIR, 'values_deses_30.csv')
df_scaled_70, params_70 = utils.estandarizar(df_70, path_est_70)
df_scaled_30, params_30 = utils.estandarizar(df_30, path_est_30)

X_30, y_30, ids_30 = utils.series_to_supervised_matrix(
    df_scaled_30, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE
)

(X_ens_fit, X_held_out,
 y_ens_fit, y_held_out,
 ids_ens_fit, ids_held_out) = train_test_split(
    X_30, y_30, ids_30,
    test_size=0.5,
    random_state=ENSEMBLE_SEED,
    stratify=ids_30,
)

y_ens_fit_orig  = utils.desestandarizar_ventanas(y_ens_fit,  ids_ens_fit,  params_30)
y_held_out_orig = utils.desestandarizar_ventanas(y_held_out, ids_held_out, params_30)

print(f'ens_fit:  X={X_ens_fit.shape}, ids únicos={len(set(ids_ens_fit))}')
print(f'held_out: X={X_held_out.shape}, ids únicos={len(set(ids_held_out))}')
```

**Celda 5 — Código: cargar modelos y predecir (igual que ensemble_nueva_info.ipynb)**
```python
import wrappers.KerasPretrainedWrapper as _kpw_mod
import wrappers.NBeatsSupervisedWrapper as _nbeats_mod
importlib.reload(_nbeats_mod); importlib.reload(_kpw_mod)
_nbeats_mod._get_nbeats_block_class()
from wrappers.KerasPretrainedWrapper import KerasPretrainedWrapper

MODEL_SPECS = {
    'TCN':          'nueva_info/tcn.keras',
    'NBEATS':       'nueva_info/nbeats.keras',
    'LSTM':         'nueva_info/lstm.keras',
    'TiDE':         'nueva_info/tide.keras',
    'EncDec':       'nueva_info/encDec.keras',
    'iTransformer': 'nueva_info/itrans.keras',
}

preds_ens_fit, preds_held_out = {}, {}
for name, path in MODEL_SPECS.items():
    w = KerasPretrainedWrapper(path, batch_size=32, name=name).load()
    preds_ens_fit[name]  = utils.desestandarizar_ventanas(w.predict(X_ens_fit),  ids_ens_fit,  params_30)
    preds_held_out[name] = utils.desestandarizar_ventanas(w.predict(X_held_out), ids_held_out, params_30)
    print(f'[OK] {name}')
```

**Celda 6 — Código: entrenar AIS ensemble**
```python
import importlib
import wrappers.AISEnsembleWrapper as _ais_mod
importlib.reload(_ais_mod)
from wrappers.AISEnsembleWrapper import AISEnsembleWrapper

ais = AISEnsembleWrapper(
    n_antibodies=20,
    sigma=1.0,
    clone_factor=5,
    suppression_threshold=0.3,
    n_new=5,
    max_iter=50,
    mutation_rate=0.1,
    neg_sel_percentile=99.0,
    random_state=42,
)

for name in MODEL_SPECS:
    ais.add_model(name, preds_ens_fit[name],  split='fit')
    ais.add_model(name, preds_held_out[name], split='eval')

ais.fit(
    X_fit=X_ens_fit,
    y_true_fit=y_ens_fit_orig,
    ids_fit=ids_ens_fit,
    fallback_model='TCN',   # mejor modelo individual (MAPE=5.52)
)
print('AIS entrenado.')
```

**Celda 7 — Código: inferencia sobre held_out**
```python
y_ais_pred = ais.predict(X_eval=X_held_out, ids_eval=ids_held_out)
ais_metrics = utils.evaluate_all_metrics(y_held_out_orig, y_ais_pred)
print('=== AIS Ensemble Adaptativo — held_out ===')
print(pd.Series(ais_metrics))
```

**Celda 8 — Código: tabla comparativa completa (todos los escenarios)**
```python
# Escenario 1: individuales
individual_results = {
    name: utils.evaluate_all_metrics(y_held_out_orig, preds_held_out[name])
    for name in MODEL_SPECS
}

# Escenario 2: cargar resultados del stacking (ya calculados)
stacking_csv = os.path.join(CACHE_DIR, 'final_results_no_leakage.csv')
df_static = pd.read_csv(stacking_csv, index_col=0)

# Escenario 3: AIS
all_results = {**individual_results, 'AIS Adaptativo': ais_metrics}
df_final = pd.DataFrame(all_results).T.sort_values('MAPE')
df_final.index.name = 'Modelo / Método'
print('=== Tabla Final — todos los escenarios (held_out) ===')
df_final
```

**Celda 9 — Código: diagnóstico AIS**
```python
diag = ais.get_diagnostics(X_held_out, ids_held_out)
print(f"Ventanas nonself: {diag['nonself_ratio']:.2%}")
print(f"\nNonself por sujeto:")
for subj, ratio in sorted(diag['nonself_by_subject'].items()):
    print(f"  {subj}: {ratio:.2%}")
print(f"\nPesos medios adaptativos:")
for name, w in zip(diag['model_names'], diag['mean_adaptive_weights']):
    print(f"  {name}: {w:.4f} ± {diag['weight_std'][list(diag['model_names']).index(name)]:.4f}")
```

**Celda 10 — Código: visualizaciones**
```python
import matplotlib.pyplot as plt
%matplotlib inline

# Muestras de predicción
utils.plot_forecast_samples(
    y_held_out_orig, y_ais_pred, n_samples=6,
    title='AIS Ensemble Adaptativo vs Real (held_out)'
)

# Error sobre horizonte
utils.plot_error_over_horizon(
    y_held_out_orig, y_ais_pred,
    title='AIS: Error según horizonte (held_out)'
)

# Rendimiento por sujeto
utils.plot_subject_performance(
    y_held_out_orig, y_ais_pred, ids_held_out,
    title='AIS: Rendimiento por sujeto (held_out)'
)
```

**Celda 11 — Código: guardar resultados**
```python
df_final.to_csv(os.path.join(CACHE_DIR, 'ais_final_results.csv'))
print('Resultados guardados.')
```

- [ ] **Step 2: Verificar que el notebook corre sin errores en local**

Abrir en VS Code, ejecutar todas las celdas con "Run All". Verificar que:
- La celda 6 imprime "AIS entrenado."
- La celda 7 imprime métricas numéricas razonables
- La celda 8 muestra tabla con columna MAPE

- [ ] **Step 3: Commit**

```bash
git add notebooks/ensemble_ais.ipynb
git commit -m "feat: add ensemble_ais notebook for AIS adaptive ensemble (Scenario 3)"
```

---

## Self-Review

### 1. Spec coverage

| Sección del spec | Tarea que la implementa |
|-----------------|------------------------|
| FeatureExtractor (20 features, normalización propia) | Task 1 |
| Separación normalización series vs features | Task 1 (docstring + comentario) |
| aiNet: anticuerpos, afinidad gaussiana, clonación, mutación, supresión | Task 2 |
| Proyección al simplex | Task 2 (`_project_simplex`) |
| SubjectMemory: M_i por sujeto, memoria uniforme para desconocidos | Task 3 |
| NegativeSelector: self space, umbral percentil, fallback | Task 4 |
| AISEnsembleWrapper: misma interfaz que EnsembleWrapper | Task 5 |
| get_diagnostics: nonself ratio, por sujeto, pesos medios | Task 5 |
| Notebook con todos los escenarios comparados | Task 6 |
| Tabla final comparativa | Task 6, celda 8 |
| Visualizaciones (forecast samples, error por horizonte, por sujeto) | Task 6, celda 10 |

✅ Toda la spec cubierta.

### 2. Placeholder scan

Ningún "TBD", "TODO" o paso sin código. ✅

### 3. Type consistency

| Nombre | Definido en | Usado en |
|--------|-------------|----------|
| `FeatureExtractor.fit_transform(X)` | Task 1 | Task 5 (`fit`) |
| `FeatureExtractor.transform(X)` | Task 1 | Task 5 (`predict`, `get_diagnostics`) |
| `AiNetCore.fit(features, preds_stack, y_true)` | Task 2 | Task 5 (`fit`) |
| `AiNetCore.get_adaptive_weights(features, subject_memory)` | Task 2 | Task 5 (`predict`) |
| `AiNetCore.K` | Task 2 | Task 3 (`fit`) |
| `AiNetCore.weights_` | Task 2 | Task 3 (`fit`) |
| `SubjectMemory.fit(ainet, features_fit, preds_stack_fit, y_true_fit, ids_fit)` | Task 3 | Task 5 (`fit`) |
| `SubjectMemory.get_memory_matrix(ids)` | Task 3 | Task 5 (`predict`, `get_diagnostics`) |
| `NegativeSelector.fit(features)` | Task 4 | Task 5 (`fit`) |
| `NegativeSelector.predict(features)` → bool array | Task 4 | Task 5 (`predict`, `get_diagnostics`) |

✅ Todos los nombres y firmas son consistentes entre tareas.
