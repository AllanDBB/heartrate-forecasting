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
        Valor alto (p.ej. 99) → pocos falsos positivos.
        Valor bajo → más sensible a anomalías.
    random_state : int
        Semilla para reproducibilidad (reservado para extensiones futuras).
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

        # Distancia de cada punto a su vecino más cercano (k=2 para excluir self)
        dists, _ = self._tree.query(features, k=2)
        nn_dists = dists[:, 1]

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
