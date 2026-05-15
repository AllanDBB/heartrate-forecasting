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

        for i, w in enumerate(windows.astype(np.float64)):
            # --- Estadísticas básicas (0-4) ---
            feat[i, 0] = np.mean(w)
            feat[i, 1] = np.std(w)
            feat[i, 2] = np.min(w)
            feat[i, 3] = np.max(w)
            feat[i, 4] = np.ptp(w)                               # rango

            # --- Forma (5-7) ---
            feat[i, 5] = float(skew(w))
            feat[i, 6] = float(scipy_kurtosis(w))
            feat[i, 7] = feat[i, 1] / (abs(feat[i, 0]) + 1e-8)  # CV

            # --- Tendencia (8-9) ---
            x = np.arange(L, dtype=np.float64)
            feat[i, 8] = float(np.polyfit(x, w, 1)[0])           # pendiente lineal
            feat[i, 9] = float(np.mean(np.diff(np.diff(w))))     # 2ª derivada media

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
