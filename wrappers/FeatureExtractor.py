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
        """Extrae features sin normalizar. Shape: (n_windows, 20). Totalmente vectorizado."""
        W = windows.astype(np.float64)
        n, L = W.shape
        feat = np.zeros((n, self.N_FEATURES), dtype=np.float64)

        # --- Estadísticas básicas (0-4) ---
        feat[:, 0] = W.mean(axis=1)
        feat[:, 1] = W.std(axis=1)
        feat[:, 2] = W.min(axis=1)
        feat[:, 3] = W.max(axis=1)
        feat[:, 4] = feat[:, 3] - feat[:, 2]                     # rango

        # --- Forma (5-7) ---
        mu = feat[:, 0:1]                                         # (n, 1)
        sigma = feat[:, 1:2] + 1e-8
        W_c = W - mu                                              # centrado
        feat[:, 5] = ((W_c / sigma) ** 3).mean(axis=1)           # skewness
        feat[:, 6] = ((W_c / sigma) ** 4).mean(axis=1) - 3       # excess kurtosis
        feat[:, 7] = feat[:, 1] / (np.abs(feat[:, 0]) + 1e-8)   # CV

        # --- Tendencia (8-9) ---
        x = np.arange(L, dtype=np.float64)
        x_c = x - x.mean()
        slope_denom = (x_c ** 2).sum() + 1e-8
        feat[:, 8] = (W_c * x_c).sum(axis=1) / slope_denom       # pendiente lineal
        feat[:, 9] = np.diff(np.diff(W, axis=1), axis=1).mean(axis=1)  # 2ª derivada

        # --- Autocorrelación en lags 1, 5, 10, 20 (10-13) ---
        norm = (W_c ** 2).sum(axis=1) + 1e-8                     # (n,)
        for j, lag in enumerate([1, 5, 10, 20]):
            if lag < L:
                feat[:, 10 + j] = (W_c[:, :-lag] * W_c[:, lag:]).sum(axis=1) / norm

        # --- Espectral (14-15): FFT vectorizado ---
        fft_vals = np.fft.rfft(W, axis=1)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(L)
        dom_idx = np.argmax(psd[:, 1:], axis=1) + 1              # ignorar DC
        feat[:, 14] = freqs[dom_idx]
        psd_sum = psd.sum(axis=1, keepdims=True) + 1e-8
        psd_norm = psd / psd_sum
        feat[:, 15] = -(psd_norm * np.log(psd_norm + 1e-8)).sum(axis=1)

        # --- Actividad (16-17) ---
        crossings = np.diff(np.sign(W - mu), axis=1)
        feat[:, 16] = (crossings != 0).sum(axis=1).astype(float)
        peaks_mask = (W[:, 1:-1] > W[:, :-2]) & (W[:, 1:-1] > W[:, 2:])
        feat[:, 17] = peaks_mask.sum(axis=1) / L

        # --- Percentiles (18-19) ---
        feat[:, 18] = np.percentile(W, 25, axis=1)
        feat[:, 19] = np.percentile(W, 75, axis=1)

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
