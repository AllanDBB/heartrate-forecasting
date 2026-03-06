"""
Ensemble forecasting wrapper.

Combines multiple model predictions using:
  1. Simple average
  2. Weighted average (optimized on validation set)
  3. Stacking meta-learner (Ridge regression trained on OOF predictions)
  4. Optional ARIMA/ETS classical baselines blended in

Usage in notebook:
    from wrappers.EnsembleWrapper import EnsembleWrapper
    ens = EnsembleWrapper()
    ens.add_model("MOMENT", y_pred_moment)
    ens.add_model("Moirai", y_pred_moirai)
    ens.add_model("TCN", y_pred_tcn)
    ens.add_arima_baseline(X_test)         # optional ARIMA component
    ens.fit_weights(y_test_val)            # learn optimal weights (or stacking)
    y_pred_ens = ens.predict()
    metrics = ens.evaluate(y_test)
"""

import numpy as np
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class EnsembleWrapper:
    """
    Ensemble of time series forecasting models.

    Supports:
      - Simple average (no fitting required)
      - Weighted average (weights optimized via scipy or grid search)
      - Stacking (Ridge regression meta-learner)
      - ARIMA/exponential smoothing as additional base models
    """

    def __init__(self):
        self.models: Dict[str, np.ndarray] = {}  # name → predictions (N, horizon)
        self.weights: Optional[np.ndarray] = None
        self.meta_model = None
        self.method = 'average'

    # ------------------------------------------------------------------
    # Add base model predictions
    # ------------------------------------------------------------------

    def add_model(self, name: str, predictions: np.ndarray):
        """
        Add a model's predictions to the ensemble.

        Args:
            name: model identifier (e.g. "MOMENT", "Moirai", "TCN")
            predictions: (N, horizon) numpy array
        """
        self.models[name] = predictions.copy()
        print(f"  + {name}: {predictions.shape}")

    def add_arima_baseline(self, X: np.ndarray, order=(5, 1, 0), horizon: int = 200):
        """
        Generate ARIMA forecasts as an additional ensemble member.
        Uses a simple per-sample ARIMA fit (statsmodels).

        Falls back to last-value forecast if ARIMA fails on a sample.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            print("  ⚠ statsmodels no disponible. Instalando...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'statsmodels'])
            from statsmodels.tsa.arima.model import ARIMA

        n = X.shape[0]
        preds = np.zeros((n, horizon))

        print(f"  Generando ARIMA{order} baseline para {n} muestras...")
        for i in range(n):
            try:
                model = ARIMA(X[i], order=order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=horizon)
                preds[i] = forecast
            except Exception:
                # Fallback: extend last value (naive forecast)
                preds[i] = X[i, -1]

            if (i + 1) % 500 == 0 or i == n - 1:
                print(f"    ARIMA: {i+1}/{n}")

        self.models["ARIMA"] = preds
        print(f"  + ARIMA{order}: {preds.shape}")

    def add_exponential_smoothing_baseline(self, X: np.ndarray, horizon: int = 200):
        """
        Simple exponential smoothing as a baseline.
        Much faster than ARIMA.
        """
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'statsmodels'])
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        n = X.shape[0]
        preds = np.zeros((n, horizon))

        print(f"  Generando Exp. Smoothing baseline para {n} muestras...")
        for i in range(n):
            try:
                model = SimpleExpSmoothing(X[i]).fit(optimized=True)
                preds[i] = model.forecast(horizon)
            except Exception:
                preds[i] = X[i, -1]

            if (i + 1) % 500 == 0 or i == n - 1:
                print(f"    ExpSmoothing: {i+1}/{n}")

        self.models["ExpSmoothing"] = preds
        print(f"  + ExpSmoothing: {preds.shape}")

    # ------------------------------------------------------------------
    # Fit ensemble weights / meta-learner
    # ------------------------------------------------------------------

    def fit_weights(self, y_true: np.ndarray, method: str = 'optimize'):
        """
        Learn optimal ensemble weights on a set of predictions.

        Args:
            y_true: (N, horizon) ground truth
            method: 'average' | 'optimize' | 'stacking'
              - 'average':  equal weights (no fitting)
              - 'optimize': minimize MSE via scipy optimization
              - 'stacking': train Ridge regression on model outputs
        """
        self.method = method
        names = list(self.models.keys())
        n_models = len(names)

        if n_models == 0:
            raise ValueError("No se han agregado modelos al ensemble.")

        print(f"\nEnsemble con {n_models} modelos: {names}")

        if method == 'average':
            self.weights = np.ones(n_models) / n_models
            print(f"  Método: promedio simple → pesos = {self.weights}")

        elif method == 'optimize':
            self._optimize_weights(y_true, names)

        elif method == 'stacking':
            self._fit_stacking(y_true, names)

        else:
            raise ValueError(f"Método desconocido: {method}")

    def _optimize_weights(self, y_true: np.ndarray, names: list):
        """Optimize weights to minimize MSE using scipy."""
        from scipy.optimize import minimize

        preds_stack = np.stack([self.models[n] for n in names], axis=0)  # (M, N, H)

        def objective(w):
            w = np.abs(w) / np.sum(np.abs(w))  # normalize to sum to 1
            weighted = np.einsum('m,mnh->nh', w, preds_stack)
            return np.mean((weighted - y_true) ** 2)

        # Initial: equal weights
        w0 = np.ones(len(names)) / len(names)
        result = minimize(objective, w0, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-6})

        self.weights = np.abs(result.x) / np.sum(np.abs(result.x))

        print(f"  Método: optimización de pesos (MSE)")
        for i, name in enumerate(names):
            print(f"    {name}: {self.weights[i]:.4f}")

        # Show improvement
        avg_pred = np.mean(preds_stack, axis=0)
        mse_avg = np.mean((avg_pred - y_true) ** 2)
        mse_opt = result.fun
        print(f"  MSE promedio simple: {mse_avg:.6f}")
        print(f"  MSE pesos óptimos:  {mse_opt:.6f} ({100*(mse_avg-mse_opt)/mse_avg:.1f}% mejora)")

    def _fit_stacking(self, y_true: np.ndarray, names: list):
        """Train a Ridge regression meta-learner (stacking)."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        N, H = y_true.shape

        # Feature matrix: for each sample, concatenate all model predictions
        # Shape: (N, M*H) where M = number of models
        features = np.hstack([self.models[n] for n in names])  # (N, M*H)

        # Target: flatten horizon for per-step prediction
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(features, y_true)

        # Cross-validation score
        scores = cross_val_score(Ridge(alpha=1.0), features, y_true,
                                 cv=3, scoring='neg_mean_squared_error')
        print(f"  Método: stacking (Ridge regression)")
        print(f"  CV MSE: {-np.mean(scores):.6f} ± {np.std(scores):.6f}")
        print(f"  Features por muestra: {features.shape[1]} ({len(names)} modelos × {H} horizonte)")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, model_predictions: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            model_predictions: optional dict of name→predictions for new data.
                               If None, uses predictions added via add_model().
        Returns:
            (N, horizon) numpy array
        """
        preds = model_predictions or self.models
        names = list(preds.keys())

        if self.method == 'stacking' and self.meta_model is not None:
            features = np.hstack([preds[n] for n in names])
            return self.meta_model.predict(features)

        elif self.weights is not None:
            preds_stack = np.stack([preds[n] for n in names], axis=0)
            return np.einsum('m,mnh->nh', self.weights, preds_stack)

        else:
            # Default: simple average
            preds_stack = np.stack([preds[n] for n in names], axis=0)
            return np.mean(preds_stack, axis=0)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, y_true: np.ndarray,
                 model_predictions: Dict[str, np.ndarray] = None) -> dict:
        """Evaluate ensemble with all metrics."""
        y_pred = self.predict(model_predictions)
        return utils.evaluate_all_metrics(y_true, y_pred)

    def compare_methods(self, y_true: np.ndarray) -> dict:
        """
        Compare all ensemble methods: average, optimized weights, stacking.
        Returns dict of method → metrics.
        """
        results = {}
        names = list(self.models.keys())

        # 1. Simple average
        preds_stack = np.stack([self.models[n] for n in names], axis=0)
        y_avg = np.mean(preds_stack, axis=0)
        print("=== Promedio Simple ===")
        results["Ensemble (Promedio)"] = utils.evaluate_all_metrics(y_true, y_avg)

        # 2. Optimized weights
        self.fit_weights(y_true, method='optimize')
        y_opt = self.predict()
        print("\n=== Pesos Optimizados ===")
        results["Ensemble (Pesos Óptimos)"] = utils.evaluate_all_metrics(y_true, y_opt)

        # 3. Stacking
        self.fit_weights(y_true, method='stacking')
        y_stack = self.predict()
        print("\n=== Stacking (Ridge) ===")
        results["Ensemble (Stacking)"] = utils.evaluate_all_metrics(y_true, y_stack)

        return results
