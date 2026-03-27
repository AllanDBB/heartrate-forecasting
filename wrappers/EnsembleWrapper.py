"""
Ensemble forecasting wrapper.

Combines multiple model predictions using:
  1. Simple average
  2. Weighted average (optimized on validation predictions)
  3. Stacking meta-learner (Ridge regression trained on validation predictions)
  4. Optional ARIMA/ETS classical baselines blended in

Usage in notebook:
    from wrappers.EnsembleWrapper import EnsembleWrapper
    ens = EnsembleWrapper()
    ens.add_model("MOMENT", y_pred_test_moment, split="eval")
    ens.add_model("MOMENT", y_pred_val_moment, split="fit")
    ens.add_model("TCN", y_pred_test_tcn, split="eval")
    ens.add_model("TCN", y_pred_val_tcn, split="fit")
    results = ens.compare_methods(y_true_eval=y_test, y_true_fit=y_val)
    y_pred_ens = ens.predict()
    metrics = ens.evaluate(y_test)
"""

import copy
import numpy as np
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class EnsembleWrapper:
    """
    Ensemble of time series forecasting models.

    Supports:
      - Simple average (no fitting required)
      - Weighted average (weights optimized on validation data)
      - Stacking (Ridge regression meta-learner trained on validation data)
      - ARIMA/exponential smoothing as additional base models

    Conventions:
      - split='fit': predictions aligned with validation data used to learn weights
      - split='eval': predictions aligned with the final evaluation set (typically test)
    """

    def __init__(self):
        self.models: Dict[str, np.ndarray] = {}      # eval predictions: name -> (N, horizon)
        self.fit_models: Dict[str, np.ndarray] = {}  # fit predictions:  name -> (N, horizon)
        self.weights: Optional[np.ndarray] = None
        self.meta_model = None
        self.method = 'average'
        self.active_model_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_store(self, split: str) -> Dict[str, np.ndarray]:
        if split == 'eval':
            return self.models
        if split == 'fit':
            return self.fit_models
        raise ValueError(f"Split desconocido: {split}. Usa 'eval' o 'fit'.")

    def _stack_predictions(self, store: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
        return np.stack([store[name] for name in names], axis=0)

    def _validate_target_shape(self, y_true: np.ndarray, store: Dict[str, np.ndarray], names: List[str]):
        if not names:
            raise ValueError("No hay modelos disponibles para el ensemble.")

        ref_shape = store[names[0]].shape
        if y_true.shape != ref_shape:
            raise ValueError(
                f"y_true={y_true.shape} no coincide con las predicciones {ref_shape}."
            )

        for name in names[1:]:
            if store[name].shape != ref_shape:
                raise ValueError(
                    f"Las predicciones del modelo {name} tienen shape {store[name].shape}; "
                    f"se esperaba {ref_shape}."
                )

    def _shared_model_names(self) -> List[str]:
        eval_names = list(self.models.keys())
        common_names = [name for name in eval_names if name in self.fit_models]

        if not common_names:
            raise ValueError(
                "No hay modelos con predicciones en split='fit' y split='eval'. "
                "Agrega predicciones de validacion con add_model(..., split='fit')."
            )

        dropped_eval = [name for name in eval_names if name not in self.fit_models]
        dropped_fit = [name for name in self.fit_models if name not in self.models]

        if dropped_eval:
            print(
                "  Aviso: se omiten en metodos ajustados por no tener split='fit': "
                f"{dropped_eval}"
            )
        if dropped_fit:
            print(
                "  Aviso: se omiten en evaluacion por no tener split='eval': "
                f"{dropped_fit}"
            )

        return common_names

    def _metric_is_higher_better(self, metric: str) -> bool:
        return utils.normalize_metric_name(metric) == 'Pearson'

    def _objective_value(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        metric = utils.normalize_metric_name(metric)
        if metric == 'MAPE':
            return float(np.mean(utils.compute_mape_for_all_windows(y_true, y_pred)))
        if metric == 'DTW':
            return float(np.mean(utils.compute_dtw_for_all_windows(y_true, y_pred)))
        if metric == 'Pearson':
            return float(np.mean(utils.compute_correlations_for_all_windows(y_true, y_pred)))
        raise ValueError(f"Metrica objetivo desconocida: {metric}")

    def _objective_loss(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        value = self._objective_value(y_true, y_pred, metric)
        if self._metric_is_higher_better(metric):
            return -value
        return value

    def _snapshot_state(self) -> dict:
        return {
            'method': self.method,
            'weights': None if self.weights is None else self.weights.copy(),
            'meta_model': copy.deepcopy(self.meta_model),
            'active_model_names': None if self.active_model_names is None else list(self.active_model_names),
        }

    def _restore_state(self, state: dict):
        self.method = state['method']
        self.weights = state['weights']
        self.meta_model = state['meta_model']
        self.active_model_names = state['active_model_names']

    def _select_best_method(self, results: Dict[str, dict], metric: str) -> str:
        if not results:
            raise ValueError("No hay resultados para seleccionar el mejor metodo.")

        metric = utils.normalize_metric_name(metric)
        normalized_results = {
            name: utils.normalize_metrics_dict(metrics)
            for name, metrics in results.items()
        }
        sample_metrics = next(iter(normalized_results.values()))
        if metric not in sample_metrics:
            raise ValueError(f"La metrica {metric} no esta disponible en los resultados.")

        if metric == 'Pearson':
            return max(
                normalized_results,
                key=lambda name: (
                    normalized_results[name][metric],
                    -normalized_results[name].get('MAPE', float('inf')),
                    -normalized_results[name].get('DTW', float('inf')),
                ),
            )

        return min(
            normalized_results,
            key=lambda name: (
                normalized_results[name][metric],
                -normalized_results[name].get('Pearson', float('-inf')),
                normalized_results[name].get('DTW', float('inf')),
            ),
        )

    def _resolve_active_method_name(self, active_method: str) -> str:
        aliases = {
            'average': 'Ensemble (Promedio)',
            'avg': 'Ensemble (Promedio)',
            'promedio': 'Ensemble (Promedio)',
            'optimize': 'Ensemble (Pesos Optimos)',
            'optimized': 'Ensemble (Pesos Optimos)',
            'weights': 'Ensemble (Pesos Optimos)',
            'pesos': 'Ensemble (Pesos Optimos)',
            'stacking': 'Ensemble (Stacking)',
            'ridge': 'Ensemble (Stacking)',
        }
        normalized = str(active_method).strip().lower()
        if normalized not in aliases:
            raise ValueError(
                f"Metodo activo desconocido: {active_method}. "
                "Usa 'average', 'optimize' o 'stacking'."
            )
        return aliases[normalized]

    # ------------------------------------------------------------------
    # Add base model predictions
    # ------------------------------------------------------------------

    def add_model(self, name: str, predictions: np.ndarray, split: str = 'eval'):
        """
        Add a model's predictions to the ensemble.

        Args:
            name: model identifier (e.g. "MOMENT", "Moirai", "TCN")
            predictions: (N, horizon) numpy array
            split: 'eval' or 'fit'
        """
        store = self._get_store(split)
        store[name] = predictions.copy()
        print(f"  + {name} [{split}]: {predictions.shape}")

    def add_arima_baseline(
        self,
        X: np.ndarray,
        order=(5, 1, 0),
        horizon: int = 200,
        split: str = 'eval',
    ):
        """
        Generate ARIMA forecasts as an additional ensemble member.
        Uses a simple per-sample ARIMA fit (statsmodels).

        Falls back to last-value forecast if ARIMA fails on a sample.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            print("  statsmodels no disponible. Instalando...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'statsmodels'])
            from statsmodels.tsa.arima.model import ARIMA

        n = X.shape[0]
        preds = np.zeros((n, horizon))

        print(f"  Generando ARIMA{order} baseline para {n} muestras [{split}]...")
        for i in range(n):
            try:
                model = ARIMA(X[i], order=order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=horizon)
                preds[i] = forecast
            except Exception:
                preds[i] = X[i, -1]

            if (i + 1) % 500 == 0 or i == n - 1:
                print(f"    ARIMA: {i + 1}/{n}")

        self._get_store(split)["ARIMA"] = preds
        print(f"  + ARIMA{order} [{split}]: {preds.shape}")

    def add_exponential_smoothing_baseline(
        self,
        X: np.ndarray,
        horizon: int = 200,
        split: str = 'eval',
    ):
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

        print(f"  Generando Exp. Smoothing baseline para {n} muestras [{split}]...")
        for i in range(n):
            try:
                model = SimpleExpSmoothing(X[i]).fit(optimized=True)
                preds[i] = model.forecast(horizon)
            except Exception:
                preds[i] = X[i, -1]

            if (i + 1) % 500 == 0 or i == n - 1:
                print(f"    ExpSmoothing: {i + 1}/{n}")

        self._get_store(split)["ExpSmoothing"] = preds
        print(f"  + ExpSmoothing [{split}]: {preds.shape}")

    # ------------------------------------------------------------------
    # Fit ensemble weights / meta-learner
    # ------------------------------------------------------------------

    def fit_weights(
        self,
        y_true: np.ndarray,
        method: str = 'optimize',
        objective_metric: str = 'MAPE',
    ):
        """
        Learn ensemble parameters on validation predictions.

        Args:
            y_true: validation target of shape (N, horizon)
            method: 'average' | 'optimize' | 'stacking'
            objective_metric: objective used for optimized weights ('MAPE', 'DTW' or 'Pearson')
        """
        self.method = method
        self.meta_model = None
        self.weights = None

        if method == 'average':
            if not self.models:
                raise ValueError("No se han agregado modelos al ensemble.")
            self.active_model_names = list(self.models.keys())
            print(f"\nEnsemble con {len(self.active_model_names)} modelos: {self.active_model_names}")
            print("  Metodo: promedio simple")
            return

        if not self.fit_models:
            raise ValueError(
                "No hay predicciones split='fit' para ajustar pesos. "
                "Agrega predicciones de validacion con add_model(..., split='fit')."
            )

        names = self._shared_model_names()
        self._validate_target_shape(y_true, self.fit_models, names)
        self.active_model_names = names

        print(f"\nEnsemble con {len(names)} modelos ajustables: {names}")

        if method == 'optimize':
            self._optimize_weights(y_true, names, objective_metric=objective_metric)
        elif method == 'stacking':
            self._fit_stacking(y_true, names)
        else:
            raise ValueError(f"Metodo desconocido: {method}")

    def _optimize_weights(self, y_true: np.ndarray, names: List[str], objective_metric: str = 'MAPE'):
        """Optimize non-negative weights that sum to 1 on validation predictions."""
        from scipy.optimize import minimize

        preds_stack = self._stack_predictions(self.fit_models, names)
        objective_metric = utils.normalize_metric_name(objective_metric)

        def normalize_weights(w):
            w = np.abs(w)
            total = np.sum(w)
            if total <= 1e-12:
                return np.ones_like(w) / len(w)
            return w / total

        def objective(w):
            w = normalize_weights(w)
            weighted = np.einsum('m,mnh->nh', w, preds_stack)
            return self._objective_loss(y_true, weighted, objective_metric)

        w0 = np.ones(len(names)) / len(names)
        result = minimize(
            objective,
            w0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6},
        )

        self.weights = normalize_weights(result.x)

        print(f"  Metodo: optimizacion de pesos ({objective_metric})")
        for i, name in enumerate(names):
            print(f"    {name}: {self.weights[i]:.4f}")

        avg_pred = np.mean(preds_stack, axis=0)
        baseline_value = self._objective_value(y_true, avg_pred, objective_metric)
        opt_pred = np.einsum('m,mnh->nh', self.weights, preds_stack)
        opt_value = self._objective_value(y_true, opt_pred, objective_metric)
        if self._metric_is_higher_better(objective_metric):
            scale = abs(baseline_value)
            improvement = 0.0 if scale <= 1e-12 else 100 * (opt_value - baseline_value) / scale
        else:
            improvement = 0.0 if baseline_value == 0 else 100 * (baseline_value - opt_value) / baseline_value
        print(f"  {objective_metric} promedio simple: {baseline_value:.6f}")
        print(f"  {objective_metric} pesos optimos: {opt_value:.6f} ({improvement:.1f}% mejora)")

    def _fit_stacking(self, y_true: np.ndarray, names: List[str]):
        """Train a Ridge regression meta-learner on validation predictions."""
        from sklearn.linear_model import Ridge

        _, horizon = y_true.shape
        features = np.hstack([self.fit_models[name] for name in names])

        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(features, y_true)

        print("  Metodo: stacking (Ridge regression)")
        print(f"  Features por muestra: {features.shape[1]} ({len(names)} modelos x {horizon} horizonte)")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, model_predictions: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Generate ensemble predictions on the evaluation split.

        Args:
            model_predictions: optional dict of name -> predictions for new data.
                               If None, uses predictions added via add_model(..., split='eval').
        Returns:
            (N, horizon) numpy array
        """
        preds = model_predictions or self.models
        if not preds:
            raise ValueError("No hay predicciones disponibles para evaluar el ensemble.")

        names = self.active_model_names or list(preds.keys())
        missing = [name for name in names if name not in preds]
        if missing:
            raise ValueError(
                "Faltan predicciones para algunos modelos activos en evaluacion: "
                f"{missing}"
            )

        if self.method == 'stacking' and self.meta_model is not None:
            features = np.hstack([preds[name] for name in names])
            return self.meta_model.predict(features)

        if self.weights is not None:
            preds_stack = self._stack_predictions(preds, names)
            return np.einsum('m,mnh->nh', self.weights, preds_stack)

        preds_stack = self._stack_predictions(preds, names)
        return np.mean(preds_stack, axis=0)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, y_true: np.ndarray, model_predictions: Dict[str, np.ndarray] = None) -> dict:
        """Evaluate ensemble with the paper metrics."""
        y_pred = self.predict(model_predictions)
        return utils.evaluate_all_metrics(y_true, y_pred)

    def compare_methods(
        self,
        y_true_eval: np.ndarray,
        y_true_fit: np.ndarray = None,
        select_by: str = 'MAPE',
        objective_metric: str = 'MAPE',
        active_method: str = 'optimize',
    ) -> dict:
        """
        Compare ensemble methods without selecting the final method on eval.

        Args:
            y_true_eval: target for final evaluation (typically test)
            y_true_fit: target for weight/meta-model fitting (typically validation)
            select_by: kept only for backward compatibility
            objective_metric: objective used by weighted averaging ('MAPE', 'DTW' or 'Pearson')
            active_method: method fixed in advance as the active ensemble
                           ('average', 'optimize' or 'stacking')

        Returns:
            dict of method -> metrics
        """
        if not self.models:
            raise ValueError("No se han agregado predicciones split='eval' al ensemble.")

        objective_metric = utils.normalize_metric_name(objective_metric)
        results = {}
        snapshots = {}
        active_method_name = self._resolve_active_method_name(active_method)

        comparison_names = list(self.models.keys())
        if y_true_fit is not None:
            comparison_names = self._shared_model_names()

        self._validate_target_shape(y_true_eval, self.models, comparison_names)

        print("=== Promedio Simple ===")
        self.method = 'average'
        self.weights = None
        self.meta_model = None
        self.active_model_names = comparison_names
        y_avg = self.predict()
        results["Ensemble (Promedio)"] = utils.evaluate_all_metrics(y_true_eval, y_avg)
        snapshots["Ensemble (Promedio)"] = self._snapshot_state()

        if y_true_fit is None:
            print(
                "\nAviso: no se proporciono y_true_fit. "
                "Se omiten pesos optimizados y stacking para evitar usar test en el ajuste."
            )
            self._restore_state(snapshots["Ensemble (Promedio)"])
            print("Metodo activo tras compare_methods: Ensemble (Promedio)")
            return results

        print("\n=== Pesos Optimizados ===")
        self.fit_weights(y_true_fit, method='optimize', objective_metric=objective_metric)
        y_opt = self.predict()
        results["Ensemble (Pesos Optimos)"] = utils.evaluate_all_metrics(y_true_eval, y_opt)
        snapshots["Ensemble (Pesos Optimos)"] = self._snapshot_state()

        print("\n=== Stacking (Ridge) ===")
        self.fit_weights(y_true_fit, method='stacking', objective_metric=objective_metric)
        y_stack = self.predict()
        results["Ensemble (Stacking)"] = utils.evaluate_all_metrics(y_true_eval, y_stack)
        snapshots["Ensemble (Stacking)"] = self._snapshot_state()

        self._restore_state(snapshots[active_method_name])
        print(
            f"\nMetodo activo tras compare_methods: {active_method_name} "
            f"(fijado de antemano; select_by={select_by} ya no usa eval)"
        )
        return results
