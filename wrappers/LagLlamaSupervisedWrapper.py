import gc
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class _ForecastAdapter(nn.Module):
    """Lightweight MLP adapter that refines zero-shot predictions."""
    def __init__(self, input_size, output_size, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.refiner = nn.Sequential(
            nn.Linear(hidden + output_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x_context, base_pred):
        ctx = self.encoder(x_context)
        combined = torch.cat([ctx, base_pred], dim=-1)
        correction = self.refiner(combined)
        return base_pred + correction


class LagLlamaSupervisedWrapper:

    def __init__(self, config_path: str, lag_llama_repo_path: str = "lag-llama"):
        # Agregar el repo clonado de lag-llama al path para poder importarlo
        sys.path.insert(0, lag_llama_repo_path)

        # Imports aquí adentro porque lag-llama debe estar en sys.path primero
        from gluonts.dataset.common import ListDataset
        from gluonts.evaluation.backtest import make_evaluation_predictions
        from lag_llama.gluon.estimator import LagLlamaEstimator

        self._ListDataset = ListDataset
        self._make_evaluation_predictions = make_evaluation_predictions

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.device = torch.device(config['model']['device'] if torch.cuda.is_available() else "cpu")
        self.num_samples = config['model']['num_samples']
        self.input_size = config['forecasting']['input_size']
        self.output_size = config['forecasting']['output_size']
        self.batch_size = config['forecasting']['batch_size']
        self.freq = config['model']['freq']

        # Cargar hiperparámetros del checkpoint
        ckpt = torch.load(config['model']['ckpt_path'], map_location=self.device, weights_only=False)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        from gluonts.torch.distributions.studentT import StudentTOutput
        torch.serialization.add_safe_globals([StudentTOutput])

        estimator = LagLlamaEstimator(
            ckpt_path=config['model']['ckpt_path'],
            prediction_length=self.output_size,
            context_length=self.input_size,
            device=self.device,
            num_parallel_samples=self.num_samples,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
        )
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        self.predictor = estimator.create_predictor(transformation, lightning_module)
        self.adapter = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 10, lr: float = 1e-3):
        """
        Fine-tune Lag-Llama via a residual adapter trained on zero-shot predictions.

        Strategy:
          1. Run the frozen Lag-Llama model once to cache zero-shot predictions.
          2. Train a lightweight MLP adapter that corrects the base predictions.
          3. During subsequent predict() calls the adapter refines results.
        """
        print("Paso 1/2: Generando predicciones zero-shot del modelo base (una sola vez)…")
        base_preds = self._predict_raw(X_train)

        print("Paso 2/2: Entrenando adaptador residual…")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.adapter = _ForecastAdapter(
            self.input_size, self.output_size
        ).to(device)

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(base_preds, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = Adam(self.adapter.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.adapter.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, bp_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                bp_batch = bp_batch.to(device)
                y_batch = y_batch.to(device)

                pred = self.adapter(x_batch, bp_batch)
                loss = criterion(pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f}")

        self.adapter.eval()
        print("Adaptador entrenado.")

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Zero-shot prediction without adapter."""
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.output_size))

        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i + self.batch_size]

            dataset = self._ListDataset(
                [{"start": pd.Period("2020-01-01", freq=self.freq), "target": batch[j]}
                 for j in range(batch.shape[0])],
                freq=self.freq
            )

            forecast_it, _ = self._make_evaluation_predictions(
                dataset=dataset,
                predictor=self.predictor,
            )
            forecasts = list(forecast_it)

            for j in range(len(forecasts)):
                y_pred[i + j] = forecasts[j].median

            del dataset, forecast_it, forecasts
            gc.collect()
            torch.cuda.empty_cache()

        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        base = self._predict_raw(X)
        if self.adapter is not None:
            device = next(self.adapter.parameters()).device
            x_t = torch.tensor(X, dtype=torch.float32).to(device)
            b_t = torch.tensor(base, dtype=torch.float32).to(device)
            with torch.no_grad():
                refined = self.adapter(x_t, b_t).cpu().numpy()
            return refined
        return base

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """Evaluate with the paper metrics: MAPE, DTW, Correlation."""
        y_pred = self.predict(X)
        return utils.evaluate_all_metrics(y_true, y_pred)
