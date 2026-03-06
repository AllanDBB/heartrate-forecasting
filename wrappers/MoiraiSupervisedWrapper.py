import gc
import os
import sys
import torch
import torch.nn as nn
import torchvision  # must be imported before uni2ts to register torchvision ops
import numpy as np
import yaml
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
from gluonts.dataset.common import ListDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class _ForecastAdapter(nn.Module):
    """Lightweight MLP adapter that refines zero-shot predictions using input context."""
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
        return base_pred + correction  # residual learning


class MoiraiSupervisedWrapper:

    def __init__(self, config_path: str):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.input_size = config['forecasting']['input_size']
        self.output_size = config['forecasting']['output_size']
        self.batch_size = config['forecasting']['batch_size']
        self.freq = config['model']['freq']
        self.num_samples = config['model']['num_samples']
        self.device = config['model'].get('device', 'cuda')

        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(config['model']['name']),
            prediction_length=self.output_size,
            context_length=self.input_size,
            patch_size=config['model']['patch_size'],
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        self.predictor = model.create_predictor(batch_size=self.batch_size)
        self.adapter = None  # set after fit()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 10, lr: float = 1e-3):
        """
        Fine-tune Moirai via a residual adapter trained on zero-shot predictions.

        Strategy:
          1. Run the frozen Moirai model once to cache zero-shot predictions.
          2. Train a lightweight MLP adapter that learns to correct the
             base predictions using both the context and the zero-shot output.
          3. During subsequent predict() calls the adapter refines results.

        Args:
            X_train: (N, input_size)
            y_train: (N, output_size)
            epochs:  adapter training epochs
            lr:      adapter learning rate
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
            dataset = ListDataset(
                [{"start": pd.Period("2020-01-01", freq=self.freq), "target": batch[j]}
                 for j in range(batch.shape[0])],
                freq=self.freq
            )
            forecasts = list(self.predictor.predict(dataset))
            for j, forecast in enumerate(forecasts):
                y_pred[i + j] = forecast.median
            del dataset, forecasts
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
        """Evaluate with all metrics: MAE, RMSE, MAPE, DTW, DDTW, CrossCorrelation."""
        y_pred = self.predict(X)
        return utils.evaluate_all_metrics(y_true, y_pred)
