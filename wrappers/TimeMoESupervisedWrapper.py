import gc
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# ── Polyfill for DynamicCache.from_legacy_cache ──────────────────────────────
# Removed in transformers >= 4.45. TimeMoE's HuggingFace code still uses it.
from transformers import DynamicCache
if not hasattr(DynamicCache, 'from_legacy_cache'):
    @classmethod
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    DynamicCache.from_legacy_cache = _from_legacy_cache

if not hasattr(DynamicCache, 'get_usable_length'):
    def _get_usable_length(self, new_seq_length, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]
    DynamicCache.get_usable_length = _get_usable_length
# ─────────────────────────────────────────────────────────────────────────────

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
        return base_pred + correction


class TimeMoESupervisedWrapper:

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.input_size = config['forecasting']['input_size']
        self.output_size = config['forecasting']['output_size']
        self.batch_size = config['forecasting']['batch_size']
        self.device = config['model']['device']

        self.model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.adapter = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 10, lr: float = 1e-3):
        """
        Fine-tune Time-MoE via a residual adapter trained on zero-shot predictions.

        Strategy:
          1. Run the frozen Time-MoE model once to cache zero-shot predictions.
          2. Train a lightweight MLP adapter that corrects the base predictions
             using both the input context and the zero-shot output.
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
            seqs = torch.tensor(batch, dtype=torch.float32)

            mean = seqs.mean(dim=-1, keepdim=True)
            std = seqs.std(dim=-1, keepdim=True).clamp(min=1e-8)
            normed_seqs = (seqs - mean) / std

            with torch.no_grad():
                output = self.model.generate(
                    normed_seqs.to(self.device),
                    max_new_tokens=self.output_size,
                )

            normed_preds = output[:, -self.output_size:].float().cpu()
            preds = normed_preds * std + mean
            y_pred[i:i + len(batch)] = preds.numpy()

            del seqs, normed_seqs, output, normed_preds, preds
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
