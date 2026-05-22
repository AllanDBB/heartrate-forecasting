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
    """
    Residual adapter on top of frozen Lag-Llama zero-shot predictions.
    Same architecture as MoiraiSupervisedWrapper for consistency.
    """
    def __init__(self, input_size, output_size, hidden=512):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.pred_encoder = nn.Sequential(
            nn.Linear(output_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True, dropout=0.1)
        self.refiner = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, output_size),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_context, base_pred):
        ctx = self.context_encoder(x_context)
        pred_emb = self.pred_encoder(base_pred)
        ctx_q = ctx.unsqueeze(1)
        pred_kv = pred_emb.unsqueeze(1)
        attn_out, _ = self.attention(ctx_q, pred_kv, pred_kv)
        attn_out = attn_out.squeeze(1)
        combined = torch.cat([attn_out, pred_emb], dim=-1)
        correction = self.refiner(combined)
        alpha = torch.sigmoid(self.gate)
        return base_pred + alpha * correction


class LagLlamaSupervisedWrapper:
    """
    Wrapper for time-series-foundation-models/Lag-Llama in forecasting mode.

    Fine-tuning strategy:
    - Lag-Llama base model stays frozen (zero-shot predictions cached once)
    - A multi-layer residual adapter with cross-attention + gated residual is trained
    - Adapter is saved to disk for reuse in ensemble notebooks
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.input_size = config['forecasting']['input_size']
        self.output_size = config['forecasting']['output_size']
        self.batch_size = config['forecasting']['batch_size']
        self.num_samples = config['model']['num_samples']
        self.freq = config['model'].get('freq', 'h')
        self.context_length = config['model'].get('context_length', self.input_size)

        requested_device = config['model'].get('device', 'cpu')
        self.device = requested_device
        if requested_device == 'cuda' and not torch.cuda.is_available():
            print("  Aviso: CUDA no disponible; Lag-Llama usara CPU.")
            self.device = 'cpu'

        from huggingface_hub import hf_hub_download
        print("Descargando checkpoint de Lag-Llama desde HuggingFace...")
        ckpt_path = hf_hub_download(
            "time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt"
        )
        print(f"  Checkpoint: {ckpt_path}")

        from lag_llama.gluon.estimator import LagLlamaEstimator

        device_obj = torch.device(self.device)
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=self.output_size,
            context_length=self.context_length,
            num_parallel_samples=self.num_samples,
            device=device_obj,
            batch_size=self.batch_size,
            nonnegative_pred_samples=False,
        )
        self.predictor = estimator.create_predictor(
            input_transform=estimator.create_transformation(),
            trained_network=estimator.create_lightning_module(),
        )
        self.adapter = None
        print("Lag-Llama listo para inferencia zero-shot.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 20, lr: float = 5e-4):
        """
        Fine-tune via a residual adapter on top of cached zero-shot predictions.

        1. Run frozen Lag-Llama once to cache base predictions.
        2. Train adapter with cross-attention + gated residual + cosine LR + early stopping.
        """
        print("Paso 1/2: Generando predicciones zero-shot de Lag-Llama (una sola vez)...")
        base_preds = self._predict_raw(X_train)

        print("Paso 2/2: Entrenando adaptador residual...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.adapter = _ForecastAdapter(
            self.input_size, self.output_size, hidden=512
        ).to(device)

        trainable = sum(p.numel() for p in self.adapter.parameters())
        print(f"  Parámetros del adaptador: {trainable:,}")

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(base_preds, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        from torch.optim.lr_scheduler import CosineAnnealingLR
        optimizer = Adam(self.adapter.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        self.adapter.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for x_b, bp_b, y_b in loader:
                x_b = x_b.to(device)
                bp_b = bp_b.to(device)
                y_b = y_b.to(device)

                pred = self.adapter(x_b, bp_b)
                loss = criterion(pred, y_b)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg = total_loss / len(loader)
            improve = ""
            if avg < best_loss:
                best_loss = avg
                patience_counter = 0
                improve = " ★"
            else:
                patience_counter += 1

            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}{improve}")

            if patience_counter >= patience:
                print(f"  Early stopping en epoch {epoch+1}")
                break

        self.adapter.eval()
        gate_val = torch.sigmoid(self.adapter.gate).item()
        print(f"Adaptador entrenado. Gate = {gate_val:.3f} (0=solo base, 1=corrección completa)")

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Zero-shot prediction using Lag-Llama base model (no adapter)."""
        from gluonts.dataset.common import ListDataset

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
            if torch.cuda.is_available():
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
        y_pred = self.predict(X)
        return utils.evaluate_all_metrics(y_true, y_pred)
