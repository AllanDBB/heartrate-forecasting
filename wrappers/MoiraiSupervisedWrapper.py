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
from tqdm import tqdm
from gluonts.dataset.common import ListDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class _ForecastAdapter(nn.Module):
    """
    Deeper adapter that refines zero-shot predictions.
    v2: Multi-layer with skip connections, per-horizon attention, and
    a separate trend + detail branch.
    """
    def __init__(self, input_size, output_size, hidden=512):
        super().__init__()
        # Encode context into a representation
        self.context_encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # Encode base prediction
        self.pred_encoder = nn.Sequential(
            nn.Linear(output_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # Attention: which parts of context matter for correction
        self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True, dropout=0.1)
        # Refiner: produce correction
        self.refiner = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, output_size),
        )
        # Learnable mixing weight: how much correction to apply (starts near 0)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_context, base_pred):
        ctx = self.context_encoder(x_context)         # (B, hidden)
        pred_emb = self.pred_encoder(base_pred)       # (B, hidden)

        # Cross-attention (context attending to prediction)
        ctx_q = ctx.unsqueeze(1)    # (B, 1, hidden)
        pred_kv = pred_emb.unsqueeze(1)  # (B, 1, hidden)
        attn_out, _ = self.attention(ctx_q, pred_kv, pred_kv)
        attn_out = attn_out.squeeze(1)  # (B, hidden)

        combined = torch.cat([attn_out, pred_emb], dim=-1)  # (B, hidden*2)
        correction = self.refiner(combined)

        # Gated residual: sigmoid(gate) controls how much correction is applied
        alpha = torch.sigmoid(self.gate)
        return base_pred + alpha * correction


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

        # Resolve patch_size: must be an int from {8,16,32,64,128}
        raw_ps = config['model']['patch_size']
        if isinstance(raw_ps, str) and raw_ps.lower() == 'auto':
            # Pick largest patch_size that divides context_length
            for ps in [64, 32, 16, 8]:
                if self.input_size % ps == 0:
                    raw_ps = ps
                    break
            else:
                raw_ps = 32
            print(f"  patch_size auto → {raw_ps}")
        patch_size = int(raw_ps)

        module = MoiraiModule.from_pretrained(config['model']['name'])

        try:
            model = MoiraiForecast(
                module=module,
                prediction_length=self.output_size,
                context_length=self.input_size,
                patch_size=patch_size,
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        except (ValueError, TypeError) as e:
            print(f"  ⚠ MoiraiForecast init falló con patch_size={patch_size}: {e}")
            print(f"  Reintentando sin feat dims explícitos...")
            model = MoiraiForecast(
                module=module,
                prediction_length=self.output_size,
                context_length=self.input_size,
                patch_size=patch_size,
                num_samples=self.num_samples,
                target_dim=1,
            )

        self.predictor = model.create_predictor(batch_size=self.batch_size)
        self.adapter = None  # set after fit()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 20, lr: float = 5e-4):
        """
        Fine-tune Moirai via a deeper residual adapter (v2).

        Strategy:
          1. Run the frozen Moirai model once to cache zero-shot predictions.
          2. Train a multi-layer adapter with attention + gated residual.
          3. Use cosine annealing + gradient clipping + early stopping.
        """
        print("Paso 1/2: Generando predicciones zero-shot del modelo base (una sola vez)…")
        base_preds = self._predict_raw(X_train)

        print("Paso 2/2: Entrenando adaptador residual (v2)…")
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
            for x_batch, bp_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                bp_batch = bp_batch.to(device)
                y_batch = y_batch.to(device)

                pred = self.adapter(x_batch, bp_batch)
                loss = criterion(pred, y_batch)

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
