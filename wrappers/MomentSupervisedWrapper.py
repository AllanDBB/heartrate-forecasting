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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class MomentSupervisedWrapper:
    """
    Wrapper for AutonLab/MOMENT-1-large in forecasting mode.

    MOMENT's forecasting head is a randomly-initialized linear layer,
    so it MUST be fine-tuned with fit() before predict() is useful.

    Speed strategy
    --------------
    The T5 encoder is frozen (0.3B params). Running it on every training
    batch is the main bottleneck. fit() therefore pre-computes the encoder
    outputs ONCE (in no_grad mode) and caches them in CPU RAM. Subsequent
    epoch iterations only execute the tiny linear forecasting head, making
    training orders of magnitude faster.

    Context length is fixed at 512. Input sequences of length < 512
    are left-padded with zeros; the input_mask marks valid timesteps.
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.input_size = config['forecasting']['input_size']
        self.output_size = config['forecasting']['output_size']
        self.batch_size = config['forecasting']['batch_size']
        self.context_length = config['forecasting'].get('context_length', 512)
        self.device = config['model']['device']
        self.epochs = config['training']['epochs']
        self.lr = config['training']['lr']

        from momentfm import MOMENTPipeline

        self.model = MOMENTPipeline.from_pretrained(
            config['model']['name'],
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': self.output_size,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': True,    # freeze T5 encoder weights
                'freeze_embedder': True,   # freeze patch-embedding layer
                'freeze_head': False,      # train only the linear forecasting head
            },
        )
        self.model.init()
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, X: np.ndarray):
        """
        Left-pad X from (N, input_size) to (N, context_length) and build
        the corresponding input_mask (1 = real data, 0 = padding).

        Returns:
            x_tensor  : (N, 1, context_length)  float32
            mask_tensor: (N, context_length)     float32
        """
        n = X.shape[0]
        padded = np.zeros((n, self.context_length), dtype=np.float32)
        padded[:, -self.input_size:] = X

        mask = np.zeros((n, self.context_length), dtype=np.float32)
        mask[:, -self.input_size:] = 1.0

        # Add channel dim for univariate: (N, 1, context_length)
        x_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        return x_tensor, mask_tensor

    def _precompute_head_inputs(self, X: np.ndarray) -> torch.Tensor:
        """
        Run the frozen encoder ONCE in no_grad mode and capture the tensor
        that MOMENT passes to its forecasting head (via a forward hook).

        Returns:
            head_inputs: CPU tensor, shape depends on architecture
                         (typically (N, 1, n_patches * d_model) for MOMENT-large)
        """
        x_tensor, mask_tensor = self._prepare_inputs(X)
        loader = DataLoader(
            TensorDataset(x_tensor, mask_tensor),
            batch_size=self.batch_size,
        )

        captured = []

        def _hook(module, inp, out):   # noqa: ARG001
            # inp[0] is the tensor fed into model.head
            captured.append(inp[0].detach().cpu())

        handle = self.model.head.register_forward_hook(_hook)

        self.model.eval()
        with torch.no_grad():
            for x_batch, mask_batch in tqdm(loader, desc="Pre-computing embeddings", leave=False):
                self.model(
                    x_enc=x_batch.to(self.device),
                    input_mask=mask_batch.to(self.device),
                )
                gc.collect()
                torch.cuda.empty_cache()

        handle.remove()
        return torch.cat(captured, dim=0)   # (N, ...)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fine-tune the linear forecasting head on (X_train, y_train).

        Step 1 — Pre-compute encoder embeddings (runs encoder ONCE, no grad).
        Step 2 — Train only model.head on the cached embeddings (very fast).

        Args:
            X_train: (N, input_size)  context sequences
            y_train: (N, output_size) target forecasts
        """
        print("Paso 1/2: Pre-computando embeddings del encoder (solo una vez)…")
        head_inputs = self._precompute_head_inputs(X_train)  # cached on CPU

        # MOMENT outputs (B, 1, output_size); match target shape
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(head_inputs, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss().to(self.device)
        # Only pass head parameters — encoder is frozen and not involved
        optimizer = Adam(self.model.head.parameters(), lr=self.lr)

        print("Paso 2/2: Entrenando cabeza lineal…")
        self.model.head.train()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.epochs):
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for emb_batch, y_batch in pbar:
                emb_batch = emb_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    forecast = self.model.head(emb_batch)   # (B, 1, output_size)
                    loss = criterion(forecast, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                del emb_batch, y_batch, forecast
                gc.collect()
                torch.cuda.empty_cache()

            print(f"Epoch {epoch + 1}/{self.epochs}: train_loss={total_loss / len(loader):.4f}")

        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate forecasts.

        Args:
            X: (N, input_size) context sequences

        Returns:
            y_pred: (N, output_size) forecasts
        """
        x_tensor, mask_tensor = self._prepare_inputs(X)
        dataset = TensorDataset(x_tensor, mask_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for (x_batch, mask_batch) in tqdm(loader, desc="Prediciendo"):
                x_batch = x_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)

                output = self.model(x_enc=x_batch, input_mask=mask_batch)
                # output.forecast: (B, 1, output_size) -> squeeze -> (B, output_size)
                pred = output.forecast.squeeze(1).cpu().numpy()
                preds.append(pred)

                del x_batch, mask_batch, output
                gc.collect()
                torch.cuda.empty_cache()

        return np.concatenate(preds, axis=0)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate with all metrics: MAE, RMSE, MAPE, DTW, DDTW, CrossCorrelation.
        """
        y_pred = self.predict(X)
        return utils.evaluate_all_metrics(y_true, y_pred)
