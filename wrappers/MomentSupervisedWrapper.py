import gc
import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


class MomentSupervisedWrapper:
    """
    Wrapper for AutonLab/MOMENT-1-large in forecasting mode.

    MOMENT's forecasting head is a randomly-initialized linear layer,
    so it MUST be fine-tuned with fit() before predict() is useful.
    The T5 encoder and patch-embedding layers are frozen; only the
    linear head is trained (fast, typically 1 epoch is enough).

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fine-tune the linear forecasting head on (X_train, y_train).
        The encoder and embedder remain frozen.

        Args:
            X_train: (N, input_size)  context sequences
            y_train: (N, output_size) target forecasts
        """
        x_tensor, mask_tensor = self._prepare_inputs(X_train)
        # Target shape expected by the model: (N, 1, output_size)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(x_tensor, mask_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss().to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for x_batch, mask_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(x_enc=x_batch, input_mask=mask_batch)
                # output.forecast shape: (B, 1, output_size)
                loss = criterion(output.forecast, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                del x_batch, mask_batch, y_batch, output
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
            for (x_batch, mask_batch) in loader:
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
        Evaluate MAE, RMSE, and MAPE.

        Note: MAPE is unreliable on standardized (zero-mean) data because
        the denominator can be zero or near-zero. Rely on MAE and RMSE.
        """
        y_pred = self.predict(X)

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        metrics = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "MAPE": round(float(mape), 4),
        }

        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        return metrics
