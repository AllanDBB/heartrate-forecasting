import gc
import torch
import numpy as np
import yaml
from transformers import AutoModelForCausalLM


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

    def predict(self, X: np.ndarray) -> np.ndarray:

        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.output_size))

        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i + self.batch_size]  # (B, input_size)

            seqs = torch.tensor(batch, dtype=torch.float32)

            # Time-MoE requires per-sample normalization
            mean = seqs.mean(dim=-1, keepdim=True)
            std = seqs.std(dim=-1, keepdim=True).clamp(min=1e-8)
            normed_seqs = (seqs - mean) / std

            with torch.no_grad():
                output = self.model.generate(
                    normed_seqs.to(self.device),
                    max_new_tokens=self.output_size,
                )  # shape: (B, input_size + output_size)

            normed_preds = output[:, -self.output_size:].float().cpu()

            # Inverse per-sample normalization to return to the globally-scaled space
            preds = normed_preds * std + mean
            y_pred[i:i + len(batch)] = preds.numpy()

            del seqs, normed_seqs, output, normed_preds, preds
            gc.collect()
            torch.cuda.empty_cache()

        return y_pred

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:

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
