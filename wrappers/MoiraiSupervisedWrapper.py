import gc
import torch
import numpy as np
import yaml
import pandas as pd
from gluonts.dataset.common import ListDataset


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

    def predict(self, X: np.ndarray) -> np.ndarray:

        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.output_size))

        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i + self.batch_size]

            # Convertir cada fila a formato GluonTS ListDataset
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
