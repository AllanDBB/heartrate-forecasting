import gc
import sys
import torch
import numpy as np
import yaml
import pandas as pd

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

        estimator = LagLlamaEstimator(
            ckpt_path=config['model']['ckpt_path'],
            prediction_length=self.output_size,
            context_length=self.input_size,
            device=self.device,
            num_parallel_samples=self.num_samples,
        )
        # Crear el predictor para inferencia
        self.predictor = estimator.create_predictor(
            estimator.create_training_network(self.device),
            estimator.create_instance_splitter("test"),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:

        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.output_size))

        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i + self.batch_size]

            # Convertir cada fila a formato GluonTS ListDataset
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