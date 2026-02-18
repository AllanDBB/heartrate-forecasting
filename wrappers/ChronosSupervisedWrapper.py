import gc
import torch
import numpy as np
import yaml
from chronos import ChronosPipeline

class ChronosSupervisedWrapper:

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.device = config['model']['device']
        self.num_samples = config['model']['num_samples']
        self.input_size = config['forecasting']['input_size']
        self.output_size = config['forecasting']['output_size']
        self.batch_size = config['forecasting']['batch_size']

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }

        torch_dtype = dtype_map[config["model"]["torch_dtype"]]

        self.pipeline = ChronosPipeline.from_pretrained(
            config['model']['name'],
            torch_dtype=torch_dtype,
            device_map=self.device,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:

        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.output_size))

        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i+self.batch_size]
            context = torch.tensor(batch, dtype=torch.float32).to(self.device)
            forecast = self.pipeline.predict(context= context, prediction_length=self.output_size, num_samples=self.num_samples)

            # forecast shape: (num_samples, batch_size, output_size)
            # We take the median across the num_samples dimension to get a single prediction per input sequence
            median = np.quantile(forecast.numpy(), 0.5, axis=1)
            y_pred[i:i+self.batch_size] = median

            del context, forecast, median
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
            "MAPE": round(float(mape), 4)
        }

        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        return metrics




## Questions for later:
# 1. Why use axis = 1 in median
# 2. what a context means
# 3. cleaning context, forecast, median could cause problems?
    
# 4. Explain what MAE, RMSE & MAPE are and how they are calculated.