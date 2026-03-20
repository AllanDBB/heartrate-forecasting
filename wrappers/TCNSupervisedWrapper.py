import numpy as np
import yaml
import os
import sys
import math
import matplotlib.pyplot as plt

# Add parent directory to path for utils access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class TCNSupervisedWrapper:
    """
    Wrapper for the TCN (Temporal Convolutional Network) model.

    This model was originally created by the professor (modeloCreadoProfe).
    It uses TensorFlow/Keras with the keras-tcn library.

    API:
        fit(X_train, y_train, X_val, y_val)  → train from scratch
        predict(X)                            → numpy array of predictions
        evaluate(X, y_true)                   → dict of metrics
        load(path)                            → load a pretrained .keras model
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.input_size = config['forecasting']['input_size']
        self.output_size = config['forecasting']['output_size']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.lr = config['training']['lr']
        self.patience = config['training']['patience']

        # TCN architecture hyperparameters
        self.filters = config['architecture']['filters']
        self.kernel_size = config['architecture']['kernel_size']
        self.dilations = config['architecture']['dilations']
        self.dropout = config['architecture']['dropout']
        self.activation = config['architecture']['activation']

        # Pretrained model path (optional)
        self.pretrained_path = config.get('model', {}).get('pretrained_path', None)

        self.model = None
        self._history = None

        # Load pretrained model if path is provided
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            self.load(self.pretrained_path)

    def _build_model(self):
        """Build TCN model from config hyperparameters."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tcn import TCN

        SEED = 42
        tf.keras.utils.set_random_seed(SEED)

        model = Sequential([
            TCN(
                nb_filters=self.filters,
                kernel_size=self.kernel_size,
                dilations=self.dilations,
                dropout_rate=self.dropout,
                activation=self.activation,
                return_sequences=False,
                input_shape=(self.input_size, 1),
            ),
            Dense(self.output_size),
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss='mape')
        return model

    def load(self, path: str):
        """Load a pretrained .keras model."""
        import tensorflow as tf
        from tcn import TCN  # noqa: F401 — needed so Keras can deserialize TCN layers

        self.model = tf.keras.models.load_model(path)
        print(f"Modelo TCN cargado desde: {path}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train (or fine-tune) the TCN model.

        Args:
            X_train: (N, input_size)  context windows
            y_train: (N, output_size) target forecasts
            X_val:   optional validation X
            y_val:   optional validation y

        Returns:
            history: Keras training history
            fig:     matplotlib figure of training curves
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        if self.model is None:
            self.model = self._build_model()
            print("Modelo TCN construido desde cero.")
        else:
            print("Fine-tuning del modelo TCN existente.")

        early_stop = EarlyStopping(
            monitor='val_loss', patience=self.patience, restore_best_weights=True
        )

        val_data = (X_val, y_val) if X_val is not None else None
        callbacks = [early_stop] if val_data is not None else []

        history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        self._history = history

        # Plot training curves
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_title('TCN: Training vs Validation Loss', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return history, fig

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate forecasts.

        Args:
            X: (N, input_size) context windows

        Returns:
            y_pred: (N, output_size) predictions
        """
        if self.model is None:
            raise RuntimeError("Modelo no inicializado. Usa fit() o load() primero.")

        y_pred = self.model.predict(X, batch_size=self.batch_size, verbose=0)

        # Handle 3D output from TCN → squeeze to 2D
        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)

        return y_pred

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate with the repo metrics: MAPE, DTW, Pearson.
        """
        y_pred = self.predict(X)
        return utils.evaluate_all_metrics(y_true, y_pred)

    def save(self, path: str):
        """Save the model to a .keras file."""
        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Modelo TCN guardado en: {path}")
