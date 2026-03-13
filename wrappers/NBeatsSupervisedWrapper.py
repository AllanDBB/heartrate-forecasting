import numpy as np
import yaml
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class NBeatsSupervisedWrapper:
    """
    Wrapper for a lightweight N-BEATS style Keras model.

    The pretrained `nbeats_0.keras` artifact uses a custom `NBeatsBlock`,
    so this wrapper recreates that layer to allow deserialization.
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

        self.blocks = config['architecture'].get('blocks', 2)
        self.units = config['architecture'].get('units', 256)
        self.expansion = config['architecture'].get('expansion', self.output_size)

        self.pretrained_path = config.get('model', {}).get('pretrained_path', None)

        self.model = None
        self._history = None

        if self.pretrained_path and os.path.exists(self.pretrained_path):
            self.load(self.pretrained_path)

    def _prepare_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            return X[..., None]
        if X.ndim == 3:
            return X
        raise ValueError(f"Se esperaba X con ndim 2 o 3. Recibido: {X.ndim}")

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import Add, Dense, Flatten, Layer

        @tf.keras.utils.register_keras_serializable(package='Custom', name='NBeatsBlock')
        class NBeatsBlock(Layer):
            def __init__(self, units=256, expansion=200, **kwargs):
                super().__init__(**kwargs)
                self.units = units
                self.expansion = expansion
                self.hidden_layers = [
                    Dense(units, activation='relu') for _ in range(4)
                ]
                self.theta = Dense(expansion)

            def call(self, inputs):
                x = inputs
                for layer in self.hidden_layers:
                    x = layer(x)
                return self.theta(x)

            def get_config(self):
                config = super().get_config()
                config.update({
                    'units': self.units,
                    'expansion': self.expansion,
                })
                return config

        inputs = Input(shape=(self.input_size, 1))
        x = Flatten()(inputs)
        block_outputs = [
            NBeatsBlock(units=self.units, expansion=self.output_size)(x)
            for _ in range(self.blocks)
        ]

        if len(block_outputs) == 1:
            outputs = block_outputs[0]
        else:
            outputs = Add()(block_outputs)

        model = Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss='mae')
        return model

    def load(self, path: str):
        """Load a pretrained .keras model."""
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Layer

        @tf.keras.utils.register_keras_serializable(package='Custom', name='NBeatsBlock')
        class NBeatsBlock(Layer):
            def __init__(self, units=256, expansion=200, **kwargs):
                super().__init__(**kwargs)
                self.units = units
                self.expansion = expansion
                self.hidden_layers = [
                    Dense(units, activation='relu') for _ in range(4)
                ]
                self.theta = Dense(expansion)

            def call(self, inputs):
                x = inputs
                for layer in self.hidden_layers:
                    x = layer(x)
                return self.theta(x)

            def get_config(self):
                config = super().get_config()
                config.update({
                    'units': self.units,
                    'expansion': self.expansion,
                })
                return config

        self.model = tf.keras.models.load_model(
            path,
            custom_objects={'NBeatsBlock': NBeatsBlock},
        )
        print(f"Modelo NBEATS cargado desde: {path}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train or fine-tune the NBEATS model.
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        X_train = self._prepare_inputs(X_train)
        if X_val is not None:
            X_val = self._prepare_inputs(X_val)

        if self.model is None:
            self.model = self._build_model()
            print("Modelo NBEATS construido desde cero.")
        else:
            print("Fine-tuning del modelo NBEATS existente.")

        early_stop = EarlyStopping(
            monitor='val_loss', patience=self.patience, restore_best_weights=True
        )

        val_data = (X_val, y_val) if X_val is not None else None
        callbacks = [early_stop] if val_data is not None else []

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        self._history = history

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_title('NBEATS: Training vs Validation Loss', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MAE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return history, fig

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo no inicializado. Usa fit() o load() primero.")

        X = self._prepare_inputs(X)
        y_pred = self.model.predict(X, batch_size=self.batch_size, verbose=0)

        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)

        return y_pred

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """Evaluate with the paper metrics: MAPE, DTW, Correlation."""
        y_pred = self.predict(X)
        return utils.evaluate_all_metrics(y_true, y_pred)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Modelo NBEATS guardado en: {path}")
