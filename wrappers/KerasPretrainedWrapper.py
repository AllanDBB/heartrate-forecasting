import json
import os
import sys
import zipfile
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


def _read_keras_metadata(path: str) -> Dict:
    with zipfile.ZipFile(path, 'r') as zf:
        return json.loads(zf.read('config.json').decode('utf-8'))


def _canonical_registered_name(metadata: Dict) -> Optional[str]:
    registered_name = metadata.get('registered_name')
    if isinstance(registered_name, str) and registered_name:
        return registered_name
    config = metadata.get('config', {})
    registered_name = config.get('registered_name')
    if isinstance(registered_name, str) and registered_name:
        return registered_name
    return None


def _ensure_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow/Keras no esta instalado. Instala `tensorflow` para usar los modelos .keras."
        ) from exc
    return tf


def _build_custom_objects():
    tf = _ensure_tensorflow()

    @tf.keras.utils.register_keras_serializable(package='Custom', name='TiDEResidualBlock')
    class TiDEResidualBlock(tf.keras.layers.Layer):
        def __init__(self, hidden_dim=256, dropout_rate=0.1, activation='relu', **kwargs):
            super().__init__(**kwargs)
            self.hidden_dim = hidden_dim
            self.dropout_rate = dropout_rate
            self.activation = activation
            self.dense1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name='dense1')
            self.dense2 = tf.keras.layers.Dense(hidden_dim, name='dense2')
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name='dropout1')
            self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name='dropout2')
            self.layer_norm = tf.keras.layers.LayerNormalization(name='layer_norm')

        def build(self, input_shape):
            input_shape = tf.TensorShape(input_shape)
            self.dense1.build(input_shape)
            hidden_shape = tf.TensorShape([input_shape[0], self.hidden_dim])
            self.dropout1.build(hidden_shape)
            self.dense2.build(hidden_shape)
            self.dropout2.build(hidden_shape)
            self.layer_norm.build(input_shape)
            super().build(input_shape)

        def call(self, inputs, training=None):
            x = self.dense1(inputs)
            x = self.dropout1(x, training=training)
            x = self.dense2(x)
            x = self.dropout2(x, training=training)
            return self.layer_norm(inputs + x)

        def get_config(self):
            config = super().get_config()
            config.update({
                'hidden_dim': self.hidden_dim,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation,
            })
            return config

    @tf.keras.utils.register_keras_serializable(package='Custom', name='TiDEModel')
    class TiDEModel(tf.keras.Model):
        def __init__(
            self,
            input_length=200,
            input_dim=1,
            horizon=200,
            hidden_dim=256,
            encoder_layers=2,
            decoder_layers=2,
            temporal_hidden_dim=128,
            dropout_rate=0.1,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.input_length = input_length
            self.input_dim = input_dim
            self.horizon = horizon
            self.hidden_dim = hidden_dim
            self.encoder_layers = encoder_layers
            self.decoder_layers = decoder_layers
            self.temporal_hidden_dim = temporal_hidden_dim
            self.dropout_rate = dropout_rate

            self.input_projection = tf.keras.layers.Dense(hidden_dim, activation='relu')
            self.encoder_blocks = [
                TiDEResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(encoder_layers)
            ]
            self.decoder_blocks = [
                TiDEResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(decoder_layers)
            ]
            self.temporal_projection = tf.keras.layers.Dense(horizon * temporal_hidden_dim)
            self.temporal_decoder = tf.keras.layers.Dense(temporal_hidden_dim, activation='relu')
            self.output_projection = tf.keras.layers.Dense(1)

        def build(self, input_shape):
            input_shape = tf.TensorShape(input_shape)
            feature_dim = 1
            if input_shape.rank == 3 and input_shape[-1] is not None:
                feature_dim = int(input_shape[-1])
            flat_dim = self.input_length * feature_dim
            self.input_projection.build(tf.TensorShape([input_shape[0], flat_dim]))

            current_shape = tf.TensorShape([input_shape[0], self.hidden_dim])
            for block in self.encoder_blocks:
                block.build(current_shape)
            for block in self.decoder_blocks:
                block.build(current_shape)

            self.temporal_projection.build(current_shape)
            temporal_shape = tf.TensorShape([input_shape[0], self.horizon, self.temporal_hidden_dim])
            self.temporal_decoder.build(temporal_shape)
            self.output_projection.build(temporal_shape)
            super().build(input_shape)

        def build_from_config(self, config):
            input_shape = config.get('input_shape')
            if input_shape is not None:
                self.build(input_shape)

        def call(self, inputs, training=None):
            x = tf.convert_to_tensor(inputs)
            if x.shape.rank == 3:
                x = tf.reshape(x, [tf.shape(x)[0], -1])
            elif x.shape.rank != 2:
                raise ValueError(
                    f"TiDEModel espera entradas rank 2 o 3. Recibido rank={x.shape.rank}."
                )

            x = self.input_projection(x)
            for block in self.encoder_blocks:
                x = block(x, training=training)
            for block in self.decoder_blocks:
                x = block(x, training=training)

            x = self.temporal_projection(x)
            x = tf.reshape(x, [tf.shape(x)[0], self.horizon, self.temporal_hidden_dim])
            x = self.temporal_decoder(x)
            x = self.output_projection(x)
            return tf.squeeze(x, axis=-1)

        def get_config(self):
            config = super().get_config()
            config.update({
                'input_length': self.input_length,
                'input_dim': self.input_dim,
                'horizon': self.horizon,
                'hidden_dim': self.hidden_dim,
                'encoder_layers': self.encoder_layers,
                'decoder_layers': self.decoder_layers,
                'temporal_hidden_dim': self.temporal_hidden_dim,
                'dropout_rate': self.dropout_rate,
            })
            return config

    @tf.keras.utils.register_keras_serializable(package='Custom', name='InvertibleNorm')
    class InvertibleNorm(tf.keras.layers.Layer):
        def __init__(self, num_features=1, eps=1e-5, affine=True, **kwargs):
            super().__init__(**kwargs)
            self.num_features = num_features
            self.eps = eps
            self.affine = affine
            self.affine_weight = None
            self.affine_bias = None

        def build(self, input_shape):
            if self.affine and self.affine_weight is None:
                self.affine_weight = self.add_weight(
                    name='affine_weight',
                    shape=(self.num_features,),
                    initializer='ones',
                    trainable=True,
                )
                self.affine_bias = self.add_weight(
                    name='affine_bias',
                    shape=(self.num_features,),
                    initializer='zeros',
                    trainable=True,
                )
            super().build(input_shape)

        def normalize(self, inputs):
            if self.affine and self.affine_weight is None:
                self.build(inputs.shape)
            mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
            std = tf.math.reduce_std(inputs, axis=1, keepdims=True) + self.eps
            outputs = (inputs - mean) / std
            if self.affine:
                outputs = outputs * self.affine_weight + self.affine_bias
            return outputs, mean, std

        def denormalize(self, outputs, mean, std):
            if self.affine:
                outputs = (outputs - self.affine_bias) / (self.affine_weight + self.eps)
            return outputs * std + mean

        def call(self, inputs):
            outputs, _, _ = self.normalize(inputs)
            return outputs

        def get_config(self):
            config = super().get_config()
            config.update({
                'num_features': self.num_features,
                'eps': self.eps,
                'affine': self.affine,
            })
            return config

    @tf.keras.utils.register_keras_serializable(package='Custom', name='iTransformerBlock')
    class ITransformerBlock(tf.keras.layers.Layer):
        def __init__(self, d_model=64, n_heads=8, d_ff=128, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_ff = d_ff
            self.dropout = dropout

            key_dim = max(1, d_model // max(1, n_heads))
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=key_dim,
                dropout=dropout,
            )
            self.dropout1 = tf.keras.layers.Dropout(dropout)
            self.norm1 = tf.keras.layers.LayerNormalization(name='norm1')
            self.ffn_dense1 = tf.keras.layers.Dense(d_ff, activation='gelu')
            self.ffn_dense2 = tf.keras.layers.Dense(d_model)
            self.dropout2 = tf.keras.layers.Dropout(dropout)
            self.norm2 = tf.keras.layers.LayerNormalization(name='norm2')

        def build(self, input_shape):
            input_shape = tf.TensorShape(input_shape)
            try:
                self.attention.build(input_shape, input_shape, input_shape)
            except TypeError:
                self.attention.build(input_shape, input_shape)
            self.dropout1.build(input_shape)
            self.norm1.build(input_shape)
            self.ffn_dense1.build(input_shape)
            ffn_shape = tf.TensorShape([input_shape[0], input_shape[1], self.d_ff])
            self.ffn_dense2.build(ffn_shape)
            self.dropout2.build(input_shape)
            self.norm2.build(input_shape)
            super().build(input_shape)

        def call(self, inputs, training=None):
            attn_output = self.attention(inputs, inputs, training=training)
            x = self.norm1(inputs + self.dropout1(attn_output, training=training))
            ffn = self.ffn_dense1(x)
            ffn = self.ffn_dense2(ffn)
            ffn = self.dropout2(ffn, training=training)
            return self.norm2(x + ffn)

        def get_config(self):
            config = super().get_config()
            config.update({
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'd_ff': self.d_ff,
                'dropout': self.dropout,
            })
            return config

    @tf.keras.utils.register_keras_serializable(package='Custom', name='iTransformer')
    class iTransformer(tf.keras.Model):
        def __init__(
            self,
            seq_len=200,
            pred_len=200,
            d_model=64,
            n_heads=8,
            d_ff=128,
            num_layers=4,
            dropout=0.1,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_ff = d_ff
            self.num_layers = num_layers
            self.dropout = dropout

            self.inorm = InvertibleNorm(num_features=1, affine=True)
            self.input_proj = tf.keras.layers.Dense(d_model)
            self.blocks = [
                ITransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
            self.head = tf.keras.layers.Dense(pred_len)

        def build(self, input_shape):
            input_shape = tf.TensorShape(input_shape)
            if input_shape.rank == 2:
                current_shape = tf.TensorShape([input_shape[0], input_shape[1], 1])
            elif input_shape.rank == 3:
                current_shape = input_shape
            else:
                raise ValueError(
                    f"iTransformer espera input_shape rank 2 o 3. Recibido: {input_shape}"
                )

            self.inorm.build(current_shape)
            self.input_proj.build(current_shape)
            current_shape = tf.TensorShape([current_shape[0], current_shape[1], self.d_model])
            for block in self.blocks:
                block.build(current_shape)
            self.head.build(tf.TensorShape([current_shape[0], self.d_model]))
            super().build(input_shape)

        def build_from_config(self, config):
            input_shape = config.get('input_shape')
            if input_shape is not None:
                self.build(input_shape)

        def call(self, inputs, training=None):
            x = tf.convert_to_tensor(inputs)
            if x.shape.rank == 2:
                x = tf.expand_dims(x, axis=-1)
            elif x.shape.rank != 3:
                raise ValueError(
                    f"iTransformer espera entradas rank 2 o 3. Recibido rank={x.shape.rank}."
                )

            x, mean, std = self.inorm.normalize(x)
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x, training=training)

            x = tf.reduce_mean(x, axis=1)
            x = self.head(x)
            mean = tf.squeeze(mean, axis=[1, 2])
            std = tf.squeeze(std, axis=[1, 2])
            return x * tf.expand_dims(std, axis=-1) + tf.expand_dims(mean, axis=-1)

        def get_config(self):
            config = super().get_config()
            config.update({
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'd_ff': self.d_ff,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            })
            return config

    return {
        'TiDEResidualBlock': TiDEResidualBlock,
        'TiDEModel': TiDEModel,
        'InvertibleNorm': InvertibleNorm,
        'iTransformerBlock': ITransformerBlock,
        'iTransformer': iTransformer,
    }


def _manual_load_custom_model(model_path, metadata, custom_objects):
    """
    Fallback loader for TiDE / iTransformer when ``load_model`` fails due to
    weight-path mismatches across Keras versions.

    Strategy:
      1. Read weight shapes from the embedded ``model.weights.h5`` to infer
         the true architecture hyper-parameters (e.g. ``d_model``).
      2. Build the model with those parameters.
      3. Assign weights by shape-group matching: within each unique shape,
         ``h5`` arrays (sorted by path) are zipped with model variables
         (sorted by ``var.path``).
    """
    import h5py
    import tempfile
    import zipfile
    from collections import defaultdict

    tf = _ensure_tensorflow()

    # ---- extract h5 from .keras zip ----
    with zipfile.ZipFile(model_path) as zf:
        h5_data = zf.read('model.weights.h5')
    h5_tmp = tempfile.mktemp(suffix='.h5')
    with open(h5_tmp, 'wb') as fh:
        fh.write(h5_data)

    try:
        # ---- collect non-optimizer weight arrays ----
        h5_vars = []
        with h5py.File(h5_tmp, 'r') as hf:
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset) and 'optimizer' not in name:
                    h5_vars.append((name, np.array(obj)))
            hf.visititems(_collect)
        h5_vars.sort()

        # ---- resolve constructor kwargs ----
        registered = _canonical_registered_name(metadata)
        cfg = {
            k: v
            for k, v in metadata.get('config', {}).items()
            if k not in ('name', 'trainable', 'dtype') and not isinstance(v, dict)
        }

        # iTransformer: config may record wrong d_model — infer from weights
        if registered == 'Custom>iTransformer':
            for path, arr in h5_vars:
                if path.startswith('input_proj/') and arr.ndim == 2:
                    cfg['d_model'] = int(arr.shape[-1])
                    break
            for path, arr in h5_vars:
                if '/ffn/' in path and 'dense/vars/0' in path and arr.ndim == 2:
                    cfg['d_ff'] = int(arr.shape[-1])
                    break
            block_names = {
                p.split('/')[1] for p, _ in h5_vars if p.startswith('blocks/')
            }
            if block_names:
                cfg['num_layers'] = len(block_names)

        # ---- build model ----
        cls_map = {
            'Custom>TiDEModel': ('TiDEModel', lambda c: [None, c.get('input_length', 200), 1]),
            'Custom>iTransformer': ('iTransformer', lambda c: [None, c.get('seq_len', 200), 1]),
        }
        if registered not in cls_map:
            raise ValueError(f"No manual loader for registered_name={registered}")

        cls_name, shape_fn = cls_map[registered]
        model = custom_objects[cls_name](**cfg)
        model.build(tf.TensorShape(shape_fn(cfg)))

        # ---- assign weights by shape-group matching ----
        h5_by_shape = defaultdict(list)
        for _, arr in h5_vars:
            h5_by_shape[arr.shape].append(arr)

        model_vars_sorted = sorted(model.variables, key=lambda v: v.path)
        model_by_shape = defaultdict(list)
        for v in model_vars_sorted:
            model_by_shape[tuple(v.shape)].append(v)

        assigned = 0
        for shape, h5_list in h5_by_shape.items():
            m_list = model_by_shape.get(shape, [])
            for arr, var in zip(h5_list, m_list):
                var.assign(arr)
                assigned += 1

        print(
            f"  Fallback weight loading: {assigned}/{len(h5_vars)} weights assigned"
        )

    finally:
        os.unlink(h5_tmp)

    return model


class KerasPretrainedWrapper:
    """
    Lightweight wrapper for local .keras forecasting artifacts.

    Supports:
      - standard Sequential / Functional Keras models
      - custom TiDEModel artifacts
      - custom iTransformer artifacts
    """

    def __init__(self, model_path: str, batch_size: int = 32, name: Optional[str] = None):
        self.model_path = model_path
        self.batch_size = batch_size
        self.name = name or os.path.splitext(os.path.basename(model_path))[0]
        self.metadata = _read_keras_metadata(model_path)
        self.model = None

    def _load_model(self):
        tf = _ensure_tensorflow()
        custom_objects = _build_custom_objects()

        # Register NBeatsBlock (from NBeatsSupervisedWrapper)
        try:
            from wrappers.NBeatsSupervisedWrapper import _get_nbeats_block_class
            custom_objects['NBeatsBlock'] = _get_nbeats_block_class()
        except Exception:
            pass

        # Register TCN layer (from keras-tcn)
        try:
            from tcn import TCN
            custom_objects['TCN'] = TCN
        except ImportError:
            pass

        registered_name = _canonical_registered_name(self.metadata)

        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=custom_objects,
                compile=False,
                safe_mode=False,
            )
        except Exception:
            # Fallback for cross-version weight-path mismatches (TiDE, iTransformer)
            if registered_name not in ('Custom>TiDEModel', 'Custom>iTransformer'):
                raise
            self.model = _manual_load_custom_model(
                self.model_path, self.metadata, custom_objects,
            )

        return self.model

    def load(self):
        if self.model is None:
            self._load_model()
            print(f"Modelo {self.name} cargado desde: {self.model_path}")
        return self

    def _infer_expected_rank(self) -> Optional[int]:
        if self.model is None:
            self.load()

        input_shape = getattr(self.model, 'input_shape', None)
        if input_shape is None:
            return None
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if input_shape is None:
            return None
        return len(input_shape)

    def _prepare_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        rank = self._infer_expected_rank()

        if rank == 3 and X.ndim == 2:
            return X[..., None]
        if rank == 2 and X.ndim == 3 and X.shape[-1] == 1:
            return X.squeeze(-1)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()

        X = self._prepare_inputs(X)
        y_pred = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        y_pred = np.asarray(y_pred)

        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]

        return y_pred

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return utils.evaluate_all_metrics(y_true, y_pred)
