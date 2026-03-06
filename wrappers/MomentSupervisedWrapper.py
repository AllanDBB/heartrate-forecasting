import gc
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils


class MomentSupervisedWrapper:
    """
    Wrapper for AutonLab/MOMENT-1-large in forecasting mode.

    Fine-tuning strategy (v2):
    - Unfreeze the LAST N encoder layers (configurable) + the head
    - Use differential learning rates: low LR for encoder, higher for head
    - Cosine annealing schedule with warmup
    - Train on full forward passes (encoder + head) for better adaptation
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
        self.encoder_lr = config['training'].get('encoder_lr', 1e-5)
        self.unfreeze_last_n = config['training'].get('unfreeze_last_n', 2)
        self.warmup_ratio = config['training'].get('warmup_ratio', 0.1)

        from momentfm import MOMENTPipeline

        self.model = MOMENTPipeline.from_pretrained(
            config['model']['name'],
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': self.output_size,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': True,
                'freeze_embedder': True,
                'freeze_head': False,
            },
        )
        self.model.init()
        self.model = self.model.to(self.device)

        # Unfreeze last N encoder layers for fine-tuning
        self._unfreeze_encoder_layers(self.unfreeze_last_n)

    def _unfreeze_encoder_layers(self, n_layers: int):
        """Unfreeze the last n_layers of the T5 encoder for fine-tuning."""
        if n_layers <= 0:
            return

        # Get encoder block layers
        encoder = None
        for name, module in self.model.named_modules():
            if 'encoder' in name and hasattr(module, 'block'):
                encoder = module
                break

        if encoder is None:
            print("  Advertencia: no se encontraron bloques del encoder para descongelar.")
            return

        total_layers = len(encoder.block)
        layers_to_unfreeze = list(range(max(0, total_layers - n_layers), total_layers))

        count = 0
        for idx in layers_to_unfreeze:
            for param in encoder.block[idx].parameters():
                param.requires_grad = True
                count += 1

        # Also unfreeze layer norm
        if hasattr(encoder, 'final_layer_norm'):
            for param in encoder.final_layer_norm.parameters():
                param.requires_grad = True
                count += 1

        print(f"  Descongelados {count} params de las últimas {n_layers} capas del encoder")

    def _prepare_inputs(self, X: np.ndarray):
        """Left-pad X to context_length and build input_mask."""
        n = X.shape[0]
        padded = np.zeros((n, self.context_length), dtype=np.float32)
        padded[:, -self.input_size:] = X

        mask = np.zeros((n, self.context_length), dtype=np.float32)
        mask[:, -self.input_size:] = 1.0

        x_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        return x_tensor, mask_tensor

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Fine-tune MOMENT with differential learning rates.
        - Head (linear layer): higher LR
        - Unfrozen encoder layers: lower LR
        - Cosine annealing schedule
        """
        x_tensor, mask_tensor = self._prepare_inputs(X_train)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(x_tensor, mask_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Differential learning rates
        head_params = list(self.model.head.parameters())
        encoder_params = [p for n, p in self.model.named_parameters()
                          if p.requires_grad and 'head' not in n]

        param_groups = [
            {'params': head_params, 'lr': self.lr},
            {'params': encoder_params, 'lr': self.encoder_lr},
        ]

        optimizer = AdamW(param_groups, weight_decay=1e-4)

        total_steps = self.epochs * len(loader)
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps - warmup_steps))

        criterion = nn.MSELoss().to(self.device)
        scaler = torch.cuda.amp.GradScaler()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        best_loss = float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for x_b, m_b, y_b in pbar:
                x_b = x_b.to(self.device)
                m_b = m_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = self.model(x_enc=x_b, input_mask=m_b)
                    loss = criterion(output.forecast, y_b)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                del x_b, m_b, y_b, output
                gc.collect()
                torch.cuda.empty_cache()

            avg_loss = total_loss / len(loader)
            improve = " ★" if avg_loss < best_loss else ""
            best_loss = min(best_loss, avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}{improve}")

        self.model.eval()
        print("Fine-tuning completado.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_tensor, mask_tensor = self._prepare_inputs(X)
        dataset = TensorDataset(x_tensor, mask_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for (x_b, m_b) in tqdm(loader, desc="Prediciendo"):
                x_b = x_b.to(self.device)
                m_b = m_b.to(self.device)
                output = self.model(x_enc=x_b, input_mask=m_b)
                pred = output.forecast.squeeze(1).cpu().numpy()
                preds.append(pred)
                del x_b, m_b, output
                gc.collect()
                torch.cuda.empty_cache()

        return np.concatenate(preds, axis=0)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return utils.evaluate_all_metrics(y_true, y_pred)
