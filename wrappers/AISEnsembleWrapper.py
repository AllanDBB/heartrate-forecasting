"""
AISEnsembleWrapper: orquestador del ensemble adaptativo basado en AIS.

Sigue la misma interfaz de EnsembleWrapper (add_model / predict) para
compatibilidad con los notebooks existentes. Internamente coordina:
  1. FeatureExtractor   — features de las ventanas de entrada
  2. AiNetCore          — K anticuerpos con pesos especializados
  3. SubjectMemory      — memoria inmunológica M_i por sujeto
  4. NegativeSelector   — detección de ventanas fuera de distribución

Pipeline de inferencia por ventana:
  X → features → NegSel? → [nonself] fallback(mejor_modelo_individual)
                          → [self]   aff(f, Ab_k) × M_i[k] → w* → Σ w*·pred
"""

import numpy as np

from wrappers.FeatureExtractor import FeatureExtractor
from wrappers.AiNetCore import AiNetCore
from wrappers.SubjectMemory import SubjectMemory
from wrappers.NegativeSelector import NegativeSelector


class AISEnsembleWrapper:
    """
    Ensemble adaptativo AIS para predicción de series temporales.

    Parámetros
    ----------
    n_antibodies          : int   — K anticuerpos en la red aiNet
    sigma                 : float — ancho del kernel gaussiano de afinidad
    clone_factor          : int   — número de clones por anticuerpo seleccionado (β)
    suppression_threshold : float — umbral de supresión de red (σ_s)
    n_new                 : int   — anticuerpos aleatorios por iteración
    max_iter              : int   — iteraciones del algoritmo aiNet
    mutation_rate         : float — tasa base de mutación hipersomática (α)
    neg_sel_percentile    : float — percentil para umbral de NegativeSelector
    random_state          : int   — semilla global
    """

    def __init__(
        self,
        n_antibodies: int = 20,
        sigma: float = 1.0,
        clone_factor: int = 5,
        suppression_threshold: float = 0.3,
        n_new: int = 5,
        max_iter: int = 50,
        mutation_rate: float = 0.1,
        neg_sel_percentile: float = 99.0,
        random_state: int = 42,
    ):
        self.n_antibodies = n_antibodies
        self.sigma = sigma
        self.clone_factor = clone_factor
        self.suppression_threshold = suppression_threshold
        self.n_new = n_new
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.neg_sel_percentile = neg_sel_percentile
        self.random_state = random_state

        self._model_names: list[str] = []
        self._preds_fit: dict[str, np.ndarray] = {}
        self._preds_eval: dict[str, np.ndarray] = {}

        self.feature_extractor_: FeatureExtractor | None = None
        self.ainet_: AiNetCore | None = None
        self.subject_memory_: SubjectMemory | None = None
        self.neg_selector_: NegativeSelector | None = None
        self._fallback_model: str | None = None

    # ------------------------------------------------------------------
    # Interfaz compatible con EnsembleWrapper
    # ------------------------------------------------------------------

    def add_model(self, name: str, predictions: np.ndarray, split: str) -> None:
        """
        Registra predicciones de un modelo base.

        name:        identificador del modelo (ej. 'TCN')
        predictions: (n_windows, horizon) en escala original (desestandarizado)
        split:       'fit'  → usado para entrenar el AIS (ens_fit)
                     'eval' → usado para predecir en inferencia (held_out)
        """
        if name not in self._model_names:
            self._model_names.append(name)
        if split == 'fit':
            self._preds_fit[name] = predictions
        elif split == 'eval':
            self._preds_eval[name] = predictions
        else:
            raise ValueError(f"split debe ser 'fit' o 'eval', recibido: {split!r}")

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        X_fit: np.ndarray,
        y_true_fit: np.ndarray,
        ids_fit: np.ndarray,
        fallback_model: str | None = None,
    ) -> 'AISEnsembleWrapper':
        """
        Entrena todos los módulos AIS sobre ens_fit.

        X_fit:          (n_windows, input_size) — ventanas estandarizadas (X_ens_fit)
        y_true_fit:     (n_windows, horizon) — ground truth en escala original
        ids_fit:        (n_windows,) — ID de sujeto por ventana (ids_ens_fit)
        fallback_model: nombre del modelo para ventanas nonself (default: primer modelo)
        """
        preds_stack = np.stack(
            [self._preds_fit[m] for m in self._model_names], axis=1
        )  # (n_windows, n_models, horizon)

        # 1. FeatureExtractor
        self.feature_extractor_ = FeatureExtractor()
        features = self.feature_extractor_.fit_transform(X_fit)

        # 2. aiNet
        self.ainet_ = AiNetCore(
            n_antibodies=self.n_antibodies,
            sigma=self.sigma,
            clone_factor=self.clone_factor,
            suppression_threshold=self.suppression_threshold,
            n_new=self.n_new,
            max_iter=self.max_iter,
            mutation_rate=self.mutation_rate,
            random_state=self.random_state,
        )
        self.ainet_.fit(features, preds_stack, y_true_fit)

        # 3. Negative Selection
        self.neg_selector_ = NegativeSelector(
            threshold_percentile=self.neg_sel_percentile,
            random_state=self.random_state,
        )
        self.neg_selector_.fit(features)

        # 4. Subject Memory
        self.subject_memory_ = SubjectMemory()
        self.subject_memory_.fit(
            self.ainet_, features, preds_stack, y_true_fit, ids_fit
        )

        # 5. Modelo de fallback
        self._fallback_model = fallback_model or self._model_names[0]
        if self._fallback_model not in self._model_names:
            raise ValueError(
                f"fallback_model '{self._fallback_model}' no está en los modelos registrados."
            )

        return self

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict(
        self,
        X_eval: np.ndarray,
        ids_eval: np.ndarray,
    ) -> np.ndarray:
        """
        Genera predicciones adaptativas sobre held_out.

        X_eval:   (n_windows, input_size) — ventanas estandarizadas (X_held_out)
        ids_eval: (n_windows,) — ID de sujeto por ventana (ids_held_out)
        returns:  (n_windows, horizon) — predicciones en escala original
        """
        assert self.feature_extractor_ is not None, \
            "Llama fit() antes de predict()."

        features = self.feature_extractor_.transform(X_eval)
        is_nonself = self.neg_selector_.predict(features)
        memory = self.subject_memory_.get_memory_matrix(ids_eval)
        adaptive_w = self.ainet_.get_adaptive_weights(features, memory)

        preds_stack = np.stack(
            [self._preds_eval[m] for m in self._model_names], axis=1
        )  # (n_windows, n_models, horizon)

        horizon = preds_stack.shape[2]
        y_pred = np.zeros((len(X_eval), horizon))

        # Ventanas self: combinación ponderada adaptativa
        self_mask = ~is_nonself
        if self_mask.sum() > 0:
            w_self = adaptive_w[self_mask]
            p_self = preds_stack[self_mask]
            y_pred[self_mask] = np.einsum('nm,nmh->nh', w_self, p_self)

        # Ventanas nonself: modelo de fallback
        if is_nonself.sum() > 0:
            fb_idx = self._model_names.index(self._fallback_model)
            y_pred[is_nonself] = preds_stack[is_nonself, fb_idx, :]

        return y_pred

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    def get_diagnostics(
        self,
        X_eval: np.ndarray,
        ids_eval: np.ndarray,
    ) -> dict:
        """
        Información diagnóstica sobre una pasada de inferencia.

        Útil para el análisis cualitativo del paper:
          - ¿Qué % de ventanas son nonself?
          - ¿Qué sujetos tienen más ventanas anómalas?
          - ¿Cuál es la distribución media de pesos adaptativos?

        returns: dict con claves:
          'nonself_ratio'         float  — fracción de ventanas nonself
          'nonself_by_subject'    dict   — {subj_id: fracción nonself}
          'mean_adaptive_weights' array  — (n_models,) peso medio por modelo
          'weight_std'            array  — (n_models,) desviación de pesos
          'model_names'           list   — nombres de modelos en orden
        """
        assert self.feature_extractor_ is not None, \
            "Llama fit() antes de get_diagnostics()."

        features = self.feature_extractor_.transform(X_eval)
        is_nonself = self.neg_selector_.predict(features)
        memory = self.subject_memory_.get_memory_matrix(ids_eval)
        adaptive_w = self.ainet_.get_adaptive_weights(features, memory)

        nonself_by_subject = {
            str(subj): float(is_nonself[ids_eval == subj].mean())
            for subj in np.unique(ids_eval)
        }

        return {
            'nonself_ratio': float(is_nonself.mean()),
            'nonself_by_subject': nonself_by_subject,
            'mean_adaptive_weights': adaptive_w.mean(axis=0),
            'weight_std': adaptive_w.std(axis=0),
            'model_names': self._model_names,
        }
