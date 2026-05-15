"""
SubjectMemory: construye y mantiene la memoria inmunológica por sujeto.

Para cada sujeto i, M_i[k] representa qué tan bien el anticuerpo k
predice sus patrones de frecuencia cardíaca. Se construye evaluando
los pesos w_k de cada anticuerpo sobre las ventanas del sujeto en ens_fit.

Visión a futuro: en producción, M_i se acumula del historial real de
una persona, personalizando el sistema sin reentrenar los modelos base.
"""

import numpy as np
from wrappers.AiNetCore import AiNetCore


class SubjectMemory:
    """
    Memoria inmunológica por sujeto.

    Uso:
        sm = SubjectMemory()
        sm.fit(ainet, features_fit, preds_stack_fit, y_true_fit, ids_fit)
        memory_matrix = sm.get_memory_matrix(ids_held_out)  # (n_windows, K)
    """

    def __init__(self):
        self._memory: dict[str, np.ndarray] = {}   # subject_id -> (K,)
        self._K: int | None = None

    def fit(
        self,
        ainet: AiNetCore,
        features_fit: np.ndarray,
        preds_stack_fit: np.ndarray,
        y_true_fit: np.ndarray,
        ids_fit: np.ndarray,
    ) -> 'SubjectMemory':
        """
        Construye M_i para cada sujeto usando las ventanas de ens_fit.

        ainet:           AiNetCore entrenado — provee K anticuerpos y sus pesos w_k
        features_fit:    (n_windows, n_features) — salida de FeatureExtractor.transform()
        preds_stack_fit: (n_windows, n_models, horizon) — predicciones en escala original
        y_true_fit:      (n_windows, horizon) — ground truth en escala original
        ids_fit:         (n_windows,) — ID de sujeto por ventana
        """
        K = ainet.K
        self._K = K

        for subj_id in np.unique(ids_fit):
            mask = ids_fit == subj_id
            y_subj = y_true_fit[mask]           # (n_subj, horizon)
            p_subj = preds_stack_fit[mask]      # (n_subj, n_models, horizon)

            M_i = np.zeros(K)
            for k in range(K):
                w_k = ainet.weights_[k]         # (n_models,)
                y_pred = np.einsum('m,nmh->nh', w_k, p_subj)

                valid = np.abs(y_subj) > 1e-8
                if valid.sum() > 0:
                    mape_k = float(
                        np.mean(np.abs(
                            (y_subj[valid] - y_pred[valid]) / y_subj[valid]
                        )) * 100
                    )
                else:
                    mape_k = 100.0

                # Menor MAPE → mayor memoria
                M_i[k] = np.exp(-mape_k / 100.0)

            s = M_i.sum()
            self._memory[subj_id] = M_i / (s + 1e-12)

        return self

    def get_memory_matrix(self, ids: np.ndarray) -> np.ndarray:
        """
        Devuelve matriz de memoria para un conjunto de ventanas.

        ids:     (n_windows,) — ID de sujeto por ventana
        returns: (n_windows, K) — M_i por ventana
                 Sujetos no vistos en fit() reciben memoria uniforme (1/K).
        """
        assert self._K is not None, "Llama fit() antes de get_memory_matrix()."
        K = self._K
        result = np.zeros((len(ids), K))
        uniform = np.ones(K) / K

        for i, subj_id in enumerate(ids):
            result[i] = self._memory.get(subj_id, uniform)

        return result

    def subject_ids(self) -> list:
        """Lista de sujetos conocidos por el sistema."""
        return list(self._memory.keys())
