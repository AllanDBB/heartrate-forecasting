"""
AiNetCore: implementación del algoritmo aiNet para ensemble adaptativo.

Cada anticuerpo Ab_k = (centroid_k, w_k):
  - centroid_k: vector (n_features,) en espacio de features — qué patrón reconoce
  - w_k:        vector (n_models,) de pesos sobre modelos base — cómo combinarlos
                restricción simplex: sum(w_k) = 1, w_k[i] >= 0

Entrenamiento: clonal selection + hypersomtic mutation + network suppression.
Inferencia: afinidad gaussiana ponderada por memoria de sujeto → pesos adaptativos.
"""

import numpy as np
from scipy.spatial.distance import cdist


class AiNetCore:
    """
    Red inmune artificial para ensemble adaptativo de series temporales.

    Parámetros
    ----------
    n_antibodies : int
        Número de anticuerpos en la red final (K).
    sigma : float
        Ancho del kernel gaussiano de afinidad.
    clone_factor : int
        Número de clones por anticuerpo seleccionado (β).
    suppression_threshold : float
        Distancia mínima entre centroides antes de suprimir (σ_s).
    n_new : int
        Anticuerpos aleatorios nuevos insertados por iteración.
    max_iter : int
        Número máximo de iteraciones del algoritmo.
    mutation_rate : float
        Tasa base de mutación hipersomática (α).
    top_p_frac : float
        Fracción de ventanas cercanas usadas para evaluar clones.
    random_state : int
        Semilla aleatoria para reproducibilidad.
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
        top_p_frac: float = 0.05,
        random_state: int = 42,
    ):
        self.K = n_antibodies
        self.sigma = sigma
        self.beta = clone_factor
        self.sigma_s = suppression_threshold
        self.d = n_new
        self.max_iter = max_iter
        self.alpha = mutation_rate
        self.top_p_frac = top_p_frac
        self.rng = np.random.RandomState(random_state)

        self.centroids_: np.ndarray | None = None  # (K, n_features)
        self.weights_: np.ndarray | None = None    # (K, n_models)
        self.n_models_: int | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _affinity(self, features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calcula afinidad gaussiana entre features y centroides.

        features:  (n_windows, n_features)
        centroids: (n_ab, n_features)
        returns:   (n_windows, n_ab)  valores en (0, 1]
        """
        dist_sq = cdist(features, centroids, metric='sqeuclidean')
        return np.exp(-dist_sq / (self.sigma ** 2 + 1e-12))

    def _project_simplex(self, w: np.ndarray) -> np.ndarray:
        """Proyecta vector de pesos al simplex de probabilidad (sum=1, w>=0)."""
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s > 1e-12:
            return w / s
        return np.ones_like(w) / len(w)

    def _eval_weights(
        self, w: np.ndarray, preds_stack: np.ndarray, y_true: np.ndarray
    ) -> float:
        """
        Evalúa MAPE de un vector de pesos w sobre un subconjunto de ventanas.

        w:           (n_models,)
        preds_stack: (n_windows, n_models, horizon)
        y_true:      (n_windows, horizon)
        """
        y_pred = np.einsum('m,nmh->nh', w, preds_stack)
        mask = np.abs(y_true) > 1e-8
        if mask.sum() == 0:
            return 100.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        preds_stack: np.ndarray,
        y_true: np.ndarray,
    ) -> 'AiNetCore':
        """
        Entrena la red aiNet sobre datos de ens_fit.

        features:    (n_windows, n_features) — salida de FeatureExtractor.transform()
        preds_stack: (n_windows, n_models, horizon) — predicciones en escala original
        y_true:      (n_windows, horizon) — ground truth en escala original
        """
        n_windows, n_features = features.shape
        n_models = preds_stack.shape[1]
        self.n_models_ = n_models

        # Inicializar población aleatoria (3×K anticuerpos)
        N_init = self.K * 3
        f_min = features.min(axis=0)
        f_max = features.max(axis=0)
        f_range = f_max - f_min + 1e-8

        centroids = f_min + self.rng.rand(N_init, n_features) * f_range
        weights = np.array([
            self._project_simplex(self.rng.rand(n_models))
            for _ in range(N_init)
        ])

        top_p = max(10, int(self.top_p_frac * n_windows))

        for _ in range(self.max_iter):
            N = len(centroids)

            # a. Afinidad media de cada anticuerpo sobre todo ens_fit
            aff = self._affinity(features, centroids)  # (n_windows, N)
            mean_aff = aff.mean(axis=0)                # (N,)

            # b. Seleccionar top min(K, N) anticuerpos para clonar
            n_select = min(self.K, N)
            top_idx = np.argsort(mean_aff)[-n_select:]

            new_centroids = list(centroids)
            new_weights = list(weights)

            for ab_idx in top_idx:
                ab_aff = float(mean_aff[ab_idx])
                mut_rate = self.alpha / (ab_aff + 1e-8)

                best_clone_mape = float('inf')
                best_clone_c = centroids[ab_idx].copy()
                best_clone_w = weights[ab_idx].copy()

                for _ in range(self.beta):
                    # c. Mutación hipersomática:
                    #    w' = w + N(0, α/afinidad), proyección al simplex
                    #    c' = c + N(0, α/afinidad * 0.1)
                    w_mut = weights[ab_idx] + self.rng.randn(n_models) * mut_rate
                    w_mut = self._project_simplex(w_mut)
                    c_mut = centroids[ab_idx] + self.rng.randn(n_features) * mut_rate * 0.1

                    # Ventanas más cercanas al centroide mutado
                    dist_to_clone = np.sum((features - c_mut) ** 2, axis=1)
                    near_idx = np.argsort(dist_to_clone)[:top_p]

                    clone_mape = self._eval_weights(
                        w_mut, preds_stack[near_idx], y_true[near_idx]
                    )
                    if clone_mape < best_clone_mape:
                        best_clone_mape = clone_mape
                        best_clone_c = c_mut
                        best_clone_w = w_mut

                # d. El mejor clon actualiza al padre
                new_centroids[ab_idx] = best_clone_c
                new_weights[ab_idx] = best_clone_w

            centroids = np.array(new_centroids)
            weights = np.array(new_weights)

            # e. Supresión de red
            dist_matrix = cdist(centroids, centroids)
            keep = np.ones(N, dtype=bool)
            aff_updated = self._affinity(features, centroids).mean(axis=0)

            for i in range(N):
                if not keep[i]:
                    continue
                for j in range(i + 1, N):
                    if not keep[j]:
                        continue
                    if dist_matrix[i, j] < self.sigma_s:
                        if aff_updated[i] >= aff_updated[j]:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break

            centroids = centroids[keep]
            weights = weights[keep]

            # f. Renovación: insertar d anticuerpos aleatorios nuevos
            new_rand_c = f_min + self.rng.rand(self.d, n_features) * f_range
            new_rand_w = np.array([
                self._project_simplex(self.rng.rand(n_models))
                for _ in range(self.d)
            ])
            centroids = np.vstack([centroids, new_rand_c])
            weights = np.vstack([weights, new_rand_w])

        # Selección final: los K de mayor afinidad media
        aff_final = self._affinity(features, centroids).mean(axis=0)
        k_actual = min(self.K, len(centroids))
        top_k = np.argsort(aff_final)[-k_actual:]

        self.centroids_ = centroids[top_k]
        self.weights_ = weights[top_k]
        self.K = k_actual

        # Garantizar simplex en todos los anticuerpos (puede haber deriva numérica)
        for k in range(len(self.weights_)):
            self.weights_[k] = self._project_simplex(self.weights_[k])

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_adaptive_weights(
        self,
        features: np.ndarray,
        subject_memory: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calcula pesos adaptativos por ventana.

        features:       (n_windows, n_features) — salida de FeatureExtractor.transform()
        subject_memory: (n_windows, K) — M_i[k] por ventana, o None (sin memoria)
        returns:        (n_windows, n_models) — pesos en simplex por ventana
        """
        assert self.centroids_ is not None, "Llama fit() antes de get_adaptive_weights()."

        aff = self._affinity(features, self.centroids_)  # (n_windows, K)

        if subject_memory is not None:
            activation = aff * subject_memory
        else:
            activation = aff

        # Normalizar activaciones → suma 1 por ventana
        # Si todas las afinidades son ~0 (ventana muy lejana), usar activación uniforme
        act_sum = activation.sum(axis=1, keepdims=True)
        uniform = np.ones_like(activation) / activation.shape[1]
        activation_norm = np.where(act_sum > 1e-12, activation / act_sum, uniform)

        # Combinar pesos de anticuerpos ponderados por activación
        adaptive_w = activation_norm @ self.weights_     # (n_windows, n_models)

        # Garantizar restricción simplex (re-normalización robusta)
        adaptive_w = np.maximum(adaptive_w, 0.0)
        row_sums = adaptive_w.sum(axis=1, keepdims=True)
        adaptive_w = adaptive_w / np.where(row_sums > 1e-12, row_sums, 1.0)

        return adaptive_w
