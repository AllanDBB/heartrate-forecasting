# Diseño: AIS Ensemble Adaptativo para Predicción de Frecuencia Cardíaca

**Fecha:** 2026-05-14  
**Contexto:** Investigación / paper académico  
**Baseline a superar:** Stacking Ridge MAPE=4.221, Pearson=0.837 (ventana 200, 6 modelos)

---

## 1. Visión General del Paper

El paper presenta tres escenarios en progresión narrativa:

| Escenario | Descripción | Notebooks actuales |
|-----------|-------------|-------------------|
| **1. Modelos individuales** | Cada modelo base evaluado solo sobre held_out | `ensemble_nueva_info.ipynb`, `ensemble_modelos_45.ipynb` |
| **2. Ensemble estático** | Weighted optimize, Stacking Ridge, búsqueda exhaustiva de combinaciones | `ensemble_nueva_info.ipynb`, `ensemble_permutaciones.ipynb` |
| **3. AIS Ensemble adaptativo** | aiNet + memoria por sujeto + Negative Selection | `ensemble_ais.ipynb` (nuevo) |

La progresión justifica cada nivel: los modelos individuales muestran el límite individual, el ensemble estático mejora combinando, y el AIS adaptativo va más allá al personalizar los pesos por patrón y por sujeto.

---

## 2. Partición de Datos y Protocolo sin Leakage

```
Dataset completo (132 series, 1841 timesteps)
        │
        ├── df_70 (70%, seed=42) ──► Entrenamiento de modelos base (ya hecho, no se toca)
        │
        └── df_30 (30%, seed=42)
                │
                ├── ens_fit  (50% del 30% = ~15% total) ──► Entrenar AIS
                │    ├── Extraer features → normalizar → guardar params
                │    ├── Entrenar aiNet → K anticuerpos
                │    ├── Construir memoria M_i por sujeto (sliding window denso)
                │    └── Entrenar detectores Negative Selection
                │
                └── held_out (50% del 30% = ~15% total) ──► Evaluación final NUNCA vista
                     ├── Ventanas: 18,257 × 200
                     └── Sujetos: 39 (mismos que ens_fit, identificados por ids_held_out)
```

**Importante:** `ens_fit` ya se usó para optimizar los pesos del ensemble estático (escenario 2).
Reutilizarlo para entrenar el AIS es correcto — ambos son meta-learners sobre las predicciones
de los modelos base. El `held_out` permanece intacto en todos los escenarios.

**Semilla del split de ensemble:** `ENSEMBLE_SEED = 123`

---

## 3. Arquitectura del Sistema AIS

```
ENTRENAMIENTO (sobre ens_fit)
─────────────────────────────
X_ens_fit ──► FeatureExtractor ──► features_ens_fit (18256 × 20)
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                         aiNetTrainer         NegSelTrainer
                              │                     │
                        K Anticuerpos          Detectores Self
                    (w_k, centroid_k)         (r_d, radio_d)
                              │
                    SubjectMemoryBuilder
                    (M_i por sujeto i=1..39, usando sliding window)
                              │
                    ──► AISEnsembleWrapper (serializado)

INFERENCIA (sobre held_out)
────────────────────────────
X_held_out ──► FeatureExtractor ──► f_nueva
                                      │
                              NegSelModule
                             ┌────────┴────────┐
                          nonself            self
                             │                 │
                     fallback: TCN      aiNet activation
                                        × subject memory M_i
                                              │
                                    pesos adaptativos w* (por ventana)
                                              │
                            Σ w*_m · pred_m(X) ──► ŷ
```

---

## 4. Bloque 1: FeatureExtractor

**Propósito:** Convertir cada ventana `X` de shape `(200,)` en un vector compacto de ~20 features
que representa el "antígeno" — el patrón de ritmo cardíaco en ese segmento temporal.

**Input:** `X_ens_fit` o `X_held_out` — ventanas **ya estandarizadas** por `utils.estandarizar`
(z-score por serie). No se re-estandariza la ventana.

**Output:** matriz de features `(n_ventanas, 20)`, con su propia normalización interna.

**Separación importante:**
- `utils.estandarizar` → normaliza las series temporales para los modelos base (una vez, sobre df_70/df_30)
- FeatureExtractor → extrae estadísticos de cada ventana y normaliza **esas features** (no las ventanas)
- Son dos normalizaciones sobre cosas distintas: series vs estadísticos de series

**Features extraídas:**

| Grupo | Features |
|-------|---------|
| Estadísticas básicas | media, std, min, max, rango |
| Forma | skewness, kurtosis, coeficiente de variación |
| Tendencia | pendiente lineal (regresión OLS), segunda derivada media |
| Autocorrelación | ACF en lags 1, 5, 10, 20 |
| Espectral | frecuencia dominante (FFT), entropía espectral |
| Actividad | cruces por la media, proporción de picos |

**Normalización de features:**
- Parámetros (media, std por feature) calculados **solo sobre ens_fit**
- Aplicados sin modificación sobre held_out → sin leakage

**Costo computacional:** determinista, corre en CPU, ~0.5ms por ventana.

---

## 5. Bloque 2: aiNet Core

**Propósito:** Aprender una población estable de K anticuerpos, cada uno especializado en un
tipo de patrón de ritmo cardíaco, con un vector de pesos óptimo para ese patrón.

### Estructura de un anticuerpo

```
Ab_k = {
    centroid_k:  vector (20,) en espacio de features
                 → describe el tipo de patrón que "reconoce"
    w_k:         vector (6,) de pesos sobre los modelos base
                 → restricción simplex: Σw = 1, w_m ≥ 0
}
```

### Función de afinidad

```
affinity(f, Ab_k) = exp( -||f - centroid_k||² / σ² )
```

Donde `f` es el vector de features de una ventana y `σ` es el ancho del kernel (hiperparámetro).
Valor en [0, 1]: 1 = patrón idéntico al centroide del anticuerpo, 0 = patrón muy distinto.

### Algoritmo de entrenamiento (sobre ens_fit)

```
1. Inicializar N anticuerpos aleatorios (centroides en el rango de features_ens_fit)

2. Para cada iteración:
   a. SELECCIÓN: calcular afinidad de cada Ab_k con cada ventana de ens_fit
   b. CLONACIÓN: los top-n anticuerpos por afinidad media se clonan β veces
   c. MUTACIÓN HIPERSOMÁTICA: cada clon muta su w_k hacia menor MAPE
         - Mecanismo: perturbación gaussiana w_k' = w_k + N(0, α/afinidad)
           seguida de proyección al simplex (clip a [0,1] + renormalizar)
         - Las ventanas más cercanas al centroide del clon (top-p% por afinidad)
           se usan para evaluar el MAPE del clon mutado
         - Tasa de mutación α inversamente proporcional a la afinidad
           (anticuerpos muy afines mutan poco — somatic hypermutation fiel a la biología)
   d. SELECCIÓN DE CLON: el mejor clon reemplaza al padre
   e. SUPRESIÓN DE RED: eliminar anticuerpos con distancia entre centroides < σ_s
         - Se preserva solo el de mayor afinidad media en el vecindario
   f. RENOVACIÓN: insertar d anticuerpos aleatorios nuevos (mantener diversidad)

3. Criterio de parada: convergencia de la afinidad media de la red o máx iteraciones

4. Resultado: K anticuerpos estables con sus centroides y pesos especializados
```

### Hiperparámetros principales

| Parámetro | Descripción | Rango sugerido |
|-----------|-------------|----------------|
| K | Tamaño final de la red (anticuerpos) | 10–30 |
| N | Población inicial | 3×K |
| σ | Ancho del kernel de afinidad | tunable |
| β | Factor de clonación | 5–10 |
| σ_s | Umbral de supresión | < σ |
| d | Anticuerpos nuevos por iteración | 5–10% de N |

---

## 6. Bloque 3: Memoria por Sujeto (SubjectMemory)

**Propósito:** Capturar que cada individuo tiene un "perfil inmunológico" propio —
ciertos anticuerpos responden mejor a sus patrones específicos de frecuencia cardíaca.

**Visión a futuro (para paper):** En producción, `M_i` se construiría con el historial de
una persona real. Las primeras semanas de datos "inmunizarían" al sistema con su perfil
cardiovascular, personalizando las predicciones futuras sin reentrenar ningún modelo base.

### Estructura

Cada sujeto `i` tiene un vector `M_i ∈ ℝ^K` (uno por anticuerpo):

```
M_i[k] = qué tan bien el anticuerpo k predice
          las ventanas históricas del sujeto i
```

### Construcción (sobre ens_fit con sliding window denso)

**¿Por qué sliding window denso aquí?**
Con paso=OUTPUT_SIZE (200) tendríamos ~468 ventanas por sujeto, suficiente para el aiNet global
pero limitado para estimar la memoria individual. El sliding con paso=1 genera ventanas
correlacionadas pero da diversidad suficiente para estimar el rendimiento de cada anticuerpo
por sujeto. Los parámetros de normalización del FeatureExtractor NO se recalculan.

```python
# Los ids ya existen en el pipeline
for sujeto in set(ids_ens_fit):
    # 1. Generar ventanas con sliding denso (paso=1) sobre la serie del sujeto en ens_fit
    #    NOTA: esto requiere re-inferencia de los modelos base sobre estas nuevas ventanas.
    #    No podemos reusar preds_ens_fit_orig (que usan paso=OUTPUT_SIZE).
    #    Costo: inferencia adicional por sujeto, manejable en Colab con batch_size grande.
    X_sujeto_aug, y_sujeto_aug = sliding_window(serie_sujeto_ens_fit, step=1)
    preds_aug = {m: modelo_m.predict(X_sujeto_aug) for m in modelos}

    for k in range(K):
        # Aplicar pesos del anticuerpo k sobre ventanas aumentadas del sujeto
        pred_k = sum(w_k[m] * preds_aug[m] for m in modelos)
        mape_k = calcular_mape(y_sujeto_aug_orig, pred_k)
        M_i[k] = exp(-mape_k)  # alta memoria donde el anticuerpo funciona bien

    M_i = M_i / M_i.sum()  # normalizar → suma 1
```

### Uso en inferencia

```
affinity_k   = exp(-||f - centroid_k||² / σ²)   # similitud geométrica
activation_k = affinity_k × M_i[k]               # modulada por memoria del sujeto
w*           = Σ_k (activation_k / Σ activation) × w_k   # pesos adaptativos finales
ŷ            = Σ_m w*_m × pred_m                          # predicción final
```

**Propiedad clave:** Si un sujeto tiene ritmo muy estable, `M_i` concentra peso en anticuerpos
especializados en ritmo estable. Si tiene alta variabilidad, activa anticuerpos distintos.
El sistema se personaliza sin reentrenar ningún modelo base.

---

## 7. Bloque 4: Negative Selection

**Propósito:** Detectar ventanas que caen fuera de la distribución normal del dataset
("nonself") y manejarlas gracefully en vez de forzar una predicción ensemble sobre
datos fuera de distribución.

### Entrenamiento (sobre ens_fit)

```
1. Extraer features de todas las ventanas de ens_fit → "self space"
2. Generar detectores aleatorios en el espacio de features
3. Eliminar detectores que caigan dentro del self space (radio de tolerancia r)
4. Los detectores supervivientes = detectores nonself
```

### Inferencia

```
f_nueva = features(X_held_out_i)

Si algún detector nonself tiene afinidad alta con f_nueva:
    → ventana "nonself" (artefacto, transición extrema, etc.)
    → fallback: usar predicción de TCN (mejor modelo individual, MAPE=5.52)

Si ningún detector activa:
    → ventana "self" (patrón reconocible)
    → pasar al aiNet + SubjectMemory
```

### Valor analítico para el paper

- ¿Qué porcentaje del held_out es nonself?
- ¿De qué sujetos provienen las ventanas anómalas?
- ¿Coinciden con eventos fisiológicos conocidos (ejercicio extremo, artefactos)?
- Muestra que el sistema es interpretable, no solo una caja negra

---

## 8. Nuevo Archivo: `wrappers/AISEnsembleWrapper.py`

Misma interfaz que `EnsembleWrapper` para compatibilidad con el pipeline existente.

```python
# Uso en notebook
ens = AISEnsembleWrapper(
    n_antibodies=20,
    sigma=0.5,
    suppression_threshold=0.1,
    clone_factor=5,
)

ens.add_model('TCN',   preds_ens_fit_orig['TCN'],   split='fit')
ens.add_model('NBEATS', preds_ens_fit_orig['NBEATS'], split='fit')
# ... resto de modelos

ens.fit(
    X_ens_fit=X_ens_fit,
    y_true_fit=y_ens_fit_orig,
    ids_fit=ids_ens_fit,
)

# Inferencia adaptativa
y_pred = ens.predict(
    X_eval=X_held_out,
    ids_eval=ids_held_out,
)
```

**Estructura interna:**

```
AISEnsembleWrapper
    ├── FeatureExtractor       (extracción + normalización de features)
    ├── AiNetCore              (población de anticuerpos)
    ├── SubjectMemory          (M_i por sujeto)
    └── NegativeSelector       (detectores nonself + fallback)
```

---

## 9. Protocolo de Evaluación

Todos los experimentos evalúan sobre el mismo `held_out` (18,257 ventanas, nunca visto).

| Experimento | Tipo | Propósito |
|-------------|------|-----------|
| Modelos individuales (6) | Baseline | Límite individual |
| Ensemble Promedio Simple | Baseline | Naive combination |
| Ensemble Pesos Optimizados | Baseline | Escenario 2 |
| Ensemble Stacking Ridge | **Baseline principal** (MAPE=4.221) | Mejor estático |
| AIS sin memoria (Enfoque A) | Ablation | Valor del aiNet solo |
| **AIS con memoria (Enfoque B)** | **Sistema propuesto** | Resultado central |
| Análisis de anticuerpos | Cualitativo | Interpretabilidad |
| Análisis nonself | Cualitativo | Robustez |

---

## 10. Notebook: `ensemble_ais.ipynb`

Estructura de secciones:

```
1. Setup y carga de datos (igual que notebooks anteriores)
2. Predicciones de modelos base (reusar preds cacheadas si existen)
3. FeatureExtractor: extracción y normalización
4. aiNet: entrenamiento de anticuerpos
5. SubjectMemory: construcción por sujeto (sliding window)
6. NegativeSelection: entrenamiento de detectores
7. Inferencia adaptativa sobre held_out
8. Tabla de resultados (todos los escenarios juntos)
9. Análisis de anticuerpos (qué patrones especializó cada uno)
10. Análisis nonself (qué ventanas son anómalas)
11. Comparación final: AIS vs baseline estático
```
