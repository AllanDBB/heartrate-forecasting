import numpy as np
import pandas as pd
import os
import joblib
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# EVALUATION METRICS: DTW, Pearson, MAPE
# ============================================================================

SUPPORTED_EVAL_METRICS = ("MAPE", "DTW", "Pearson")
DEFAULT_DTW_WINDOW = 20


def normalize_metric_name(metric_name: str) -> str:
    """
    Normalize metric aliases so the repo exposes only MAPE, DTW and Pearson.
    """
    if not isinstance(metric_name, str):
        raise TypeError(f"El nombre de la metrica debe ser str. Recibido: {type(metric_name)!r}")

    normalized = metric_name.strip().upper()
    aliases = {
        "MAPE": "MAPE",
        "DWT": "DTW",
        "DTW": "DTW",
        "PEARSON": "Pearson",
        "PEARSON CORRELATION": "Pearson",
        "CORRELATION": "Pearson",
        "CORRELACION": "Pearson",
        "CORRELACIÓN": "Pearson",
        "CORRELACION DE PEARSON": "Pearson",
        "CORRELACIÓN DE PEARSON": "Pearson",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Metrica no soportada: {metric_name}. "
            f"Usa solo {', '.join(SUPPORTED_EVAL_METRICS)}."
        )
    return aliases[normalized]


def normalize_metrics_dict(metrics_dict):
    """
    Normalize metric keys to the canonical names used across the repo.
    """
    normalized = {}
    for metric_name, metric_value in metrics_dict.items():
        normalized[normalize_metric_name(metric_name)] = metric_value
    return normalized


def _validate_metric_inputs(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch en evaluate_all_metrics: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if y_true.ndim != 2:
        raise ValueError(
            f"evaluate_all_metrics espera arrays 2D (N, horizon). Recibido ndim={y_true.ndim}"
        )

    return y_true, y_pred

def _dtw_distance(s1, s2, window=None):
    """
    Compute Dynamic Time Warping distance between two 1D sequences.
    Uses Sakoe-Chiba band constraint for performance.

    Args:
        s1, s2: 1D numpy arrays
        window: int, Sakoe-Chiba band radius (None = no constraint)

    Returns:
        float: DTW distance (Euclidean-based)
    """
    n, m = len(s1), len(s2)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return np.sqrt(D[n, m])


def compute_dtw(y_true_row, y_pred_row, window=DEFAULT_DTW_WINDOW):
    """
    DTW distance between a single (true, pred) pair.
    """
    return _dtw_distance(y_true_row, y_pred_row, window=window)


def compute_dtw_for_all_windows(y_true, y_pred, window=DEFAULT_DTW_WINDOW):
    """
    DTW computed independently for every available evaluation window.
    """
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)
    n_windows = y_true.shape[0]
    return np.array(
        [compute_dtw(y_true[i], y_pred[i], window=window) for i in range(n_windows)],
        dtype=float,
    )


def compute_mape_for_all_windows(y_true, y_pred, eps=1e-8):
    """
    MAPE computed independently for every available evaluation window.
    """
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)
    y_safe = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / y_safe), axis=1) * 100


def compute_correlation(y_true_row, y_pred_row, eps=1e-8):
    """
    Pearson correlation for one forecast window.

    If one of the windows is constant, Pearson is undefined. For reporting:
      - return 1.0 if both windows are effectively identical
      - return 0.0 otherwise
    """
    y_true_row = np.asarray(y_true_row, dtype=float)
    y_pred_row = np.asarray(y_pred_row, dtype=float)

    true_centered = y_true_row - np.mean(y_true_row)
    pred_centered = y_pred_row - np.mean(y_pred_row)
    denom = np.sqrt(np.sum(true_centered ** 2) * np.sum(pred_centered ** 2))

    if denom <= eps:
        return 1.0 if np.allclose(y_true_row, y_pred_row, atol=eps, rtol=0.0) else 0.0

    corr = np.sum(true_centered * pred_centered) / denom
    return float(np.clip(corr, -1.0, 1.0))


def compute_correlations_for_all_windows(y_true, y_pred, eps=1e-8):
    """
    Pearson correlation computed independently for every forecast window.
    """
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)

    true_centered = y_true - np.mean(y_true, axis=1, keepdims=True)
    pred_centered = y_pred - np.mean(y_pred, axis=1, keepdims=True)
    numerator = np.sum(true_centered * pred_centered, axis=1)
    denominator = np.sqrt(
        np.sum(true_centered ** 2, axis=1) * np.sum(pred_centered ** 2, axis=1)
    )

    corr = np.zeros(y_true.shape[0], dtype=float)
    valid = denominator > eps
    corr[valid] = numerator[valid] / denominator[valid]

    equal_constant = (~valid) & np.all(
        np.isclose(y_true, y_pred, atol=eps, rtol=0.0), axis=1
    )
    corr[equal_constant] = 1.0
    return np.clip(corr, -1.0, 1.0)


def evaluate_all_metrics(y_true, y_pred, dtw_window=DEFAULT_DTW_WINDOW):
    """
    Compute the repo-wide metrics on (y_true, y_pred) arrays of shape (N, horizon).
    Every row is one evaluation window, so aggregation spans all available
    windows generated upstream.

    Metrics:
      - MAPE: mean window-level MAPE across all evaluation windows
      - DTW: mean DTW distance across all evaluation windows
      - Pearson: mean Pearson correlation across all evaluation windows


    Returns:
        dict with metric names -> rounded values
    """
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)

    mape_vals = compute_mape_for_all_windows(y_true, y_pred)
    dtw_vals = compute_dtw_for_all_windows(y_true, y_pred, window=dtw_window)
    corr_vals = compute_correlations_for_all_windows(y_true, y_pred)

    metrics = {
        "MAPE": round(float(np.mean(mape_vals)), 4),
        "DTW": round(float(np.mean(dtw_vals)), 4),
        "Pearson": round(float(np.mean(corr_vals)), 4),
    }

    print("=" * 50)
    print("  Evaluation Metrics")
    print("=" * 50)
    print(f"  Ventanas evaluadas   : {y_true.shape[0]}")
    for name, val in metrics.items():
        print(f"  {name:20s}: {val}")
    print("=" * 50)
    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_forecast_samples(y_true, y_pred, n_samples=6, title="Predicción vs Real"):
    """
    Plot sample forecast windows showing predicted vs actual.
    Designed to be understandable for non-technical audiences.
    """
    n_total = y_true.shape[0]
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    n_cols = min(3, n_samples)
    n_rows = math.ceil(n_samples / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for k, idx in enumerate(indices):
        ax = axes[k]
        t = np.arange(y_true.shape[1])
        ax.plot(t, y_true[idx], label='Real', color='steelblue', linewidth=2)
        ax.plot(t, y_pred[idx], label='Predicción', color='darkorange', linewidth=2, linestyle='--')
        ax.fill_between(t, y_true[idx], y_pred[idx], alpha=0.15, color='red')
        y_safe_i = np.maximum(np.abs(y_true[idx]), 1e-8)
        mape_i = np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_safe_i)) * 100
        corr_i = compute_correlation(y_true[idx], y_pred[idx])
        ax.set_title(f'Ventana #{idx}  |  MAPE={mape_i:.2f}%  |  Pearson={corr_i:.2f}', fontsize=10)
        ax.set_xlabel('Paso de tiempo')
        ax.set_ylabel('Valor')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(indices), len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig


def plot_error_over_horizon(y_true, y_pred, title="Error según horizonte de predicción"):
    """
    Shows how prediction error increases over the forecast horizon.
    Very intuitive for non-technical people: 'further in the future = harder to predict'.
    """
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)
    horizon = y_true.shape[1]
    y_safe = np.maximum(np.abs(y_true), 1e-8)
    mape_per_step = np.mean(np.abs((y_true - y_pred) / y_safe), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(horizon), mape_per_step, color='steelblue', linewidth=2, label='MAPE')
    ax.fill_between(range(horizon), mape_per_step, alpha=0.2, color='steelblue')
    ax.set_xlabel('Paso en el futuro', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_metrics_comparison(results_dict, title="Comparación de Modelos"):
    """
    Bar chart comparing multiple models across all metrics.

    Args:
        results_dict: dict of {"Model Name": metrics_dict, ...}
                      e.g. {"MOMENT": {"MAPE": 8.1, "DTW": 3.2, ...}, "Moirai": {...}}
    """
    if not results_dict:
        print("No hay resultados para comparar.")
        return

    results_dict = {
        model_name: normalize_metrics_dict(metrics_dict)
        for model_name, metrics_dict in results_dict.items()
    }
    model_names = list(results_dict.keys())
    metric_names = list(list(results_dict.values())[0].keys())

    # Separate correlation (higher is better) from the rest (lower is better)
    higher_better_names = {"Pearson"}
    lower_better = [m for m in metric_names if m not in higher_better_names]
    higher_better = [m for m in metric_names if m in higher_better_names]

    fig, axes = plt.subplots(1, 2 if higher_better else 1,
                             figsize=(7 * (2 if higher_better else 1), 6))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Plot 1: metrics where lower is better
    x = np.arange(len(model_names))
    width = 0.8 / len(lower_better)
    for i, metric in enumerate(lower_better):
        vals = [results_dict[m].get(metric, 0) for m in model_names]
        axes[0].bar(x + i * width, vals, width, label=metric, alpha=0.85)
    axes[0].set_xticks(x + width * len(lower_better) / 2)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0].set_title('Métricas de Error (↓ menor = mejor)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: metrics where higher is better
    if higher_better:
        for metric in higher_better:
            vals = [results_dict[m].get(metric, 0) for m in model_names]
            colors = ['#2ecc71' if v == max(vals) else '#95a5a6' for v in vals]
            axes[1].bar(model_names, vals, color=colors, alpha=0.85)
            for j, v in enumerate(vals):
                axes[1].text(j, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
        axes[1].set_title('Pearson (↑ mayor = mejor)', fontsize=12, fontweight='bold')
        axes[1].set_ylim(-1.0, 1.1)
        axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig


def plot_subject_performance(y_true, y_pred, ids_list, title="Rendimiento por Sujeto"):
    """
    Per-subject performance barplot showing MAPE and Pearson.
    Shows which subjects are easier/harder to predict.
    """
    ids_arr = np.array(ids_list)
    unique_ids = sorted(set(ids_list))

    mapes = []
    corrs = []
    for uid in unique_ids:
        mask = ids_arr == uid
        yt = y_true[mask]
        yp = y_pred[mask]
        mape_i = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100
        corr_vals = compute_correlations_for_all_windows(yt, yp)
        mapes.append(mape_i)
        corrs.append(np.mean(corr_vals))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(unique_ids) * 0.8), 10))

    # Short labels
    labels = [str(uid)[-15:] for uid in unique_ids]

    colors_mape = ['#e74c3c' if m > np.mean(mapes) else '#2ecc71' for m in mapes]
    ax1.bar(labels, mapes, color=colors_mape, alpha=0.8)
    ax1.axhline(np.mean(mapes), color='navy', linestyle='--', label=f'Promedio={np.mean(mapes):.1f}%')
    ax1.set_title('MAPE por Sujeto (↓ menor = mejor)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAPE (%)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    colors_corr = ['#2ecc71' if x > np.mean(corrs) else '#e74c3c' for x in corrs]
    ax2.bar(labels, corrs, color=colors_corr, alpha=0.8)
    ax2.axhline(np.mean(corrs), color='navy', linestyle='--', label=f'Promedio={np.mean(corrs):.2f}')
    ax2.set_title('Pearson por Sujeto (↑ mayor = mejor)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pearson')
    ax2.set_ylim(-1.0, 1.1)
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig


def plot_prediction_timeline(y_true, y_pred, ids_list, inp, subject_idx=0,
                             title="Línea de Tiempo: Real vs Predicción"):
    """
    Reconstructs the full timeline for a subject (non-overlapping windows)
    and plots actual vs predicted. Uses the same logic as graphModel from modeloCreadoProfe.
    """
    ids_arr = np.array(ids_list)
    unique_ids = sorted(set(ids_list))

    if subject_idx >= len(unique_ids):
        subject_idx = 0

    uid = unique_ids[subject_idx]
    mask = ids_arr == uid
    yt = y_true[mask]
    yp = y_pred[mask]

    # Extract non-overlapping windows
    n_windows = len(yt) // inp
    remainder = len(yt) - n_windows * inp

    real_parts = []
    pred_parts = []
    for i in range(n_windows):
        idx = i * inp
        real_parts.append(yt[idx])
        pred_parts.append(yp[idx])
    if remainder > 0:
        real_parts.append(yt[-1][-remainder:])
        pred_parts.append(yp[-1][-remainder:])

    real_full = np.concatenate(real_parts)
    pred_full = np.concatenate(pred_parts)

    fig, ax = plt.subplots(figsize=(16, 5))
    t = np.arange(len(real_full))
    ax.plot(t, real_full, label='Real', color='steelblue', linewidth=1.5)
    ax.plot(t, pred_full, label='Predicción', color='darkorange', linewidth=1.5, alpha=0.8)
    ax.fill_between(t, real_full, pred_full, alpha=0.1, color='red')
    ax.set_title(f'{title} — Sujeto: {uid}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tiempo (pasos)')
    ax.set_ylabel('Valor')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig

def plot_all_subjects_timeline(y_true, y_pred, ids_list, inp,
                               title="Línea de Tiempo: Real vs Predicción"):
    """
    Plot prediction timeline for EVERY subject in ids_list.
    """
    ids_arr = np.array(ids_list)
    unique_ids = sorted(set(ids_list))

    for subject_idx, uid in enumerate(unique_ids):
        mask = ids_arr == uid
        yt = y_true[mask]
        yp = y_pred[mask]

        n_windows = len(yt) // inp
        remainder = len(yt) - n_windows * inp

        real_parts, pred_parts = [], []
        for i in range(n_windows):
            idx = i * inp
            real_parts.append(yt[idx])
            pred_parts.append(yp[idx])
        if remainder > 0:
            real_parts.append(yt[-1][-remainder:])
            pred_parts.append(yp[-1][-remainder:])

        real_full = np.concatenate(real_parts)
        pred_full = np.concatenate(pred_parts)

        fig, ax = plt.subplots(figsize=(16, 4))
        t = np.arange(len(real_full))
        ax.plot(t, real_full, label='Real', color='steelblue', linewidth=1.5)
        ax.plot(t, pred_full, label='Predicción', color='darkorange', linewidth=1.5, alpha=0.8)
        ax.fill_between(t, real_full, pred_full, alpha=0.1, color='red')
        ax.set_title(f'{title} — Sujeto: {uid}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tiempo (pasos)')
        ax.set_ylabel('Valor')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def loadAllFiles(folder_path):
    # --- Listar archivos Excel ---
    files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    dfs = []
    for i, file in enumerate(files, start=1):

        full_path = os.path.join(folder_path, file)
        dfn = pd.read_excel(full_path)

        # Aplica la función
        df = preProcDataset(dfn)

        # Renombrar columnas para evitar duplicados
        # Ejemplo: "Columna" -> "Columna_1", "Columna" -> "Columna_2", ...
        df.columns = [f"{col}_{i}" for col in df.columns]

        dfs.append(df)

    # --- Concatenar por columnas ---
    final_df = pd.concat(dfs, axis=1)

    return final_df


def clean_outliers_iqr(df, fill_edges=True, verbose=True):
    """
    Detecta outliers (por IQR) en columnas numéricas, los reemplaza por NaN,
    aplica interpolación lineal y opcionalmente rellena bordes.

    Retorna:
      - df_clean: DataFrame limpio
      - report: diccionario {columna: cantidad_outliers}
    """

    df_clean = df.copy()
    report = {}

    for col in df_clean.select_dtypes(include=[np.number]).columns:

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 2.5 * IQR
        upper = Q3 + 2.5 * IQR

        # Detectar outliers
        mask_outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
        n_outliers = mask_outliers.sum()
        report[col] = n_outliers

        # Reemplazar con NaN
        df_clean.loc[mask_outliers, col] = np.nan

        # Interpolación lineal
        df_clean[col] = df_clean[col].interpolate(method='linear')

        # Opcional: llenar bordes
        if fill_edges:
            df_clean[col] = df_clean[col].bfill().ffill()

    if verbose:
        print("Outliers detectados por columna:")
        for col, n in report.items():
            print(f"{col}: {n}")

    return df_clean, report


def clean_spikes_window_fast(df, window=3, pct_thresh=0.2, conclude=-3, fill_edges=True, verbose=True):
    """
    Detecta y limpia valores con cambios súbitos respecto a los últimos 'window' valores anteriores.
    Vectorizado y rápido, evitando listas.
    """

    from numpy.lib.stride_tricks import sliding_window_view

    df_clean = df.iloc[:,:conclude].copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    mask_df = pd.DataFrame(False, index=df_clean.index, columns=numeric_cols)
    report = {col: 0 for col in numeric_cols}

    for col in numeric_cols:
        x = df_clean[col].astype(float).values

        if len(x) <= window:
            # si la serie es muy corta, no se puede comparar
            df_clean[col] = pd.Series(x)
            continue

        # Crear ventana deslizante sobre los valores previos (shift 1)
        x_shifted = np.roll(x, 1)  # desplaza 1 hacia adelante
        x_shifted[0] = np.nan       # primer valor no tiene previos

        # windows sobre los valores previos
        # la primera ventana será menor, completamos con NaN
        n_missing = len(x) - (len(x) - window + 1) # This needs adjustment if original len(x) < window
        if len(x) > window:
            prev_windows = sliding_window_view(x_shifted, window_shape=window)
            n_missing = len(x) - prev_windows.shape[0]
            if n_missing > 0:
                prev_windows = np.vstack([np.full((n_missing, window), np.nan), prev_windows])
        else:
            prev_windows = np.full((len(x), window), np.nan)

        # Diferencias y porcentaje
        diff_matrix = x.reshape(-1, 1) - prev_windows
        pct_change_matrix = np.abs(diff_matrix) / np.abs(prev_windows)

        pct_change_matrix[np.isnan(pct_change_matrix)] = 0
        pct_change_matrix[np.isinf(pct_change_matrix)] = 0

        # detectar si alguna ventana supera pct_thresh
        mask = (pct_change_matrix > pct_thresh).any(axis=1)

        x_clean = x.copy()
        x_clean[mask] = np.nan

        series = pd.Series(x_clean).interpolate()
        if fill_edges:
            series = series.bfill().ffill()

        df_clean[col] = series.values
        mask_df[col] = mask
        report[col] = int(mask.sum())

        if verbose:
            print(f"{col}: {report[col]} valores corregidos (vectorizado)")
    df_clean = pd.concat([df_clean,  df.iloc[:,conclude:]], axis=1)
    return df_clean, report, mask_df


def series_to_supervised_matrix(df, input_size=5, output_size=1):
    """
    Convierte cada columna de un DataFrame de series de tiempo en un problema supervisado.
    Cada columna se transforma en X (input) y y (output), y se concatenan todas las columnas.

    Parámetros:
        df : pandas DataFrame
            Cada columna es una serie de tiempo.
        input_size : int
            Número de pasos de entrada (lags) para cada muestra.
        output_size : int
            Número de pasos de salida (horizonte de pronóstico).

    Retorna:
        X_concat : np.ndarray
            Matriz de inputs (num_samples, input_size)
        y_concat : np.ndarray
            Matriz de outputs (num_samples, output_size)
    """
    X_list = []
    y_list = []
    ids_list=[]
    for col in df.columns:
        series = df[col].values
        series = series[~np.isnan(series)]
        n_samples = len(series) - input_size - output_size + 1
        if n_samples <= 0:
            continue  # columna muy corta

        X_col = np.zeros((n_samples, input_size))
        y_col = np.zeros((n_samples, output_size))

        for i in range(n_samples):
            X_col[i, :] = series[i:i+input_size]
            y_col[i, :] = series[i+input_size:i+input_size+output_size]

        # create id
        ids_list+=[col] * n_samples

        X_list.append(X_col)
        y_list.append(y_col)

    # Concatenar todas las columnas verticalmente (por filas)
    if X_list:
        X_concat = np.vstack(X_list)
        y_concat = np.vstack(y_list)
    else:
        X_concat = np.array([])
        y_concat = np.array([])

    return X_concat, y_concat, ids_list


def preProcDataset(df_data):
    '''
    # Extract variables from SELECT
    df['fase']=pd.factorize(df['SELECT '])[0]
    df['faseTrabajo']=np.where(df['SELECT '].str.contains('Trabajo'), 1,
                       np.where(df['SELECT '].str.contains('Recuperación'), 0, 2))
    '''
    df=df_data.copy()
    df.drop(['SELECT '], axis=1, inplace=True)

    # 2. Convertir la columna 'TIME' a datetime usando el formato
    df['TIME_dt'] = pd.to_datetime(df['TIME '], format='%H:%M:%S %f')

    # 3. Crear una columna que contenga solo los segundos (sin milisegundos)
    df['segundo_agrupado'] = df['TIME_dt'].dt.floor('s')

    # 4. Generar el consecutivo para cada grupo de segundos
    df['consecutivo']=df.groupby('segundo_agrupado').ngroup()

    '''
    # 5 select columns of interest
    dfSelected=df[['WIMU_01_Marta_Baños   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_03_Lorena_Caparros   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_04_Lucia_Garcia   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_05_Salvador_Cavas   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_06_Carlota_Micallef   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_07_Juan_Miguel_Morales   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_09_Sara_Morlanes   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_15_Lucia_Vidal   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_16_Julia_Baños   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_17_Carla_Martinez   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_19_Alicia_Barreiro   HRM HeartRate(bpm)(SYNC) ',
       'WIMU_10_Pepe   HRM HeartRate(bpm)(SYNC) ', 'fase', 'faseTrabajo', 'consecutivo']]
    '''
    # 6 inputar valores con cambios repentinos
    dfSelected2, report, mask_df=clean_spikes_window_fast(df, window=30, pct_thresh=0.10, fill_edges=True, verbose=True)

    # 7 get average by second
    #dfSelectedMean=dfSelected2.groupby('consecutivo').mean()
    dfSelectedMean = dfSelected2.groupby("consecutivo").mean(numeric_only=True)

    return dfSelectedMean


def read_split_dataframe(csv_path):
    """
    Read a saved split CSV and drop the implicit index column if it exists.
    """
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def load_predefined_split(split_70_path, split_30_path):
    """
    Load the saved 70/30 split exactly as it was exported.
    """
    return read_split_dataframe(split_70_path), read_split_dataframe(split_30_path)


def sanitize_split_dataframes(df_70, df_30):
    """
    Ensure train/eval split frames are disjoint and have unique column names.

    Legacy saved splits may contain overlapping columns because an older
    pipeline appended the same tail columns to both partitions. Those
    overlapping columns are removed from both splits so the effective
    train/eval partitions are disjoint.
    """
    df_70 = df_70.copy()
    df_30 = df_30.copy()

    dup_70 = df_70.columns[df_70.columns.duplicated()].tolist()
    dup_30 = df_30.columns[df_30.columns.duplicated()].tolist()
    if dup_70 or dup_30:
        raise ValueError(
            f"Columnas duplicadas dentro del split. split_70={dup_70[:10]}, split_30={dup_30[:10]}"
        )

    overlap = [col for col in df_70.columns if col in set(df_30.columns)]
    if overlap:
        df_70 = df_70[[col for col in df_70.columns if col not in overlap]].copy()
        df_30 = df_30[[col for col in df_30.columns if col not in overlap]].copy()

    if df_70.shape[1] == 0 or df_30.shape[1] == 0:
        raise ValueError("Uno de los splits quedo vacio tras sanear columnas solapadas.")

    return df_70, df_30, overlap


def compare_split_dataframes(expected_70, expected_30, observed_70, observed_30, atol=1e-10):
    """
    Compare two 70/30 split pairs. Values are compared after aligning columns,
    so the report distinguishes between split content and column ordering.
    """

    def _compare(expected_df, observed_df, label):
        same_shape = expected_df.shape == observed_df.shape
        same_columns_exact = list(expected_df.columns) == list(observed_df.columns)
        same_column_set = set(expected_df.columns) == set(observed_df.columns)
        values_allclose = False
        max_abs_diff = None

        if same_shape and same_column_set:
            observed_aligned = observed_df[expected_df.columns]
            expected_values = expected_df.to_numpy(dtype=float)
            observed_values = observed_aligned.to_numpy(dtype=float)
            values_allclose = bool(
                np.allclose(expected_values, observed_values, equal_nan=True, atol=atol)
            )
            if expected_values.size:
                max_abs_diff = float(np.nanmax(np.abs(expected_values - observed_values)))
            else:
                max_abs_diff = 0.0

        return {
            f'{label}_shape': expected_df.shape,
            f'{label}_same_shape': same_shape,
            f'{label}_same_columns_exact': same_columns_exact,
            f'{label}_same_column_set': same_column_set,
            f'{label}_values_allclose': values_allclose,
            f'{label}_max_abs_diff': max_abs_diff,
        }

    comparison = {}
    comparison.update(_compare(expected_70, observed_70, 'split_70'))
    comparison.update(_compare(expected_30, observed_30, 'split_30'))
    comparison['matches_by_content'] = (
        comparison['split_70_same_shape']
        and comparison['split_70_same_column_set']
        and comparison['split_70_values_allclose']
        and comparison['split_30_same_shape']
        and comparison['split_30_same_column_set']
        and comparison['split_30_values_allclose']
    )
    comparison['matches_exact_layout'] = (
        comparison['matches_by_content']
        and comparison['split_70_same_columns_exact']
        and comparison['split_30_same_columns_exact']
    )
    return comparison


def _legacy_selectRandomColumns(df, seed=42):
  '''
  Function to select validation from rows sampled randomly
  '''
  # Identificar las últimas dos columnas
  last_two = df.columns[-2:].tolist()

  # Todas las demás columnas
  other_cols = df.columns[:-2].tolist()

  # Seleccionar aleatoriamente 70% de columnas (distintas de las dos últimas)
  rng = np.random.RandomState(seed)
  n_70 = int(len(other_cols) * 0.7)
  cols_70 = rng.choice(other_cols, size=n_70, replace=False).tolist()

  # El resto serán 30%
  cols_70_set = set(cols_70)
  cols_30 = [col for col in other_cols if col not in cols_70_set]


  # Construcción de los dos nuevos DataFrames conservando las últimas dos al final
  df_70 = df[list(cols_70) + last_two]
  df_30 = df[list(cols_30) + last_two]

  return df_70, df_30


def selectRandomColumns(df, seed=42):
  '''
  Function to split columns randomly into 70% / 30%.
  '''
  all_cols = df.columns.tolist()

  rng = np.random.RandomState(seed)
  n_70 = int(len(all_cols) * 0.7)
  cols_70 = rng.choice(all_cols, size=n_70, replace=False).tolist()

  cols_70_set = set(cols_70)
  cols_30 = [col for col in all_cols if col not in cols_70_set]

  df_70 = df[list(cols_70)]
  df_30 = df[list(cols_30)]

  return df_70, df_30


def estandarizar(data, pathEst):
    # Convertir numpy → pandas para facilitar manejo de columnas
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Calcular medias y desviaciones
    medias = df[numeric_cols].mean()
    desvios = df[numeric_cols].std(ddof=0)  # ddof=0 para std poblacional

    # Estandarizar
    df_std = df.copy()
    df_std[numeric_cols] = (df[numeric_cols] - medias) / desvios

    # Crear tabla parámetros
    parametros = pd.DataFrame({
        "names":numeric_cols,
        "media": medias,
        "desviacion": desvios
    })
    parametros.to_csv(pathEst)
    return df_std, parametros


def cargar_parametros_estandarizacion(params_source):
    """
    Load normalization parameters generated by estandarizar().
    """
    if isinstance(params_source, pd.DataFrame):
        params = params_source.copy()
    else:
        params = pd.read_csv(params_source)

    unnamed_cols = [col for col in params.columns if str(col).startswith('Unnamed:')]
    if unnamed_cols:
        params = params.drop(columns=unnamed_cols)

    required_cols = {'names', 'media', 'desviacion'}
    missing = required_cols - set(params.columns)
    if missing:
        raise ValueError(
            f"Parametros de estandarizacion incompletos. Faltan columnas: {sorted(missing)}"
        )

    return params[['names', 'media', 'desviacion']].copy()


def desestandarizar_ventanas(data, ids_list, params_source):
    """
    Restore standardized forecast windows back to their original scale.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"desestandarizar_ventanas espera un array 2D (N, horizon). Recibido ndim={data.ndim}"
        )

    if len(ids_list) != data.shape[0]:
        raise ValueError(
            f"ids_list tiene longitud {len(ids_list)} pero data tiene {data.shape[0]} filas."
        )

    params = cargar_parametros_estandarizacion(params_source)
    params = params.drop_duplicates(subset='names').set_index('names')

    missing_ids = sorted(set(ids_list) - set(params.index))
    if missing_ids:
        raise KeyError(
            f"No se encontraron parametros de estandarizacion para ids: {missing_ids[:10]}"
        )

    medias = np.array([params.at[item_id, 'media'] for item_id in ids_list], dtype=float)
    desvios = np.array([params.at[item_id, 'desviacion'] for item_id in ids_list], dtype=float)
    return data * desvios[:, None] + medias[:, None]


def desestandarizar_predicciones(predictions_dict, ids_list, params_source):
    """
    Apply desestandarizar_ventanas() to every prediction array in a dict.
    """
    return {
        name: desestandarizar_ventanas(predictions, ids_list, params_source)
        for name, predictions in predictions_dict.items()
    }


def deses(objetivo, ids_list, parametros):
    """
    Desestandariza un array 2D de predicciones usando la tabla de parametros guardada.

    Formula: objetivo * desviacion + media  (inversa de estandarizar)

    Args:
        objetivo   : np.ndarray de shape (N, horizonte) — predicciones estandarizadas
        ids_list   : lista de N strings con el nombre de la serie para cada fila
        parametros : DataFrame con columnas ['names', 'media', 'desviacion']

    Returns:
        np.ndarray de shape (N, horizonte) en escala original
    """
    df_ids = pd.DataFrame({"names": ids_list})
    df_resultado = df_ids.merge(parametros, on="names", how="left")
    return (
        (objetivo * np.array(df_resultado['desviacion']).reshape(-1, 1))
        + np.array(df_resultado['media']).reshape(-1, 1)
    )
