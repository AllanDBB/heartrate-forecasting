import numpy as np
import pandas as pd
import os
import joblib
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    df['segundo_agrupado'] = df['TIME_dt'].dt.floor('S')

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


def estandarization(df, pathEst):

    # get columns excluding two last columns
    cols = df.select_dtypes(include='number').columns
    cols=cols[:-2]
    # Crear scaler
    scaler = StandardScaler()

    # Entrenar scaler y transformar datos
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])

    # Guardar scaler para usarlo después
    #joblib.dump(scaler, pathEst)

    return df_scaled

def selectRandomColumns(df):
  '''
  Function to select validation from rows sampled randomly
  '''
  # Identificar las últimas dos columnas
  last_two = df.columns[-2:].tolist()

  # Todas las demás columnas
  other_cols = df.columns[:-2].tolist()

  # Seleccionar aleatoriamente 70% de columnas (distintas de las dos últimas)
  np.random.seed(42)
  n_70 = int(len(other_cols) * 0.7)
  cols_70 = np.random.choice(other_cols, size=n_70, replace=False)

  # El resto serán 30%
  cols_30 = list(set(other_cols) - set(cols_70))


  # Construcción de los dos nuevos DataFrames conservando las últimas dos al final
  df_70 = df[list(cols_70) + last_two]
  df_30 = df[list(cols_30) + last_two]

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
