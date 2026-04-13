import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
from wrappers.EnsembleWrapper import EnsembleWrapper
from wrappers.KerasPretrainedWrapper import KerasPretrainedWrapper


LOCAL_PRETRAINED_MODELS = [
    'tcn',
    'nbeats',
    'lstm',
    'tide',
    'encdec',
    'itrans',
]
FOUNDATION_MODELS = [
    'moment',
    'moirai',
]
DEFAULT_MODELS = list(LOCAL_PRETRAINED_MODELS)
ALL_MODELS = list(FOUNDATION_MODELS) + list(LOCAL_PRETRAINED_MODELS)
DEFAULT_SPLIT_70_CSV = os.path.join(PROJECT_DIR, 'nueva_info', 'df_70.csv')
DEFAULT_SPLIT_30_CSV = os.path.join(PROJECT_DIR, 'nueva_info', 'df_30.csv')
DEFAULT_ENSEMBLE_ACTIVE_METHOD = 'optimize'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main local para ensemble de forecasting con MAPE, DTW y Pearson.',
    )
    parser.add_argument('--dataset-dir', default='dataset', help='Directorio con los Excel.')
    parser.add_argument('--cache-dir', default='cache', help='Directorio para cachear predicciones.')
    parser.add_argument('--input-size', type=int, default=200, help='Longitud de la ventana de entrada.')
    parser.add_argument('--output-size', type=int, default=200, help='Horizonte de prediccion.')
    parser.add_argument('--validation-size', type=float, default=0.1, help='Fraccion para validacion.')
    parser.add_argument('--random-state', type=int, default=42, help='Semilla de train/val split.')
    parser.add_argument(
        '--split-seed',
        type=int,
        default=42,
        help='Semilla usada para regenerar el split 70/30 cuando no se cargan CSVs predefinidos.',
    )
    parser.add_argument(
        '--split-70-csv',
        default=None,
        help='CSV con el 70% usado para entrenar los modelos preentrenados.',
    )
    parser.add_argument(
        '--split-30-csv',
        default=None,
        help='CSV con el 30% usado para evaluacion y desescalizacion.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=DEFAULT_MODELS,
        help='Subconjunto de modelos a ejecutar. Por defecto usa solo los modelos locales ya entrenados.',
    )
    parser.add_argument(
        '--select-by',
        default='MAPE',
        help='Compatibilidad legacy. Ya no selecciona el ensemble final usando eval.',
    )
    parser.add_argument(
        '--objective-metric',
        default='MAPE',
        help='Metrica objetivo para optimizacion de pesos.',
    )
    parser.add_argument(
        '--ensemble-active-method',
        default=DEFAULT_ENSEMBLE_ACTIVE_METHOD,
        choices=['average', 'optimize', 'stacking'],
        help='Metodo del ensemble que queda activo al final; se fija de antemano y no se elige sobre eval.',
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Ignora cache existente y recomputa predicciones.',
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Genera graficos al final de la corrida.',
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_DIR, *parts)


def normalize_project_path(path: str | None) -> str | None:
    if path is None:
        return None

    abs_path = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(abs_path, PROJECT_DIR)
    except ValueError:
        return abs_path

    if rel_path.startswith('..'):
        return abs_path
    return rel_path.replace('\\', '/')


def stable_hash_text(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def stable_hash_ids(ids_list: List[str]) -> str:
    payload = json.dumps([str(item) for item in ids_list], ensure_ascii=False)
    return stable_hash_text(payload)


def stable_hash_params(params_source) -> str:
    params = utils.cargar_parametros_estandarizacion(params_source)
    params = params.sort_values('names').reset_index(drop=True)
    payload = params.to_csv(index=False, float_format='%.17g')
    return stable_hash_text(payload)


def build_artifact_signature(artifact) -> dict:
    if isinstance(artifact, str) and os.path.exists(artifact):
        stat = os.stat(artifact)
        return {
            'path': normalize_project_path(artifact),
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }

    return {'artifact': artifact}


def build_cache_metadata(model_key: str, spec: Dict, datasets: Dict) -> Dict:
    return {
        'schema_version': 2,
        'model_key': model_key,
        'model_label': spec['label'],
        'model_source': spec.get('source'),
        'artifact_signature': build_artifact_signature(spec.get('artifact')),
        'input_size': int(datasets['input_size']),
        'output_size': int(datasets['output_size']),
        'eval_shape': list(datasets['y_test'].shape),
        'fit_shape': list(datasets['y_va'].shape),
        'ids_eval_hash': stable_hash_ids(datasets['ids_test']),
        'ids_fit_hash': stable_hash_ids(datasets['ids_va']),
        'params_eval_hash': stable_hash_params(datasets['params_eval']),
        'params_fit_hash': stable_hash_params(datasets['params_fit']),
        'split_source': datasets.get('split_source'),
        'split_seed': datasets.get('split_seed'),
        'split_70_path': normalize_project_path(datasets.get('split_70_path')),
        'split_30_path': normalize_project_path(datasets.get('split_30_path')),
        'dropped_overlap_columns': list(datasets.get('dropped_overlap_columns', [])),
    }


def resolve_split_paths(split_70_path: str = None, split_30_path: str = None) -> Tuple[str | None, str | None]:
    if split_70_path is None and split_30_path is None:
        if os.path.exists(DEFAULT_SPLIT_70_CSV) and os.path.exists(DEFAULT_SPLIT_30_CSV):
            return DEFAULT_SPLIT_70_CSV, DEFAULT_SPLIT_30_CSV
        return None, None

    if bool(split_70_path) != bool(split_30_path):
        raise ValueError('Debes indicar ambos CSVs del split: --split-70-csv y --split-30-csv.')

    if not os.path.exists(split_70_path):
        raise FileNotFoundError(f'No existe el split 70%: {split_70_path}')
    if not os.path.exists(split_30_path):
        raise FileNotFoundError(f'No existe el split 30%: {split_30_path}')

    return split_70_path, split_30_path



def load_split_dataframes(
    dataset_dir: str,
    split_seed: int = 42,
    split_70_path: str = None,
    split_30_path: str = None,
):
    split_70_path, split_30_path = resolve_split_paths(split_70_path, split_30_path)

    if split_70_path is not None and split_30_path is not None:
        print(f'Usando split predefinido: {split_70_path} | {split_30_path}')
        df_70, df_30 = utils.load_predefined_split(split_70_path, split_30_path)
        df_70, df_30, dropped_overlap_columns = utils.sanitize_split_dataframes(df_70, df_30)
        if dropped_overlap_columns:
            print(
                'Aviso: se excluyen columnas solapadas del split legacy para evitar leakage: '
                f'{dropped_overlap_columns}'
            )
        metadata = {
            'split_source': 'predefined_csv',
            'split_seed': split_seed,
            'split_70_path': split_70_path,
            'split_30_path': split_30_path,
            'dropped_overlap_columns': dropped_overlap_columns,
        }
        return df_70, df_30, metadata

    df_selected_mean = utils.loadAllFiles(dataset_dir)
    df_70, df_30 = utils.selectRandomColumns(df_selected_mean, seed=split_seed)
    df_70, df_30, dropped_overlap_columns = utils.sanitize_split_dataframes(df_70, df_30)
    print(f'Usando split regenerado desde dataset con seed={split_seed}.')
    metadata = {
        'split_source': 'dataset_random_split',
        'split_seed': split_seed,
        'split_70_path': None,
        'split_30_path': None,
        'dropped_overlap_columns': dropped_overlap_columns,
    }
    return df_70, df_30, metadata


def prepare_datasets(
    dataset_dir: str,
    cache_dir: str,
    input_size: int,
    output_size: int,
    validation_size: float,
    random_state: int,
    split_seed: int = 42,
    split_70_path: str = None,
    split_30_path: str = None,
):
    print('=' * 60)
    print('Preparando dataset')
    print('=' * 60)
    ensure_dir(cache_dir)

    df_70, df_30, split_metadata = load_split_dataframes(
        dataset_dir=dataset_dir,
        split_seed=split_seed,
        split_70_path=split_70_path,
        split_30_path=split_30_path,
    )

    path_est_70 = os.path.join(cache_dir, 'values_deses_70.csv')
    path_est_30 = os.path.join(cache_dir, 'values_deses_30.csv')
    df_scaled_70, params_fit = utils.estandarizar(df_70, path_est_70)
    df_scaled_30, params_eval = utils.estandarizar(df_30, path_est_30)

    X_train, y_train, ids_train = utils.series_to_supervised_matrix(
        df_scaled_70, input_size=input_size, output_size=output_size
    )
    X_tr, X_va, y_tr, y_va, ids_tr, ids_va = train_test_split(
        X_train,
        y_train,
        ids_train,
        test_size=validation_size,
        random_state=random_state,
    )
    X_test, y_test, ids_list_te = utils.series_to_supervised_matrix(
        df_scaled_30, input_size=input_size, output_size=output_size
    )

    print(f'X_tr={X_tr.shape}, y_tr={y_tr.shape}')
    print(f'X_va={X_va.shape}, y_va={y_va.shape}')
    print(f'X_test={X_test.shape}, y_test={y_test.shape}')

    return {
        'X_tr': X_tr,
        'X_va': X_va,
        'y_tr': y_tr,
        'y_va': y_va,
        'X_test': X_test,
        'y_test': y_test,
        'ids_tr': ids_tr,
        'ids_va': ids_va,
        'ids_test': ids_list_te,
        'params_fit': params_fit,
        'params_eval': params_eval,
        'input_size': input_size,
        'output_size': output_size,
        **split_metadata,
    }


def build_model_registry() -> Dict[str, Dict]:
    return {
        'moment': {
            'label': 'MOMENT',
            'factory': _build_moment_wrapper,
            'needs_fit': True,
            'source': 'foundation',
            'artifact': 'AutonLab/MOMENT-1-large',
        },
        'moirai': {
            'label': 'Moirai',
            'factory': _build_moirai_wrapper,
            'needs_fit': True,
            'source': 'foundation',
            'artifact': 'Salesforce/moirai-1.1-R-small',
        },
        'tcn': {
            'label': 'TCN',
            'factory': _build_tcn_wrapper,
            'needs_fit': False,
            'source': 'local_pretrained',
            'artifact': project_path('nueva_info', 'tcn.keras'),
        },
        'nbeats': {
            'label': 'NBEATS',
            'factory': _build_nbeats_wrapper,
            'needs_fit': False,
            'source': 'local_pretrained',
            'artifact': project_path('nueva_info', 'nbeats.keras'),
        },
        'lstm': {
            'label': 'LSTM',
            'factory': lambda: KerasPretrainedWrapper(
                project_path('nueva_info', 'lstm.keras'),
                batch_size=32,
                name='LSTM',
            ).load(),
            'needs_fit': False,
            'source': 'local_pretrained',
            'artifact': project_path('nueva_info', 'lstm.keras'),
        },
        'tide': {
            'label': 'TiDE',
            'factory': lambda: KerasPretrainedWrapper(
                project_path('nueva_info', 'tide.keras'),
                batch_size=32,
                name='TiDE',
            ).load(),
            'needs_fit': False,
            'source': 'local_pretrained',
            'artifact': project_path('nueva_info', 'tide.keras'),
        },
        'encdec': {
            'label': 'Encoder-Decoder LSTM',
            'factory': lambda: KerasPretrainedWrapper(
                project_path('nueva_info', 'encDec.keras'),
                batch_size=32,
                name='Encoder-Decoder LSTM',
            ).load(),
            'needs_fit': False,
            'source': 'local_pretrained',
            'artifact': project_path('nueva_info', 'encDec.keras'),
        },
        'itrans': {
            'label': 'i-Transformer',
            'factory': lambda: KerasPretrainedWrapper(
                project_path('nueva_info', 'itrans.keras'),
                batch_size=32,
                name='i-Transformer',
            ).load(),
            'needs_fit': False,
            'source': 'local_pretrained',
            'artifact': project_path('nueva_info', 'itrans.keras'),
        },
    }


def _build_moment_wrapper():
    from wrappers.MomentSupervisedWrapper import MomentSupervisedWrapper

    return MomentSupervisedWrapper(config_path=project_path('configs', 'moment_config.yaml'))


def _build_moirai_wrapper():
    from wrappers.MoiraiSupervisedWrapper import MoiraiSupervisedWrapper

    return MoiraiSupervisedWrapper(config_path=project_path('configs', 'moirai_config.yaml'))


def _build_tcn_wrapper():
    from wrappers.TCNSupervisedWrapper import TCNSupervisedWrapper

    return TCNSupervisedWrapper(config_path=project_path('configs', 'tcn_config.yaml'))


def _build_nbeats_wrapper():
    from wrappers.NBeatsSupervisedWrapper import NBeatsSupervisedWrapper

    return NBeatsSupervisedWrapper(config_path=project_path('configs', 'nbeats_config.yaml'))


def prediction_cache_paths(cache_dir: str, model_key: str) -> Dict[str, str]:
    model_dir = os.path.join(cache_dir, model_key)
    ensure_dir(model_dir)
    return {
        'eval': os.path.join(model_dir, 'y_pred_eval.npy'),
        'fit': os.path.join(model_dir, 'y_pred_fit.npy'),
        'metrics': os.path.join(model_dir, 'metrics_eval.json'),
    }


def load_cached_predictions(paths: Dict[str, str], expected_metadata: Dict = None):
    if not all(os.path.exists(paths[key]) for key in ['eval', 'fit', 'metrics']):
        return None

    with open(paths['metrics'], 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or 'metrics' not in payload or 'metadata' not in payload:
        print('Cache legacy/incompleto detectado; se recomputa para evitar inconsistencias.')
        return None

    cached_metadata = payload['metadata']
    if expected_metadata is not None and cached_metadata != expected_metadata:
        print('Cache incompatible con el dataset/modelo actual; se recomputa.')
        return None

    eval_pred = np.load(paths['eval'])
    fit_pred = np.load(paths['fit'])
    if expected_metadata is not None:
        if list(eval_pred.shape) != expected_metadata['eval_shape']:
            print('Cache invalido: shape de eval no coincide; se recomputa.')
            return None
        if list(fit_pred.shape) != expected_metadata['fit_shape']:
            print('Cache invalido: shape de fit no coincide; se recomputa.')
            return None

    return {
        'eval': eval_pred,
        'fit': fit_pred,
        'metrics': payload['metrics'],
        'metadata': cached_metadata,
    }


def save_cached_predictions(
    paths: Dict[str, str],
    y_eval: np.ndarray,
    y_fit: np.ndarray,
    metrics: dict,
    metadata: Dict,
):
    np.save(paths['eval'], y_eval)
    np.save(paths['fit'], y_fit)
    with open(paths['metrics'], 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'metadata': metadata}, f, indent=2)


def run_model(
    model_key: str,
    spec: Dict,
    datasets: Dict,
    cache_dir: str,
    force_recompute: bool = False,
):
    label = spec['label']
    print(f'\n{"=" * 60}\nModelo: {label}\n{"=" * 60}')
    paths = prediction_cache_paths(cache_dir, model_key)
    expected_cache_metadata = build_cache_metadata(model_key, spec, datasets)

    if not force_recompute:
        cached = load_cached_predictions(paths, expected_metadata=expected_cache_metadata)
        if cached is not None:
            print(f'Usando cache local para {label}.')
            return cached

    model = spec['factory']()

    if spec.get('needs_fit', False):
        print(f'Ajustando {label} sobre train...')
        model.fit(
            datasets['X_tr'],
            datasets['y_tr'],
            X_val=datasets['X_va'],
            y_val=datasets['y_va'],
        )

    y_eval = model.predict(datasets['X_test'])
    y_fit = model.predict(datasets['X_va'])
    y_eval_original = utils.desestandarizar_ventanas(
        y_eval,
        datasets['ids_test'],
        datasets['params_eval'],
    )
    y_test_original = utils.desestandarizar_ventanas(
        datasets['y_test'],
        datasets['ids_test'],
        datasets['params_eval'],
    )
    metrics = utils.evaluate_all_metrics(y_test_original, y_eval_original)

    save_cached_predictions(paths, y_eval, y_fit, metrics, expected_cache_metadata)
    return {
        'eval': y_eval,
        'fit': y_fit,
        'metrics': metrics,
        'metadata': expected_cache_metadata,
    }


def align_predictions(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    if not predictions:
        return y_true, predictions, y_true.shape[0]

    expected_shape = y_true.shape
    mismatches = {
        name: pred.shape
        for name, pred in predictions.items()
        if pred.shape != expected_shape
    }
    if mismatches:
        raise ValueError(
            f'Predicciones desalineadas. y_true={expected_shape}, predicciones={mismatches}'
        )

    return y_true, predictions, y_true.shape[0]


def trim_ids(ids_list: List[str], length: int) -> List[str]:
    if len(ids_list) <= length:
        return ids_list
    return ids_list[:length]


def compute_individual_results(y_true: np.ndarray, eval_predictions: Dict[str, np.ndarray]) -> Dict[str, dict]:
    return {
        label: utils.evaluate_all_metrics(y_true, y_pred)
        for label, y_pred in eval_predictions.items()
    }


def align_and_restore_original_scale(
    datasets: Dict,
    eval_predictions: Dict[str, np.ndarray],
    fit_predictions: Dict[str, np.ndarray],
):
    y_test_scaled, eval_predictions_scaled, eval_n = align_predictions(
        datasets['y_test'],
        eval_predictions,
    )
    y_va_scaled, fit_predictions_scaled, fit_n = align_predictions(
        datasets['y_va'],
        fit_predictions,
    )

    ids_eval = trim_ids(datasets['ids_test'], eval_n)
    ids_fit = trim_ids(datasets['ids_va'], fit_n)

    y_true_eval = utils.desestandarizar_ventanas(
        y_test_scaled,
        ids_eval,
        datasets['params_eval'],
    )
    y_true_fit = utils.desestandarizar_ventanas(
        y_va_scaled,
        ids_fit,
        datasets['params_fit'],
    )
    eval_predictions_original = utils.desestandarizar_predicciones(
        eval_predictions_scaled,
        ids_eval,
        datasets['params_eval'],
    )
    fit_predictions_original = utils.desestandarizar_predicciones(
        fit_predictions_scaled,
        ids_fit,
        datasets['params_fit'],
    )

    return {
        'y_true_eval': y_true_eval,
        'y_true_fit': y_true_fit,
        'eval_predictions': eval_predictions_original,
        'fit_predictions': fit_predictions_original,
        'ids_eval': ids_eval,
        'ids_fit': ids_fit,
        'eval_n': eval_n,
        'fit_n': fit_n,
    }


def save_final_tables(cache_dir: str, individual_results: Dict, ensemble_results: Dict):
    final_results = {}
    final_results.update(individual_results)
    final_results.update(ensemble_results)

    if not final_results:
        return

    final_df = pd.DataFrame(final_results).T
    final_df.index.name = 'Modelo / Metodo'
    if 'MAPE' in final_df.columns:
        final_df = final_df.sort_values('MAPE')

    csv_path = os.path.join(cache_dir, 'final_results.csv')
    json_path = os.path.join(cache_dir, 'final_results.json')
    final_df.to_csv(csv_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

    print(f'\nResultados guardados en:\n  {csv_path}\n  {json_path}')
    print(final_df)


def maybe_plot(
    datasets: Dict,
    ids_test: List[str],
    individual_predictions: Dict[str, np.ndarray],
    ensemble_prediction: np.ndarray,
):
    for label, y_pred in individual_predictions.items():
        utils.plot_forecast_samples(datasets['y_test'], y_pred, n_samples=6, title=f'{label}: Prediccion vs Real')
        utils.plot_error_over_horizon(datasets['y_test'], y_pred, title=f'{label}: Error segun horizonte')
        utils.plot_subject_performance(datasets['y_test'], y_pred, ids_test, title=f'{label}: Rendimiento por sujeto')

    if ensemble_prediction is not None:
        utils.plot_forecast_samples(
            datasets['y_test'],
            ensemble_prediction,
            n_samples=6,
            title='Ensemble: Prediccion vs Real',
        )
        utils.plot_error_over_horizon(
            datasets['y_test'],
            ensemble_prediction,
            title='Ensemble: Error segun horizonte',
        )
        utils.plot_subject_performance(
            datasets['y_test'],
            ensemble_prediction,
            ids_test,
            title='Ensemble: Rendimiento por sujeto',
        )


def main():
    args = parse_args()
    ensure_dir(args.cache_dir)

    registry = build_model_registry()
    unknown_models = [name for name in args.models if name not in registry]
    if unknown_models:
        raise ValueError(
            f'Modelos desconocidos: {unknown_models}. Disponibles: {sorted(registry.keys())}'
        )

    datasets = prepare_datasets(
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
        input_size=args.input_size,
        output_size=args.output_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        split_seed=args.split_seed,
        split_70_path=args.split_70_csv,
        split_30_path=args.split_30_csv,
    )

    individual_results = {}
    eval_predictions = {}
    fit_predictions = {}

    for model_key in args.models:
        spec = registry[model_key]
        label = spec['label']
        try:
            bundle = run_model(
                model_key=model_key,
                spec=spec,
                datasets=datasets,
                cache_dir=args.cache_dir,
                force_recompute=args.force_recompute,
            )
        except Exception as exc:
            print(f'[SKIP] {label}: {exc}')
            continue

        eval_predictions[label] = bundle['eval']
        fit_predictions[label] = bundle['fit']

    if not eval_predictions:
        raise RuntimeError(
            'No se pudo ejecutar ningun modelo. Revisa dependencias o usa el cache existente.'
        )

    original_scale = align_and_restore_original_scale(
        datasets,
        eval_predictions,
        fit_predictions,
    )
    ids_test = original_scale['ids_eval']
    eval_n = original_scale['eval_n']

    datasets['y_test_scaled'] = datasets['y_test']
    datasets['y_va_scaled'] = datasets['y_va']
    datasets['y_test'] = original_scale['y_true_eval']
    datasets['y_va'] = original_scale['y_true_fit']
    eval_predictions = original_scale['eval_predictions']
    fit_predictions = original_scale['fit_predictions']
    datasets['ids_test'] = ids_test
    datasets['ids_va'] = original_scale['ids_fit']

    individual_results = compute_individual_results(datasets['y_test'], eval_predictions)

    print(f'\nModelos individuales disponibles: {list(individual_results.keys())}')
    print(pd.DataFrame(individual_results).T)

    ensemble_results = {}
    ensemble_prediction = None

    if len(eval_predictions) >= 2:
        print(f'\n{"=" * 60}\nEnsemble\n{"=" * 60}')
        ens = EnsembleWrapper()

        for label, pred in eval_predictions.items():
            ens.add_model(label, pred, split='eval')
        for label, pred in fit_predictions.items():
            ens.add_model(label, pred, split='fit')

        ensemble_results = ens.compare_methods(
            y_true_eval=datasets['y_test'],
            y_true_fit=datasets['y_va'],
            select_by=args.select_by,
            objective_metric=args.objective_metric,
            active_method=args.ensemble_active_method,
        )
        ensemble_prediction = ens.predict()

        print(pd.DataFrame(ensemble_results).T)
    else:
        print('\nNo hay suficientes modelos para construir ensemble. Se requieren al menos 2.')

    save_final_tables(args.cache_dir, individual_results, ensemble_results)

    if args.plots:
        maybe_plot(datasets, ids_test, eval_predictions, ensemble_prediction)


if __name__ == '__main__':
    main()
