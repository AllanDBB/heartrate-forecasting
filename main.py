import argparse
import json
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils
from wrappers.EnsembleWrapper import EnsembleWrapper
from wrappers.KerasPretrainedWrapper import KerasPretrainedWrapper


DEFAULT_MODELS = [
    'moment',
    'moirai',
    'tcn',
    'nbeats',
    'lstm',
    'tide',
    'encdec',
    'itrans',
]


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
        '--models',
        nargs='+',
        default=DEFAULT_MODELS,
        help='Subconjunto de modelos a ejecutar.',
    )
    parser.add_argument(
        '--select-by',
        default='MAPE',
        help='Metrica usada para dejar activo el mejor ensemble.',
    )
    parser.add_argument(
        '--objective-metric',
        default='MAPE',
        help='Metrica objetivo para optimizacion de pesos.',
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


def prepare_datasets(
    dataset_dir: str,
    cache_dir: str,
    input_size: int,
    output_size: int,
    validation_size: float,
    random_state: int,
):
    print('=' * 60)
    print('Preparando dataset')
    print('=' * 60)

    df_selected_mean = utils.loadAllFiles(dataset_dir)
    df_70, df_30 = utils.selectRandomColumns(df_selected_mean)

    path_est_70 = os.path.join(cache_dir, 'values_deses_70.csv')
    path_est_30 = os.path.join(cache_dir, 'values_deses_30.csv')
    df_scaled_70, _ = utils.estandarizar(df_70, path_est_70)
    df_scaled_30, _ = utils.estandarizar(df_30, path_est_30)

    X_train, y_train, _ = utils.series_to_supervised_matrix(
        df_scaled_70.iloc[:, :-2], input_size=input_size, output_size=output_size
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
    )
    X_test, y_test, ids_list_te = utils.series_to_supervised_matrix(
        df_scaled_30.iloc[:, :-2], input_size=input_size, output_size=output_size
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
        'ids_test': ids_list_te,
        'input_size': input_size,
        'output_size': output_size,
    }


def build_model_registry() -> Dict[str, Dict]:
    return {
        'moment': {
            'label': 'MOMENT',
            'factory': _build_moment_wrapper,
            'needs_fit': True,
        },
        'moirai': {
            'label': 'Moirai',
            'factory': _build_moirai_wrapper,
            'needs_fit': True,
        },
        'tcn': {
            'label': 'TCN',
            'factory': _build_tcn_wrapper,
            'needs_fit': False,
        },
        'nbeats': {
            'label': 'NBEATS',
            'factory': _build_nbeats_wrapper,
            'needs_fit': False,
        },
        'lstm': {
            'label': 'LSTM',
            'factory': lambda: KerasPretrainedWrapper('modelos/lstm_2.keras', batch_size=32, name='LSTM').load(),
            'needs_fit': False,
        },
        'tide': {
            'label': 'TiDE',
            'factory': lambda: KerasPretrainedWrapper('modelos/tide_0.keras', batch_size=32, name='TiDE').load(),
            'needs_fit': False,
        },
        'encdec': {
            'label': 'Encoder-Decoder LSTM',
            'factory': lambda: KerasPretrainedWrapper(
                'modelos/encDec_2.keras',
                batch_size=32,
                name='Encoder-Decoder LSTM',
            ).load(),
            'needs_fit': False,
        },
        'itrans': {
            'label': 'i-Transformer',
            'factory': lambda: KerasPretrainedWrapper(
                'modelos/itrans_2.keras',
                batch_size=32,
                name='i-Transformer',
            ).load(),
            'needs_fit': False,
        },
    }


def _build_moment_wrapper():
    from wrappers.MomentSupervisedWrapper import MomentSupervisedWrapper

    return MomentSupervisedWrapper(config_path='configs/moment_config.yaml')


def _build_moirai_wrapper():
    from wrappers.MoiraiSupervisedWrapper import MoiraiSupervisedWrapper

    return MoiraiSupervisedWrapper(config_path='configs/moirai_config.yaml')


def _build_tcn_wrapper():
    from wrappers.TCNSupervisedWrapper import TCNSupervisedWrapper

    return TCNSupervisedWrapper(config_path='configs/tcn_config.yaml')


def _build_nbeats_wrapper():
    from wrappers.NBeatsSupervisedWrapper import NBeatsSupervisedWrapper

    return NBeatsSupervisedWrapper(config_path='configs/nbeats_config.yaml')


def prediction_cache_paths(cache_dir: str, model_key: str) -> Dict[str, str]:
    model_dir = os.path.join(cache_dir, model_key)
    ensure_dir(model_dir)
    return {
        'eval': os.path.join(model_dir, 'y_pred_eval.npy'),
        'fit': os.path.join(model_dir, 'y_pred_fit.npy'),
        'metrics': os.path.join(model_dir, 'metrics_eval.json'),
    }


def load_cached_predictions(paths: Dict[str, str]):
    if not all(os.path.exists(paths[key]) for key in ['eval', 'fit', 'metrics']):
        return None

    with open(paths['metrics'], 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return {
        'eval': np.load(paths['eval']),
        'fit': np.load(paths['fit']),
        'metrics': metrics,
    }


def save_cached_predictions(paths: Dict[str, str], y_eval: np.ndarray, y_fit: np.ndarray, metrics: dict):
    np.save(paths['eval'], y_eval)
    np.save(paths['fit'], y_fit)
    with open(paths['metrics'], 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


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

    if not force_recompute:
        cached = load_cached_predictions(paths)
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
    metrics = utils.evaluate_all_metrics(datasets['y_test'], y_eval)

    save_cached_predictions(paths, y_eval, y_fit, metrics)
    return {
        'eval': y_eval,
        'fit': y_fit,
        'metrics': metrics,
    }


def align_predictions(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    if not predictions:
        return y_true, predictions, y_true.shape[0]

    sizes = [y_true.shape[0]]
    sizes.extend(pred.shape[0] for pred in predictions.values())
    aligned_n = min(sizes)
    y_true_aligned = y_true[:aligned_n]
    predictions_aligned = {
        name: pred[:aligned_n]
        for name, pred in predictions.items()
    }
    return y_true_aligned, predictions_aligned, aligned_n


def trim_ids(ids_list: List[str], length: int) -> List[str]:
    if len(ids_list) <= length:
        return ids_list
    return ids_list[:length]


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

        individual_results[label] = utils.normalize_metrics_dict(bundle['metrics'])
        eval_predictions[label] = bundle['eval']
        fit_predictions[label] = bundle['fit']

    if not eval_predictions:
        raise RuntimeError(
            'No se pudo ejecutar ningun modelo. Revisa dependencias o usa el cache existente.'
        )

    y_test_aligned, eval_predictions, eval_n = align_predictions(datasets['y_test'], eval_predictions)
    y_va_aligned, fit_predictions, _ = align_predictions(datasets['y_va'], fit_predictions)
    ids_test = trim_ids(datasets['ids_test'], eval_n)

    if eval_n != datasets['y_test'].shape[0]:
        print(f'\nAviso: recortando eval a {eval_n} ventanas para alinear predicciones.')

    datasets['y_test'] = y_test_aligned
    datasets['y_va'] = y_va_aligned

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
