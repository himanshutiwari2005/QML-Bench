
"""Unified classical benchmark runner."""

import time
import numpy as np
from typing import Dict, Any, List
from src.utils.config import ClassicalConfig, ExperimentResult


def run_classical_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    configs: List[ClassicalConfig],
    dataset_name: str,
) -> List[ExperimentResult]:
    """
    Run classical baselines on a dataset.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        configs: List of ClassicalConfig objects
        dataset_name: Name of the dataset
    
    Returns:
        List of ExperimentResult objects
    """
    from .logistic_regression import train_logistic_regression
    from .svm import train_svm
    from .random_forest import train_random_forest
    from .mlp import train_mlp
    from .xgboost_model import train_xgboost
    
    results = []
    
    for config in configs:
        start_time = time.time()
        
        # Accept either ClassicalConfig objects or plain dicts
        if hasattr(config, 'model_type'):
            model_type = config.model_type
            cv_folds = config.cv_folds
            hyperparameters = config.hyperparameters
        else:
            model_type = config['model_type']
            cv_folds = config.get('cv_folds', 5)
            hyperparameters = config.get('hyperparameters', {})
        
        if model_type == 'logreg':
            result_dict = train_logistic_regression(
                X_train, y_train, X_test, y_test,
                cv_folds=cv_folds
            )
        elif model_type == 'svm':
            kernel = hyperparameters.get('kernel', 'rbf')
            result_dict = train_svm(
                X_train, y_train, X_test, y_test,
                kernel=kernel, cv_folds=cv_folds
            )
        elif model_type == 'rf':
            result_dict = train_random_forest(
                X_train, y_train, X_test, y_test,
                cv_folds=cv_folds
            )
        elif model_type == 'mlp':
            result_dict = train_mlp(
                X_train, y_train, X_test, y_test,
                cv_folds=cv_folds
            )
        elif model_type == 'xgb':
            result_dict = train_xgboost(
                X_train, y_train, X_test, y_test,
                cv_folds=cv_folds
            )
        else:
            raise ValueError(f"Unknown classical model type: {model_type}")
        
        runtime = time.time() - start_time
        
        result = ExperimentResult(
            experiment_id=f"{dataset_name}_{model_type}",
            dataset_name=dataset_name,
            model_type=model_type,
            n_features=X_train.shape[1],
            accuracy=result_dict['metrics']['accuracy'],
            balanced_accuracy=result_dict['metrics']['balanced_accuracy'],
            macro_f1=result_dict['metrics']['macro_f1'],
            precision=result_dict['metrics']['precision'],
            recall=result_dict['metrics']['recall'],
            confusion_matrix=result_dict['metrics']['confusion_matrix'],
            runtime_seconds=runtime,
            n_trainable_params=result_dict.get('n_trainable_params'),
        )
        results.append(result)
    
    return results
