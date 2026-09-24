
"""SVM baseline with multiple kernel options."""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import Dict, Any, Optional


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kernel: str = 'rbf',
    cv_folds: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Train an SVM classifier with validation-tuned hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        kernel: Kernel type ('linear', 'rbf', 'poly')
        cv_folds: Number of cross-validation folds
        random_seed: Random seed
    
    Returns:
        Dictionary with model, metrics, and training info
    """
    if kernel == 'linear':
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        }
    elif kernel == 'rbf':
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        }
    elif kernel == 'poly':
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'degree': [2, 3, 4],
        }
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    svm = SVC(kernel=kernel, random_state=random_seed)
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv_folds, scoring='balanced_accuracy',
        n_jobs=-1, refit=True
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'precision': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='macro')),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return {
        'model': best_model,
        'metrics': metrics,
        'best_params': grid_search.best_params_,
        'kernel': kernel,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
        },
    }
