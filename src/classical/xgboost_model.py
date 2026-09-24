
"""XGBoost baseline (optional)."""

import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from typing import Dict, Any


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier with validation-tuned hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        cv_folds: Number of cross-validation folds
        random_seed: Random seed
    
    Returns:
        Dictionary with model, metrics, and training info
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
    }
    
    xgb = XGBClassifier(
        random_state=random_seed, use_label_encoder=False,
        eval_metric='logloss', n_jobs=-1
    )
    grid_search = GridSearchCV(
        xgb, param_grid, cv=cv_folds, scoring='balanced_accuracy',
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
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
        },
    }
