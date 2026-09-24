
"""MLP (Multi-Layer Perceptron) baseline."""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import Dict, Any


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Train an MLP classifier with validation-tuned hyperparameters.
    
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
    # Scale features for MLP
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['adaptive'],
        'max_iter': [500],
    }
    
    mlp = MLPClassifier(
        random_state=random_seed, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=10
    )
    grid_search = GridSearchCV(
        mlp, param_grid, cv=cv_folds, scoring='balanced_accuracy',
        n_jobs=-1, refit=True
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )
    
    n_params = 0
    layer_sizes = [X_train.shape[1]] + list(best_model.hidden_layer_sizes) + [len(np.unique(y_train))]
    for i in range(len(layer_sizes) - 1):
        n_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]  # weights + biases
    
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
        'scaler': scaler,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
        },
        'n_trainable_params': n_params,
    }
