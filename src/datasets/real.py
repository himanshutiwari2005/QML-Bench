
"""Real-world dataset loaders for QML-Bench."""

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Dict, Any


def load_iris_dataset(
    n_features: int = 4,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Load the Iris dataset.
    
    Args:
        n_features: Number of features to use (max 4)
        train_ratio: Fraction for training
        random_seed: Random seed
    
    Returns:
        Dictionary with data splits and metadata
    """
    data = load_iris()
    X = data.data[:, :n_features]
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=random_seed
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': {
            'name': 'iris',
            'n_samples': len(X),
            'n_features': n_features,
            'n_classes': len(np.unique(y)),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_balance': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            'feature_names': list(data.feature_names[:n_features]),
            'class_names': list(data.target_names),
            'source': 'sklearn.datasets.load_iris',
            'preprocessing': 'standardize -> minmax [0, pi]',
        }
    }


def load_breast_cancer_dataset(
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Load the Breast Cancer Wisconsin dataset.
    
    Args:
        train_ratio: Fraction for training
        random_seed: Random seed
    
    Returns:
        Dictionary with data splits and metadata
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=random_seed
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': {
            'name': 'breast_cancer',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_balance': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            'feature_names': list(data.feature_names),
            'class_names': list(data.target_names),
            'source': 'sklearn.datasets.load_breast_cancer',
            'preprocessing': 'standardize -> minmax [0, pi]',
        }
    }
