
"""Synthetic dataset generators for QML-Bench."""

import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


def generate_linearly_separable(
    n_samples: int = 200,
    n_features: int = 4,
    n_informative: int = 4,
    class_sep: float = 1.5,
    random_seed: int = 42,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Generate a linearly separable classification dataset.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_informative: Number of informative features
        class_sep: Separation between classes (higher = easier)
        random_seed: Random seed for reproducibility
        train_ratio: Fraction of data for training
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test, and metadata
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=0,
        random_state=random_seed,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=random_seed
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': {
            'name': 'linearly_separable',
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': len(np.unique(y)),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_balance': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            'n_informative': n_informative,
            'class_sep': class_sep,
            'preprocessing': 'standardize -> minmax [0, pi]',
        }
    }


def generate_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    random_seed: int = 42,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Generate a non-linear moons dataset.
    
    Args:
        n_samples: Total number of samples
        noise: Noise level in the moons
        random_seed: Random seed for reproducibility
        train_ratio: Fraction of data for training
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test, and metadata
    """
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_seed,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=random_seed
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': {
            'name': 'moons',
            'n_samples': n_samples,
            'n_features': 2,
            'n_classes': len(np.unique(y)),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_balance': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            'noise': noise,
            'preprocessing': 'standardize -> minmax [0, pi]',
        }
    }
