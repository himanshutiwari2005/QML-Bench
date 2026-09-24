
"""Classical ML baseline implementations."""

from .logistic_regression import train_logistic_regression
from .svm import train_svm
from .random_forest import train_random_forest
from .mlp import train_mlp
from .xgboost_model import train_xgboost
from .benchmark import run_classical_benchmark

__all__ = [
    'train_logistic_regression',
    'train_svm',
    'train_random_forest',
    'train_mlp',
    'train_xgboost',
    'run_classical_benchmark',
]
