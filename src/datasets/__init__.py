
"""Dataset loaders and synthetic data generators."""

from .synthetic import generate_linearly_separable, generate_moons
from .real import load_iris_dataset, load_breast_cancer_dataset
from .data_info import describe_dataset

__all__ = [
    'generate_linearly_separable',
    'generate_moons',
    'load_iris_dataset',
    'load_breast_cancer_dataset',
    'describe_dataset',
]
