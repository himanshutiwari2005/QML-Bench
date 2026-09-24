
"""Configuration dataclasses for experiments."""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    train_ratio: float = 0.8
    random_seed: int = 42
    noise: float = 0.0
    # For synthetic datasets
    n_informative: int = 2
    class_sep: float = 1.0


@dataclass
class ClassicalConfig:
    """Configuration for classical models."""
    model_type: str  # 'logreg', 'svm', 'rf', 'mlp', 'xgb'
    hyperparameters: dict = field(default_factory=dict)
    cv_folds: int = 5


@dataclass
class QuantumConfig:
    """Configuration for quantum models."""
    n_qubits: int
    feature_map_type: str  # 'zz', 'pauli', 'linear'
    feature_map_depth: int = 1
    ansatz_type: str = 'hardware_efficient'
    ansatz_depth: int = 1
    optimizer: str = 'SPSA'
    optimizer_kwargs: dict = field(default_factory=dict)
    shots: int = 1024
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    dataset: DatasetConfig
    classical: List[ClassicalConfig]
    quantum: List[QuantumConfig]
    noise: Optional[dict] = None  # Noise model parameters
    n_seeds: int = 1
    execution_mode: str = 'simulator'  # 'simulator', 'noisy_simulator', 'hardware'


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""
    experiment_id: str
    dataset_name: str
    model_type: str
    n_features: int
    n_qubits: Optional[int] = None
    circuit_depth: Optional[int] = None
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    confusion_matrix: Optional[List[List[int]]] = None
    runtime_seconds: float = 0.0
    n_trainable_params: Optional[int] = None
    n_circuit_evaluations: Optional[int] = None
    n_shots: Optional[int] = None
    kernel_evaluation_cost: Optional[float] = None
    seed: int = 42
    noise_level: Optional[float] = None
    extra: dict = field(default_factory=dict)


def result_to_dict(result: ExperimentResult) -> dict:
    """Convert ExperimentResult to a JSON-serializable dict."""
    return {
        'experiment_id': result.experiment_id,
        'dataset_name': result.dataset_name,
        'model_type': result.model_type,
        'n_features': result.n_features,
        'n_qubits': result.n_qubits,
        'circuit_depth': result.circuit_depth,
        'accuracy': result.accuracy,
        'balanced_accuracy': result.balanced_accuracy,
        'macro_f1': result.macro_f1,
        'precision': result.precision,
        'recall': result.recall,
        'confusion_matrix': result.confusion_matrix,
        'runtime_seconds': result.runtime_seconds,
        'n_trainable_params': result.n_trainable_params,
        'n_circuit_evaluations': result.n_circuit_evaluations,
        'n_shots': result.n_shots,
        'kernel_evaluation_cost': result.kernel_evaluation_cost,
        'seed': result.seed,
        'noise_level': result.noise_level,
        'extra': result.extra,
    }


def save_results(results: List[ExperimentResult], path: str) -> None:
    """Save a list of ExperimentResults to a JSON file."""
    with open(path, 'w') as f:
        json.dump([result_to_dict(r) for r in results], f, indent=2)


def load_results(path: str) -> List[ExperimentResult]:
    """Load ExperimentResults from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [ExperimentResult(**d) for d in data]
