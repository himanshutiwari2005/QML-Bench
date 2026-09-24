"""Quantum kernel method implementation with caching support."""

import time
import numpy as np
from typing import Optional
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit_aer.noise import NoiseModel

from src.quantum.feature_maps.factory import create_feature_map, get_feature_map_info
from src.utils.config import QuantumConfig, ExperimentResult


class QuantumKernelMethod:
    """
    Quantum kernel method with SVM classification.
    
    Computes the quantum kernel matrix and uses it with a classical SVM.
    """
    
    def __init__(
        self,
        n_qubits: int,
        feature_map_type: str = 'zz',
        feature_map_reps: int = 1,
        noise_model: Optional[NoiseModel] = None,
        shots: int = 1024,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.feature_map_reps = feature_map_reps
        self.noise_model = noise_model
        self.shots = shots
        self.seed = seed
        
        # Create feature map
        self.feature_map = create_feature_map(
            feature_map_type=feature_map_type,
            n_qubits=n_qubits,
            reps=feature_map_reps,
        )
        
        # Set up the kernel
        self.kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            enforce_psd=True,
        )
        
        self.feature_map_info = get_feature_map_info(self.feature_map)
        self._kernel_matrix_cache = {}
        self._svm_model = None
        self._kernel_evaluation_cost = 0.0
    
    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the quantum kernel matrix K(X1, X2)."""
        if X2 is None:
            X2 = X1
        
        cache_key = f"k_{id(X1)}_{id(X2)}"
        
        if cache_key in self._kernel_matrix_cache:
            return self._kernel_matrix_cache[cache_key]
        
        start = time.time()
        K = self.kernel.evaluate(X1, X2)
        self._kernel_evaluation_cost += time.time() - start
        
        if cache_key is not None:
            self._kernel_matrix_cache[cache_key] = K
        
        return K
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the quantum kernel SVM."""
        K_train = self.compute_kernel_matrix(X_train)
        self._svm_model = SVC(kernel='precomputed', random_state=self.seed)
        self._svm_model.fit(K_train, y_train)
    
    def predict(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> dict:
        """Predict using precomputed kernel. Need K(test, train)."""
        if self._svm_model is None:
            raise RuntimeError("Call train() first.")
        
        # For precomputed kernel SVM, prediction needs K(test_i, train_j)
        # Shape: (n_test, n_train)
        K_cross = self.compute_kernel_matrix(X_test, X_train)
        y_pred = self._svm_model.predict(K_cross)
        
        result = {
            'predictions': list(y_pred),
            'n_trainable_params': 0,
            'n_circuit_evaluations': 0,
            'kernel_evaluation_cost': self._kernel_evaluation_cost,
        }
        
        if y_test is not None:
            result['metrics'] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
                'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
                'precision': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='macro')),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            }
        
        return result


def run_quantum_kernel_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config,
    dataset_name: str,
    noise_model=None,
) -> ExperimentResult:
    """Run one quantum kernel experiment."""
    start_time = time.time()
    
    # Accept dict or QuantumConfig
    if hasattr(config, 'n_qubits'):
        n_qubits = config.n_qubits
        fm_type = config.feature_map_type
        fm_depth = config.feature_map_depth
        shots = config.shots
        seed = config.seed
    else:
        n_qubits = config['n_qubits']
        fm_type = config.get('feature_map_type', 'zz')
        fm_depth = config.get('feature_map_depth', 1)
        shots = config.get('shots', 1024)
        seed = config.get('seed', 42)
    
    qkm = QuantumKernelMethod(
        n_qubits=n_qubits,
        feature_map_type=fm_type,
        feature_map_reps=fm_depth,
        noise_model=noise_model,
        shots=shots,
        seed=seed,
    )
    
    qkm.train(X_train, y_train)
    result_dict = qkm.predict(X_train, X_test, y_test)
    
    runtime = time.time() - start_time
    
    noise_level = None
    if noise_model is not None:
        try:
            noise_level = float(noise_model.noise_algorithms['depolarizing']()[0].probability)
        except Exception:
            pass
    
    return ExperimentResult(
        experiment_id=f"{dataset_name}_QK_{fm_type}_q{n_qubits}_d{fm_depth}",
        dataset_name=dataset_name,
        model_type='Quantum Kernel',
        n_features=X_train.shape[1],
        n_qubits=n_qubits,
        circuit_depth=fm_depth,
        accuracy=result_dict['metrics']['accuracy'],
        balanced_accuracy=result_dict['metrics']['balanced_accuracy'],
        macro_f1=result_dict['metrics']['macro_f1'],
        precision=result_dict['metrics']['precision'],
        recall=result_dict['metrics']['recall'],
        confusion_matrix=result_dict['metrics']['confusion_matrix'],
        runtime_seconds=runtime,
        n_trainable_params=0,
        n_circuit_evaluations=0,
        n_shots=shots,
        kernel_evaluation_cost=result_dict['kernel_evaluation_cost'],
        seed=seed,
        noise_level=noise_level,
        extra={'feature_map_type': fm_type, 'feature_map_info': qkm.feature_map_info},
    )
