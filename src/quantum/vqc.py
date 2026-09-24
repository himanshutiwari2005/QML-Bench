"""VQC experiment runner using Qiskit ML's EstimatorQNN."""

import time
import numpy as np
from typing import Optional
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import SPSA, COBYLA

from src.quantum.feature_maps.factory import create_feature_map
from src.quantum.circuits.ansatz import create_ansatz, get_ansatz_info
from src.utils.config import QuantumConfig, ExperimentResult


def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config,
    dataset_name: str,
    n_classes: int = 2,
) -> ExperimentResult:
    """Run a VQC experiment and return results."""
    start_time = time.time()
    
    # Accept dict or QuantumConfig
    if hasattr(config, 'n_qubits'):
        n_qubits = config.n_qubits
        fm_type = config.feature_map_type
        fm_depth = config.feature_map_depth
        ansatz_type = config.ansatz_type
        ansatz_depth = config.ansatz_depth
        optimizer_name = config.optimizer
        opt_kwargs = config.optimizer_kwargs or {}
        shots = config.shots
        seed = config.seed
    else:
        n_qubits = config['n_qubits']
        fm_type = config.get('feature_map_type', 'zz')
        fm_depth = config.get('feature_map_depth', 1)
        ansatz_type = config.get('ansatz_type', 'hardware_efficient')
        ansatz_depth = config.get('ansatz_depth', 1)
        optimizer_name = config.get('optimizer', 'COBYLA')
        opt_kwargs = config.get('optimizer_kwargs', {})
        shots = config.get('shots', 1024)
        seed = config.get('seed', 42)
    
    # Build feature map
    feature_map = create_feature_map(
        feature_map_type=fm_type,
        n_qubits=n_qubits,
        reps=fm_depth,
    )
    
    # Build ansatz
    ansatz = create_ansatz(
        ansatz_type=ansatz_type,
        n_qubits=n_qubits,
        reps=ansatz_depth,
    )
    
    # Combine: feature_map -> ansatz
    full_circuit = QuantumCircuit(n_qubits)
    full_circuit.compose(feature_map, inplace=True)
    full_circuit.compose(ansatz, inplace=True)
    
    # Observables using from_list to match number of qubits
    if n_classes == 2:
        obs_list = [SparsePauliOp.from_list([('Z', 1)], num_qubits=n_qubits)]
    else:
        obs_list = [
            SparsePauliOp.from_list([(f'Z{i}', 1)], num_qubits=n_qubits)
            for i in range(min(n_classes, n_qubits))
        ]
    
    # Build QNN
    qnn = EstimatorQNN(
        circuit=full_circuit,
        estimator=StatevectorEstimator(),
        observables=obs_list[0] if n_classes == 2 else obs_list,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    
    from scipy.optimize import minimize
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    n_params = len(ansatz.parameters)
    
    def unpack_weights(vec):
        return {p: v for p, v in zip(ansatz.parameters, vec)}
    
    def vqc_loss(weights_vec):
        """Classification loss using QNN predictions."""
        weights_dict = unpack_weights(weights_vec)
        preds = qnn.forward(X_train, weights_dict)
        
        if n_classes == 2:
            preds_binary = (preds > 0).astype(int).flatten()
            loss = np.mean((preds_binary - y_train_enc) ** 2)
        else:
            preds_class = np.argmax(preds, axis=1)
            loss = np.mean((preds_class - y_train_enc) ** 2)
        
        return float(loss)
    
    # Initial weights
    np.random.seed(seed)
    initial_weights = np.random.randn(n_params) * 0.1
    
    # Optimize
    opt_start = time.time()
    maxiter = opt_kwargs.get('maxiter', 50)
    
    if optimizer_name == 'SPSA':
        try:
            optimizer = SPSA(maxiter=maxiter)
            opt_result = optimizer.minimize(vqc_loss, x0=initial_weights)
        except Exception:
            optimizer = COBYLA(maxiter=maxiter)
            opt_result = optimizer.minimize(vqc_loss, x0=initial_weights)
    else:
        optimizer = COBYLA(maxiter=maxiter)
        opt_result = optimizer.minimize(vqc_loss, x0=initial_weights)
    
    opt_time = time.time() - opt_start
    
    best_weights = opt_result.x
    
    # Evaluate on test set
    test_preds = qnn.forward(X_test, unpack_weights(best_weights))
    
    if n_classes == 2:
        test_preds_binary = (test_preds > 0).astype(int).flatten()
    else:
        test_preds_binary = np.argmax(test_preds, axis=1)
    
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )
    
    metrics = {
        'accuracy': float(accuracy_score(y_test_enc, test_preds_binary)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test_enc, test_preds_binary)),
        'macro_f1': float(f1_score(y_test_enc, test_preds_binary, average='macro')),
        'precision': float(precision_score(y_test_enc, test_preds_binary, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test_enc, test_preds_binary, average='macro')),
        'confusion_matrix': confusion_matrix(y_test_enc, test_preds_binary).tolist(),
    }
    
    runtime = time.time() - start_time
    
    ansatz_info = get_ansatz_info(ansatz)
    
    return ExperimentResult(
        experiment_id=f"{dataset_name}_VQC_{fm_type}_q{n_qubits}_d{ansatz_depth}",
        dataset_name=dataset_name,
        model_type='VQC',
        n_features=X_train.shape[1],
        n_qubits=n_qubits,
        circuit_depth=ansatz_depth,
        accuracy=metrics['accuracy'],
        balanced_accuracy=metrics['balanced_accuracy'],
        macro_f1=metrics['macro_f1'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        confusion_matrix=metrics['confusion_matrix'],
        runtime_seconds=runtime,
        n_trainable_params=n_params,
        n_circuit_evaluations=getattr(opt_result, 'nfev', 0),
        n_shots=shots,
        kernel_evaluation_cost=opt_time,
        seed=seed,
        extra={
            'feature_map_type': fm_type,
            'feature_map_depth': fm_depth,
            'ansatz_type': ansatz_type,
            'ansatz_depth': ansatz_depth,
            'optimizer': optimizer_name,
            'ansatz_info': ansatz_info,
        },
    )
