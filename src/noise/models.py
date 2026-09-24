"""Noise model construction for QML-Bench."""

from typing import Optional, Dict
import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error


def create_depolarizing_noise_model(depolarizing_prob: float = 0.01) -> NoiseModel:
    """Create a noise model with depolarizing errors on all gates."""
    noise_model = NoiseModel()
    error = depolarizing_error(depolarizing_prob, 1)
    noise_model.add_all_qubit_quantum_error(error, ['id', 'rx', 'ry', 'rz', 'paulix', 'pauliy', 'pauliz'])
    noise_model.add_all_qubit_quantum_error(error, ['cx', 'cp'])
    return noise_model


def create_readout_noise_model(readout_error_prob: float = 0.05) -> NoiseModel:
    """Create a noise model with readout errors."""
    noise_model = NoiseModel()
    error = depolarizing_error(readout_error_prob, 1)
    noise_model.add_all_qubit_quantum_error(error, ['measure'])
    return noise_model


def create_combined_noise_model(
    gate_error_prob: float = 0.01,
    readout_error_prob: float = 0.05,
) -> NoiseModel:
    """Create a combined noise model with gate errors and readout errors."""
    noise_model = NoiseModel()
    
    if gate_error_prob > 0:
        # 1-qubit depolarizing error for single-qubit gates
        error1 = depolarizing_error(gate_error_prob, 1)
        single_gates = ['id', 'rx', 'ry', 'rz', 'paulix', 'pauliy', 'pauliz']
        noise_model.add_all_qubit_quantum_error(error1, single_gates)
        
        # 2-qubit depolarizing error for 2-qubit gates
        if gate_error_prob <= 0.25:  # Valid range for 2-qubit depolarizing
            error2 = depolarizing_error(gate_error_prob, 2)
            two_qubit_gates = ['cx', 'cp']
            noise_model.add_all_qubit_quantum_error(error2, two_qubit_gates)
    
    if readout_error_prob > 0:
        error_meas = depolarizing_error(readout_error_prob, 1)
        noise_model.add_all_qubit_quantum_error(error_meas, ['measure'])
    
    return noise_model


def get_noise_model_info(noise_model: NoiseModel) -> Dict:
    """Extract summary info about a noise model."""
    try:
        n_quantum = len(noise_model._local_quantum_errors())
    except Exception:
        n_quantum = 0
    try:
        n_all = len(noise_model._all_qubit_quantum_errors_equal())
    except Exception:
        n_all = 0
    try:
        n_readout = len(noise_model._local_readout_errors())
    except Exception:
        n_readout = 0
    return {
        'num_qubit_errors': n_quantum + n_all,
        'num_classical_errors': n_readout,
    }