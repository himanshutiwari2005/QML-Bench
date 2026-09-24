"""Noise model construction for QML-Bench."""

from typing import Optional, Dict
import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error


def create_depolarizing_noise_model(depolarizing_prob: float = 0.01) -> NoiseModel:
    """
    Create a noise model with depolarizing errors on all gates.
    
    Args:
        depolarizing_prob: Probability of depolarizing error per gate
    
    Returns:
        NoiseModel instance
    """
    noise_model = NoiseModel()
    
    # Add depolarizing error to all gates
    error = depolarizing_error(depolarizing_prob, 1)
    noise_model.add_all_qubit_quantum_error(error, ['id', 'rx', 'ry', 'rz', 'paulix', 'pauliy', 'pauliz'])
    noise_model.add_all_qubit_quantum_error(error, ['cx', 'cp'])
    
    return noise_model


def create_readout_noise_model(readout_error_prob: float = 0.05) -> NoiseModel:
    """
    Create a noise model with readout errors.
    
    Args:
        readout_error_prob: Probability of bit-flip during readout
    
    Returns:
        NoiseModel instance
    """
    noise_model = NoiseModel()
    
    # Add readout error
    error = depolarizing_error(readout_error_prob, 1)
    noise_model.add_all_qubit_quantum_error(error, ['measure'])
    
    return noise_model


def create_combined_noise_model(
    gate_error_prob: float = 0.01,
    readout_error_prob: float = 0.05,
    thermal_relaxation_t1: Optional[float] = None,
    thermal_relaxation_t2: Optional[float] = None,
    gate_time: float = 1e-9,
) -> NoiseModel:
    """
    Create a combined noise model with gate errors, readout errors, and optional
    thermal relaxation.
    
    Args:
        gate_error_prob: Depolarizing error probability for gates
        readout_error_prob: Readout error probability
        thermal_relaxation_t1: T1 relaxation time (seconds). If None, skip.
        thermal_relaxation_t2: T2 relaxation time (seconds). If None, skip.
        gate_time: Typical gate time in seconds
    
    Returns:
        NoiseModel instance
    """
    noise_model = NoiseModel()
    
    # Gate errors: depolarizing
    if gate_error_prob > 0:
        error = depolarizing_error(gate_error_prob, 1)
        single_qubit_gates = ['id', 'rx', 'ry', 'rz', 'paulix', 'pauliy', 'pauliz']
        two_qubit_gates = ['cx', 'cp']
        noise_model.add_all_qubit_quantum_error(error, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error, two_qubit_gates)
    
    # Readout errors
    if readout_error_prob > 0:
        error = depolarizing_error(readout_error_prob, 1)
        noise_model.add_all_qubit_quantum_error(error, ['measure'])
    
    # Thermal relaxation (if requested)
    if thermal_relaxation_t1 is not None and thermal_relaxation_t2 is not None:
        t1_error = thermal_relaxation_error(t1=thermal_relaxation_t1, t2=thermal_relaxation_t2, time=gate_time)
        noise_model.add_all_qubit_quantum_error(t1_error, ['id', 'rx', 'ry', 'rz', 'paulix', 'pauliy', 'pauliz', 'measure'])
    
    return noise_model


def get_noise_model_info(noise_model: NoiseModel) -> Dict:
    """Extract summary info about a noise model."""
    info = {
        'num_qubit_errors': len(noise_model.qubit_errors),
        'num_classical_errors': len(noise_model.classical_errors),
        'error_names': list(set(
            err.name for err, _, _ in noise_model.qubit_errors
        )),
    }
    
    if noise_model.noise_algorithms:
        dep = noise_model.noise_algorithms.get('depolarizing')
        if dep:
            info['depolarizing_probs'] = [float(d.prob) for d in dep]
    
    return info