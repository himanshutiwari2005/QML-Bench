
"""Feature map factory for creating configurable quantum feature maps."""

from typing import Optional
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit.circuit import QuantumCircuit
import numpy as np


def create_feature_map(
    feature_map_type: str = 'zz',
    n_qubits: int = 2,
    reps: int = 1,
    entropy: float = 0.5,
    pauli_block: str = 'ZZ',
) -> QuantumCircuit:
    """
    Create a configurable quantum feature map.
    
    Args:
        feature_map_type: Type of feature map ('zz', 'pauli', 'z', 'linear')
        n_qubits: Number of qubits (must match feature dimension)
        reps: Number of repetitions (circuit depth)
        entropy: Entropy parameter for PauliFeatureMap
        pauli_block: Pauli block string for PauliFeatureMap
    
    Returns:
        Configured QuantumCircuit feature map
    """
    if feature_map_type == 'zz':
        return ZZFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement='full',
        )
    
    elif feature_map_type == 'pauli':
        return PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement='full',
            pauli_block=pauli_block,
            entropy=entropy,
        )
    
    elif feature_map_type == 'z':
        return ZFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement='linear',
        )
    
    elif feature_map_type == 'linear':
        # Simple linear (angle) encoding: one RX per qubit
        from qiskit.circuit.library import PauliFeatureMap
        # Use PauliFeatureMap with minimal entanglement as a linear-like map
        return PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement='linear',
            pauli_block='X',
            entropy=0.0,
        )
    
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}. "
                         f"Choose from 'zz', 'pauli', 'z', 'linear'.")


def get_feature_map_info(feature_map) -> dict:
    """Extract information about a feature map."""
    return {
        'num_qubits': feature_map.num_qubits,
        'num_parameters': feature_map.num_parameters,
        'depth': feature_map.depth(),
        'num_gates': feature_map.size(),
        'type': type(feature_map).__name__,
    }
