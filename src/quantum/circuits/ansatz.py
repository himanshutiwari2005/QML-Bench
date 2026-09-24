
"""Parameterized ansatz circuit implementations."""

from typing import Optional
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
import numpy as np


def create_ansatz(
    ansatz_type: str = 'hardware_efficient',
    n_qubits: int = 2,
    reps: int = 1,
    entanglement: str = 'full',
    param_prefix: str = 'θ',
) -> QuantumCircuit:
    """
    Create a parameterized ansatz circuit.
    
    Args:
        ansatz_type: Type of ansatz ('hardware_efficient', 'real_amplitudes', 'efficient_su2', 'strong_entangler')
        n_qubits: Number of qubits
        reps: Number of repetitions (circuit depth)
        entanglement: Entanglement pattern ('full', 'linear', 'circular')
        param_prefix: Prefix for parameter names
    
    Returns:
        Parameterized QuantumCircuit
    """
    if ansatz_type == 'hardware_efficient':
        # RX-RZ-RX rotation layers with CNOT entanglers
        qc = QuantumCircuit(n_qubits)
        
        for _ in range(reps):
            # Rotation layer
            for i in range(n_qubits):
                qc.rx(Parameter(f'{param_prefix}_rx_{_}_{i}'), i)
                qc.rz(Parameter(f'{param_prefix}_rz_{_}_{i}'), i)
                qc.rx(Parameter(f'{param_prefix}_rx2_{_}_{i}'), i)
            
            # Entanglement layer
            if entanglement == 'full':
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qc.cx(i, j)
            elif entanglement == 'linear':
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
            elif entanglement == 'circular':
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
                qc.cx(n_qubits - 1, 0)
        
        return qc
    
    elif ansatz_type == 'real_amplitudes':
        return RealAmplitudes(
            num_qubits=n_qubits,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=param_prefix,
        )
    
    elif ansatz_type == 'efficient_su2':
        return EfficientSU2(
            num_qubits=n_qubits,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=param_prefix,
        )
    
    elif ansatz_type == 'strong_entangler':
        return StrongEntangler(
            num_qubits=n_qubits,
            reps=reps,
        )
    
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")


def get_ansatz_info(ansatz: QuantumCircuit) -> dict:
    """Extract information about an ansatz circuit."""
    return {
        'num_qubits': ansatz.num_qubits,
        'num_parameters': ansatz.num_parameters,
        'depth': ansatz.depth(),
        'num_gates': ansatz.size(),
        'num_cx_gates': sum(1 for inst in ansatz.data if inst.operation.name == 'cx'),
        'type': type(ansatz).__name__,
    }


def count_trainable_parameters(ansatz: QuantumCircuit) -> int:
    """Count the number of trainable parameters in an ansatz."""
    return ansatz.num_parameters
