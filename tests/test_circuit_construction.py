"""Tests for quantum circuit construction."""
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from src.quantum.feature_maps.factory import create_feature_map


def test_zz_feature_map_basic():
    fm = create_feature_map(feature_map_type="zz", n_qubits=2, reps=1)
    assert fm.num_qubits == 2
    assert fm.num_parameters == 2  # x0, x1


def test_feature_map_dimension_match():
    """Test feature map with different qubit counts."""
    for nq in [2, 4, 8]:
        fm = create_feature_map(feature_map_type="zz", n_qubits=nq, reps=1)
        assert fm.num_qubits == nq
