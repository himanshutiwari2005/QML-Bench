"""Tests for configuration dataclasses."""
from src.utils.config import (
    DatasetConfig, ClassicalConfig, QuantumConfig,
    ExperimentConfig, ExperimentResult, result_to_dict, save_results, load_results
)
import os
import json


def test_dataset_config():
    cfg = DatasetConfig(name="test", n_samples=100, n_features=4)
    assert cfg.name == "test"
    assert cfg.n_samples == 100


def test_classical_config():
    cfg = ClassicalConfig(model_type="logreg", hyperparameters={"C": 1.0})
    assert cfg.model_type == "logreg"


def test_quantum_config():
    cfg = QuantumConfig(n_qubits=2, feature_map_type="zz")
    assert cfg.n_qubits == 2
    assert cfg.feature_map_type == "zz"


def test_result_serialization():
    result = ExperimentResult(
        experiment_id="test_exp",
        dataset_name="iris",
        model_type="logreg",
        n_features=4,
        accuracy=0.95,
        balanced_accuracy=0.94,
        macro_f1=0.93,
        precision=0.92,
        recall=0.91,
        runtime_seconds=0.5,
        seed=42,
    )
    d = result_to_dict(result)
    assert d["accuracy"] == 0.95
    assert d["experiment_id"] == "test_exp"
    
    path = "results/test_result.json"
    save_results([result], path)
    assert os.path.exists(path)
    
    loaded = load_results(path)
    assert len(loaded) == 1
    assert loaded[0].accuracy == 0.95
    
    os.remove(path)
