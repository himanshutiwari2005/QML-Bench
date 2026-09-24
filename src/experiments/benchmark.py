"""Experiment orchestration: configuration-driven benchmark runner."""

import time
import os
import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from src.utils.config import (
    DatasetConfig, ClassicalConfig, QuantumConfig,
    ExperimentConfig, ExperimentResult, save_results, load_results,
)
from src.utils.preprocessing import preprocess_pipeline, compute_pca_explained_variance
from src.utils.version import get_versions, save_versions
from src.datasets import (
    generate_linearly_separable, generate_moons,
    load_iris_dataset, load_breast_cancer_dataset,
)
from src.classical.benchmark import run_classical_benchmark
from src.quantum.kernels.quantum_kernel import run_quantum_kernel_experiment
from src.quantum.vqc import train_vqc
from src.noise.models import (
    create_combined_noise_model, get_noise_model_info,
)


class QMLBenchmark:
    """
    Main benchmark orchestrator.
    
    Manages the full experiment lifecycle: data loading, preprocessing,
    classical baselines, quantum methods, noise studies, and result serialization.
    """
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: List[ExperimentResult] = []
        
        # Save environment versions
        versions = get_versions()
        save_versions(self.results_dir / 'versions.json')
        self._log(f"Environment: {json.dumps(versions, indent=2)}")
    
    def _log(self, msg: str):
        """Log a message with timestamp."""
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {msg}")
    
    def load_dataset(self, dataset_name: str, config: DatasetConfig) -> Dict:
        """Load a dataset by name."""
        if dataset_name == 'linearly_separable':
            data = generate_linearly_separable(
                n_samples=config.n_samples,
                n_features=config.n_features,
                n_informative=config.n_informative,
                class_sep=config.class_sep,
                random_seed=config.random_seed,
                train_ratio=config.train_ratio,
            )
        elif dataset_name == 'moons':
            data = generate_moons(
                n_samples=config.n_samples,
                noise=config.noise,
                random_seed=config.random_seed,
                train_ratio=config.train_ratio,
            )
        elif dataset_name == 'iris':
            data = load_iris_dataset(
                n_features=config.n_features,
                train_ratio=config.train_ratio,
                random_seed=config.random_seed,
            )
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer_dataset(
                train_ratio=config.train_ratio,
                random_seed=config.random_seed,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self._log(f"Loaded dataset '{dataset_name}': {data['metadata']['n_samples']} samples, "
                  f"{data['metadata']['n_features']} features, "
                  f"{data['metadata']['n_classes']} classes")
        return data
    
    def run_classical_experiments(
        self,
        data: Dict,
        classical_configs: List[ClassicalConfig],
        dataset_name: str,
    ) -> List[ExperimentResult]:
        """Run all classical baselines on a dataset."""
        results = run_classical_benchmark(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test'],
            classical_configs,
            dataset_name,
        )
        self.all_results.extend(results)
        for r in results:
            self._log(f"  Classical [{r.model_type}]: acc={r.accuracy:.3f}, "
                      f"f1={r.macro_f1:.3f}, time={r.runtime_seconds:.2f}s")
        return results
    
    def run_quantum_experiments(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        quantum_configs: List[QuantumConfig],
        dataset_name: str,
        n_classes: int,
        noise_model: Optional = None,
    ) -> List[ExperimentResult]:
        """Run quantum kernel and VQC experiments."""
        results = []
        
        for qc in quantum_configs:
            # Quantum Kernel
            self._log(f"  Running Quantum Kernel: n_qubits={qc.n_qubits}, "
                      f"map={qc.feature_map_type}, depth={qc.feature_map_depth}")
            try:
                qr = run_quantum_kernel_experiment(
                    X_train, y_train, X_test, y_test,
                    qc, dataset_name, noise_model,
                )
                results.append(qr)
                self._log(f"    QK: acc={qr.accuracy:.3f}, f1={qr.macro_f1:.3f}, "
                          f"time={qr.runtime_seconds:.2f}s")
            except Exception as e:
                self._log(f"    QK FAILED: {e}")
            
            # VQC
            if n_classes <= qc.n_qubits:
                self._log(f"  Running VQC: n_qubits={qc.n_qubits}, "
                          f"ansatz={qc.ansatz_type}, depth={qc.ansatz_depth}")
                try:
                    vr = train_vqc(
                        X_train, y_train, X_test, y_test,
                        qc, dataset_name, n_classes,
                    )
                    results.append(vr)
                    self._log(f"    VQC: acc={vr.accuracy:.3f}, f1={vr.macro_f1:.3f}, "
                              f"time={vr.runtime_seconds:.2f}s")
                except Exception as e:
                    self._log(f"    VQC FAILED: {e}")
        
        self.all_results.extend(results)
        return results
    
    def run_full_benchmark(self, config: ExperimentConfig):
        """
        Run a complete benchmark with all configured experiments.
        
        This is the main entry point for a full benchmark run.
        """
        self._log(f"Starting benchmark: {config.dataset.name}")
        self._log(f"  Dataset: {config.dataset.n_samples} samples, "
                  f"{config.dataset.n_features} features")
        self._log(f"  Classical models: {[c.model_type for c in config.classical]}")
        self._log(f"  Quantum configs: {len(config.quantum)} configurations")
        self._log(f"  Seeds: {config.n_seeds}")
        
        # Load dataset
        data = self.load_dataset(config.dataset.name, config.dataset)
        
        # Preprocess
        n_features = config.dataset.n_features
        
        for seed_idx in range(config.n_seeds):
            actual_seed = config.dataset.random_seed + seed_idx
            self._log(f"\n--- Seed {seed_idx + 1}/{config.n_seeds} (seed={actual_seed}) ---")
            
            # Reload with different seed
            ds_config = DatasetConfig(
                name=config.dataset.name,
                n_samples=config.dataset.n_samples,
                n_features=config.dataset.n_features,
                n_classes=config.dataset.n_classes,
                train_ratio=config.dataset.train_ratio,
                random_seed=actual_seed,
                noise=config.dataset.noise,
            )
            data_seed = self.load_dataset(config.dataset.name, ds_config)
            
            for n_components in [None, min(4, n_features), min(2, n_features)]:
                pca_label = f"PCA{n_components}" if n_components else "Full"
                self._log(f"  Encoding: {pca_label} ({n_features} -> {n_components or n_features} dims)")
                
                X_tr, X_te, scaler, pca, rs = preprocess_pipeline(
                    data_seed['X_train'], data_seed['X_test'],
                    n_components=n_components,
                )
                
                # PCA variance info
                if pca is not None:
                    var_ratio = compute_pca_explained_variance(
                        data_seed['X_train'], n_components
                    )
                    self._log(f"    PCA variance explained: {var_ratio:.3f}")
                
                # Classical baselines
                self.run_classical_experiments(
                    {'X_train': X_tr, 'y_train': data_seed['y_train'],
                     'X_test': X_te, 'y_test': data_seed['y_test']},
                    config.classical,
                    f"{config.dataset.name}_{pca_label}_s{actual_seed}",
                )
                
                # Quantum experiments
                n_classes = len(np.unique(data_seed['y_train']))
                
                for qc in config.quantum:
                    if qc.n_qubits <= (n_components or n_features):
                        self.run_quantum_experiments(
                            X_tr, data_seed['y_train'],
                            X_te, data_seed['y_test'],
                            [qc],
                            f"{config.dataset.name}_{pca_label}_s{actual_seed}",
                            n_classes,
                            noise_model=None,  # Add noise study separately
                        )
        
        # Save results
        results_path = self.results_dir / 'all_results.json'
        save_results(self.all_results, str(results_path))
        self._log(f"\nSaved {len(self.all_results)} results to {results_path}")
    
    def run_noise_study(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        base_config: QuantumConfig,
        dataset_name: str,
        n_classes: int,
        noise_levels: List[float] = None,
    ) -> List[ExperimentResult]:
        """Run noise sensitivity experiments."""
        if noise_levels is None:
            noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
        
        self._log(f"\n--- Noise Study: {dataset_name} ---")
        results = []
        
        for noise_level in noise_levels:
            self._log(f"  Noise level: {noise_level}")
            
            if noise_level > 0:
                noise_model = create_combined_noise_model(
                    gate_error_prob=noise_level,
                    readout_error_prob=noise_level * 0.5,
                )
            else:
                noise_model = None
            
            for model_type in ['Quantum Kernel', 'VQC']:
                try:
                    if model_type == 'Quantum Kernel':
                        qr = run_quantum_kernel_experiment(
                            X_train, y_train, X_test, y_test,
                            base_config, dataset_name, noise_model,
                        )
                        qr.noise_level = noise_level
                        results.append(qr)
                        self._log(f"    QK: acc={qr.accuracy:.3f}")
                    else:
                        vr = train_vqc(
                            X_train, y_train, X_test, y_test,
                            base_config, dataset_name, n_classes,
                        )
                        vr.noise_level = noise_level
                        results.append(vr)
                        self._log(f"    VQC: acc={vr.accuracy:.3f}")
                except Exception as e:
                    self._log(f"    {model_type} FAILED: {e}")
        
        self.all_results.extend(results)
        return results
    
    def save_results_csv(self, path: str = None):
        """Export results to CSV for analysis."""
        if path is None:
            path = self.results_dir / 'all_results.csv'
        
        import csv
        
        if not self.all_results:
            self._log("No results to save.")
            return
        
        fieldnames = list(self.all_results[0].__dict__.keys())
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.all_results:
                writer.writerow(r.__dict__)
        
        self._log(f"Saved CSV to {path}")


def create_default_config() -> ExperimentConfig:
    """Create a sensible default experiment configuration."""
    return ExperimentConfig(
        dataset=DatasetConfig(
            name='iris',
            n_samples=150,
            n_features=4,
            n_classes=3,
            train_ratio=0.8,
            random_seed=42,
        ),
        classical=[
            ClassicalConfig(model_type='logreg', cv_folds=5),
            ClassicalConfig(model_type='svm', hyperparameters={'kernel': 'linear'}, cv_folds=5),
            ClassicalConfig(model_type='svm', hyperparameters={'kernel': 'rbf'}, cv_folds=5),
            ClassicalConfig(model_type='rf', cv_folds=5),
            ClassicalConfig(model_type='mlp', cv_folds=5),
        ],
        quantum=[
            QuantumConfig(n_qubits=2, feature_map_type='zz', feature_map_depth=1,
                         ansatz_type='hardware_efficient', ansatz_depth=1,
                         optimizer='COBYLA', optimizer_kwargs={'maxiter': 50}),
            QuantumConfig(n_qubits=2, feature_map_type='zz', feature_map_depth=2,
                         ansatz_type='hardware_efficient', ansatz_depth=2,
                         optimizer='COBYLA', optimizer_kwargs={'maxiter': 50}),
            QuantumConfig(n_qubits=4, feature_map_type='zz', feature_map_depth=1,
                         ansatz_type='hardware_efficient', ansatz_depth=1,
                         optimizer='COBYLA', optimizer_kwargs={'maxiter': 50}),
        ],
        n_seeds=1,
    )
