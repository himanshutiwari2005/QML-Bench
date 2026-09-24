#!/usr/bin/env python
"""
QML-Bench: Quantum-Classical Machine Learning Benchmark

A rigorous, scientifically honest comparison of quantum and classical ML
on small and medium-scale classification problems.

Usage:
    python run_benchmark.py                    # Run default benchmark
    python run_benchmark.py --dataset iris     # Run on Iris
    python run_benchmark.py --noise-study      # Run noise sensitivity study
    python run_benchmark.py --plots            # Generate all plots from saved results
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from experiments.benchmark import (
    QMLBenchmark, create_default_config,
)
from visualization.plots import generate_all_plots
from noise.models import create_combined_noise_model


def main():
    parser = argparse.ArgumentParser(description='QML-Bench: Quantum-Classical ML Benchmark')
    parser.add_argument('--dataset', type=str, default='iris',
                       choices=['iris', 'breast_cancer', 'linearly_separable', 'moons'],
                       help='Dataset to use')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file')
    parser.add_argument('--noise-study', action='store_true',
                       help='Run noise sensitivity study instead of full benchmark')
    parser.add_argument('--plots', action='store_true',
                       help='Generate all plots from saved results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory for saving results')
    parser.add_argument('--seeds', type=int, default=1,
                       help='Number of random seeds for repetition')
    parser.add_argument('--quick', action='store_true',
                       help='Run a quick smoke test with minimal configurations')
    
    args = parser.parse_args()
    
    benchmark = QMLBenchmark(results_dir=args.results_dir)
    
    if args.plots:
        print("Generating plots from saved results...")
        generate_all_plots(
            results_path=os.path.join(args.results_dir, 'all_results.json'),
            output_dir=os.path.join(args.results_dir, 'figures'),
        )
        return
    
    if args.config:
        import json
        with open(args.config) as f:
            config_dict = json.load(f)
        # Reconstruct config from dict
        from utils.config import (
            DatasetConfig, ClassicalConfig, QuantumConfig, ExperimentConfig,
        )
        config = ExperimentConfig(
            dataset=DatasetConfig(**config_dict['dataset']),
            classical=[ClassicalConfig(**c) for c in config_dict['classical']],
            quantum=[QuantumConfig(**q) for q in config_dict['quantum']],
            n_seeds=config_dict.get('n_seeds', 1),
        )
    else:
        config = create_default_config()
    
    # Override dataset
    config.dataset.name = args.dataset
    
    if args.quick:
        print("Running QUICK smoke test...")
        config.classical = config.classical[:2]  # Just LogReg and SVM
        config.quantum = [config.quantum[0]] if config.quantum else []  # Just first config
        config.n_seeds = 1
    
    if args.noise_study:
        print("Running noise sensitivity study...")
        # Create a specific config for noise study
        noise_config = QuantumConfig(
            n_qubits=2,
            feature_map_type='zz',
            feature_map_depth=1,
            ansatz_type='hardware_efficient',
            ansatz_depth=1,
            optimizer='COBYLA',
            optimizer_kwargs={'maxiter': 30},
            shots=1024,
            seed=42,
        )
        
        # Load just the dataset
        from datasets import load_iris_dataset
        data = load_iris_dataset(n_features=4)
        
        # Preprocess
        from utils.preprocessing import preprocess_pipeline
        X_tr, X_te, _, _, _ = preprocess_pipeline(
            data['X_train'], data['X_test'], n_components=2
        )
        
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
        benchmark.run_noise_study(
            X_tr, data['y_train'],
            X_te, data['y_test'],
            noise_config,
            f"iris_noise",
            n_classes=3,
            noise_levels=noise_levels,
        )
    else:
        print(f"Starting benchmark: {config.dataset.name}")
        benchmark.run_full_benchmark(config)
    
    # Save CSV export
    benchmark.save_results_csv()
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    from evaluation.metrics import aggregate_results
    agg = aggregate_results(benchmark.all_results)
    
    for key, data in sorted(agg.items()):
        print(f"\n{key}:")
        print(f"  Accuracy:  {data.get('accuracy_mean', 0):.3f} ± {data.get('accuracy_std', 0):.3f}")
        print(f"  F1 Score:  {data.get('macro_f1_mean', 0):.3f} ± {data.get('macro_f1_std', 0):.3f}")
        print(f"  Runtime:   {data.get('runtime_mean', 0):.2f}s")
    
    print(f"\nTotal experiments: {len(benchmark.all_results)}")
    print(f"Results saved to: {args.results_dir}/")


if __name__ == '__main__':
    main()
