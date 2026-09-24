"""Visualization module: generate all required plots from saved results."""

import os
import json
from typing import List, Optional
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.utils.config import ExperimentResult, load_results
from src.evaluation.metrics import aggregate_results


def _style_axes(ax):
    """Apply consistent styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')
    ax.grid(axis='y', alpha=0.3)


def plot_model_comparison(
    results: List[ExperimentResult],
    output_path: str,
    group_by: str = 'model_type',
):
    """
    Plot model performance comparison (bar chart with error bars).
    
    Compares accuracy across all models for each dataset.
    """
    agg = aggregate_results(results)
    
    datasets = sorted(set(v['dataset'] for v in agg.values()))
    models = sorted(set(v['model'] for v in agg.values()))
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for ax, dataset in zip(axes, datasets):
        dataset_results = {m: v for m, v in agg.items() if v['dataset'] == dataset}
        
        x = np.arange(len(models))
        means = [dataset_results.get(m, {}).get('accuracy_mean', 0) for m in models]
        stds = [dataset_results.get(m, {}).get('accuracy_std', 0) for m in models]
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.tab10.colors,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset}')
        ax.set_ylim(0, 1.05)
        _style_axes(ax)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison plot: {output_path}")


def plot_performance_vs_qubits(
    results: List[ExperimentResult],
    output_path: str,
):
    """Plot accuracy vs number of qubits for quantum models."""
    quantum_results = [r for r in results if r.n_qubits is not None]
    
    if not quantum_results:
        print("No quantum results for qubit plot.")
        return
    
    datasets = sorted(set(r.dataset_name for r in quantum_results))
    models = sorted(set(r.model_type for r in quantum_results))
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10.colors
    
    for ax, dataset in zip(axes, datasets):
        for i, model in enumerate(models):
            model_results = [r for r in quantum_results
                           if r.model_type == model and r.dataset_name == dataset]
            
            if not model_results:
                continue
            
            qubits = sorted(set(r.n_qubits for r in model_results))
            means = []
            stds = []
            for q in qubits:
                q_results = [r for r in model_results if r.n_qubits == q]
                means.append(np.mean([r.accuracy for r in q_results]))
                stds.append(np.std([r.accuracy for r in q_results]))
            
            ax.errorbar(qubits, means, yerr=stds, marker='o', markersize=8,
                       linewidth=2, label=model, color=colors[i % 10],
                       capsize=5, capthick=1)
        
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Accuracy')
        ax.set_title(dataset)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        _style_axes(ax)
        ax.set_xscale('log')
    
    plt.suptitle('Performance vs Qubit Count', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved qubit comparison plot: {output_path}")


def plot_performance_vs_depth(
    results: List[ExperimentResult],
    output_path: str,
):
    """Plot accuracy vs circuit depth for quantum models."""
    quantum_results = [r for r in results if r.circuit_depth is not None]
    
    if not quantum_results:
        print("No quantum results for depth plot.")
        return
    
    datasets = sorted(set(r.dataset_name for r in quantum_results))
    models = sorted(set(r.model_type for r in quantum_results))
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10.colors
    
    for ax, dataset in zip(axes, datasets):
        for i, model in enumerate(models):
            model_results = [r for r in quantum_results
                           if r.model_type == model and r.dataset_name == dataset]
            
            if not model_results:
                continue
            
            depths = sorted(set(r.circuit_depth for r in model_results))
            means = []
            stds = []
            for d in depths:
                d_results = [r for r in model_results if r.circuit_depth == d]
                means.append(np.mean([r.accuracy for r in d_results]))
                stds.append(np.std([r.accuracy for r in d_results]))
            
            if means:
                ax.errorbar(depths, means, yerr=stds, marker='s', markersize=8,
                           linewidth=2, label=model, color=colors[i % 10],
                           capsize=5, capthick=1)
        
        ax.set_xlabel('Circuit Depth')
        ax.set_ylabel('Accuracy')
        ax.set_title(dataset)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        _style_axes(ax)
    
    plt.suptitle('Performance vs Circuit Depth', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved depth comparison plot: {output_path}")


def plot_performance_vs_noise(
    results: List[ExperimentResult],
    output_path: str,
):
    """Plot accuracy vs noise level."""
    noisy_results = [r for r in results if r.noise_level is not None]
    
    if not noisy_results:
        print("No noisy results for noise plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = sorted(set(r.model_type for r in noisy_results))
    colors = plt.cm.tab10.colors
    
    for i, model in enumerate(models):
        model_results = [r for r in noisy_results if r.model_type == model]
        
        noise_levels = sorted(set(r.noise_level for r in model_results))
        means = []
        stds = []
        for nl in noise_levels:
            nl_results = [r for r in model_results if r.noise_level == nl]
            means.append(np.mean([r.accuracy for r in nl_results]))
            stds.append(np.std([r.accuracy for r in nl_results]))
        
        if means:
            ax.errorbar(noise_levels, means, yerr=stds, marker='o', markersize=8,
                       linewidth=2, label=model, color=colors[i % 10],
                       capsize=5, capthick=1)
    
    ax.set_xlabel('Noise Level (depolarizing probability)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance vs Noise Level')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xscale('log')
    _style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved noise plot: {output_path}")


def plot_runtime_comparison(
    results: List[ExperimentResult],
    output_path: str,
):
    """Plot runtime vs model type."""
    agg = aggregate_results(results)
    
    models = sorted(set(v['model'] for v in agg.values()))
    runtimes = [agg.get(m, {}).get('runtime_mean', 0) for m in models]
    runtime_stds = [agg.get(m, {}).get('runtime_std', 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    bars = ax.bar(x, runtimes, yerr=runtime_stds, capsize=5,
                  color=plt.cm.tab10.colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison by Model')
    _style_axes(ax)
    
    # Log scale if needed
    if max(runtimes) > 100:
        ax.set_yscale('log')
    
    for bar, rt in zip(bars, runtimes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{rt:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved runtime comparison plot: {output_path}")


def plot_kernel_matrix(
    kernel_matrix: np.ndarray,
    output_path: str,
    title: str = 'Quantum Kernel Matrix',
):
    """Plot a kernel matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(kernel_matrix, cmap='viridis', norm=LogNorm(),
                   aspect='auto', interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Kernel Value')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved kernel matrix plot: {output_path}")


def plot_confusion_matrix(
    confusion_matrix_data: List[List[int]],
    class_names: Optional[List[str]] = None,
    output_path: str = None,
    title: str = 'Confusion Matrix',
):
    """Plot a confusion matrix."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    
    if output_path is None:
        output_path = 'results/confusion_matrix.png'
    
    cm = np.array(confusion_matrix_data)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count')
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    ax.set(xticks=range(n_classes),
           yticks=range(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations
    threshold = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                   color='white' if cm[i, j] > threshold else 'black',
                   fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {output_path}")
    return fig


def plot_vqc_convergence(
    history: List[dict],
    output_path: str,
):
    """Plot VQC training convergence curve."""
    losses = [h['loss'] for h in history]
    iterations = list(range(len(losses)))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, losses, marker='o', markersize=3, linewidth=1.5, color='steelblue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('VQC Training Convergence')
    ax.grid(alpha=0.3)
    _style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved VQC convergence plot: {output_path}")


def plot_pca_variance(
    explained_variance_ratios: List[float],
    n_features: int,
    output_path: str,
):
    """Plot PCA explained variance curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    components = range(1, len(explained_variance_ratios) + 1)
    cumulative = np.cumsum(explained_variance_ratios)
    
    ax.plot(components, explained_variance_ratios, 'o-', label='Individual')
    ax.plot(components, cumulative, 's-', label='Cumulative', color='red')
    ax.axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    ax.legend()
    ax.set_xticks(components)
    _style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA variance plot: {output_path}")


def plot_scaling_curves(
    results: List[ExperimentResult],
    output_path: str,
):
    """Plot scaling behavior: runtime vs number of features/qubits."""
    quantum_results = [r for r in results if r.n_qubits is not None]
    
    if not quantum_results:
        print("No quantum results for scaling plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Runtime vs qubits
    ax = axes[0]
    for model in sorted(set(r.model_type for r in quantum_results)):
        model_results = [r for r in quantum_results if r.model_type == model]
        qubits = sorted(set(r.n_qubits for r in model_results))
        means = [np.mean([r.runtime_seconds for r in model_results if r.n_qubits == q])
                 for q in qubits]
        ax.plot(qubits, means, 'o-', label=model, markersize=8)
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime vs Qubits')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    _style_axes(ax)
    
    # Runtime vs features (for classical models)
    ax = axes[1]
    classical_results = [r for r in results if r.n_qubits is None]
    for model in sorted(set(r.model_type for r in classical_results)):
        model_results = [r for r in classical_results if r.model_type == model]
        features = sorted(set(r.n_features for r in model_results))
        means = [np.mean([r.runtime_seconds for r in model_results if r.n_features == f])
                 for f in features]
        ax.plot(features, means, 'o-', label=model, markersize=8)
    
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Classical Runtime vs Features')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    _style_axes(ax)
    
    plt.suptitle('Scaling Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scaling curves plot: {output_path}")


def generate_all_plots(
    results_path: str = 'results/all_results.json',
    output_dir: str = 'results/figures',
):
    """Generate all required visualization plots from saved results."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = load_results(results_path)
    print(f"Loaded {len(results)} results for visualization.")
    
    plot_model_comparison(results, os.path.join(output_dir, 'model_comparison.png'))
    plot_performance_vs_qubits(results, os.path.join(output_dir, 'performance_vs_qubits.png'))
    plot_performance_vs_depth(results, os.path.join(output_dir, 'performance_vs_depth.png'))
    plot_performance_vs_noise(results, os.path.join(output_dir, 'performance_vs_noise.png'))
    plot_runtime_comparison(results, os.path.join(output_dir, 'runtime_comparison.png'))
    plot_scaling_curves(results, os.path.join(output_dir, 'scaling_curves.png'))
    
    # Confusion matrices for a few key results
    key_results = [
        r for r in results
        if r.model_type in ['Quantum Kernel', 'SVM'] and r.confusion_matrix is not None
    ][:4]
    
    for i, r in enumerate(key_results):
        plot_confusion_matrix(
            r.confusion_matrix,
            output_path=os.path.join(output_dir, f'confusion_{r.model_type}_{r.dataset_name}.png'),
            title=f"Confusion Matrix: {r.model_type} on {r.dataset_name}",
        )
    
    print(f"\nAll plots saved to {output_dir}/")
