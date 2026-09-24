"""Evaluation utilities for metric computation and statistical analysis."""

import numpy as np
from typing import List, Dict
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score,
)

from utils.config import ExperimentResult


def aggregate_results(results: List[ExperimentResult]) -> Dict:
    """
    Aggregate results by model type and dataset.
    
    Returns mean, std, and sample count for key metrics.
    """
    from collections import defaultdict
    
    groups = defaultdict(list)
    for r in results:
        key = (r.dataset_name, r.model_type)
        groups[key].append(r)
    
    aggregated = {}
    for (dataset, model), group_results in groups.items():
        metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'precision', 'recall']
        agg = {
            'dataset': dataset,
            'model': model,
            'n_runs': len(group_results),
            'seeds': [r.seed for r in group_results],
        }
        for m in metrics:
            values = [r.__dict__[m] for r in group_results]
            agg[f'{m}_mean'] = float(np.mean(values))
            agg[f'{m}_std'] = float(np.std(values))
            agg[f'{m}_min'] = float(np.min(values))
            agg[f'{m}_max'] = float(np.max(values))
        
        # Runtime
        runtimes = [r.runtime_seconds for r in group_results]
        agg['runtime_mean'] = float(np.mean(runtimes))
        agg['runtime_std'] = float(np.std(runtimes))
        
        aggregated[f"{dataset}_{model}"] = agg
    
    return aggregated


def paired_t_test(
    results_a: List[ExperimentResult],
    results_b: List[ExperimentResult],
    metric: str = 'accuracy',
) -> Dict:
    """
    Perform paired t-test between two sets of results.
    
    Assumes results are paired by seed/experiment_id.
    
    Returns dict with t-statistic, p-value, and interpretation.
    """
    values_a = [r.__dict__[metric] for r in results_a]
    values_b = [r.__dict__[metric] for r in results_b]
    
    if len(values_a) != len(values_b):
        return {
            'error': 'Results must have same length for paired test',
            'n_a': len(values_a),
            'n_b': len(values_b),
        }
    
    if len(values_a) < 2:
        return {
            'error': 'Need at least 2 samples for t-test',
            'n': len(values_a),
        }
    
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    diff = mean_a - mean_b
    
    return {
        'metric': metric,
        'mean_a': float(mean_a),
        'mean_b': float(mean_b),
        'difference': float(diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_at_0.05': bool(p_value < 0.05),
        'n_pairs': len(values_a),
        'interpretation': (
            f"Mean difference: {diff:+.4f}. "
            f"{'Significant' if p_value < 0.05 else 'Not significant'} "
            f"(p={p_value:.4f})"
        ),
    }


def wilcoxon_test(
    results_a: List[ExperimentResult],
    results_b: List[ExperimentResult],
    metric: str = 'accuracy',
) -> Dict:
    """Wilcoxon signed-rank test (non-parametric paired test)."""
    values_a = [r.__dict__[metric] for r in results_a]
    values_b = [r.__dict__[metric] for r in results_b]
    
    if len(values_a) != len(values_b) or len(values_a) < 3:
        return {'error': 'Need >= 3 paired samples'}
    
    statistic, p_value = stats.wilcoxon(values_a, values_b)
    
    return {
        'metric': metric,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_at_0.05': bool(p_value < 0.05),
        'n_pairs': len(values_a),
    }
