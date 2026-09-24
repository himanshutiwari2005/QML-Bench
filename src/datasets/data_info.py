
"""Dataset description and documentation utilities."""

from typing import Dict, Any


def describe_dataset(data_dict: Dict[str, Any]) -> str:
    """
    Generate a human-readable description of a dataset.
    
    Args:
        data_dict: Dictionary from a dataset loader
    
    Returns:
        Formatted description string
    """
    meta = data_dict['metadata']
    lines = [
        f"Dataset: {meta['name']}",
        f"  Total samples: {meta['n_samples']}",
        f"  Features: {meta['n_features']}",
        f"  Classes: {meta['n_classes']}",
        f"  Class balance: {meta['class_balance']}",
        f"  Train samples: {meta['n_train']}",
        f"  Test samples: {meta['n_test']}",
        f"  Preprocessing: {meta['preprocessing']}",
    ]
    
    if 'noise' in meta:
        lines.append(f"  Noise level: {meta['noise']}")
    if 'class_sep' in meta:
        lines.append(f"  Class separation: {meta['class_sep']}")
    if 'feature_names' in meta:
        lines.append(f"  Feature names: {meta['feature_names']}")
    if 'class_names' in meta:
        lines.append(f"  Class names: {meta['class_names']}")
    if 'source' in meta:
        lines.append(f"  Source: {meta['source']}")
    
    return '\n'.join(lines)
