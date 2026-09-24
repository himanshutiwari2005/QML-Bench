
"""Preprocessing pipeline utilities."""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional


def preprocess_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: Optional[int] = None,
    scale_to_pi_range: bool = True,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, Optional[PCA], MinMaxScaler]:
    """
    Full preprocessing pipeline:
    raw features -> standardization -> optional PCA -> range scaling to [0, pi]
    
    Args:
        X_train: Training features
        X_test: Test features
        n_components: Number of PCA components (None = skip PCA)
        scale_to_pi_range: If True, scale to [0, pi] for quantum encoding
    
    Returns:
        X_train_processed, X_test_processed, scaler, pca, range_scaler
    """
    # Step 1: Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Optional PCA
    pca = None
    if n_components is not None and n_components < X_train_scaled.shape[1]:
        pca = PCA(n_components=n_components, random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    
    # Step 3: Range scaling to [0, pi] (or [0, 1])
    range_scaler = MinMaxScaler(feature_range=(0, np.pi) if scale_to_pi_range else (0, 1))
    X_train_final = range_scaler.fit_transform(X_train_scaled)
    X_test_final = range_scaler.transform(X_test_scaled)
    
    return X_train_final, X_test_final, scaler, pca, range_scaler


def compute_pca_explained_variance(X: np.ndarray, n_components: int) -> float:
    """Compute the fraction of variance explained by PCA with n_components."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    return float(pca.explained_variance_ratio_.sum())
