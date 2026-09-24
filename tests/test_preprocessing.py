"""Tests for preprocessing pipeline."""
import numpy as np
from src.utils.preprocessing import preprocess_pipeline, compute_pca_explained_variance


def test_preprocess_pipeline_shape():
    """Test that preprocessing preserves sample count and adjusts features."""
    X_train = np.random.randn(100, 8)
    X_test = np.random.randn(20, 8)
    
    # Without PCA
    X_tr, X_te, scaler, pca, rs = preprocess_pipeline(X_train, X_test, n_components=None)
    assert X_tr.shape == (100, 8)
    assert X_te.shape == (20, 8)
    
    # With PCA
    X_tr2, X_te2, scaler2, pca2, rs2 = preprocess_pipeline(X_train, X_test, n_components=4)
    assert X_tr2.shape == (100, 4)
    assert X_te2.shape == (20, 4)


def test_preprocess_range():
    """Test that output is in [0, pi] range."""
    X_train = np.random.randn(50, 4)
    X_test = np.random.randn(10, 4)
    X_tr, X_te, _, _, _ = preprocess_pipeline(X_train, X_test, n_components=None)
    
    assert np.all(X_tr >= 0) and np.all(X_tr <= np.pi)
    assert np.all(X_te >= 0) and np.all(X_te <= np.pi)


def test_pca_explained_variance():
    """Test that PCA variance computation works."""
    X = np.random.randn(100, 10)
    ratio = compute_pca_explained_variance(X, 5)
    assert 0 <= ratio <= 1
