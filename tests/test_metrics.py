"""Tests for metric calculations."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)


def test_metrics_basic():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    
    assert accuracy_score(y_true, y_pred) == 4/6
    assert balanced_accuracy_score(y_true, y_pred) > 0
    assert f1_score(y_true, y_pred, average='macro') > 0
    assert precision_score(y_true, y_pred, average='macro', zero_division=0) > 0
    assert recall_score(y_true, y_pred, average='macro') > 0
    
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)


def test_metrics_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert accuracy_score(y_true, y_pred) == 4/6
