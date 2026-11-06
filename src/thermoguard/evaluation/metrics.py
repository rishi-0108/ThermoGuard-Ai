"""Ground-truth scoring helpers, intentionally separate from detector APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd


class EvaluationError(ValueError):
    """Raised for invalid labels, predictions, or evaluation alignment."""


METRIC_DEFINITIONS = {
    "true_positives": "count(y_true=1 and y_pred=1)",
    "false_positives": "count(y_true=0 and y_pred=1)",
    "true_negatives": "count(y_true=0 and y_pred=0)",
    "false_negatives": "count(y_true=1 and y_pred=0)",
    "precision": "TP / (TP + FP); zero when denominator is zero",
    "recall": "TP / (TP + FN); zero when denominator is zero",
    "f1": "2 * precision * recall / (precision + recall); zero when denominator is zero",
    "accuracy": "(TP + TN) / (TP + FP + TN + FN)",
    "specificity": "TN / (TN + FP); zero when denominator is zero",
    "false_positive_rate": "FP / (FP + TN); zero when denominator is zero",
    "n_true_anomaly_points": "count(y_true=1)",
    "n_predicted_anomaly_points": "count(y_pred=1)",
}


def _binary_vector(values: pd.Series | np.ndarray | list[int], *, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        if values.empty:
            raise EvaluationError(f"{name} must not be empty.")
        if values.isna().any():
            raise EvaluationError(f"{name} contains missing values.")
        array = values.to_numpy(copy=True)
    else:
        array = np.asarray(values)
        if array.size == 0:
            raise EvaluationError(f"{name} must not be empty.")
    if array.ndim != 1:
        raise EvaluationError(f"{name} must be one-dimensional.")
    if pd.isna(array).any():
        raise EvaluationError(f"{name} contains missing values.")
    if not np.isin(array, (0, 1)).all():
        raise EvaluationError(f"{name} must contain only binary values 0 and 1.")
    return array.astype(np.uint8, copy=False)


def calculate_metrics(
    labels: pd.Series | np.ndarray | list[int],
    predictions: pd.Series | np.ndarray | list[int],
) -> dict[str, int | float]:
    """Calculate pointwise binary metrics; undefined rate denominators map to zero."""
    truth = _binary_vector(labels, name="labels")
    predicted = _binary_vector(predictions, name="predictions")
    if len(truth) != len(predicted):
        raise EvaluationError(f"Labels/predictions length mismatch: {len(truth)} != {len(predicted)}.")
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(truth),
        "specificity": specificity,
        "false_positive_rate": fpr,
        "n_true_anomaly_points": int(truth.sum()),
        "n_predicted_anomaly_points": int(predicted.sum()),
    }
