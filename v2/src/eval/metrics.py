from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    try:
        metrics["rocauc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["rocauc"] = float("nan")
    try:
        metrics["prauc"] = float(average_precision_score(y_true, y_proba))
    except Exception:
        metrics["prauc"] = float("nan")
    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_proba))
    except Exception:
        metrics["brier"] = float("nan")
    return metrics



