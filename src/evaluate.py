"""Evaluation metrics for SECOM.

  - PR-AUC (average precision)
  - Recall
  - Precision
  - MCC
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def dummy_baseline_accuracy(y):
    # accuracy of the dummy 'predict pass for everything' classifier
    y = np.asarray(y)
    return float((y == 0).mean())


def evaluate(y_true, y_pred, y_score=None):
    """Metrics for one set of predictions.

    y_pred  : hard 0/1 predictions (threshold already applied)
    y_score : predicted probability of failure, if available. Needed for the
              threshold-free metrics (PR-AUC, ROC-AUC).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp),
        "fn": int(fn),   # missed failures - the costly error in QC
        "fp": int(fp),   # false alarms
        "tn": int(tn),
    }

    if y_score is not None:
        y_score = np.asarray(y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
        out["roc_auc"] = roc_auc_score(y_true, y_score)

    return out


def summarise_cv(scores: dict):
    # turns sklearn cross_validate output into mean +/- std per metric
    rows = []
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        values = np.asarray(values, dtype=float)
        rows.append({
            "metric": key.replace("test_", ""),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        })
    return pd.DataFrame(rows).set_index("metric")


def threshold_sweep(y_true, y_score, thresholds=None):
    # recall / precision / MCC across decision thresholds
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": round(float(t), 2),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "flagged": int(tp + fp),
            "missed_failures": int(fn),
        })
    return pd.DataFrame(rows).set_index("threshold")
