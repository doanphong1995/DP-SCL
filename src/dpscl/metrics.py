"""Metrics and summary helpers for DP-SCL experiments."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import METRIC_NAMES, MODEL_NAME


def select_threshold_by_f1(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    candidates = np.unique(y_score)
    if len(candidates) == 0:
        return 0.5
    if len(candidates) > 1000:
        candidates = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 1000)))

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        score = f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics_with_threshold(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)
    auc = float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_score))
    return {
        "auc": auc,
        "acc": float(np.mean(y_pred == y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def summarize(rows):
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return []
    out = {"model": MODEL_NAME}
    for metric in METRIC_NAMES:
        values = np.array([float(row[f"test_{metric}"]) for row in ok_rows], dtype=float)
        out[f"{metric}_mean"] = float(np.nanmean(values))
        out[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    out["avg_best_epoch"] = float(np.mean([int(row["best_epoch"]) for row in ok_rows]))
    out["avg_stopped_epoch"] = float(np.mean([int(row["stopped_epoch"]) for row in ok_rows]))
    return [out]


def fmt(value):
    if value == "":
        return ""
    value = float(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def fmt_mean_std(row, metric):
    return f"{fmt(row[f'{metric}_mean'])} +/- {fmt(row[f'{metric}_std'])}"

