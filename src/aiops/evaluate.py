"""Metrics and diagnostic figures shared by training and the regression gate."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": float(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, y_score) if both_classes else float("nan"),
        "brier": brier_score_loss(y_true, y_score),
    }


def evaluate_split(y_true, y_score, onset, threshold: float, prefix: str) -> dict[str, float]:
    """All metrics, plus the same metrics on onset rows (no incident right now)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    onset = np.asarray(onset, dtype=bool)
    out = {f"{prefix}_{k}": v for k, v in compute_metrics(y_true, y_score, threshold).items()}
    onset_metrics = compute_metrics(y_true[onset], y_score[onset], threshold)
    out.update({f"{prefix}_onset_{k}": v for k, v in onset_metrics.items()})
    return out


def slice_recall(y_true, y_pred, groups, prefix: str) -> dict[str, float]:
    """Recall per group (for example per project); feeds the 'no silent failure' rule."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[f"{prefix}_recall_{g}"] = recall_score(y_true[m], y_pred[m], zero_division=0)
    return out


def plot_confusion(y_true, y_pred, title: str) -> Figure:
    fig = Figure(figsize=(4.5, 4))
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=["no incident", "incident"], ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_pr(y_true, y_score, title: str, label: str = "model") -> Figure:
    fig = Figure(figsize=(9, 4))
    ax_roc, ax_pr = fig.subplots(1, 2)
    RocCurveDisplay.from_predictions(y_true, y_score, name=label, ax=ax_roc)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
    PrecisionRecallDisplay.from_predictions(y_true, y_score, name=label, ax=ax_pr)
    ax_roc.set_title(f"ROC - {title}")
    ax_pr.set_title(f"PR - {title}")
    fig.tight_layout()
    return fig
