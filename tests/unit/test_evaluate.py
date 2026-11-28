import math

import numpy as np

from aiops import evaluate


def test_perfect_scores_give_perfect_metrics():
    m = evaluate.compute_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m["recall"] == m["precision"] == m["roc_auc"] == m["pr_auc"] == 1.0


def test_persistence_cannot_warn_on_onset_rows():
    # Onset rows have incident_now == 0, so persistence scores them all 0.
    y = np.array([1, 0, 1, 0, 1])
    now = np.array([1, 1, 0, 0, 0])
    out = evaluate.evaluate_split(y, now.astype(float), now == 0, threshold=0.5, prefix="test")
    assert out["test_onset_recall"] == 0.0
    assert out["test_onset_roc_auc"] == 0.5


def test_single_class_subset_gives_nan_auc_not_error():
    m = evaluate.compute_metrics([1, 1, 1], [0.2, 0.6, 0.9])
    assert math.isnan(m["roc_auc"])
    assert m["recall"] == 2 / 3


def test_slice_recall_per_project():
    out = evaluate.slice_recall([1, 1, 1, 1], [1, 0, 1, 1], ["a", "a", "b", "b"], "val")
    assert out == {"val_recall_a": 0.5, "val_recall_b": 1.0}


def test_figures_are_built_without_a_display():
    y, score = [0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6]
    assert len(evaluate.plot_confusion(y, [0, 1, 0, 1], "t").axes) == 2  # matrix + colour bar
    assert len(evaluate.plot_roc_pr(y, score, "t").axes) == 2
