import pandas as pd
import pytest

from aiops import train
from aiops.models import build_model


def test_persistence_predicts_current_state():
    X = pd.DataFrame({"pass__incident_now": [0, 1, 1, 0]})
    model = build_model("persistence").fit(X, None)
    assert model.predict(X).tolist() == [0, 1, 1, 0]
    assert model.predict_proba(X)[:, 1].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_model("xgboost")


def test_logreg_beats_persistence_on_onset_rows(data_dir):
    base = train.train_and_evaluate(data_dir, "persistence")
    cand = train.train_and_evaluate(data_dir, "logreg")
    assert base.metrics["val_onset_recall"] == 0.0
    assert cand.metrics["val_onset_pr_auc"] > base.metrics["val_onset_pr_auc"]
    assert {"val_confusion_matrix", "test_roc_pr"} <= cand.figures.keys()
    assert cand.params["iteration"] == 1


def test_training_stops_on_time_leakage(data_dir):
    abt = pd.read_csv(data_dir / "abt_clean.csv")
    abt.loc[abt["split"] == "test", "timestamp"] = abt["timestamp"].iloc[0]
    abt.to_csv(data_dir / "abt_clean.csv", index=False)
    with pytest.raises(ValueError, match="leakage"):
        train.train_and_evaluate(data_dir, "logreg")
