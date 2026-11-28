"""Candidate models for the pipeline iterations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aiops.features import NOW_COL

MODEL_NAMES = ("persistence", "logreg")
ITERATION = {"persistence": 0, "logreg": 1}


class PersistenceModel:
    """Iter 0 floor: predict that the next hour looks like this hour."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PersistenceModel:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[NOW_COL].astype(float).to_numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[NOW_COL].astype(int).to_numpy()


def build_model(name: str, seed: int = 42):
    if name == "persistence":
        return PersistenceModel()
    if name == "logreg":
        # CW1 features are already scaled/encoded, so no extra preprocessing here.
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
