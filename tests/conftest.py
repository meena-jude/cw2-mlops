"""Synthetic stand-in for the CW1 data folder (the real data is not in git)."""

import numpy as np
import pandas as pd
import pytest

from aiops.features import NOW_COL, TARGET

SIZES = {"train": 400, "val": 150, "test": 150}


def _make_split(rng, n, start, split):
    now = rng.integers(0, 2, n)
    signal = rng.normal(size=n)
    # Next-hour incident depends on the current state AND a monitoring signal,
    # so a real model can warn on onset rows where persistence cannot.
    prob = 1 / (1 + np.exp(-(2.5 * now + 2.0 * signal - 1.0)))
    y = (rng.random(n) < prob).astype(int)
    model_ready = pd.DataFrame(
        {
            NOW_COL: now,
            "std__error_rate": signal,
            "minmax__cpu": rng.random(n),
            "ohe__project_fraud_detection": rng.integers(0, 2, n).astype(float),
            TARGET: y,
        }
    )
    keys = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="h"),
            "project": rng.choice(["fraud_detection", "rag_qa_service"], n),
            "environment": "prod",
            "deployment_id": "dep-1",
            "incident_now": now,
            TARGET: y.astype(float),
            "split": split,
        }
    )
    return model_ready, keys


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2024-01-01")
    parts = []
    for split, n in SIZES.items():
        model_ready, keys = _make_split(rng, n, start, split)
        model_ready.to_csv(tmp_path / f"model_ready_{split}.csv", index=False)
        parts.append(keys)
        start = keys["timestamp"].max() + pd.Timedelta(hours=1)
    abt = pd.concat(parts, ignore_index=True)
    # CW1 dropped rows with a null target; put one inside 'val' to prove we skip it.
    null_row = abt.iloc[[450]].assign(**{TARGET: np.nan})
    abt = pd.concat([abt.iloc[:450], null_row, abt.iloc[450:]], ignore_index=True)
    abt.to_csv(tmp_path / "abt_clean.csv", index=False)
    return tmp_path
