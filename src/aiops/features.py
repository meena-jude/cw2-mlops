"""Load the CW1 model-ready splits and run the data checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "incident_next_1h"
NOW_COL = "pass__incident_now"
SPLITS = ("train", "val", "test")
KEY_COLS = ["timestamp", "project", "environment", "deployment_id", "incident_now"]


def load_split(data_dir: str | Path, split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return features X and target y for one split."""
    df = pd.read_csv(Path(data_dir) / f"model_ready_{split}.csv")
    y = df.pop(TARGET).astype(int)
    return df, y


def load_keys(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Recover timestamp/project/environment for a split from abt_clean.csv.

    The model-ready files carry no keys. In abt_clean, rows with a null target
    were dropped by CW1, and the remaining rows of each split are in the same
    order as model_ready_<split>.csv.
    """
    abt = pd.read_csv(
        Path(data_dir) / "abt_clean.csv",
        usecols=[*KEY_COLS, TARGET, "split"],
        parse_dates=["timestamp"],
    )
    keys = abt[abt[TARGET].notna() & (abt["split"] == split)]
    return keys.drop(columns="split").reset_index(drop=True)


def validate_frame(X: pd.DataFrame, y: pd.Series, expected_columns: list[str]) -> None:
    """Schema checks: same columns as train, numeric, no nulls, binary target."""
    if list(X.columns) != list(expected_columns):
        raise ValueError("feature columns differ from the training split")
    if X.isna().any().any():
        raise ValueError("features contain nulls")
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"non-numeric features: {non_numeric[:5]}")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("target must be 0/1")


def check_alignment(keys: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    """Confirm the abt keys line up 1:1 with the model-ready rows."""
    if len(keys) != len(X):
        raise ValueError(f"row count mismatch: keys {len(keys)} vs features {len(X)}")
    if not np.array_equal(keys[TARGET].astype(int).to_numpy(), y.to_numpy()):
        raise ValueError("target does not match abt_clean row order")
    if not np.array_equal(
        keys["incident_now"].astype(int).to_numpy(), X[NOW_COL].astype(int).to_numpy()
    ):
        raise ValueError("incident_now does not match abt_clean row order")


def check_no_leakage(keys_by_split: dict[str, pd.DataFrame]) -> None:
    """Chronological split: all train times < all val times < all test times."""
    for earlier, later in zip(SPLITS, SPLITS[1:], strict=False):
        last = keys_by_split[earlier]["timestamp"].max()
        first = keys_by_split[later]["timestamp"].min()
        if last >= first:
            raise ValueError(f"time leakage: {earlier} ends {last}, {later} starts {first}")


def onset_mask(X: pd.DataFrame) -> np.ndarray:
    """Rows with no incident right now: the hard early-warning cases."""
    return (X[NOW_COL] == 0).to_numpy()
