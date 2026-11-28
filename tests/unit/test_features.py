import numpy as np
import pytest

from aiops import features
from aiops.features import TARGET


def test_load_split_separates_target(data_dir):
    X, y = features.load_split(data_dir, "train")
    assert TARGET not in X.columns
    assert len(X) == len(y) == 400
    assert set(y.unique()) <= {0, 1}


def test_keys_skip_null_target_rows_and_align(data_dir):
    X, y = features.load_split(data_dir, "val")
    keys = features.load_keys(data_dir, "val")
    features.check_alignment(keys, X, y)  # fails if the null-target row were kept


def test_alignment_detects_shifted_rows(data_dir):
    X, y = features.load_split(data_dir, "val")
    keys = features.load_keys(data_dir, "val").iloc[1:].reset_index(drop=True)
    with pytest.raises(ValueError):
        features.check_alignment(keys, X.iloc[:-1], y.iloc[:-1])


def test_chronological_splits_pass_leakage_check(data_dir):
    keys = {s: features.load_keys(data_dir, s) for s in features.SPLITS}
    features.check_no_leakage(keys)


def test_overlapping_splits_fail_leakage_check(data_dir):
    keys = {s: features.load_keys(data_dir, s) for s in features.SPLITS}
    keys["test"].loc[0, "timestamp"] = keys["val"]["timestamp"].min()
    with pytest.raises(ValueError, match="leakage"):
        features.check_no_leakage(keys)


def test_validate_frame_rejects_bad_data(data_dir):
    X, y = features.load_split(data_dir, "train")
    cols = list(X.columns)
    features.validate_frame(X, y, cols)
    with_null = X.copy()
    with_null.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match="nulls"):
        features.validate_frame(with_null, y, cols)
    with pytest.raises(ValueError, match="0/1"):
        features.validate_frame(X, y.replace(1, 2), cols)
    with pytest.raises(ValueError, match="columns"):
        features.validate_frame(X[cols[::-1]], y, cols)
