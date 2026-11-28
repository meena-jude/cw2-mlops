from make_manifest import build_manifest


def test_manifest_records_rows_and_md5(data_dir):
    m = build_manifest(data_dir, "aiops-model-ready", "1")
    train = m["files"]["model_ready_train.csv"]
    assert train["rows"] == 400
    assert len(train["md5"]) == 32
    assert m["files"]["abt_clean.csv"]["rows"] == 701  # 700 rows + 1 null-target row
