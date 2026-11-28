"""Train one candidate model, evaluate it on val and test, and log it to MLflow.

Run as an Azure ML command job (see jobs/train.yml) or locally:
    python -m aiops.train --data <folder with model_ready_*.csv> --model logreg
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from aiops import evaluate, features, models


@dataclass
class Result:
    model: object
    params: dict
    metrics: dict[str, float]
    figures: dict = field(default_factory=dict)


def load_checked_data(data_dir: str | Path):
    """Load all splits and run the data tests; any failure stops the job."""
    data = {s: features.load_split(data_dir, s) for s in features.SPLITS}
    columns = list(data["train"][0].columns)
    keys = {}
    for split, (X, y) in data.items():
        features.validate_frame(X, y, columns)
        keys[split] = features.load_keys(data_dir, split)
        features.check_alignment(keys[split], X, y)
    features.check_no_leakage(keys)
    return data, keys


def train_and_evaluate(
    data_dir: str | Path, model_name: str, threshold: float = 0.5, seed: int = 42
) -> Result:
    data, keys = load_checked_data(data_dir)
    X_train, y_train = data["train"]

    model = models.build_model(model_name, seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    metrics = {"train_seconds": time.perf_counter() - start}

    figures = {}
    for split in ("val", "test"):
        X, y = data[split]
        score = model.predict_proba(X)[:, 1]
        pred = (score >= threshold).astype(int)
        metrics.update(evaluate.evaluate_split(y, score, features.onset_mask(X), threshold, split))
        metrics.update(evaluate.slice_recall(y, pred, keys[split]["project"], split))
        title = f"iter{models.ITERATION[model_name]} {model_name} ({split})"
        figures[f"{split}_confusion_matrix"] = evaluate.plot_confusion(y, pred, title)
        figures[f"{split}_roc_pr"] = evaluate.plot_roc_pr(y, score, title, model_name)

    params = {
        "model": model_name,
        "iteration": models.ITERATION[model_name],
        "threshold": threshold,
        "seed": seed,
        "n_features": X_train.shape[1],
        **{f"n_{s}": len(data[s][1]) for s in features.SPLITS},
    }
    return Result(model, params, metrics, figures)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="folder containing model_ready_*.csv")
    p.add_argument("--model", choices=models.MODEL_NAMES, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs", help="Azure ML uploads ./outputs automatically")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import mlflow  # imported here so unit tests and CI do not need MLflow

    result = train_and_evaluate(args.data, args.model, args.threshold, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2))

    with mlflow.start_run():
        mlflow.set_tags({"model": args.model, "data_checks": "passed"})
        mlflow.log_params(result.params)
        mlflow.log_metrics({k: v for k, v in result.metrics.items() if math.isfinite(v)})
        for name, fig in result.figures.items():
            mlflow.log_figure(fig, f"figures/{name}.png")

    summary = {k: round(v, 4) for k, v in result.metrics.items() if k.startswith("test_")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
