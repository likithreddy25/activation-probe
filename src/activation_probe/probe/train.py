"""Sweep logistic regression probes across every layer's mean-pooled residual
stream, pick the best layer by cross-validated AUC on the train set (never
the test set, to avoid selecting the layer based on the data we later report
numbers on), fit the final probe on the full train set at that layer, and
report held-out test + out-of-distribution-benign metrics.

Usage: python -m activation_probe.probe.train
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from activation_probe.eval.harness import evaluate
from activation_probe.probe.model import ProbeArtifact, fit_pipeline

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
N_CV_FOLDS = 5
RANDOM_SEED = 42


@dataclass
class LayerSweepResult:
    layer: int
    cv_auc_mean: float
    cv_auc_std: float


def sweep_layers(train_acts: np.ndarray, train_labels: np.ndarray, n_layers: int) -> list[LayerSweepResult]:
    results = []
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for layer in range(n_layers):
        X = train_acts[:, layer, :]
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)),
            ]
        )
        scores = cross_val_score(pipeline, X, train_labels, cv=cv, scoring="roc_auc")
        results.append(LayerSweepResult(layer=layer, cv_auc_mean=float(scores.mean()), cv_auc_std=float(scores.std())))
    return results


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    train_data = np.load(os.path.join(CACHE_DIR, "train.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(CACHE_DIR, "test.npz"), allow_pickle=True)
    ood_data = np.load(os.path.join(CACHE_DIR, "ood_benign.npz"), allow_pickle=True)

    train_acts, train_labels = train_data["activations"], train_data["labels"]
    n_layers = int(train_data["n_layers"])
    model_name = str(train_data["model_name"])

    print(f"sweeping {n_layers} layers with {N_CV_FOLDS}-fold CV on {len(train_labels)} train examples...")
    sweep = sweep_layers(train_acts, train_labels, n_layers)
    for r in sweep:
        print(f"  layer {r.layer:2d}: cv_auc = {r.cv_auc_mean:.4f} +/- {r.cv_auc_std:.4f}")

    best = max(sweep, key=lambda r: r.cv_auc_mean)
    print(f"\nbest layer by CV AUC: {best.layer} ({best.cv_auc_mean:.4f})")

    pipeline = fit_pipeline(train_acts[:, best.layer, :], train_labels)
    artifact = ProbeArtifact(model_name=model_name, best_layer=best.layer, pipeline=pipeline)
    artifact_path = os.path.join(ARTIFACT_DIR, "probe.joblib")
    artifact.save(artifact_path)
    print(f"saved probe artifact to {artifact_path}")

    test_report = evaluate(
        artifact, test_data["activations"], test_data["labels"], test_data["categories"], split_name="test"
    )
    ood_report = evaluate(
        artifact, ood_data["activations"], ood_data["labels"], ood_data["categories"], split_name="ood_benign"
    )

    print("\n" + test_report.summary_line())
    print(ood_report.summary_line())

    sweep_path = os.path.join(ARTIFACT_DIR, "layer_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "best_layer": best.layer,
                "sweep": [asdict(r) for r in sweep],
                "test_report": json.loads(test_report.to_json()),
                "ood_benign_report": json.loads(ood_report.to_json()),
            },
            f,
            indent=2,
        )
    print(f"saved sweep + eval report to {sweep_path}")


if __name__ == "__main__":
    main()
