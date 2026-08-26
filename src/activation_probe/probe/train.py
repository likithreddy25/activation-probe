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
from sklearn.metrics import roc_auc_score
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
    cv_accuracy_mean: float
    cv_accuracy_std: float
    # Diagnostic only -- fit on full train, scored on held-out test. NOT used
    # for layer selection (that's cv_auc_mean above); reported for transparency
    # about how CV-based selection compares to what the test set would have said.
    test_accuracy: float
    test_auc: float


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)),
        ]
    )


def sweep_layers(
    train_acts: np.ndarray,
    train_labels: np.ndarray,
    test_acts: np.ndarray,
    test_labels: np.ndarray,
    n_layers: int,
) -> list[LayerSweepResult]:
    results = []
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for layer in range(n_layers):
        X_train, X_test = train_acts[:, layer, :], test_acts[:, layer, :]

        auc_scores = cross_val_score(_make_pipeline(), X_train, train_labels, cv=cv, scoring="roc_auc")
        acc_scores = cross_val_score(_make_pipeline(), X_train, train_labels, cv=cv, scoring="accuracy")

        diagnostic_pipeline = _make_pipeline()
        diagnostic_pipeline.fit(X_train, train_labels)
        test_preds = diagnostic_pipeline.predict(X_test)
        test_scores = diagnostic_pipeline.predict_proba(X_test)[:, 1]

        results.append(
            LayerSweepResult(
                layer=layer,
                cv_auc_mean=float(auc_scores.mean()),
                cv_auc_std=float(auc_scores.std()),
                cv_accuracy_mean=float(acc_scores.mean()),
                cv_accuracy_std=float(acc_scores.std()),
                test_accuracy=float((test_preds == test_labels).mean()),
                test_auc=float(roc_auc_score(test_labels, test_scores)),
            )
        )
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
    sweep = sweep_layers(train_acts, train_labels, test_data["activations"], test_data["labels"], n_layers)
    for r in sweep:
        print(
            f"  layer {r.layer:2d}: cv_auc = {r.cv_auc_mean:.4f} +/- {r.cv_auc_std:.4f}  "
            f"cv_acc = {r.cv_accuracy_mean:.4f} +/- {r.cv_accuracy_std:.4f}  "
            f"test_auc = {r.test_auc:.4f}  test_acc = {r.test_accuracy:.4f}"
        )

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
