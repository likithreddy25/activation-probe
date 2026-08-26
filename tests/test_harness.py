import numpy as np

from activation_probe.eval.harness import evaluate
from activation_probe.probe.model import ProbeArtifact, fit_pipeline


def _perfect_artifact(n_layers=2, hidden_dim=4):
    rng = np.random.default_rng(1)
    n = 20
    X_benign = rng.normal(loc=-3.0, scale=0.3, size=(n, hidden_dim))
    X_harmful = rng.normal(loc=3.0, scale=0.3, size=(n, hidden_dim))
    X = np.concatenate([X_benign, X_harmful])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    pipeline = fit_pipeline(X, y)
    return ProbeArtifact(model_name="fake/model", best_layer=0, pipeline=pipeline), X, y


def test_evaluate_perfect_separation():
    artifact, X, y = _perfect_artifact()
    # activations shape (n, n_layers, hidden_dim); best_layer=0
    activations = np.stack([X, X], axis=1)
    categories = np.array(["catA"] * 20 + ["catB"] * 20)

    report = evaluate(artifact, activations, y, categories, split_name="unit_test")

    assert report.n_examples == 40
    assert report.accuracy > 0.9
    assert report.auc is not None and report.auc > 0.9
    assert report.confusion_matrix[0][1] < 3  # very few false positives
    assert len(report.per_category) == 2


def test_evaluate_per_category_fpr_only_for_all_benign_category():
    artifact, X, y = _perfect_artifact()
    activations = np.stack([X, X], axis=1)
    # first 20 rows are benign (label 0), last 20 harmful (label 1); tag accordingly
    categories = np.array(["benign_only"] * 20 + ["harmful_only"] * 20)

    report = evaluate(artifact, activations, y, categories, split_name="unit_test")
    by_cat = {c.category: c for c in report.per_category}

    assert by_cat["benign_only"].false_positive_rate is not None
    assert by_cat["benign_only"].false_negative_rate is None
    assert by_cat["harmful_only"].false_negative_rate is not None
    assert by_cat["harmful_only"].false_positive_rate is None


def test_evaluate_handles_single_class_split():
    artifact, X, y = _perfect_artifact()
    activations = np.stack([X, X], axis=1)
    only_benign_mask = y == 0
    report = evaluate(
        artifact,
        activations[only_benign_mask],
        y[only_benign_mask],
        np.array(["benign_only"] * int(only_benign_mask.sum())),
        split_name="benign_only_split",
    )
    assert report.auc is None  # AUC undefined with a single class present
    assert report.n_positive == 0
