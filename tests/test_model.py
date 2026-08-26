import numpy as np

from activation_probe.probe.model import ProbeArtifact, fit_pipeline


def _toy_data():
    rng = np.random.default_rng(0)
    n = 40
    X_benign = rng.normal(loc=0.0, scale=1.0, size=(n, 8))
    X_harmful = rng.normal(loc=4.0, scale=1.0, size=(n, 8))
    X = np.concatenate([X_benign, X_harmful])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


def test_fit_pipeline_separates_toy_classes():
    X, y = _toy_data()
    pipeline = fit_pipeline(X, y)
    preds = pipeline.predict(X)
    assert (preds == y).mean() > 0.95


def test_probe_artifact_roundtrip(tmp_path):
    X, y = _toy_data()
    pipeline = fit_pipeline(X, y)
    artifact = ProbeArtifact(model_name="fake/model", best_layer=3, pipeline=pipeline)

    path = str(tmp_path / "probe.joblib")
    artifact.save(path)
    loaded = ProbeArtifact.load(path)

    assert loaded.model_name == "fake/model"
    assert loaded.best_layer == 3
    scores_before = artifact.predict_proba(X)
    scores_after = loaded.predict_proba(X)
    np.testing.assert_allclose(scores_before, scores_after)


def test_predict_proba_shape():
    X, y = _toy_data()
    pipeline = fit_pipeline(X, y)
    artifact = ProbeArtifact(model_name="fake/model", best_layer=0, pipeline=pipeline)
    scores = artifact.predict_proba(X)
    assert scores.shape == (X.shape[0],)
    assert ((scores >= 0) & (scores <= 1)).all()
