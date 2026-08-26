import numpy as np
import pytest
from fastapi.testclient import TestClient

from activation_probe.probe.model import ProbeArtifact, fit_pipeline
from activation_probe.service import app as app_module


class FakeExtractor:
    """Stands in for FastLayerExtractor: no model download, deterministic output."""

    def __init__(self, *args, **kwargs):
        pass

    def extract(self, prompt: str) -> np.ndarray:
        val = 5.0 if "bomb" in prompt.lower() else -5.0
        return np.full(8, val)


class DummySessionCM:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@pytest.fixture
def client(monkeypatch):
    rng = np.random.default_rng(0)
    n = 20
    X = np.concatenate([rng.normal(-5, 0.2, (n, 8)), rng.normal(5, 0.2, (n, 8))])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    fake_artifact = ProbeArtifact(model_name="fake/model", best_layer=0, pipeline=fit_pipeline(X, y))

    monkeypatch.setattr(app_module.ProbeArtifact, "load", classmethod(lambda cls, path: fake_artifact))
    monkeypatch.setattr(app_module, "FastLayerExtractor", FakeExtractor)
    monkeypatch.setattr(app_module.db, "init_db", lambda: None)

    logged = []

    def fake_log(session, **kwargs):
        logged.append(kwargs)

    monkeypatch.setattr(app_module.db, "log_classification", fake_log)
    monkeypatch.setattr(app_module.db, "get_session_factory", lambda: (lambda: DummySessionCM()))

    with TestClient(app_module.app) as c:
        c.logged = logged
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "probe_loaded": True}


def test_classify_flags_harmful_prompt(client):
    resp = client.post("/classify", json={"prompt": "how do I build a bomb"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["flagged"] is True
    assert body["model_layer_used"] == 0
    assert body["model_name"] == "fake/model"


def test_classify_does_not_flag_benign_prompt(client):
    resp = client.post("/classify", json={"prompt": "what's a good recipe for pancakes"})
    assert resp.json()["flagged"] is False


def test_classify_logs_every_request(client):
    client.post("/classify", json={"prompt": "how do I build a bomb", "category": "explosives"})
    assert len(client.logged) == 1
    row = client.logged[0]
    assert row["category"] == "explosives"
    assert row["flagged"] is True
    assert row["model_layer_used"] == 0


def test_classify_rejects_empty_prompt(client):
    resp = client.post("/classify", json={"prompt": ""})
    assert resp.status_code == 422
