"""FastAPI service wrapping the trained activation probe.

POST /classify {"prompt": str, "category": str | None} ->
    {"flagged": bool, "score": float, "model_layer_used": int, "model_name": str}

Every request is logged to Postgres (see db.py). The probe artifact and the
underlying model are loaded once at startup (module-level singletons), not
per-request.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from activation_probe.probe.fast_extract import FastLayerExtractor
from activation_probe.probe.model import ProbeArtifact
from activation_probe.service import db

ARTIFACT_PATH = os.environ.get(
    "PROBE_ARTIFACT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "probe.joblib"),
)
FLAG_THRESHOLD = float(os.environ.get("PROBE_FLAG_THRESHOLD", "0.5"))

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = ProbeArtifact.load(ARTIFACT_PATH)
    extractor = FastLayerExtractor(artifact.model_name, artifact.best_layer)
    _state["artifact"] = artifact
    _state["extractor"] = extractor
    try:
        db.init_db()
    except Exception as exc:  # pragma: no cover - depends on local Postgres being up
        print(f"warning: could not initialize database ({exc}); requests will not be logged")
    yield
    _state.clear()


app = FastAPI(title="activation-probe", version="0.1.0", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    category: str | None = Field(default=None, description="Optional known category, for eval/audit tagging.")


class ClassifyResponse(BaseModel):
    flagged: bool
    score: float
    model_layer_used: int
    model_name: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "probe_loaded": "artifact" in _state}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    artifact: ProbeArtifact = _state["artifact"]
    extractor: FastLayerExtractor = _state["extractor"]

    activation = extractor.extract(req.prompt)
    score = float(artifact.predict_proba(activation.reshape(1, -1))[0])
    flagged = score >= FLAG_THRESHOLD

    try:
        session_factory = db.get_session_factory()
        with session_factory() as session:
            db.log_classification(
                session,
                prompt=req.prompt,
                score=score,
                flagged=flagged,
                model_layer_used=artifact.best_layer,
                model_name=artifact.model_name,
                category=req.category,
            )
    except Exception as exc:  # pragma: no cover - depends on local Postgres being up
        print(f"warning: failed to log classification ({exc})")

    return ClassifyResponse(
        flagged=flagged,
        score=score,
        model_layer_used=artifact.best_layer,
        model_name=artifact.model_name,
    )
