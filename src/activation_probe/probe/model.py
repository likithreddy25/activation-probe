"""The trained probe artifact: a StandardScaler + LogisticRegression pipeline
bound to a specific layer of a specific model, serialized as one object so
the eval harness and the FastAPI service always load a matched
(scaler, classifier, layer, model_name) tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeArtifact:
    model_name: str
    best_layer: int
    pipeline: Pipeline  # StandardScaler -> LogisticRegression

    def predict_proba(self, layer_activations: np.ndarray) -> np.ndarray:
        """layer_activations: (n, hidden_dim) mean-pooled residual stream at best_layer."""
        return self.pipeline.predict_proba(layer_activations)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(
            {"model_name": self.model_name, "best_layer": self.best_layer, "pipeline": self.pipeline},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ProbeArtifact":
        obj = joblib.load(path)
        return cls(model_name=obj["model_name"], best_layer=obj["best_layer"], pipeline=obj["pipeline"])


def fit_pipeline(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline
