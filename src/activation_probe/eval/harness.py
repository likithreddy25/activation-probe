"""Rerunnable evaluation harness for the trained probe.

Loads a probe artifact + a cached activation split, scores every example,
and produces a structured `EvalReport` (aggregate stats + per-category
breakdown) rather than a single accuracy number. Meant to be re-run against
new held-out data (e.g. nightly, in Phase 3) to catch regressions -- the
dataclасses give a stable shape to diff between runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from activation_probe.probe.model import ProbeArtifact


@dataclass
class CategoryBreakdown:
    category: str
    n: int
    n_positive: int  # n labeled harmful
    accuracy: float
    false_positive_rate: float | None  # only meaningful for benign-only categories
    false_negative_rate: float | None  # only meaningful for harmful-only categories
    mean_score: float


@dataclass
class EvalReport:
    split_name: str
    model_name: str
    probe_layer: int
    n_examples: int
    n_positive: int
    threshold: float
    accuracy: float
    balanced_accuracy: float
    auc: float | None
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: list[list[int]]  # [[tn, fp], [fn, tp]]
    per_category: list[CategoryBreakdown] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def summary_line(self) -> str:
        return (
            f"[{self.split_name}] n={self.n_examples} acc={self.accuracy:.3f} "
            f"bal_acc={self.balanced_accuracy:.3f} "
            f"auc={'n/a' if self.auc is None else f'{self.auc:.3f}'} "
            f"f1={self.f1:.3f} fpr={self.false_positive_rate:.3f} fnr={self.false_negative_rate:.3f}"
        )


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def evaluate(
    artifact: ProbeArtifact,
    activations: np.ndarray,  # (n, n_layers, hidden_dim)
    labels: np.ndarray,  # (n,) 0/1
    categories: np.ndarray,  # (n,) str
    split_name: str,
    threshold: float = 0.5,
) -> EvalReport:
    layer_acts = activations[:, artifact.best_layer, :]
    scores = artifact.predict_proba(layer_acts)
    preds = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = _safe_rate(fp, fp + tn) or 0.0
    fnr = _safe_rate(fn, fn + tp) or 0.0
    recall_pos = (fn + tp) > 0
    recall_neg = (fp + tn) > 0
    balanced_acc = 0.5 * (
        (tp / (tp + fn) if recall_pos else 0.0) + (tn / (tn + fp) if recall_neg else 0.0)
    )

    try:
        auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else None
    except ValueError:
        auc = None

    per_category: list[CategoryBreakdown] = []
    for cat in sorted(set(categories)):
        mask = categories == cat
        cat_labels = labels[mask]
        cat_preds = preds[mask]
        cat_scores = scores[mask]
        n_pos = int(cat_labels.sum())
        n_total = int(mask.sum())
        cat_fp = int(((cat_preds == 1) & (cat_labels == 0)).sum())
        cat_neg_total = int((cat_labels == 0).sum())
        cat_fn = int(((cat_preds == 0) & (cat_labels == 1)).sum())
        per_category.append(
            CategoryBreakdown(
                category=str(cat),
                n=n_total,
                n_positive=n_pos,
                accuracy=float(accuracy_score(cat_labels, cat_preds)),
                false_positive_rate=_safe_rate(cat_fp, cat_neg_total) if n_pos == 0 else None,
                false_negative_rate=_safe_rate(cat_fn, n_pos) if n_pos == n_total else None,
                mean_score=float(cat_scores.mean()),
            )
        )

    return EvalReport(
        split_name=split_name,
        model_name=artifact.model_name,
        probe_layer=artifact.best_layer,
        n_examples=len(labels),
        n_positive=int(labels.sum()),
        threshold=threshold,
        accuracy=float(accuracy_score(labels, preds)),
        balanced_accuracy=float(balanced_acc),
        auc=float(auc) if auc is not None else None,
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
        f1=float(f1_score(labels, preds, zero_division=0)),
        false_positive_rate=float(fpr),
        false_negative_rate=float(fnr),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        per_category=per_category,
    )


def evaluate_npz(artifact: ProbeArtifact, npz_path: str, split_name: str) -> EvalReport:
    data = np.load(npz_path, allow_pickle=True)
    return evaluate(
        artifact,
        activations=data["activations"],
        labels=data["labels"],
        categories=data["categories"],
        split_name=split_name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=os.path.join("artifacts", "probe.joblib"))
    parser.add_argument("--npz", required=True)
    parser.add_argument("--name", default="eval")
    args = parser.parse_args()

    art = ProbeArtifact.load(args.artifact)
    report = evaluate_npz(art, args.npz, args.name)
    print(report.summary_line())
    print(report.to_json())
