"""The Airflow-callable half of the nightly eval rerun: score the current
probe artifact against the held-out JBB test split (using cached
activations, not a live model -- this task deliberately needs no
torch/transformers/nnsight, only numpy/sklearn/joblib, to keep the Airflow
image light) and persist the result to Postgres so accuracy/AUC can be
tracked over time.

Callable directly (used by the Airflow DAG's PythonOperator) or from the
command line for a one-off manual run.
"""

from __future__ import annotations

import os

from activation_probe.eval.harness import EvalReport, evaluate_npz
from activation_probe.probe.model import ProbeArtifact
from activation_probe.service import db

ARTIFACT_PATH = os.environ.get(
    "PROBE_ARTIFACT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "probe.joblib"),
)
TEST_NPZ_PATH = os.environ.get(
    "PROBE_TEST_NPZ_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache", "test.npz"),
)


def run_nightly_eval(artifact_path: str = ARTIFACT_PATH, test_npz_path: str = TEST_NPZ_PATH) -> EvalReport:
    artifact = ProbeArtifact.load(artifact_path)
    report = evaluate_npz(artifact, test_npz_path, split_name="test")

    db.init_db()
    session_factory = db.get_session_factory()
    with session_factory() as session:
        session.add(
            db.EvalHistory(
                split_name=report.split_name,
                model_name=report.model_name,
                probe_layer=report.probe_layer,
                n_examples=report.n_examples,
                accuracy=report.accuracy,
                auc=report.auc,
                f1=report.f1,
                false_positive_rate=report.false_positive_rate,
                false_negative_rate=report.false_negative_rate,
            )
        )
        session.commit()

    print(f"recorded eval_history row: {report.summary_line()}")
    return report


if __name__ == "__main__":
    run_nightly_eval()
