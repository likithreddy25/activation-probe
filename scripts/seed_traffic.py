"""Populate classification_log with real probe outputs on the labeled
dataset, spread across the last N simulated days, so the Phase 2 dbt marts
(daily flag rate, category breakdown, rolling FPR trend) have more than a
couple of manually-curled rows to model.

Important: these are NOT fabricated results. Every score/flag is the real
trained probe's output on a real prompt (the same prompts used for
training/eval in Phase 1). The only synthetic part is the `created_at`
timestamp, backdated across the last N days so downstream time-series
marts have something to show. This is documented, not hidden -- see
docs/PHASE2.md.

Usage: python scripts/seed_traffic.py [--days 14] [--clear]
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from activation_probe.probe.fast_extract import FastLayerExtractor
from activation_probe.probe.model import ProbeArtifact
from activation_probe.service import db

load_dotenv()

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "probe.joblib")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="spread replayed traffic across this many past days")
    parser.add_argument("--clear", action="store_true", help="delete existing classification_log rows first")
    args = parser.parse_args()

    db.init_db()
    artifact = ProbeArtifact.load(ARTIFACT_PATH)
    extractor = FastLayerExtractor(artifact.model_name, artifact.best_layer)

    dfs = [pd.read_csv(os.path.join(PROCESSED_DIR, f"{s}.csv")) for s in ["train", "test", "ood_benign"]]
    all_df = pd.concat(dfs, ignore_index=True)

    rng = random.Random(RANDOM_SEED)
    now = datetime.now(timezone.utc)

    session_factory = db.get_session_factory()
    with session_factory() as session:
        if args.clear:
            deleted = session.query(db.ClassificationLog).delete()
            session.commit()
            print(f"cleared {deleted} existing rows")

        for _, row in all_df.iterrows():
            activation = extractor.extract(row["text"])
            score = float(artifact.predict_proba(activation.reshape(1, -1))[0])
            flagged = score >= 0.5
            created_at = now - timedelta(days=rng.uniform(0, args.days))

            session.add(
                db.ClassificationLog(
                    prompt=row["text"],
                    score=score,
                    flagged=flagged,
                    category=row["category"],
                    model_layer_used=artifact.best_layer,
                    model_name=artifact.model_name,
                    created_at=created_at,
                )
            )
        session.commit()

    print(f"seeded {len(all_df)} rows into classification_log, spread across the last {args.days} days")


if __name__ == "__main__":
    main()
