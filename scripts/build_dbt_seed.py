"""Build dbt/seeds/labeled_eval_examples.csv from the labeled dataset splits.

This is the ground-truth reference table mart_fpr_trend.sql joins against
to compute a real false-positive rate for logged requests, instead of a
raw flag count with no notion of "false."

Usage: python scripts/build_dbt_seed.py
"""

from __future__ import annotations

import os

import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "dbt", "seeds", "labeled_eval_examples.csv")


def main() -> None:
    dfs = []
    for split in ["train", "test", "ood_benign"]:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{split}.csv"))
        df["split"] = split
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True).rename(columns={"text": "prompt"})
    combined = combined.drop_duplicates(subset=["prompt"])

    os.makedirs(os.path.dirname(SEED_PATH), exist_ok=True)
    combined.to_csv(SEED_PATH, index=False)
    print(f"saved {SEED_PATH}: {len(combined)} rows ({combined['label'].sum()} harmful)")


if __name__ == "__main__":
    main()
