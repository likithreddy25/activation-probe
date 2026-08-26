"""Build the labeled jailbreak/benign dataset used to train and evaluate the probe.

Primary set: JailbreakBench JBB-Behaviors (`behaviors` config), which ships 100
harmful requests and 100 category-matched benign counterparts -- a clean,
balanced core set with per-category structure.

Secondary set: a larger out-of-distribution benign sample pulled from the
Alpaca instruction dataset. This is NOT mixed into the primary train/test
split; it exists so Phase 1's eval harness can report a false-positive rate
on benign prompts the probe never trained on, which is a more honest signal
than FPR measured only on JBB's own (small, curated) benign split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
PROCESSED_DIR = os.path.abspath(os.path.join(DATA_DIR, "processed"))


@dataclass
class Example:
    text: str
    label: int  # 1 = harmful/jailbreak, 0 = benign
    category: str
    source: str


def load_jbb_behaviors() -> pd.DataFrame:
    """Load JBB-Behaviors harmful + benign splits into one labeled DataFrame."""
    harmful = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    benign = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")

    harmful_df = harmful.to_pandas()
    benign_df = benign.to_pandas()

    harmful_df = pd.DataFrame(
        {
            "text": harmful_df["Goal"],
            "label": 1,
            "category": harmful_df["Category"],
            "source": "jbb_harmful_" + harmful_df["Source"].str.lower(),
        }
    )
    benign_df = pd.DataFrame(
        {
            "text": benign_df["Goal"],
            "label": 0,
            "category": benign_df["Category"],
            "source": "jbb_benign",
        }
    )
    return pd.concat([harmful_df, benign_df], ignore_index=True)


def load_ood_benign(n: int = 200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Sample n benign instructions from Alpaca as an out-of-distribution benign set."""
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    df = alpaca.to_pandas()
    # Keep only instructions with no separate input context, so `text` is self-contained.
    df = df[df["input"].str.len() == 0]
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return pd.DataFrame(
        {
            "text": df["instruction"],
            "label": 0,
            "category": "ood_benign_alpaca",
            "source": "alpaca",
        }
    )


def build_primary_dataset(test_size: float = 0.2, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the stratified train/test split used for probe training and eval.

    Stratifies on (label, category) so every category and class is represented
    in both splits despite the small size (200 total examples).
    """
    df = load_jbb_behaviors()
    strat_key = df["label"].astype(str) + "_" + df["category"]
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat_key
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_datasets() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df, test_df = build_primary_dataset()
    ood_df = load_ood_benign()

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    ood_df.to_csv(os.path.join(PROCESSED_DIR, "ood_benign.csv"), index=False)

    print(f"train: {len(train_df)} rows ({train_df['label'].sum()} harmful)")
    print(f"test:  {len(test_df)} rows ({test_df['label'].sum()} harmful)")
    print(f"ood_benign: {len(ood_df)} rows")


if __name__ == "__main__":
    save_datasets()
