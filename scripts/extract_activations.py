"""Extract and cache activations for train/test/ood_benign CSVs built by
`src/activation_probe/data/datasets.py`. Run once; downstream training and
eval scripts load the cached .npz files instead of re-running the model.

Usage: python scripts/extract_activations.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from activation_probe.probe.extract import ActivationExtractor

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def process_split(extractor: ActivationExtractor, name: str) -> None:
    csv_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df = pd.read_csv(csv_path)
    batch = extractor.extract_batch(df["text"].tolist())
    out_path = os.path.join(CACHE_DIR, f"{name}.npz")
    np.savez(
        out_path,
        activations=batch.activations,
        labels=df["label"].to_numpy(),
        categories=df["category"].to_numpy(),
        sources=df["source"].to_numpy(),
        model_name=batch.model_name,
        n_layers=batch.n_layers,
        hidden_dim=batch.hidden_dim,
    )
    print(f"saved {out_path}: activations {batch.activations.shape}")


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    extractor = ActivationExtractor()
    for split in ["train", "test", "ood_benign"]:
        process_split(extractor, split)


if __name__ == "__main__":
    main()
