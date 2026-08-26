import pandas as pd
import pytest
from datasets import Dataset

from activation_probe.data import datasets as ds_mod


def _fake_jbb(split: str, config_name: str = "behaviors"):
    if split == "harmful":
        return Dataset.from_pandas(
            pd.DataFrame(
                {
                    "Index": [0, 1],
                    "Goal": ["harmful goal one", "harmful goal two"],
                    "Target": ["t1", "t2"],
                    "Behavior": ["b1", "b2"],
                    "Category": ["Malware", "Fraud"],
                    "Source": ["Original", "AdvBench"],
                }
            )
        )
    return Dataset.from_pandas(
        pd.DataFrame(
            {
                "Index": [0, 1],
                "Goal": ["benign goal one", "benign goal two"],
                "Target": ["t1", "t2"],
                "Behavior": ["b1", "b2"],
                "Category": ["Malware", "Fraud"],
                "Source": ["Original", "Original"],
            }
        )
    )


def test_load_jbb_behaviors_shapes_labels(monkeypatch):
    def fake_load_dataset(name, config, split):
        assert name == "JailbreakBench/JBB-Behaviors"
        assert config == "behaviors"
        return _fake_jbb(split)

    monkeypatch.setattr(ds_mod, "load_dataset", fake_load_dataset)
    df = ds_mod.load_jbb_behaviors()

    assert len(df) == 4
    assert set(df.columns) == {"text", "label", "category", "source"}
    assert df[df["label"] == 1]["text"].tolist() == ["harmful goal one", "harmful goal two"]
    assert (df[df["label"] == 0]["source"] == "jbb_benign").all()
    assert df[df["label"] == 1]["source"].tolist() == ["jbb_harmful_original", "jbb_harmful_advbench"]


def test_build_primary_dataset_is_stratified(monkeypatch):
    def fake_load_jbb_behaviors():
        n = 20
        return pd.DataFrame(
            {
                "text": [f"t{i}" for i in range(n)],
                "label": ([1] * 10) + ([0] * 10),
                "category": (["catA"] * 5 + ["catB"] * 5) * 2,
                "source": "jbb",
            }
        )

    monkeypatch.setattr(ds_mod, "load_jbb_behaviors", fake_load_jbb_behaviors)
    train_df, test_df = ds_mod.build_primary_dataset(test_size=0.5, seed=0)

    assert len(train_df) + len(test_df) == 20
    # every (label, category) combo present in the full set should appear in both splits
    for split in (train_df, test_df):
        combos = set(zip(split["label"], split["category"]))
        assert combos == {(1, "catA"), (1, "catB"), (0, "catA"), (0, "catB")}
