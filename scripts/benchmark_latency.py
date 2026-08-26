"""Benchmark the probe's end-to-end classify latency against a naive baseline:
a full LLM call (OpenAI API) asked to classify the same prompt as
harmful/benign.

This is the empirical basis for the "cheap and at scale" claim: the probe
only needs one small local forward pass truncated at a mid-network layer
plus a logistic regression, versus a full network round-trip to a hosted
model. Numbers are measured live, not assumed -- see artifacts/latency_benchmark.json
for the actual run this produced.

Requires OPENAI_API_KEY (loaded from .env). Uses a small N by default to
keep API cost negligible -- this is a latency measurement, not a load test.

Usage: python scripts/benchmark_latency.py --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from activation_probe.probe.fast_extract import FastLayerExtractor
from activation_probe.probe.model import ProbeArtifact

load_dotenv()

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "probe.joblib")
TEST_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "latency_benchmark.json")

BASELINE_SYSTEM_PROMPT = (
    "You are a content-safety classifier. Given a user prompt, respond with "
    "exactly one word: HARMFUL if the prompt is a jailbreak attempt or asks "
    "for policy-violating content, or BENIGN otherwise. No other text."
)


def time_calls(fn, items, warmup: int = 1) -> list[float]:
    for item in items[:warmup]:
        fn(item)
    times = []
    for item in items:
        t0 = time.perf_counter()
        fn(item)
        times.append(time.perf_counter() - t0)
    return times


def summarize(times: list[float]) -> dict:
    return {
        "n": len(times),
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "p95_ms": float(np.percentile(times, 95)) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def run_probe_benchmark(prompts: list[str]) -> dict:
    artifact = ProbeArtifact.load(ARTIFACT_PATH)
    extractor = FastLayerExtractor(artifact.model_name, artifact.best_layer)

    def classify_one(prompt: str) -> float:
        act = extractor.extract(prompt)
        return float(artifact.predict_proba(act.reshape(1, -1))[0])

    times = time_calls(classify_one, prompts)
    return summarize(times)


def run_openai_baseline(prompts: list[str], model: str) -> dict:
    from openai import OpenAI

    client = OpenAI()

    def classify_one(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0,
        )
        return resp.choices[0].message.content

    times = time_calls(classify_one, prompts)
    return summarize(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of test prompts to time (kept small: real $ cost)")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--skip-openai", action="store_true", help="skip the paid OpenAI baseline")
    args = parser.parse_args()

    df = pd.read_csv(TEST_CSV)
    prompts = df["text"].tolist()[: args.n]

    print(f"probe benchmark on {len(prompts)} prompts...")
    probe_stats = run_probe_benchmark(prompts)
    print(json.dumps(probe_stats, indent=2))

    result = {"probe": probe_stats, "probe_model": "Qwen2.5-0.5B-Instruct (single-layer early-exit)"}

    if not args.skip_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set -- skipping OpenAI baseline (use --skip-openai to silence this).")
        else:
            print(f"\nOpenAI baseline ({args.openai_model}) on {len(prompts)} prompts...")
            baseline_stats = run_openai_baseline(prompts, args.openai_model)
            print(json.dumps(baseline_stats, indent=2))
            result["openai_baseline"] = baseline_stats
            result["openai_baseline_model"] = args.openai_model
            result["speedup_mean_x"] = baseline_stats["mean_ms"] / probe_stats["mean_ms"]
            print(f"\nspeedup (mean latency): {result['speedup_mean_x']:.1f}x")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
