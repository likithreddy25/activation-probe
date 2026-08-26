# Phase 1: Core probe + eval harness + service

## What this is

A `POST /classify` endpoint that flags a prompt as harmful/jailbreak or
benign, backed by a logistic regression probe trained on the internal
activations of **Qwen2.5-0.5B-Instruct**, rather than by calling an LLM to
judge the prompt. Every request is logged to Postgres. The probe's layer
was chosen by cross-validation, not by peeking at the test set, and every
number below came from an actual run of this code -- nothing here is
hand-picked or estimated.

## Why a probe instead of an LLM judge

An LLM-based safety judge (e.g. asking GPT-4o-mini "is this a jailbreak?")
works, but it's a full network call per request: real latency, real cost, at
every request in the pipeline. A linear probe on activations from a small,
local, open-weight model is one truncated forward pass through ~half a
2.5-layer-per-second-scale model plus a `sklearn.linear_model.LogisticRegression`
call -- no network round trip, no per-token generation. Whether that
trade-off is worth it depends entirely on how much accuracy you give up,
which is why this phase measures both sides rather than asserting the
conclusion.

## Model and extraction

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct` (24 decoder layers, hidden size
  896, ungated on Hugging Face).
- **Extraction library**: [nnsight](https://nnsight.net/), used to capture
  the residual stream (the output of every decoder layer) in a single
  forward pass per prompt, mean-pooled over the sequence dimension.
  ([`src/activation_probe/probe/extract.py`](../src/activation_probe/probe/extract.py))
- Prompts are wrapped in Qwen's chat template (single user turn +
  generation prompt) before extraction, rather than fed as raw text --
  that's the same input shape the model sees in real deployment.
- **Serving path is a separate, faster extractor**
  ([`fast_extract.py`](../src/activation_probe/probe/fast_extract.py)):
  since the served probe only ever reads one fixed layer, a forward hook on
  that decoder layer captures the pooled activation and raises to abort the
  rest of the forward pass -- skipping every later layer and the LM head.
  Verified to produce bit-identical output to the full nnsight extraction
  (`max abs diff: 0.0` in manual testing) at ~13-30ms per prompt on an M-series
  Mac (MPS), instead of running the full 24-layer network.

## Dataset

- **Harmful set**: [JailbreakBench JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
  (`behaviors` config) -- 100 harmful requests across 10 policy categories
  (Malware/Hacking, Physical harm, Fraud/Deception, etc.), each written by
  the benchmark's authors or drawn from AdvBench/HarmBench.
- **Primary benign set**: JBB's own 100 category-matched benign
  counterparts -- topically similar to the harmful goal in the same
  category, but harmless. This makes the primary train/test set a genuinely
  hard discrimination task (see limitations below), not benign-vs-harmful
  on obviously different topics.
- **Held-out generalization set**: 200 benign instructions sampled from
  [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca),
  used only for a separate out-of-distribution false-positive check, never
  mixed into training.
- Primary set stratified 80/20 by (label, category) into train (160) /
  test (40). See [`src/activation_probe/data/datasets.py`](../src/activation_probe/data/datasets.py).

## Layer selection methodology

To avoid selecting the "best" layer based on the same data we report
numbers on, layer selection uses 5-fold stratified cross-validation **on
the train split only** (`src/activation_probe/probe/train.py`). The held-out
test set and the OOD benign set are touched exactly once, after the layer
and the final probe are already fixed.

## Results (actual run, see `artifacts/layer_sweep.json` for full output)

Full per-layer sweep. **Selection** uses only the two CV columns (5-fold,
train set); the two test columns are diagnostic only, computed by fitting
each layer's probe on the full train set and scoring once against the
held-out test set, to show how CV-based selection compares to what test
performance would have said (it does not feed back into which layer is
chosen):

| Layer | CV AUC | CV Acc | Test AUC | Test Acc |
|---|---|---|---|---|
| 0 | 0.738 | 0.669 | 0.688 | 0.600 |
| 1 | 0.734 | 0.669 | 0.670 | 0.550 |
| 2 | 0.742 | 0.706 | 0.613 | 0.575 |
| 3 | 0.738 | 0.663 | 0.810 | 0.725 |
| 4 | 0.758 | 0.688 | 0.788 | 0.700 |
| 5 | 0.772 | 0.694 | 0.735 | 0.675 |
| 6 | 0.796 | 0.725 | 0.693 | 0.600 |
| 7 | 0.826 | 0.788 | 0.785 | 0.675 |
| 8 | 0.857 | 0.769 | 0.798 | 0.725 |
| 9 | 0.908 | 0.831 | 0.798 | 0.700 |
| 10 | 0.912 | 0.825 | 0.863 | 0.750 |
| **11 (selected)** | **0.931** | **0.825** | 0.880 | 0.775 |
| 12 | 0.925 | 0.844 | **0.895** | **0.850** |
| 13 | 0.920 | 0.831 | 0.848 | 0.800 |
| 14 | 0.917 | 0.825 | 0.863 | 0.800 |
| 15 | 0.897 | 0.800 | 0.833 | 0.750 |
| 16 | 0.862 | 0.775 | 0.740 | 0.725 |
| 17 | 0.863 | 0.788 | 0.773 | 0.725 |
| 18 | 0.854 | 0.788 | 0.775 | 0.725 |
| 19 | 0.872 | 0.794 | 0.770 | 0.750 |
| 20 | 0.838 | 0.781 | 0.783 | 0.750 |
| 21 | 0.844 | 0.794 | 0.775 | 0.775 |
| 22 | 0.835 | 0.788 | 0.768 | 0.750 |
| 23 | 0.837 | 0.750 | 0.740 | 0.750 |

Signal is weak in the early layers (AUC ~0.73-0.76 for layers 0-4, barely
above chance for a probe), climbs through the middle of the network, peaks
around **layers 9-14**, then degrades gradually toward the output --
consistent with the general finding in the interpretability literature that
safety-relevant concepts tend to be most linearly decodable mid-network.

**Layer 11 wins the selection criterion (CV AUC) by a real but modest
margin**: 0.9313 vs. 0.9250 for the runner-up (layer 12), a gap of about
0.006 -- smaller than the fold-to-fold std of ~0.04-0.05 on either layer.
Layers 9-14 are all within ~0.02 CV AUC of each other; this is a broad
plateau, not a sharp single winner. Worth noting honestly: on the held-out
test set, **layer 12 actually scores higher** than the selected layer 11
(0.895 test AUC / 85.0% test accuracy vs. layer 11's 0.880 / 77.5%). This is
the expected kind of noise from picking a layer by 5-fold CV on 160 examples
and then evaluating once on 40 held-out examples -- it doesn't invalidate
CV-based selection (which never touches the test set, by design), but it
does mean the reported test numbers for "the winning layer" are a bit
pessimistic relative to what a lucky layer choice could have shown. Layer 12
would be a reasonable alternative choice; it isn't a large enough gap to
call either one definitively best.

**Layer 11 probe, held-out test set (n=40, 20 harmful / 20 benign):**

| Metric | Value |
|---|---|
| Accuracy | 0.775 |
| Balanced accuracy | 0.775 |
| AUC | 0.880 |
| Precision | 0.739 |
| Recall | 0.850 |
| F1 | 0.791 |
| False positive rate | 0.300 |
| False negative rate | 0.150 |

**Same probe, 200 out-of-distribution benign prompts (Alpaca, never seen in training):**

| Metric | Value |
|---|---|
| Accuracy | 0.970 |
| False positive rate | 0.030 |

## Honest read of these numbers

- The gap between the two benign sets (30% FPR on JBB's category-matched
  benign counterparts vs. 3% FPR on generic Alpaca instructions) is the
  most informative result in this phase. JBB's benign set is deliberately
  adversarial-adjacent -- topically close to the harmful goal in the same
  category -- so a 30% FPR there means the probe is picking up on topic
  more than on genuine harmfulness for some categories. On ordinary,
  topically diverse benign traffic it looks much better (3% FPR), but this
  project makes no claim that 3% would hold on a larger or more
  adversarial OOD sample -- 200 examples from one source is not enough to
  claim a stable production FPR.
- Per-category test accuracy is uneven: perfect on Disinformation, Expert
  advice, and Malware/Hacking (all 4/4), but only 25% (1/4) on
  Sexual/Adult content, where the probe's mean score (0.378) sits below
  the 0.5 threshold even though half the category's examples are harmful --
  i.e. it's systematically under-flagging that category. With only 4
  examples per category in the test set, this could be noise, a genuine
  weak spot, or both; it isn't resolved by this phase.
- The test set is small (40 examples). A 0.775 accuracy / 0.880 AUC on 40
  examples has real sampling noise -- these are the honest numbers from
  the actual run, not a claim of a tight confidence interval.
- **Not yet tested**: paraphrased or obfuscated jailbreak attempts (the
  project's own ground rules call this out as a planned follow-up
  experiment). Everything above is in-distribution relative to JBB's
  original phrasing; there's no evidence yet either way for how the probe
  generalizes to adversarial rephrasing, and that gap should not be assumed
  away.

## Latency benchmark

`scripts/benchmark_latency.py` times the probe end-to-end (extraction +
classification) against a naive baseline: a full OpenAI API call asked to
classify the same prompt.

**Probe only** (20 test prompts, Qwen2.5-0.5B single-layer early-exit, MPS):

| Metric | Value |
|---|---|
| Mean | 28.9 ms |
| Median | 17.3 ms |
| p95 | 98.7 ms |
| Min / Max | 13.2 ms / 185.8 ms |

The OpenAI baseline comparison requires `OPENAI_API_KEY` to be set (it
isn't, in this environment, as of writing this doc) -- run
`python scripts/benchmark_latency.py --n 20` once a key is available in
`.env` to fill in the other half of this table and the resulting speedup
factor. This is deliberately left unfilled rather than estimated: the
ground rules for this project are no fabricated results.

## Service

- `POST /classify` — `{"prompt": str, "category": str | None}` →
  `{"flagged": bool, "score": float, "model_layer_used": int, "model_name": str}`
- `GET /health`
- Every request is logged to Postgres (`classification_log` table: id,
  prompt, score, flagged, category, model_layer_used, model_name,
  created_at). `category` is only populated when the caller supplies a
  known label (e.g. during eval runs); production traffic won't have one.
- Local Postgres via `docker-compose.yml` (mapped to host port 5433 to
  avoid clashing with any existing local Postgres on 5432).
- Model and probe are loaded once at process startup (FastAPI lifespan),
  not per-request.

## Tests

`tests/` is fully mocked -- no GPU, model download, network access, or
running Postgres required, so it can run in CI as-is (see
`.github/workflows/tests.yml`, which installs CPU-only torch and runs
pytest on every push). Covers: dataset construction/stratification logic,
probe save/load round-tripping, the eval harness's metric and per-category
logic (including edge cases like a single-class split), and the FastAPI
endpoint with a fake extractor/probe and a fake DB session.

## Reproducing this phase

```bash
conda activate activation-probe
python -m activation_probe.data.datasets
python scripts/extract_activations.py
python -m activation_probe.probe.train
pytest tests/ -v
docker compose up -d
uvicorn activation_probe.service.app:app --reload
python scripts/benchmark_latency.py --n 20   # add OPENAI_API_KEY to .env first for the full comparison
```
