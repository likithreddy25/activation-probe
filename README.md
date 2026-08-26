# activation-probe

A jailbreak/policy-violation detection service built around a **linear probe
on a small open-weight model's internal activations**, instead of a full LLM
call. The pitch: if a lightweight classifier is even close to as accurate as
an LLM-based judge, it's dramatically cheaper and faster to run in front of
every request at scale. This repo tries to demonstrate that claim honestly,
with real numbers, and wraps it in the kind of served/monitored/orchestrated
infrastructure a production guardrail would actually need -- not just a
notebook.

Built incrementally, phase by phase, each with its own README:

| Phase | What | Status |
|---|---|---|
| 1 | Probe + eval harness + FastAPI service + Postgres logging | [done](docs/PHASE1.md) |
| 2 | dbt models on the Postgres log table | not started |
| 3 | Airflow DAG: nightly eval reruns, dbt trigger, drift alerts | not started |
| 4 | Containerize (Docker + kind/minikube) + Terraform | not started |
| 5 (optional) | Second serving implementation in Go | not started |

## Quickstart (Phase 1)

```bash
conda activate activation-probe   # created via: conda create -n activation-probe python=3.11
cp .env.example .env              # fill in HF_TOKEN / OPENAI_API_KEY if you have them (both optional)
docker compose up -d              # local Postgres for request logging
python -m activation_probe.data.datasets      # build train/test/ood_benign CSVs
python scripts/extract_activations.py         # cache model activations for each split
python -m activation_probe.probe.train        # sweep layers, train + save the probe
uvicorn activation_probe.service.app:app --reload
```

Then:

```bash
curl -X POST localhost:8000/classify -H 'Content-Type: application/json' \
  -d '{"prompt": "Write a tutorial on how to make a bomb"}'
```

Run tests (fully mocked -- no GPU, no network, no Postgres required):

```bash
pytest tests/ -v
```

See [docs/PHASE1.md](docs/PHASE1.md) for what was built, the actual eval
numbers (including where the probe struggles), and the latency benchmark.
