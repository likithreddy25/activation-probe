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
| 2 | dbt models on the Postgres log table | [done](docs/PHASE2.md) |
| 3 | Airflow DAG: nightly eval reruns, dbt trigger, drift alerts | [done](docs/PHASE3.md) |
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

## Quickstart (Phase 2)

Builds on Phase 1's Postgres instance -- adds a dbt project that turns the
raw `classification_log` table into a daily flag-rate rollup, a
per-category breakdown, and a rolling false-positive-rate trend computed
against labeled ground truth.

```bash
python scripts/build_dbt_seed.py                    # builds dbt/seeds/labeled_eval_examples.csv
python scripts/seed_traffic.py --days 14 --clear     # replays labeled data through the real probe into classification_log

cd dbt
set -a && source ../.env && set +a
DBT_PROFILES_DIR=. dbt seed
DBT_PROFILES_DIR=. dbt run
DBT_PROFILES_DIR=. dbt test
```

See [docs/PHASE2.md](docs/PHASE2.md) for the model layout, the actual mart
output, and why the FPR trend is computed against a labeled seed table
rather than raw flag counts.

## Quickstart (Phase 3)

A local Airflow stack (LocalExecutor, not the default CeleryExecutor --
see docs/PHASE3.md for why) running one nightly DAG: rerun the eval
harness, trigger the Phase 2 dbt run, then check for accuracy/AUC/flag-rate
drift and log an alert if anything crosses a threshold.

```bash
cd airflow
cp .env.example .env
docker compose build
docker compose up airflow-init                              # one-time
docker compose up -d airflow-webserver airflow-scheduler
docker compose exec airflow-scheduler airflow dags unpause activation_probe_nightly
docker compose exec airflow-scheduler airflow dags trigger activation_probe_nightly
```

UI at http://localhost:8081 (airflow/airflow). See
[docs/PHASE3.md](docs/PHASE3.md) for the two real bugs this phase found
(a SQLAlchemy version conflict with Airflow's own pinned version, and a
genuine dbt race condition between overlapping DAG runs) and the actual
run evidence, including a deliberate synthetic test proving the drift
alerting really fires.
