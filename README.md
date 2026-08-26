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
| 4 | Containerize (Docker + kind/minikube) + Terraform | [done](docs/PHASE4.md) |
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

## Quickstart (Phase 4)

Dockerfile for the FastAPI service (476MB, CPU-only torch -- down from
3.19GB with the default CUDA-bundled wheel), a basic Kubernetes Deployment
+ Service, and a Terraform configuration managing those same two resources
against a local `kind` cluster. No real cloud spend -- see docs/PHASE4.md
for why local kind was chosen and what that trades off.

```bash
docker build -t activation-probe:latest .
kind create cluster --config k8s/kind-config.yaml
kind load docker-image activation-probe:latest --name activation-probe

# either apply the raw manifests directly...
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
# ...or manage the same resources with Terraform instead (not both at once)
cd terraform && terraform init && terraform apply

curl localhost:30080/health
```

Both paths were actually run against a real kind cluster and verified end
to end (pod healthy, `/classify` returns a real score, and the request
lands in the same Postgres `classification_log` table from Phase 1/2/3) --
see [docs/PHASE4.md](docs/PHASE4.md) for the verification evidence and the
honestly-stated limitations (no PVC for the model cache, no registry, no
autoscaling -- this is a "basic Deployment + Service," not a production
deployment).
