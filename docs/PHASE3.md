# Phase 3: Airflow DAG

## What this is

A single Airflow DAG (`activation_probe_nightly`, `@daily`) with three tasks,
run in a local Airflow stack started from an adapted version of the
official docker-compose setup:

1. **`rerun_eval_harness`** -- scores the current probe against the
   held-out JBB test split and writes accuracy/AUC/F1/FPR/FNR to a new
   `eval_history` Postgres table.
2. **`trigger_dbt_run`** -- reruns the Phase 2 dbt project (`dbt run` +
   `dbt test`) so the marts reflect whatever's new in `classification_log`
   or `eval_history`.
3. **`check_drift_and_alert`** -- compares the fresh eval numbers against
   an absolute floor and a rolling baseline of prior runs, and the fresh
   `mart_daily_flag_rate` against its own rolling baseline; logs an `Alert`
   row (+ a stub notification print) for anything that crosses a threshold.

## Why LocalExecutor instead of the stock CeleryExecutor setup

The official `docker-compose.yaml` defaults to CeleryExecutor: Postgres +
Redis + webserver + scheduler + worker + triggerer + optional flower, 7
services. That's real infrastructure for a multi-worker production
deployment, but this machine's Docker VM has 7.75GB allocated and there's
already another Postgres container running for the app. `LocalExecutor`
runs tasks as subprocesses of the scheduler directly -- no broker, no
separate worker -- which drops the stack to 3 services (Postgres, webserver,
scheduler) and is still a fully-supported, documented Airflow executor, not
a toy substitute. This is the one deliberate deviation from "official
docker-compose setup," and it's called out here rather than silently
switched.

The custom image (`airflow/Dockerfile`) also adds an **isolated venv for
dbt** (`/opt/dbt_venv`) rather than pip-installing `dbt-core` into Airflow's
own environment -- dbt's dependency tree (jinja2, click, pyyaml, protobuf...)
overlaps heavily with Airflow's own pins, and this is the standard way to
avoid that conflict rather than fight it. The whole repo is bind-mounted
into the containers at `/opt/activation-probe`, so the DAG imports
`activation_probe` directly (via `PYTHONPATH`) and shells out to the dbt
project in place, instead of copying code into the image at build time.

## Why two of the tasks don't need torch

`rerun_eval_harness` and `check_drift_and_alert` run in Airflow's own
Python environment (`airflow/requirements-airflow.txt`: numpy,
scikit-learn, joblib, SQLAlchemy, psycopg2-binary, python-dotenv -- nothing
else). They're able to stay this light because the eval harness scores
*cached activations* (`data/cache/test.npz`), not raw prompts -- no
tokenizer, no forward pass, no torch/transformers/nnsight anywhere in this
image. That's a real, not incidental, benefit of separating activation
extraction (Phase 1, needs the model) from scoring (Phases 1 and 3, only
needs the cached vectors and the trained logistic regression).

## Connecting to the app's Postgres from a separate compose project

`airflow/docker-compose.yaml` is a second, independent compose project from
the top-level one, so its containers aren't on the same Docker network as
`activation-probe-postgres`. Rather than merge the two compose files (which
would couple Airflow's lifecycle to the app's), the Airflow containers reach
the app's Postgres through the host, via `host.docker.internal:5433` (the
same port the app publishes to the host in Phase 1/2) -- set as
`POSTGRES_HOST=host.docker.internal` and a matching `DATABASE_URL` in
`airflow/docker-compose.yaml`'s environment block, plus
`extra_hosts: host.docker.internal:host-gateway` so that name resolves.

## Two real bugs this phase found

**1. SQLAlchemy 1.4 vs. 2.0 API conflict.** `db.py` used SQLAlchemy 2.0's
`DeclarativeBase`/`Mapped`/`mapped_column` syntax, written and tested only
against the project's own conda env (SQLAlchemy 2.x). Airflow 2.10.5 pins
SQLAlchemy to the 1.4 line, and 1.4 doesn't have `DeclarativeBase` --
`rerun_eval_harness` failed on import with
`ImportError: cannot import name 'DeclarativeBase'` the first time it ran
inside the Airflow container. Forcing an upgrade of Airflow's own pinned
SQLAlchemy felt like the wrong fix (real risk to Airflow's DB layer for a
portfolio project's convenience). Fixed by rewriting `db.py`'s models with
the legacy `Column`/`declarative_base()` API, which works unmodified under
both SQLAlchemy 1.4 and 2.0 -- confirmed by rerunning the full Phase 1/2
test suite (still 13/13 passing) under the project's own SQLAlchemy 2.x env
after the change.

**2. A real race condition between overlapping DAG runs.** Unpausing the
DAG caused Airflow to schedule a catch-up run for the most recent missed
`@daily` interval, which started at nearly the same moment as a manual
trigger issued for testing. Both ran `trigger_dbt_run` concurrently, and
one failed with a genuine Postgres error:
`relation "stg_classification_log__dbt_backup" already exists` -- dbt's
view-swap mechanism isn't safe against two concurrent `dbt run`s touching
the same relation. Fixed by adding `max_active_runs=1` to the DAG (a real
production concern independent of how this collision was triggered in
testing: a slow run any given night could otherwise still overlap with the
next scheduled trigger). Verified the fix by clearing the failed run and
letting Airflow re-attempt it: it queued behind nothing (the manual run had
already finished) and completed cleanly in 6 seconds. This bug and its fix
are the more interesting finding of this phase; it's the kind of thing that
only shows up under real concurrent execution, not by reading the DAG file.

## Actual run evidence

Four real DAG-triggered eval reruns are in `eval_history`, all identical:

| id | accuracy | auc | f1 | fpr |
|---|---|---|---|---|
| 1 | 0.775 | 0.880 | 0.791 | 0.300 |
| 3 | 0.775 | 0.880 | 0.791 | 0.300 |
| 4 | 0.775 | 0.880 | 0.791 | 0.300 |
| 5 | 0.775 | 0.880 | 0.791 | 0.300 |

These match Phase 1's original numbers exactly, which is the expected and
correct result: it's the same static probe re-scoring the same static
`test.npz` every time. **No drift occurred, and none should have** --
Phase 3's job in this repo is to prove the drift-detection *mechanism*
works, not to manufacture drift that doesn't exist. That mechanism was
verified separately and honestly: a synthetic bad `eval_history` row
(accuracy 0.50, AUC 0.55) was inserted by hand, `check_and_alert()` was run
against it, and it correctly raised three alerts (absolute accuracy floor,
absolute AUC floor, and relative rolling-mean drop) with real stub-webhook
output:

```
[ALERT STUB] would notify: [eval_accuracy_drift] eval accuracy 0.5000 fell below absolute floor 0.65 (value=0.5, threshold=0.65)
[ALERT STUB] would notify: [eval_auc_drift] eval AUC 0.5500 fell below absolute floor 0.75 (value=0.55, threshold=0.75)
[ALERT STUB] would notify: [eval_accuracy_drift] eval accuracy 0.5000 dropped 0.2750 below the 1-run rolling mean 0.7750 (value=0.5, threshold=0.675)
```

The synthetic row and the resulting alert rows were deleted afterward so
`eval_history` and `alerts` only contain genuine runs; this test is
recorded here, not left silently in the tables, so the eval_history table
in a fresh clone of this repo won't show numbers no one can explain.

`mart_daily_flag_rate`, `mart_category_breakdown`, and `mart_fpr_trend` all
rebuilt successfully as part of `trigger_dbt_run` (11, 15, and 15 rows
respectively, matching Phase 2's output -- no new classification traffic
landed between phases, so no change was expected, only confirmation that
the DAG can actually invoke dbt end to end).

## Reproducing this phase

```bash
cd airflow
cp .env.example .env                 # sets AIRFLOW_UID
docker compose build
docker compose up airflow-init       # one-time: migrates Airflow's metadata DB, creates admin user
docker compose up -d airflow-webserver airflow-scheduler
```

UI at http://localhost:8081 (user/pass: `airflow`/`airflow`, the docker-compose default -- change before using this anywhere but localhost). Or from the CLI:

```bash
docker compose exec airflow-scheduler airflow dags unpause activation_probe_nightly
docker compose exec airflow-scheduler airflow dags trigger activation_probe_nightly
docker compose exec airflow-scheduler airflow dags list-runs -d activation_probe_nightly
```

Tear down: `docker compose down` (add `-v` to also drop Airflow's own metadata Postgres volume).
