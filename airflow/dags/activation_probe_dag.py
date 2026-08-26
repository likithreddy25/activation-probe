"""Nightly pipeline for the activation-probe project.

1. rerun_eval_harness -- score the current probe against the held-out JBB
   test split and record accuracy/AUC to Postgres (eval_history table).
2. trigger_dbt_run -- rebuild the Phase 2 dbt marts (staging + daily flag
   rate + category breakdown + FPR trend) so they reflect anything new in
   classification_log or eval_history.
3. check_drift_and_alert -- compare the freshly written eval_history row
   against absolute floors and a rolling baseline, and the freshly rebuilt
   mart_daily_flag_rate against its own rolling baseline; log an Alert row
   (+ stub notification print) for anything that crosses a threshold.

Tasks 1 and 3 run in Airflow's own Python environment (they only need
numpy/sklearn/joblib/sqlalchemy -- see airflow/Dockerfile for why that's
deliberately kept separate from dbt's dependency tree). Task 2 shells out to
an isolated dbt venv also baked into the image, to avoid dbt-core's
dependencies conflicting with Airflow's own.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

DBT_PROJECT_DIR = "/opt/activation-probe/dbt"
DBT_BIN = "/opt/dbt_venv/bin/dbt"


@dag(
    dag_id="activation_probe_nightly",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 8, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,  # two overlapping runs both `dbt run`-ing race on the
    # same relation and can fail with "relation ..._dbt_backup already exists"
    tags=["activation-probe"],
)
def activation_probe_nightly():
    @task(task_id="rerun_eval_harness")
    def rerun_eval_harness() -> dict:
        from activation_probe.eval.nightly import run_nightly_eval

        report = run_nightly_eval()
        return {"accuracy": report.accuracy, "auc": report.auc, "n_examples": report.n_examples}

    trigger_dbt_run = BashOperator(
        task_id="trigger_dbt_run",
        bash_command=(
            f"{DBT_BIN} run --project-dir {DBT_PROJECT_DIR} --profiles-dir {DBT_PROJECT_DIR} && "
            f"{DBT_BIN} test --project-dir {DBT_PROJECT_DIR} --profiles-dir {DBT_PROJECT_DIR}"
        ),
    )

    @task(task_id="check_drift_and_alert")
    def check_drift_and_alert(eval_result: dict) -> list[str]:
        from activation_probe.eval.drift import check_and_alert

        print(f"eval result feeding this check: {eval_result}")
        raised = check_and_alert()
        return [a.alert_type for a in raised]

    eval_result = rerun_eval_harness()
    eval_result >> trigger_dbt_run >> check_drift_and_alert(eval_result)


activation_probe_nightly()
