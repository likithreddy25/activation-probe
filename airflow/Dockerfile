# Extends the official Airflow image (see docker-compose.yaml) with:
#   1. An isolated venv for dbt, so dbt-core's dependency tree (jinja2, click,
#      pyyaml, etc., all things Airflow itself also pins) never touches
#      Airflow's own environment. This is the standard fix for the
#      well-known "pip install dbt-core into your Airflow image" conflict
#      problem -- see https://docs.getdbt.com/docs/deploy/deployment-tools
#   2. The light, torch-free dependencies the eval/drift PythonOperator tasks
#      need (numpy, scikit-learn, joblib, sqlalchemy, psycopg2, dotenv).
#      Those tasks only score cached activations and query Postgres, so they
#      deliberately don't need torch/transformers/nnsight in this image.
FROM apache/airflow:2.10.5-python3.11

USER root
RUN python -m venv /opt/dbt_venv \
    && /opt/dbt_venv/bin/pip install --no-cache-dir dbt-core==1.8.* dbt-postgres==1.8.* \
    && chmod -R a+rX /opt/dbt_venv

USER airflow
COPY requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt
