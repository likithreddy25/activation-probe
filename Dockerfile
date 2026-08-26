# Serves the trained activation probe over HTTP. Deliberately does not
# install nnsight, datasets, or openai -- see requirements-service.txt for
# why the serving image doesn't need the research/offline dependencies.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-service.txt .
# CPU-only torch wheel -- the default PyPI wheel bundles CUDA libraries even
# though this container has no GPU and never will (kind runs on the host's
# Docker Desktop VM), which otherwise bloats the image by several GB for
# nothing.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-service.txt

COPY src/ src/
COPY artifacts/probe.joblib artifacts/probe.joblib

ENV PYTHONPATH=/app/src
ENV PROBE_ARTIFACT_PATH=/app/artifacts/probe.joblib

EXPOSE 8000

CMD ["uvicorn", "activation_probe.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
