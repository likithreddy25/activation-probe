#!/usr/bin/env bash
# One-shot script for the Phase 4 local deploy: build the image, create the
# kind cluster if it doesn't exist, load the image in (kind clusters don't
# see the host's Docker image cache), then apply the k8s manifests.
#
# Usage: ./scripts/deploy_to_kind.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== building image =="
docker build -t activation-probe:latest .

if ! kind get clusters | grep -q '^activation-probe$'; then
  echo "== creating kind cluster =="
  kind create cluster --config k8s/kind-config.yaml
else
  echo "== kind cluster already exists =="
fi

echo "== loading image into kind =="
kind load docker-image activation-probe:latest --name activation-probe

echo "== applying manifests =="
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

echo "== waiting for rollout =="
kubectl rollout status deployment/activation-probe --timeout=180s

echo "== done. service at http://localhost:30080 =="
