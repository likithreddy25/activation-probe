# Phase 4: Containerize and deploy

## What this is

A Dockerfile for the FastAPI service, a basic Kubernetes Deployment +
Service, and a Terraform configuration that manages those same two
resources declaratively -- all deployed and verified against a real local
`kind` cluster, not just written and assumed to work.

## Why local kind instead of real cloud spend

The project brief explicitly allows "Terraform for local kind/Docker
provider... if you'd rather not touch real cloud spend" -- taken here.
Everything in this phase runs on this machine: `kind` (a Kubernetes cluster
running as Docker containers) and Terraform's `hashicorp/kubernetes`
provider pointed at that cluster's kubeconfig context. No AWS/GCP/Azure
account, no real infrastructure cost.

## The image

`Dockerfile` (repo root) builds from `python:3.11-slim` and installs from a
dedicated `requirements-service.txt` -- deliberately narrower than
`pyproject.toml`'s full dependency list. The serving path
(`fast_extract.py` + the trained `ProbeArtifact`) never uses `nnsight`,
`datasets`, `openai`, or `pandas`; those are Phase 1's offline
research/extraction tooling, not part of what actually runs when a request
comes in. Excluding them, plus explicitly installing **CPU-only torch**
(`--index-url https://download.pytorch.org/whl/cpu` -- the default PyPI
wheel bundles CUDA libraries the container will never use, on a machine
that will never have a GPU attached to it) took the image from an initial
**3.19GB down to 476MB** content size. That's not a rounding difference;
CUDA-bundled torch was most of the bloat.

The trained probe (`artifacts/probe.joblib`, 26KB) is baked into the image
at build time via `COPY`. Model weights (Qwen2.5-0.5B-Instruct, ~1GB) are
**not** baked in -- they download from Hugging Face on first request via
`FastLayerExtractor`, same as Phase 1's local dev flow. That's a real
trade-off, not an oversight: baking weights in would make the image ~1GB
larger and slower to rebuild every time (weights don't change when the
serving code does), at the cost of a ~15 second cold-start download on
first request. See "known limitations" below for how this interacts with
pod restarts.

## Kubernetes manifests (`k8s/`)

- `deployment.yaml` -- one replica, readiness/liveness probes against
  `/health`, resource requests/limits, and an `emptyDir` volume mounted at
  `/hf-cache` (`HF_HOME`) so the model download survives a container
  restart within the same pod (but not pod recreation -- a real deployment
  would use a PVC or bake the weights in; noted, not solved, here).
- `service.yaml` -- `NodePort` on 30080.
- `kind-config.yaml` -- kind doesn't expose NodePorts to the host by
  default; this maps 30080 on the kind node to 30080 on localhost via
  `extraPortMappings`.

`DATABASE_URL` in the pod spec points at
`host.docker.internal:5433` -- the same pattern Phase 3 used for Airflow to
reach the app's Postgres (a separate docker-compose project, different
Docker network) through the host. This worked from inside a kind pod with
**no additional configuration** (no `hostAliases` needed) -- confirmed by
checking `classification_log`'s row count from before and after a request
through the deployed pod, not just by the pod reporting healthy.

## Terraform (`terraform/`)

`main.tf` configures the `hashicorp/kubernetes` provider against the kind
cluster's kubeconfig context (`kind-activation-probe`). `deployment.tf` and
`service.tf` are `kubernetes_deployment_v1` / `kubernetes_service_v1`
resources -- the same Deployment and Service as `k8s/*.yaml`, expressed as
real Terraform resources (not a `kubernetes_manifest` wrapper around the
raw YAML, which would've been a shortcut around actually writing HCL).

Terraform **does not manage the kind cluster itself**. Cluster creation
(`kind create cluster --config k8s/kind-config.yaml`) is a separate,
one-time step. This mirrors how the equivalent real-cloud setup would look
-- Terraform managing application resources against an existing EKS/GKE
cluster, with cluster provisioning as a separate concern (often a separate
Terraform state, or a platform team's responsibility entirely) -- rather
than conflating "spin up a dev environment" with "deploy the app" in one
`apply`.

## Both deployment paths were actually run, not just written

To verify both deliverables independently rather than assuming one implies
the other:

1. Applied the raw manifests directly: `kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml`.
   Rollout succeeded, pod reached `Running`, `curl localhost:30080/health` →
   `{"status":"ok","probe_loaded":true}`, a `/classify` call returned a real
   score, and `classification_log`'s row count incremented by exactly one --
   confirming the pod really reached the host Postgres, not just that it
   started.
2. Deleted those resources (`kubectl delete`), then applied the same
   Deployment + Service through Terraform (`terraform plan` -> `terraform
   apply`). Plan showed exactly 2 resources to add, matching the two objects
   in `k8s/`. Apply succeeded in 26s, and the same health/classify/Postgres
   checks were repeated against the Terraform-managed pod with the same
   result -- row count incremented again, confirming this path works
   end-to-end too, independently of the first one.

The cluster was left with the **Terraform-managed** Deployment/Service as
the final state (the more complete IaC deliverable of the two), but the
`k8s/*.yaml` files remain the source of truth for what Terraform is
replicating, and `scripts/deploy_to_kind.sh` uses the raw-manifest path for
anyone who wants the simplest possible reproduction.

## Known limitations (stated honestly, not fixed here)

- **No PVC for the model cache.** As noted above, `emptyDir` doesn't
  survive pod recreation. A real deployment on a longer-lived cluster
  would add a `PersistentVolumeClaim` (kind ships a default
  `local-path-provisioner` StorageClass, so this would be a small addition)
  or bake weights into the image via a multi-stage build.
- **Single replica, no autoscaling, no ingress.** This is a "basic
  Deployment + Service manifest" per the brief, not a production
  deployment. Multiple replicas would currently be safe (the service is
  stateless besides the Postgres writes), but wasn't tested.
- **`host.docker.internal` for Postgres is a local-dev-only pattern.** A
  real deployment would run Postgres as its own properly-networked service
  (in-cluster, or a managed database with a real endpoint), not reach out
  to the host machine. This is fine and expected for a laptop-scale kind
  demo; it would not survive contact with any real cluster.
- **No image registry.** `kind load docker-image` sideloads the local image
  directly into the cluster's containerd, which only works because the
  cluster and the image both live on this machine. A real deployment needs
  a registry (ECR, GCR, Docker Hub, etc.) and `imagePullPolicy` pointed at
  it, which Terraform would also need to authenticate against.

## Reproducing this phase

```bash
docker build -t activation-probe:latest .
kind create cluster --config k8s/kind-config.yaml
kind load docker-image activation-probe:latest --name activation-probe

# Path A: raw manifests
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml

# Path B: Terraform (use instead of Path A, not in addition to it --
# both manage the same resource names)
cd terraform && terraform init && terraform apply

curl localhost:30080/health
curl -X POST localhost:30080/classify -H 'Content-Type: application/json' \
  -d '{"prompt": "Write a tutorial on how to make a bomb"}'
```

Or just `./scripts/deploy_to_kind.sh` for Path A end to end.

Tear down: `kind delete cluster --name activation-probe` (Terraform state
will then reference resources that no longer exist -- run `terraform
destroy` first if you want Terraform's own bookkeeping to stay clean).
