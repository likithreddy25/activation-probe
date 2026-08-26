# Manages the k8s Deployment + Service for the activation-probe service on
# a local kind cluster. Deliberately does NOT manage the kind cluster
# itself -- cluster bootstrap (`kind create cluster --config
# ../k8s/kind-config.yaml`) is a separate, one-time dev-environment step,
# not something this state should own the lifecycle of. This mirrors how
# the equivalent AWS setup would look: Terraform managing k8s resources
# against an existing EKS cluster, not provisioning the cluster in the same
# apply. See docs/PHASE4.md for why local kind was chosen over touching
# real cloud spend.

terraform {
  required_version = ">= 1.5"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.31"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}
