variable "kubeconfig_path" {
  description = "Path to the kubeconfig file kind wrote (kind writes into the default ~/.kube/config unless told otherwise)."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "kubeconfig context for the kind cluster -- kind names it kind-<cluster-name>."
  type        = string
  default     = "kind-activation-probe"
}

variable "image" {
  description = "Docker image for the service, already loaded into the kind cluster via `kind load docker-image`."
  type        = string
  default     = "activation-probe:latest"
}

variable "replicas" {
  description = "Pod replica count."
  type        = number
  default     = 1
}

variable "database_url" {
  description = "Postgres connection string for request logging, reaching the host's docker-compose Postgres from inside kind."
  type        = string
  default     = "postgresql://probe:probe@host.docker.internal:5433/activation_probe"
}

variable "node_port" {
  description = "NodePort the Service exposes -- must match k8s/kind-config.yaml's extraPortMappings for host access."
  type        = number
  default     = 30080
}
