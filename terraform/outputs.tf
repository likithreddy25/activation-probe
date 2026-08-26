output "service_url" {
  description = "URL the service is reachable at on the host, via kind's extraPortMappings."
  value       = "http://localhost:${var.node_port}"
}

output "deployment_name" {
  value = kubernetes_deployment_v1.activation_probe.metadata[0].name
}
