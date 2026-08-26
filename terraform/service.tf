resource "kubernetes_service_v1" "activation_probe" {
  metadata {
    name = "activation-probe"
  }

  spec {
    selector = { app = "activation-probe" }
    type     = "NodePort"

    port {
      port        = 8000
      target_port = 8000
      node_port   = var.node_port
    }
  }
}
