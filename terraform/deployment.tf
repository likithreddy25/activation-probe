resource "kubernetes_deployment_v1" "activation_probe" {
  metadata {
    name   = "activation-probe"
    labels = { app = "activation-probe" }
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = { app = "activation-probe" }
    }

    template {
      metadata {
        labels = { app = "activation-probe" }
      }

      spec {
        container {
          name              = "activation-probe"
          image             = var.image
          image_pull_policy = "IfNotPresent"

          port {
            container_port = 8000
          }

          env {
            name  = "DATABASE_URL"
            value = var.database_url
          }
          env {
            name  = "HF_HOME"
            value = "/hf-cache"
          }

          volume_mount {
            name       = "hf-cache"
            mount_path = "/hf-cache"
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "1Gi"
            }
            limits = {
              cpu    = "1"
              memory = "2Gi"
            }
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 15
            period_seconds        = 10
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 30
            period_seconds         = 20
          }
        }

        volume {
          name = "hf-cache"
          empty_dir {}
        }
      }
    }
  }
}
