# A2A World Platform - Terraform Outputs

# Network Information
output "vpc_id" {
  description = "ID of the VPC"
  value       = digitalocean_vpc.main.id
}

output "vpc_ip_range" {
  description = "IP range of the VPC"
  value       = digitalocean_vpc.main.ip_range
}

# Load Balancer
output "load_balancer_ip" {
  description = "IP address of the load balancer"
  value       = digitalocean_loadbalancer.web.ip
}

output "load_balancer_status" {
  description = "Status of the load balancer"
  value       = digitalocean_loadbalancer.web.status
}

# Database Information
output "database_host" {
  description = "Database host endpoint"
  value       = digitalocean_database_cluster.postgres.host
  sensitive   = true
}

output "database_port" {
  description = "Database port"
  value       = digitalocean_database_cluster.postgres.port
}

output "database_name" {
  description = "Database name"
  value       = digitalocean_database_db.a2a_world.name
}

output "database_user" {
  description = "Database username"
  value       = digitalocean_database_user.a2a_user.name
  sensitive   = true
}

output "database_password" {
  description = "Database password"
  value       = digitalocean_database_user.a2a_user.password
  sensitive   = true
}

output "database_connection_string" {
  description = "Full database connection string"
  value       = "postgresql://${digitalocean_database_user.a2a_user.name}:${digitalocean_database_user.a2a_user.password}@${digitalocean_database_cluster.postgres.private_host}:${digitalocean_database_cluster.postgres.port}/${digitalocean_database_db.a2a_world.name}?sslmode=require"
  sensitive   = true
}

# Application Servers
output "app_server_ips" {
  description = "IP addresses of application servers"
  value = {
    public  = digitalocean_droplet.app[*].ipv4_address
    private = digitalocean_droplet.app[*].ipv4_address_private
  }
}

output "app_server_names" {
  description = "Names of application servers"
  value       = digitalocean_droplet.app[*].name
}

# Monitoring Server
output "monitoring_server_ip" {
  description = "IP address of monitoring server"
  value       = var.enable_monitoring ? digitalocean_droplet.monitoring[0].ipv4_address : "Not deployed"
}

# Container Registry
output "container_registry_endpoint" {
  description = "Container registry endpoint"
  value       = digitalocean_container_registry.a2a_registry.endpoint
}

output "container_registry_server_url" {
  description = "Container registry server URL"
  value       = digitalocean_container_registry.a2a_registry.server_url
}

# Spaces (Object Storage)
output "spaces_bucket_name" {
  description = "Spaces bucket name for storage"
  value       = digitalocean_spaces_bucket.storage.name
}

output "spaces_bucket_domain" {
  description = "Spaces bucket domain name"
  value       = digitalocean_spaces_bucket.storage.bucket_domain_name
}

output "spaces_endpoint" {
  description = "Spaces endpoint URL"
  value       = "https://${var.region}.digitaloceanspaces.com"
}

# DNS Information
output "domain_records" {
  description = "DNS records created"
  value = {
    root        = "${var.domain_name} -> ${digitalocean_loadbalancer.web.ip}"
    www         = "www.${var.domain_name} -> ${digitalocean_loadbalancer.web.ip}"
    api         = "api.${var.domain_name} -> ${digitalocean_loadbalancer.web.ip}"
    monitoring  = var.enable_monitoring ? "monitoring.${var.domain_name} -> ${digitalocean_droplet.monitoring[0].ipv4_address}" : "Not configured"
  }
}

# SSL Certificate
output "ssl_certificate_id" {
  description = "SSL certificate ID"
  value       = digitalocean_certificate.cert.id
}

output "ssl_certificate_status" {
  description = "SSL certificate status"
  value       = digitalocean_certificate.cert.state
}

# Project Information
output "project_id" {
  description = "DigitalOcean project ID"
  value       = digitalocean_project.a2a_world.id
}

# Cost Estimation
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown"
  value = {
    droplets          = "$${var.droplet_count * (var.droplet_size == "s-2vcpu-4gb" ? 24 : 12)}"
    database          = "$${var.db_size == "db-s-1vcpu-1gb" ? 15 : 30}"
    load_balancer     = "$${var.lb_size == "lb-small" ? 12 : 20}"
    container_registry = "$5"
    monitoring        = var.enable_monitoring ? "$12" : "$0"
    spaces            = "$5 + usage"
    backup_storage    = "$1-5 (estimated)"
    total_estimated   = "$${var.droplet_count * (var.droplet_size == "s-2vcpu-4gb" ? 24 : 12) + (var.db_size == "db-s-1vcpu-1gb" ? 15 : 30) + (var.lb_size == "lb-small" ? 12 : 20) + 5 + (var.enable_monitoring ? 12 : 0) + 8}-${var.droplet_count * (var.droplet_size == "s-2vcpu-4gb" ? 24 : 12) + (var.db_size == "db-s-1vcpu-1gb" ? 15 : 30) + (var.lb_size == "lb-small" ? 12 : 20) + 5 + (var.enable_monitoring ? 12 : 0) + 15}"
  }
}

# Environment Configuration
output "environment_variables" {
  description = "Environment variables for application deployment"
  value = {
    DATABASE_URL = "postgresql://${digitalocean_database_user.a2a_user.name}:${digitalocean_database_user.a2a_user.password}@${digitalocean_database_cluster.postgres.private_host}:${digitalocean_database_cluster.postgres.port}/${digitalocean_database_db.a2a_world.name}?sslmode=require"
    REDIS_URL    = "redis://localhost:6379"
    NATS_URL     = "nats://localhost:4222"
    CONSUL_HOST  = "localhost"
    CONSUL_PORT  = "8500"
    ENVIRONMENT  = var.environment
    DOMAIN_NAME  = var.domain_name
    CDN_URL      = "https://${digitalocean_spaces_bucket.storage.bucket_domain_name}"
  }
  sensitive = true
}