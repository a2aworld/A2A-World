# A2A World Platform - Main Infrastructure
# DigitalOcean Resources

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
  
  name_prefix = "${var.project_name}-${var.environment}"
}

# VPC Network
resource "digitalocean_vpc" "main" {
  name     = "${local.name_prefix}-vpc"
  region   = var.region
  ip_range = "10.0.0.0/16"
}

# SSH Key
resource "digitalocean_ssh_key" "main" {
  count      = length(var.ssh_keys) > 0 ? 0 : 1
  name       = "${local.name_prefix}-key"
  public_key = file("~/.ssh/id_rsa.pub")
}

# Database Cluster (PostgreSQL with PostGIS)
resource "digitalocean_database_cluster" "postgres" {
  name       = "${local.name_prefix}-postgres"
  engine     = "pg"
  version    = "15"
  size       = var.db_size
  region     = var.region
  node_count = var.db_node_count
  
  private_network_uuid = digitalocean_vpc.main.id
  
  # Enable automatic backups
  backup_restore {
    database_name = "a2a_world"
  }
  
  tags = [var.project_name, var.environment, "database"]
}

# Database
resource "digitalocean_database_db" "a2a_world" {
  cluster_id = digitalocean_database_cluster.postgres.id
  name       = "a2a_world"
}

# Database User
resource "digitalocean_database_user" "a2a_user" {
  cluster_id = digitalocean_database_cluster.postgres.id
  name       = "a2a_user"
}

# Container Registry
resource "digitalocean_container_registry" "a2a_registry" {
  name                   = "${var.project_name}registry"
  subscription_tier_slug = "basic" # $5/month
  region                 = var.region
}

# Spaces (Object Storage)
resource "digitalocean_spaces_bucket" "storage" {
  name   = "${local.name_prefix}-storage"
  region = var.region
  
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE"]
    allowed_origins = ["https://${var.domain_name}", "https://www.${var.domain_name}"]
    max_age_seconds = 3600
  }
  
  lifecycle_rule {
    id      = "cleanup"
    enabled = true
    
    expiration {
      days = 365
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  versioning {
    enabled = true
  }
}

# Spaces for Terraform State
resource "digitalocean_spaces_bucket" "terraform_state" {
  name   = "${local.name_prefix}-terraform-state"
  region = var.region
  
  versioning {
    enabled = true
  }
}

# Load Balancer
resource "digitalocean_loadbalancer" "web" {
  name   = "${local.name_prefix}-lb"
  region = var.region
  size   = var.lb_size
  
  vpc_uuid = digitalocean_vpc.main.id
  
  forwarding_rule {
    entry_protocol  = "https"
    entry_port      = 443
    target_protocol = "http"
    target_port     = 80
    certificate_name = digitalocean_certificate.cert.name
  }
  
  forwarding_rule {
    entry_protocol  = "http"
    entry_port      = 80
    target_protocol = "http"
    target_port     = 80
  }
  
  healthcheck {
    protocol                 = "http"
    port                     = 80
    path                     = "/health"
    check_interval_seconds   = 10
    response_timeout_seconds = 5
    unhealthy_threshold      = 3
    healthy_threshold        = 2
  }
  
  droplet_ids = digitalocean_droplet.app[*].id
}

# SSL Certificate
resource "digitalocean_certificate" "cert" {
  name    = "${local.name_prefix}-cert"
  type    = "lets_encrypt"
  domains = [var.domain_name, "www.${var.domain_name}", "api.${var.domain_name}"]
}

# Application Droplets
resource "digitalocean_droplet" "app" {
  count  = var.droplet_count
  image  = "docker-20-04" # Ubuntu 20.04 with Docker pre-installed
  name   = "${local.name_prefix}-app-${count.index + 1}"
  region = var.region
  size   = var.droplet_size
  
  vpc_uuid = digitalocean_vpc.main.id
  ssh_keys = length(var.ssh_keys) > 0 ? var.ssh_keys : [digitalocean_ssh_key.main[0].fingerprint]
  
  tags = [var.project_name, var.environment, "app"]
  
  user_data = templatefile("${path.module}/cloud-init/app-server.yml", {
    project_name = var.project_name
    environment  = var.environment
  })
  
  # Enable backups
  backups = true
  
  # Enable monitoring
  monitoring = true
}

# Monitoring Droplet (optional)
resource "digitalocean_droplet" "monitoring" {
  count = var.enable_monitoring ? 1 : 0
  
  image  = "ubuntu-22-04-x64"
  name   = "${local.name_prefix}-monitoring"
  region = var.region
  size   = var.monitoring_size
  
  vpc_uuid = digitalocean_vpc.main.id
  ssh_keys = length(var.ssh_keys) > 0 ? var.ssh_keys : [digitalocean_ssh_key.main[0].fingerprint]
  
  tags = [var.project_name, var.environment, "monitoring"]
  
  user_data = templatefile("${path.module}/cloud-init/monitoring-server.yml", {
    project_name = var.project_name
    environment  = var.environment
  })
  
  # Enable monitoring
  monitoring = true
}

# Firewall - Web Traffic
resource "digitalocean_firewall" "web" {
  name = "${local.name_prefix}-web-fw"
  
  droplet_ids = digitalocean_droplet.app[*].id
  
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.allowed_ips
  }
  
  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # Internal communication
  inbound_rule {
    protocol    = "tcp"
    port_range  = "1-65535"
    source_tags = [var.project_name]
  }
  
  inbound_rule {
    protocol    = "udp"
    port_range  = "1-65535"
    source_tags = [var.project_name]
  }
  
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

# Firewall - Monitoring
resource "digitalocean_firewall" "monitoring" {
  count = var.enable_monitoring ? 1 : 0
  name  = "${local.name_prefix}-monitoring-fw"
  
  droplet_ids = [digitalocean_droplet.monitoring[0].id]
  
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.allowed_ips
  }
  
  # Prometheus
  inbound_rule {
    protocol    = "tcp"
    port_range  = "9090"
    source_tags = [var.project_name]
  }
  
  # Grafana
  inbound_rule {
    protocol    = "tcp"
    port_range  = "3000"
    source_tags = [var.project_name]
  }
  
  # Node Exporter
  inbound_rule {
    protocol    = "tcp"
    port_range  = "9100"
    source_tags = [var.project_name]
  }
  
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

# Project Resource Assignment
resource "digitalocean_project" "a2a_world" {
  name        = "${local.name_prefix} Platform"
  description = "A2A World Platform Infrastructure"
  purpose     = "Web Application"
  environment = var.environment
  
  resources = concat(
    [
      digitalocean_loadbalancer.web.urn,
      digitalocean_database_cluster.postgres.urn,
      digitalocean_spaces_bucket.storage.urn,
      digitalocean_spaces_bucket.terraform_state.urn,
    ],
    digitalocean_droplet.app[*].urn,
    var.enable_monitoring ? digitalocean_droplet.monitoring[*].urn : []
  )
}