# A2A World Platform - Terraform Variables
# Environment Configuration

variable "do_token" {
  description = "DigitalOcean API Token"
  type        = string
  sensitive   = true
}

variable "cloudflare_api_token" {
  description = "Cloudflare API Token"
  type        = string
  sensitive   = true
}

variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "a2a-world"
}

variable "domain_name" {
  description = "Primary domain name"
  type        = string
  default     = "a2aworld.ai"
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "nyc3"
}

# Database Configuration
variable "db_size" {
  description = "Database cluster size"
  type        = string
  default     = "db-s-1vcpu-1gb" # $15/month
}

variable "db_node_count" {
  description = "Number of database nodes"
  type        = number
  default     = 1
}

# Droplet Configuration
variable "droplet_size" {
  description = "Main application droplet size"
  type        = string
  default     = "s-2vcpu-4gb" # $24/month
}

variable "droplet_count" {
  description = "Number of application droplets"
  type        = number
  default     = 2
}

variable "lb_size" {
  description = "Load balancer size"
  type        = string
  default     = "lb-small" # $12/month
}

# Security Configuration
variable "ssh_keys" {
  description = "List of SSH key fingerprints"
  type        = list(string)
  default     = []
}

variable "allowed_ips" {
  description = "List of IPs allowed for admin access"
  type        = list(string)
  default     = ["0.0.0.0/0"] # Restrict in production
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Database backup retention in days"
  type        = number
  default     = 7
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable monitoring droplet"
  type        = bool
  default     = true
}

variable "monitoring_size" {
  description = "Monitoring droplet size"
  type        = string
  default     = "s-1vcpu-2gb" # $12/month
}

# Cost Optimization
variable "auto_scale_min" {
  description = "Minimum number of application droplets"
  type        = number
  default     = 1
}

variable "auto_scale_max" {
  description = "Maximum number of application droplets"
  type        = number
  default     = 3
}