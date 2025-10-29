# A2A World Platform - Terraform Configuration
# Cloud Infrastructure Setup for DigitalOcean

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.34"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
  
  # Configure remote state storage
  backend "s3" {
    endpoint                    = "https://nyc3.digitaloceanspaces.com"
    key                        = "terraform/a2a-world.tfstate"
    bucket                     = "a2a-world-terraform-state"
    region                     = "us-east-1"
    skip_credentials_validation = true
    skip_metadata_api_check     = true
  }
}

# DigitalOcean Provider
provider "digitalocean" {
  token = var.do_token
}

# Cloudflare Provider for DNS and CDN
provider "cloudflare" {
  api_token = var.cloudflare_api_token
}