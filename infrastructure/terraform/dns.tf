# A2A World Platform - DNS Configuration
# Cloudflare DNS Records

# Main domain
resource "cloudflare_record" "root" {
  zone_id = data.cloudflare_zone.domain.id
  name    = "@"
  value   = digitalocean_loadbalancer.web.ip
  type    = "A"
  ttl     = 300
}

# www subdomain
resource "cloudflare_record" "www" {
  zone_id = data.cloudflare_zone.domain.id
  name    = "www"
  value   = digitalocean_loadbalancer.web.ip
  type    = "A"
  ttl     = 300
}

# API subdomain
resource "cloudflare_record" "api" {
  zone_id = data.cloudflare_zone.domain.id
  name    = "api"
  value   = digitalocean_loadbalancer.web.ip
  type    = "A"
  ttl     = 300
}

# Monitoring subdomain (if enabled)
resource "cloudflare_record" "monitoring" {
  count   = var.enable_monitoring ? 1 : 0
  zone_id = data.cloudflare_zone.domain.id
  name    = "monitoring"
  value   = digitalocean_droplet.monitoring[0].ipv4_address
  type    = "A"
  ttl     = 300
}

# Data source for the domain
data "cloudflare_zone" "domain" {
  name = var.domain_name
}

# CAA records for SSL certificate validation
resource "cloudflare_record" "caa_letsencrypt" {
  zone_id = data.cloudflare_zone.domain.id
  name    = "@"
  type    = "CAA"
  ttl     = 300
  
  data = {
    flags = "0"
    tag   = "issue"
    value = "letsencrypt.org"
  }
}

resource "cloudflare_record" "caa_digicert" {
  zone_id = data.cloudflare_zone.domain.id
  name    = "@"
  type    = "CAA"
  ttl     = 300
  
  data = {
    flags = "0"
    tag   = "issue"
    value = "digicert.com"
  }
}

# Page Rules for performance optimization
resource "cloudflare_page_rule" "api_cache" {
  zone_id  = data.cloudflare_zone.domain.id
  target   = "api.${var.domain_name}/api/v1/health"
  priority = 1
  
  actions {
    cache_level = "cache_everything"
    edge_cache_ttl = 300
  }
}

resource "cloudflare_page_rule" "static_assets" {
  zone_id  = data.cloudflare_zone.domain.id
  target   = "${var.domain_name}/_next/static/*"
  priority = 2
  
  actions {
    cache_level = "cache_everything"
    edge_cache_ttl = 31536000 # 1 year
  }
}

# Security settings
resource "cloudflare_zone_settings_override" "security" {
  zone_id = data.cloudflare_zone.domain.id
  
  settings {
    ssl                      = "strict"
    always_use_https         = "on"
    automatic_https_rewrites = "on"
    security_level           = "medium"
    browser_check            = "on"
    challenge_ttl            = 1800
    development_mode         = "off"
    email_obfuscation        = "on"
    hotlink_protection       = "off"
    ip_geolocation           = "on"
    ipv6                     = "on"
    min_tls_version          = "1.2"
    opportunistic_encryption = "on"
    privacy_pass             = "on"
    server_side_exclude      = "on"
    tls_1_3                  = "zrt"
    waf                      = "on"
    
    minify {
      css  = "on"
      html = "on"
      js   = "on"
    }
    
    mobile_redirect {
      status           = "off"
      mobile_subdomain = ""
      strip_uri        = false
    }
    
    security_header {
      enabled            = true
      preload            = true
      max_age            = 31536000
      include_subdomains = true
      nosniff            = true
    }
  }
}