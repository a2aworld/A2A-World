#!/bin/bash
# A2A World Platform - SSL Certificate Renewal Script
# Automated Let's Encrypt certificate renewal with zero downtime

set -euo pipefail

# Configuration
DOMAIN="a2aworld.ai"
EMAIL="admin@a2aworld.ai"
WEBROOT="/var/www/html"
NGINX_CONFIG="/etc/nginx/sites-available/a2a-world"
LOG_FILE="/var/log/a2a-world/ssl-renewal.log"
SLACK_WEBHOOK="${SLACK_WEBHOOK_SECURITY:-}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Notification function
notify() {
    local message="$1"
    local level="${2:-info}"
    
    log "$message"
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"SSL Renewal - $level: $message\"}" \
            "$SLACK_WEBHOOK" || true
    fi
}

# Check if certificate needs renewal
check_certificate() {
    local cert_path="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"
    
    if [[ ! -f "$cert_path" ]]; then
        log "Certificate not found, need to obtain new certificate"
        return 1
    fi
    
    # Check if certificate expires in less than 30 days
    local expiry_date=$(openssl x509 -enddate -noout -in "$cert_path" | cut -d= -f2)
    local expiry_epoch=$(date -d "$expiry_date" +%s)
    local current_epoch=$(date +%s)
    local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    log "Certificate expires in $days_until_expiry days"
    
    if [[ $days_until_expiry -lt 30 ]]; then
        return 1
    fi
    
    return 0
}

# Backup current certificates
backup_certificates() {
    local backup_dir="/backup/ssl/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    if [[ -d "/etc/letsencrypt/live/$DOMAIN" ]]; then
        cp -r "/etc/letsencrypt/live/$DOMAIN" "$backup_dir/"
        log "Certificates backed up to $backup_dir"
    fi
}

# Test nginx configuration
test_nginx_config() {
    if nginx -t; then
        log "Nginx configuration test passed"
        return 0
    else
        log "ERROR: Nginx configuration test failed"
        return 1
    fi
}

# Reload nginx with zero downtime
reload_nginx() {
    if systemctl reload nginx; then
        log "Nginx reloaded successfully"
        return 0
    else
        log "ERROR: Failed to reload nginx"
        return 1
    fi
}

# Obtain or renew certificate
obtain_certificate() {
    log "Starting certificate renewal process"
    
    # Ensure webroot exists
    mkdir -p "$WEBROOT"
    
    # Run certbot
    if certbot certonly \
        --webroot \
        --webroot-path="$WEBROOT" \
        --email "$EMAIL" \
        --agree-tos \
        --no-eff-email \
        --force-renewal \
        -d "$DOMAIN" \
        -d "www.$DOMAIN" \
        -d "api.$DOMAIN" \
        --deploy-hook "/usr/local/bin/ssl-deploy-hook.sh"; then
        
        log "Certificate obtained/renewed successfully"
        return 0
    else
        log "ERROR: Certificate renewal failed"
        return 1
    fi
}

# Verify certificate installation
verify_certificate() {
    local cert_path="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"
    
    if [[ ! -f "$cert_path" ]]; then
        log "ERROR: Certificate file not found after renewal"
        return 1
    fi
    
    # Check certificate validity
    if openssl x509 -in "$cert_path" -text -noout > /dev/null 2>&1; then
        local expiry_date=$(openssl x509 -enddate -noout -in "$cert_path" | cut -d= -f2)
        log "Certificate is valid, expires: $expiry_date"
        
        # Test HTTPS connectivity
        if curl -I "https://$DOMAIN" >/dev/null 2>&1; then
            log "HTTPS connectivity test passed"
            return 0
        else
            log "WARNING: HTTPS connectivity test failed"
            return 1
        fi
    else
        log "ERROR: Certificate validation failed"
        return 1
    fi
}

# Update security headers
update_security_headers() {
    local nginx_ssl_config="/etc/nginx/snippets/ssl-params.conf"
    
    cat > "$nginx_ssl_config" << 'EOF'
# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES256-SHA256:DHE-RSA-AES128-SHA:DHE-RSA-AES256-SHA:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA;
ssl_prefer_server_ciphers off;
ssl_dhparam /etc/nginx/dhparam.pem;

# OCSP Stapling
ssl_stapling on;
ssl_stapling_verify on;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;

# Security Headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
add_header X-Frame-Options DENY always;
add_header X-Content-Type-Options nosniff always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https://api.a2aworld.ai; frame-ancestors 'none';" always;
add_header Permissions-Policy "camera=(), microphone=(), geolocation=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()" always;

# Session Security
ssl_session_cache shared:le_nginx_SSL:10m;
ssl_session_timeout 1440m;
ssl_session_tickets off;
EOF
    
    log "Security headers updated"
}

# Generate DH parameters if not exists
generate_dhparam() {
    local dhparam_file="/etc/nginx/dhparam.pem"
    
    if [[ ! -f "$dhparam_file" ]]; then
        log "Generating DH parameters (this may take a while)..."
        openssl dhparam -out "$dhparam_file" 2048
        log "DH parameters generated"
    fi
}

# Main execution
main() {
    log "Starting SSL certificate renewal check"
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        log "ERROR: This script must be run as root"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p /backup/ssl
    
    # Generate DH parameters if needed
    generate_dhparam
    
    # Check if renewal is needed
    if check_certificate; then
        log "Certificate is still valid, no renewal needed"
        exit 0
    fi
    
    # Start renewal process
    notify "Starting SSL certificate renewal for $DOMAIN"
    
    # Backup current certificates
    backup_certificates
    
    # Test current nginx config
    if ! test_nginx_config; then
        notify "Nginx configuration test failed, aborting renewal" "error"
        exit 1
    fi
    
    # Obtain/renew certificate
    if ! obtain_certificate; then
        notify "Certificate renewal failed" "error"
        exit 1
    fi
    
    # Update security headers
    update_security_headers
    
    # Test nginx config with new certificate
    if ! test_nginx_config; then
        notify "Nginx configuration test failed after certificate renewal" "error"
        exit 1
    fi
    
    # Reload nginx
    if ! reload_nginx; then
        notify "Failed to reload nginx after certificate renewal" "error"
        exit 1
    fi
    
    # Verify certificate installation
    if verify_certificate; then
        notify "SSL certificate renewed and verified successfully" "success"
    else
        notify "SSL certificate renewal completed but verification failed" "warning"
    fi
    
    log "SSL certificate renewal process completed"
}

# Run main function
main "$@"