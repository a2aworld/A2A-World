#!/bin/bash
# A2A World Platform - Secrets Management Script
# Secure handling of production secrets and environment variables

set -euo pipefail

# Configuration
SECRETS_DIR="/opt/a2a-world/secrets"
BACKUP_DIR="/backup/secrets"
LOG_FILE="/var/log/a2a-world/secrets-manager.log"
ENCRYPTION_KEY_FILE="/etc/a2a-world/encryption.key"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    error_exit "This script must be run as root"
fi

# Create necessary directories
setup_directories() {
    log "Setting up directories"
    mkdir -p "$SECRETS_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$(dirname "$ENCRYPTION_KEY_FILE")"
    
    # Set secure permissions
    chmod 700 "$SECRETS_DIR"
    chmod 700 "$BACKUP_DIR"
    chmod 755 "$(dirname "$ENCRYPTION_KEY_FILE")"
}

# Generate encryption key if it doesn't exist
generate_encryption_key() {
    if [[ ! -f "$ENCRYPTION_KEY_FILE" ]]; then
        log "Generating new encryption key"
        openssl rand -base64 32 > "$ENCRYPTION_KEY_FILE"
        chmod 600 "$ENCRYPTION_KEY_FILE"
        chown root:root "$ENCRYPTION_KEY_FILE"
    fi
}

# Encrypt a secret
encrypt_secret() {
    local secret_name="$1"
    local secret_value="$2"
    local encrypted_file="$SECRETS_DIR/${secret_name}.enc"
    
    echo -n "$secret_value" | openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 -pass file:"$ENCRYPTION_KEY_FILE" -out "$encrypted_file"
    chmod 600 "$encrypted_file"
    chown root:root "$encrypted_file"
    
    log "Secret '$secret_name' encrypted and stored"
}

# Decrypt a secret
decrypt_secret() {
    local secret_name="$1"
    local encrypted_file="$SECRETS_DIR/${secret_name}.enc"
    
    if [[ ! -f "$encrypted_file" ]]; then
        error_exit "Secret '$secret_name' not found"
    fi
    
    openssl enc -aes-256-cbc -d -pbkdf2 -iter 100000 -pass file:"$ENCRYPTION_KEY_FILE" -in "$encrypted_file"
}

# Store Docker secrets
store_docker_secrets() {
    log "Setting up Docker secrets"
    
    # Database password
    if [[ -n "${POSTGRES_PASSWORD:-}" ]]; then
        echo -n "$POSTGRES_PASSWORD" | docker secret create db_password - 2>/dev/null || true
    fi
    
    # Redis password
    if [[ -n "${REDIS_PASSWORD:-}" ]]; then
        echo -n "$REDIS_PASSWORD" | docker secret create redis_password - 2>/dev/null || true
    fi
    
    # API secret key
    if [[ -n "${SECRET_KEY:-}" ]]; then
        echo -n "$SECRET_KEY" | docker secret create secret_key - 2>/dev/null || true
    fi
    
    # DigitalOcean Spaces credentials
    if [[ -n "${SPACES_ACCESS_KEY:-}" ]]; then
        echo -n "$SPACES_ACCESS_KEY" | docker secret create spaces_key - 2>/dev/null || true
    fi
    
    if [[ -n "${SPACES_SECRET_KEY:-}" ]]; then
        echo -n "$SPACES_SECRET_KEY" | docker secret create spaces_secret - 2>/dev/null || true
    fi
    
    log "Docker secrets configured"
}

# Generate secure passwords
generate_password() {
    local length="${1:-32}"
    openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-"$length"
}

# Initialize production secrets
initialize_secrets() {
    log "Initializing production secrets"
    
    # Generate database password if not set
    if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
        POSTGRES_PASSWORD=$(generate_password 32)
        encrypt_secret "postgres_password" "$POSTGRES_PASSWORD"
        export POSTGRES_PASSWORD
    fi
    
    # Generate Redis password if not set
    if [[ -z "${REDIS_PASSWORD:-}" ]]; then
        REDIS_PASSWORD=$(generate_password 32)
        encrypt_secret "redis_password" "$REDIS_PASSWORD"
        export REDIS_PASSWORD
    fi
    
    # Generate API secret key if not set
    if [[ -z "${SECRET_KEY:-}" ]]; then
        SECRET_KEY=$(generate_password 64)
        encrypt_secret "secret_key" "$SECRET_KEY"
        export SECRET_KEY
    fi
    
    # Generate JWT secret if not set
    if [[ -z "${JWT_SECRET:-}" ]]; then
        JWT_SECRET=$(generate_password 64)
        encrypt_secret "jwt_secret" "$JWT_SECRET"
        export JWT_SECRET
    fi
    
    # Generate encryption key for application
    if [[ -z "${APP_ENCRYPTION_KEY:-}" ]]; then
        APP_ENCRYPTION_KEY=$(generate_password 32)
        encrypt_secret "app_encryption_key" "$APP_ENCRYPTION_KEY"
        export APP_ENCRYPTION_KEY
    fi
}

# Create environment file for Docker Compose
create_env_file() {
    local env_file="$1"
    
    log "Creating environment file: $env_file"
    
    cat > "$env_file" << EOF
# A2A World Platform - Production Environment Variables
# Generated on $(date)

# Database Configuration
POSTGRES_DB=a2a_world
POSTGRES_USER=a2a_user
POSTGRES_PASSWORD=$(decrypt_secret "postgres_password")
DATABASE_URL=postgresql://a2a_user:$(decrypt_secret "postgres_password")@postgres:5432/a2a_world?sslmode=require

# Redis Configuration
REDIS_PASSWORD=$(decrypt_secret "redis_password")
REDIS_URL=redis://:$(decrypt_secret "redis_password")@redis:6379

# API Configuration
SECRET_KEY=$(decrypt_secret "secret_key")
JWT_SECRET=$(decrypt_secret "jwt_secret")
APP_ENCRYPTION_KEY=$(decrypt_secret "app_encryption_key")
ENVIRONMENT=production
LOG_LEVEL=INFO

# CORS Configuration
BACKEND_CORS_ORIGINS=https://a2aworld.ai,https://www.a2aworld.ai,https://api.a2aworld.ai

# Frontend Configuration
NEXT_PUBLIC_API_URL=https://api.a2aworld.ai

# External Services
SMTP_HOST=${SMTP_HOST:-smtp.gmail.com}
SMTP_PORT=${SMTP_PORT:-587}
SMTP_USER=${SMTP_USER:-}
SMTP_PASSWORD=${SMTP_PASSWORD:-}

# Monitoring Configuration
PROMETHEUS_ENABLED=true

# Container Registry
REGISTRY_URL=registry.digitalocean.com/a2aregistry
VERSION=latest

# Domain Configuration
DOMAIN_NAME=a2aworld.ai
EOF
    
    chmod 600 "$env_file"
    chown root:root "$env_file"
    
    log "Environment file created successfully"
}

# Backup secrets
backup_secrets() {
    local backup_file="$BACKUP_DIR/secrets-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    log "Creating secrets backup: $backup_file"
    
    tar -czf "$backup_file" -C "$(dirname "$SECRETS_DIR")" "$(basename "$SECRETS_DIR")" "$ENCRYPTION_KEY_FILE"
    chmod 600 "$backup_file"
    
    # Keep only last 10 backups
    find "$BACKUP_DIR" -name "secrets-backup-*.tar.gz" -type f | sort -r | tail -n +11 | xargs rm -f
    
    log "Secrets backup completed"
}

# Restore secrets from backup
restore_secrets() {
    local backup_file="$1"
    
    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi
    
    log "Restoring secrets from backup: $backup_file"
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    
    # Extract backup
    tar -xzf "$backup_file" -C "$temp_dir"
    
    # Restore secrets
    cp -r "$temp_dir/$(basename "$SECRETS_DIR")"/* "$SECRETS_DIR/"
    cp "$temp_dir$(dirname "$ENCRYPTION_KEY_FILE")/$(basename "$ENCRYPTION_KEY_FILE")" "$ENCRYPTION_KEY_FILE"
    
    # Set permissions
    chmod -R 600 "$SECRETS_DIR"/*
    chmod 600 "$ENCRYPTION_KEY_FILE"
    chown -R root:root "$SECRETS_DIR" "$ENCRYPTION_KEY_FILE"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "Secrets restored successfully"
}

# Rotate secrets
rotate_secrets() {
    log "Starting secrets rotation"
    
    # Backup current secrets before rotation
    backup_secrets
    
    # Generate new passwords
    NEW_POSTGRES_PASSWORD=$(generate_password 32)
    NEW_REDIS_PASSWORD=$(generate_password 32)
    NEW_SECRET_KEY=$(generate_password 64)
    NEW_JWT_SECRET=$(generate_password 64)
    NEW_APP_ENCRYPTION_KEY=$(generate_password 32)
    
    # Store new secrets
    encrypt_secret "postgres_password" "$NEW_POSTGRES_PASSWORD"
    encrypt_secret "redis_password" "$NEW_REDIS_PASSWORD"
    encrypt_secret "secret_key" "$NEW_SECRET_KEY"
    encrypt_secret "jwt_secret" "$NEW_JWT_SECRET"
    encrypt_secret "app_encryption_key" "$NEW_APP_ENCRYPTION_KEY"
    
    log "Secrets rotation completed - restart services to apply changes"
}

# List stored secrets
list_secrets() {
    log "Listing stored secrets"
    
    echo "Encrypted secrets in $SECRETS_DIR:"
    for secret_file in "$SECRETS_DIR"/*.enc; do
        if [[ -f "$secret_file" ]]; then
            echo "  - $(basename "$secret_file" .enc)"
        fi
    done
}

# Validate secrets
validate_secrets() {
    log "Validating secrets"
    
    local errors=0
    
    # Check encryption key
    if [[ ! -f "$ENCRYPTION_KEY_FILE" ]]; then
        log "ERROR: Encryption key file not found"
        ((errors++))
    fi
    
    # Check required secrets
    required_secrets=("postgres_password" "redis_password" "secret_key" "jwt_secret" "app_encryption_key")
    
    for secret in "${required_secrets[@]}"; do
        if [[ ! -f "$SECRETS_DIR/${secret}.enc" ]]; then
            log "ERROR: Required secret missing: $secret"
            ((errors++))
        else
            # Try to decrypt to validate
            if ! decrypt_secret "$secret" >/dev/null 2>&1; then
                log "ERROR: Cannot decrypt secret: $secret"
                ((errors++))
            fi
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        log "All secrets validated successfully"
        return 0
    else
        log "Validation failed with $errors errors"
        return 1
    fi
}

# Main function
main() {
    case "${1:-help}" in
        init)
            setup_directories
            generate_encryption_key
            initialize_secrets
            store_docker_secrets
            ;;
        backup)
            backup_secrets
            ;;
        restore)
            restore_secrets "$2"
            ;;
        rotate)
            rotate_secrets
            ;;
        list)
            list_secrets
            ;;
        validate)
            validate_secrets
            ;;
        env)
            create_env_file "${2:-/opt/a2a-world/.env.production}"
            ;;
        decrypt)
            decrypt_secret "$2"
            ;;
        encrypt)
            encrypt_secret "$2" "$3"
            ;;
        help|*)
            echo "Usage: $0 {init|backup|restore|rotate|list|validate|env|decrypt|encrypt|help}"
            echo ""
            echo "Commands:"
            echo "  init              Initialize secrets management system"
            echo "  backup            Create backup of all secrets"
            echo "  restore <file>    Restore secrets from backup file"
            echo "  rotate            Rotate all secrets (generates new ones)"
            echo "  list              List all stored secrets"
            echo "  validate          Validate all secrets can be decrypted"
            echo "  env [file]        Create environment file from secrets"
            echo "  decrypt <secret>  Decrypt and display a secret"
            echo "  encrypt <name> <value>  Encrypt and store a secret"
            echo "  help              Show this help message"
            ;;
    esac
}

# Run main function with all arguments
main "$@"