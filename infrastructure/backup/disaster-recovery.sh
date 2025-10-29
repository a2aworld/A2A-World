#!/bin/bash
# A2A World Platform - Disaster Recovery Script
# Complete system restoration from backups

set -euo pipefail

# Configuration
BACKUP_ROOT="/backup"
RESTORE_ROOT="/restore"
LOG_FILE="/var/log/a2a-world/disaster-recovery.log"

# DigitalOcean Spaces configuration
SPACES_BUCKET="${SPACES_BUCKET:-a2a-world-backups}"
SPACES_ENDPOINT="${SPACES_ENDPOINT:-https://nyc3.digitaloceanspaces.com}"
SPACES_ACCESS_KEY="${SPACES_ACCESS_KEY:-}"
SPACES_SECRET_KEY="${SPACES_SECRET_KEY:-}"

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-a2a_world}"
DB_USER="${DB_USER:-a2a_user}"
DB_PASSWORD="${DB_PASSWORD:-}"

# Notification configuration
SLACK_WEBHOOK="${SLACK_WEBHOOK_BACKUP:-}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    notify "Disaster recovery failed: $1" "error"
    exit 1
}

# Notification function
notify() {
    local message="$1"
    local level="${2:-info}"
    
    log "$message"
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local color="good"
        [[ "$level" == "error" ]] && color="danger"
        [[ "$level" == "warning" ]] && color="warning"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 DISASTER RECOVERY - $level: $message\", \"color\":\"$color\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

# Setup recovery environment
setup_recovery_env() {
    log "Setting up disaster recovery environment"
    
    # Create necessary directories
    mkdir -p "$RESTORE_ROOT"/{downloads,extracted,temp}
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root for system restoration"
    fi
    
    log "Recovery environment setup completed"
}

# List available backups
list_backups() {
    local backup_type="$1"
    
    log "Listing available backups for type: $backup_type"
    
    # List local backups
    echo "=== LOCAL BACKUPS ==="
    if [[ -d "$BACKUP_ROOT/$backup_type" ]]; then
        ls -la "$BACKUP_ROOT/$backup_type/"*.tar.gz 2>/dev/null || echo "No local backups found"
    else
        echo "No local backup directory found"
    fi
    
    # List remote backups if configured
    if [[ -n "$SPACES_ACCESS_KEY" ]] && command -v s3cmd &> /dev/null; then
        echo ""
        echo "=== REMOTE BACKUPS (Spaces) ==="
        list_remote_backups "$backup_type"
    fi
}

# List remote backups from Spaces
list_remote_backups() {
    local backup_type="$1"
    
    # Configure s3cmd
    local s3cfg_file="/tmp/.s3cfg_$$"
    cat > "$s3cfg_file" << EOF
[default]
access_key = $SPACES_ACCESS_KEY
secret_key = $SPACES_SECRET_KEY
host_base = $(echo "$SPACES_ENDPOINT" | sed 's|https://||')
host_bucket = %(bucket)s.$(echo "$SPACES_ENDPOINT" | sed 's|https://||')
use_https = True
signature_v2 = False
EOF
    
    # List remote files
    s3cmd -c "$s3cfg_file" ls "s3://$SPACES_BUCKET/system-backups/$backup_type/" 2>/dev/null || \
        s3cmd -c "$s3cfg_file" ls "s3://$SPACES_BUCKET/database/" 2>/dev/null || \
        echo "No remote backups found or connection failed"
    
    # Cleanup
    rm -f "$s3cfg_file"
}

# Download backup from Spaces
download_backup_from_spaces() {
    local remote_path="$1"
    local local_file="$2"
    
    log "Downloading backup from Spaces: $remote_path"
    
    if [[ -z "$SPACES_ACCESS_KEY" ]]; then
        error_exit "Spaces credentials not configured"
    fi
    
    if ! command -v s3cmd &> /dev/null; then
        error_exit "s3cmd not available for downloading from Spaces"
    fi
    
    # Configure s3cmd
    local s3cfg_file="/tmp/.s3cfg_$$"
    cat > "$s3cfg_file" << EOF
[default]
access_key = $SPACES_ACCESS_KEY
secret_key = $SPACES_SECRET_KEY
host_base = $(echo "$SPACES_ENDPOINT" | sed 's|https://||')
host_bucket = %(bucket)s.$(echo "$SPACES_ENDPOINT" | sed 's|https://||')
use_https = True
signature_v2 = False
EOF
    
    # Download file
    if s3cmd -c "$s3cfg_file" get "$remote_path" "$local_file"; then
        log "Backup downloaded successfully: $local_file"
    else
        error_exit "Failed to download backup from Spaces"
    fi
    
    # Cleanup
    rm -f "$s3cfg_file"
}

# Restore database from backup
restore_database() {
    local backup_file="$1"
    local restore_db_name="${2:-$DB_NAME}"
    
    log "Starting database restoration from: $(basename "$backup_file")"
    
    # Verify backup file
    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi
    
    # Check if compressed
    local temp_file="$backup_file"
    if [[ "$backup_file" == *.gz ]]; then
        log "Decompressing backup file"
        temp_file="$RESTORE_ROOT/temp/$(basename "$backup_file" .gz)"
        gunzip -c "$backup_file" > "$temp_file"
    fi
    
    # Test database connectivity
    if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT 1;" &>/dev/null; then
        error_exit "Cannot connect to database server"
    fi
    
    # Create restore database if it doesn't exist
    log "Creating/preparing database for restoration"
    PGPASSWORD="$DB_PASSWORD" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$restore_db_name" 2>/dev/null || true
    
    # Terminate existing connections
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c \
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$restore_db_name' AND pid <> pg_backend_pid();" || true
    
    # Drop existing database if requested
    read -p "Drop existing database '$restore_db_name' before restore? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Dropping existing database: $restore_db_name"
        PGPASSWORD="$DB_PASSWORD" dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$restore_db_name" || true
        PGPASSWORD="$DB_PASSWORD" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$restore_db_name"
    fi
    
    # Restore database
    log "Restoring database from backup"
    if PGPASSWORD="$DB_PASSWORD" pg_restore \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$restore_db_name" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        --verbose \
        "$temp_file"; then
        
        log "Database restoration completed successfully"
        
        # Verify restoration
        local table_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$restore_db_name" -t -c \
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
        log "Restored database contains $table_count tables"
        
    else
        error_exit "Database restoration failed"
    fi
    
    # Cleanup temporary file
    if [[ "$temp_file" != "$backup_file" ]]; then
        rm -f "$temp_file"
    fi
    
    log "Database restoration process completed"
}

# Restore application data
restore_application_data() {
    local backup_file="$1"
    local target_dir="${2:-/}"
    
    log "Starting application data restoration from: $(basename "$backup_file")"
    
    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi
    
    # Create extraction directory
    local extract_dir="$RESTORE_ROOT/extracted/app_$(date +%s)"
    mkdir -p "$extract_dir"
    
    # Extract backup
    log "Extracting application backup"
    if tar -xzf "$backup_file" -C "$extract_dir"; then
        log "Backup extracted successfully"
        
        # Stop services before restoration
        log "Stopping application services"
        systemctl stop nginx || true
        docker-compose -f /opt/a2a-world/docker-compose.production.yml down || true
        
        # Restore files
        log "Restoring application files to $target_dir"
        
        # Backup existing files if they exist
        local backup_existing="$RESTORE_ROOT/temp/existing_backup_$(date +%s)"
        mkdir -p "$backup_existing"
        
        if [[ -d "/opt/a2a-world" ]]; then
            log "Backing up existing application files"
            cp -r /opt/a2a-world "$backup_existing/"
        fi
        
        # Restore from backup
        if rsync -av "$extract_dir/" "$target_dir/"; then
            log "Application files restored successfully"
            
            # Set proper permissions
            chown -R root:root /opt/a2a-world
            chmod -R 755 /opt/a2a-world
            
            # Restart services
            log "Restarting application services"
            docker-compose -f /opt/a2a-world/docker-compose.production.yml up -d
            systemctl start nginx
            
        else
            log "WARNING: File restoration may have been incomplete"
        fi
        
    else
        error_exit "Failed to extract backup file"
    fi
    
    # Cleanup
    rm -rf "$extract_dir"
    
    log "Application data restoration completed"
}

# Restore Docker volumes
restore_docker_volumes() {
    local backup_file="$1"
    
    log "Starting Docker volumes restoration from: $(basename "$backup_file")"
    
    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi
    
    # Stop all containers
    log "Stopping all Docker containers"
    docker stop $(docker ps -aq) 2>/dev/null || true
    
    # Create extraction directory
    local extract_dir="$RESTORE_ROOT/extracted/docker_$(date +%s)"
    mkdir -p "$extract_dir"
    
    # Extract backup
    log "Extracting Docker volumes backup"
    if tar -xzf "$backup_file" -C "$extract_dir"; then
        
        # Backup existing volumes
        local existing_backup="/backup/docker_volumes_existing_$(date +%s)"
        if [[ -d "/var/lib/docker/volumes" ]]; then
            log "Backing up existing Docker volumes"
            mkdir -p "$(dirname "$existing_backup")"
            cp -r /var/lib/docker/volumes "$existing_backup"
        fi
        
        # Restore volumes
        log "Restoring Docker volumes"
        if [[ -d "$extract_dir/volumes" ]]; then
            rsync -av "$extract_dir/volumes/" "/var/lib/docker/volumes/"
        fi
        
        # Restore compose files
        if [[ -d "$extract_dir/compose" ]]; then
            log "Restoring Docker Compose files"
            rsync -av "$extract_dir/compose/" "/opt/a2a-world/"
        fi
        
        log "Docker volumes restoration completed"
        
    else
        error_exit "Failed to extract Docker volumes backup"
    fi
    
    # Cleanup
    rm -rf "$extract_dir"
    
    log "Docker volumes restoration process completed"
}

# Restore system configuration
restore_system_config() {
    local backup_file="$1"
    
    log "Starting system configuration restoration from: $(basename "$backup_file")"
    
    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi
    
    # Create extraction directory
    local extract_dir="$RESTORE_ROOT/extracted/system_$(date +%s)"
    mkdir -p "$extract_dir"
    
    # Extract backup
    log "Extracting system configuration backup"
    if tar -xzf "$backup_file" -C "$extract_dir"; then
        
        # Restore configuration files
        log "Restoring system configuration files"
        
        # Nginx configuration
        if [[ -d "$extract_dir/etc/nginx" ]]; then
            log "Restoring Nginx configuration"
            rsync -av "$extract_dir/etc/nginx/" "/etc/nginx/"
            nginx -t && systemctl reload nginx
        fi
        
        # SSL certificates
        if [[ -d "$extract_dir/etc/letsencrypt" ]]; then
            log "Restoring SSL certificates"
            rsync -av "$extract_dir/etc/letsencrypt/" "/etc/letsencrypt/"
        fi
        
        # Monitoring configurations
        if [[ -d "$extract_dir/etc/prometheus" ]]; then
            log "Restoring Prometheus configuration"
            rsync -av "$extract_dir/etc/prometheus/" "/etc/prometheus/"
        fi
        
        if [[ -d "$extract_dir/etc/grafana" ]]; then
            log "Restoring Grafana configuration"
            rsync -av "$extract_dir/etc/grafana/" "/etc/grafana/"
        fi
        
        # Systemd services
        if [[ -d "$extract_dir/systemd" ]]; then
            log "Restoring systemd services"
            cp "$extract_dir/systemd"/*.service /etc/systemd/system/ 2>/dev/null || true
            systemctl daemon-reload
        fi
        
        # Crontab
        if [[ -f "$extract_dir/crontab.txt" ]]; then
            log "Restoring crontab"
            crontab "$extract_dir/crontab.txt" 2>/dev/null || true
        fi
        
        log "System configuration restoration completed"
        
    else
        error_exit "Failed to extract system configuration backup"
    fi
    
    # Cleanup
    rm -rf "$extract_dir"
}

# Full system recovery
full_system_recovery() {
    log "Starting FULL SYSTEM RECOVERY"
    
    notify "🚨 FULL SYSTEM RECOVERY INITIATED" "warning"
    
    echo "=== FULL SYSTEM RECOVERY ==="
    echo "This will restore:"
    echo "1. Database"
    echo "2. Application data"  
    echo "3. Docker volumes"
    echo "4. System configuration"
    echo ""
    
    read -p "Are you sure you want to proceed? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Full system recovery cancelled by user"
        exit 0
    fi
    
    # List available backups
    echo "=== AVAILABLE DATABASE BACKUPS ==="
    list_backups "database"
    echo ""
    
    echo "=== AVAILABLE SYSTEM BACKUPS ==="
    list_backups "application"
    echo ""
    
    # Get backup file paths
    read -p "Enter database backup file path: " db_backup
    read -p "Enter application backup file path: " app_backup
    read -p "Enter docker volumes backup file path: " docker_backup
    read -p "Enter system config backup file path: " system_backup
    
    # Verify all files exist
    for backup_file in "$db_backup" "$app_backup" "$docker_backup" "$system_backup"; do
        if [[ ! -f "$backup_file" ]]; then
            # Try to download from Spaces if it looks like a remote path
            if [[ "$backup_file" == s3://* ]]; then
                local local_file="$RESTORE_ROOT/downloads/$(basename "$backup_file")"
                download_backup_from_spaces "$backup_file" "$local_file"
                backup_file="$local_file"
            else
                error_exit "Backup file not found: $backup_file"
            fi
        fi
    done
    
    # Execute recovery steps
    log "Starting recovery sequence"
    
    restore_docker_volumes "$docker_backup"
    restore_system_config "$system_backup"
    restore_application_data "$app_backup"
    restore_database "$db_backup"
    
    notify "🎉 FULL SYSTEM RECOVERY COMPLETED SUCCESSFULLY" "success"
    log "Full system recovery completed successfully"
}

# Interactive recovery menu
interactive_recovery() {
    echo "=== A2A World Disaster Recovery Menu ==="
    echo "1. List available backups"
    echo "2. Restore database only"
    echo "3. Restore application data only"
    echo "4. Restore Docker volumes only"
    echo "5. Restore system configuration only"
    echo "6. Full system recovery"
    echo "7. Download backup from Spaces"
    echo "8. Exit"
    echo ""
    
    read -p "Select option [1-8]: " choice
    
    case $choice in
        1)
            echo "Select backup type:"
            echo "1. Database backups"
            echo "2. Application backups"
            echo "3. Docker backups"
            echo "4. System backups"
            read -p "Select type [1-4]: " backup_type_choice
            
            case $backup_type_choice in
                1) list_backups "database" ;;
                2) list_backups "application" ;;
                3) list_backups "docker" ;;
                4) list_backups "system" ;;
                *) echo "Invalid choice" ;;
            esac
            ;;
        2)
            read -p "Enter database backup file path: " backup_file
            restore_database "$backup_file"
            ;;
        3)
            read -p "Enter application backup file path: " backup_file
            restore_application_data "$backup_file"
            ;;
        4)
            read -p "Enter Docker volumes backup file path: " backup_file
            restore_docker_volumes "$backup_file"
            ;;
        5)
            read -p "Enter system config backup file path: " backup_file
            restore_system_config "$backup_file"
            ;;
        6)
            full_system_recovery
            ;;
        7)
            read -p "Enter Spaces backup path (s3://bucket/path): " remote_path
            read -p "Enter local file path: " local_path
            download_backup_from_spaces "$remote_path" "$local_path"
            ;;
        8)
            log "Exiting disaster recovery menu"
            exit 0
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
}

# Main function
main() {
    local operation="${1:-interactive}"
    
    log "Starting disaster recovery process"
    
    # Setup environment
    setup_recovery_env
    
    case "$operation" in
        "list")
            list_backups "${2:-all}"
            ;;
        "restore-db")
            restore_database "$2" "${3:-$DB_NAME}"
            ;;
        "restore-app")
            restore_application_data "$2" "${3:-/}"
            ;;
        "restore-docker")
            restore_docker_volumes "$2"
            ;;
        "restore-system")
            restore_system_config "$2"
            ;;
        "full-recovery")
            full_system_recovery
            ;;
        "download")
            download_backup_from_spaces "$2" "$3"
            ;;
        "interactive"|*)
            interactive_recovery
            ;;
    esac
    
    log "Disaster recovery process completed"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi