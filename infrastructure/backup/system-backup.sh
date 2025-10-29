#!/bin/bash
# A2A World Platform - System Backup Script
# Complete system backup including application data, configs, and logs

set -euo pipefail

# Configuration
BACKUP_ROOT="/backup"
LOG_FILE="/var/log/a2a-world/system-backup.log"
RETENTION_DAYS=14
EXCLUDE_FILE="/tmp/backup-exclude.txt"

# Application directories
APP_DIRS=(
    "/opt/a2a-world"
    "/etc/nginx/sites-available"
    "/etc/nginx/sites-enabled"
    "/etc/letsencrypt"
    "/var/log/a2a-world"
    "/etc/systemd/system/a2a-*.service"
)

# Docker data
DOCKER_VOLUMES_DIR="/var/lib/docker/volumes"
DOCKER_COMPOSE_DIR="/opt/a2a-world"

# Configuration files
CONFIG_DIRS=(
    "/etc/prometheus"
    "/etc/grafana"
    "/etc/loki"
    "/etc/fail2ban/jail.d"
    "/etc/ufw"
)

# DigitalOcean Spaces configuration
SPACES_BUCKET="${SPACES_BUCKET:-a2a-world-backups}"
SPACES_ENDPOINT="${SPACES_ENDPOINT:-https://nyc3.digitaloceanspaces.com}"
SPACES_ACCESS_KEY="${SPACES_ACCESS_KEY:-}"
SPACES_SECRET_KEY="${SPACES_SECRET_KEY:-}"

# Notification configuration
SLACK_WEBHOOK="${SLACK_WEBHOOK_BACKUP:-}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    notify "System backup failed: $1" "error"
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
            --data "{\"text\":\"System Backup - $level: $message\", \"color\":\"$color\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

# Setup backup environment
setup_backup_env() {
    log "Setting up backup environment"
    
    # Create backup directories
    mkdir -p "$BACKUP_ROOT"/{application,system,docker,configs}
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Create exclude file for rsync
    cat > "$EXCLUDE_FILE" << 'EOF'
*.tmp
*.temp
*.log
*.cache
node_modules/
.git/
__pycache__/
*.pyc
.DS_Store
Thumbs.db
EOF
    
    log "Backup environment setup completed"
}

# Backup application data
backup_application_data() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="$BACKUP_ROOT/application/app_data_$timestamp"
    
    log "Starting application data backup"
    
    mkdir -p "$backup_dir"
    
    # Backup each application directory
    for app_dir in "${APP_DIRS[@]}"; do
        if [[ -d "$app_dir" ]]; then
            log "Backing up: $app_dir"
            
            local dest_dir="$backup_dir$(dirname "$app_dir")"
            mkdir -p "$dest_dir"
            
            if rsync -av --exclude-from="$EXCLUDE_FILE" "$app_dir/" "$dest_dir/$(basename "$app_dir")/"; then
                log "Successfully backed up: $app_dir"
            else
                log "WARNING: Failed to backup: $app_dir"
            fi
        else
            log "Directory not found, skipping: $app_dir"
        fi
    done
    
    # Backup user data and uploads
    local data_dir="/opt/a2a-world/data"
    if [[ -d "$data_dir" ]]; then
        log "Backing up user data"
        rsync -av --exclude-from="$EXCLUDE_FILE" "$data_dir/" "$backup_dir/user_data/"
    fi
    
    # Create archive
    local archive_file="$BACKUP_ROOT/application/app_data_$timestamp.tar.gz"
    if tar -czf "$archive_file" -C "$backup_dir" .; then
        rm -rf "$backup_dir"
        log "Application data backup completed: $archive_file"
        echo "$archive_file"
    else
        error_exit "Failed to create application data archive"
    fi
}

# Backup Docker volumes
backup_docker_data() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="$BACKUP_ROOT/docker/docker_volumes_$timestamp"
    
    log "Starting Docker data backup"
    
    # Stop containers gracefully if running
    if docker ps -q > /dev/null 2>&1; then
        log "Stopping Docker containers for consistent backup"
        cd "$DOCKER_COMPOSE_DIR" && docker-compose down || true
    fi
    
    mkdir -p "$backup_dir"
    
    # Backup Docker volumes
    if [[ -d "$DOCKER_VOLUMES_DIR" ]]; then
        log "Backing up Docker volumes"
        rsync -av "$DOCKER_VOLUMES_DIR/" "$backup_dir/volumes/"
    fi
    
    # Backup Docker Compose files
    if [[ -d "$DOCKER_COMPOSE_DIR" ]]; then
        log "Backing up Docker Compose configurations"
        rsync -av --include="*.yml" --include="*.yaml" --include="*.env*" --exclude="*" \
            "$DOCKER_COMPOSE_DIR/" "$backup_dir/compose/"
    fi
    
    # Restart containers
    if [[ -f "$DOCKER_COMPOSE_DIR/docker-compose.production.yml" ]]; then
        log "Restarting Docker containers"
        cd "$DOCKER_COMPOSE_DIR" && docker-compose -f docker-compose.production.yml up -d || true
    fi
    
    # Create archive
    local archive_file="$BACKUP_ROOT/docker/docker_volumes_$timestamp.tar.gz"
    if tar -czf "$archive_file" -C "$backup_dir" .; then
        rm -rf "$backup_dir"
        log "Docker data backup completed: $archive_file"
        echo "$archive_file"
    else
        error_exit "Failed to create Docker data archive"
    fi
}

# Backup system configuration
backup_system_config() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="$BACKUP_ROOT/system/system_config_$timestamp"
    
    log "Starting system configuration backup"
    
    mkdir -p "$backup_dir"
    
    # Backup configuration directories
    for config_dir in "${CONFIG_DIRS[@]}"; do
        if [[ -d "$config_dir" ]]; then
            log "Backing up config: $config_dir"
            
            local dest_dir="$backup_dir$(dirname "$config_dir")"
            mkdir -p "$dest_dir"
            
            rsync -av "$config_dir/" "$dest_dir/$(basename "$config_dir")/"
        fi
    done
    
    # Backup system files
    log "Backing up system files"
    
    # Crontab
    crontab -l > "$backup_dir/crontab.txt" 2>/dev/null || echo "No crontab" > "$backup_dir/crontab.txt"
    
    # Systemd services
    mkdir -p "$backup_dir/systemd"
    cp /etc/systemd/system/a2a-*.service "$backup_dir/systemd/" 2>/dev/null || true
    
    # SSH configuration
    mkdir -p "$backup_dir/ssh"
    cp /etc/ssh/sshd_config "$backup_dir/ssh/" 2>/dev/null || true
    
    # UFW rules
    mkdir -p "$backup_dir/ufw"
    ufw status verbose > "$backup_dir/ufw/status.txt" 2>/dev/null || true
    
    # Installed packages
    dpkg --get-selections > "$backup_dir/installed_packages.txt" 2>/dev/null || true
    
    # Network configuration
    ip addr show > "$backup_dir/network_interfaces.txt" 2>/dev/null || true
    
    # Create archive
    local archive_file="$BACKUP_ROOT/system/system_config_$timestamp.tar.gz"
    if tar -czf "$archive_file" -C "$backup_dir" .; then
        rm -rf "$backup_dir"
        log "System configuration backup completed: $archive_file"
        echo "$archive_file"
    else
        error_exit "Failed to create system configuration archive"
    fi
}

# Backup SSL certificates
backup_ssl_certificates() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_ROOT/system/ssl_certificates_$timestamp.tar.gz"
    
    log "Starting SSL certificates backup"
    
    if [[ -d "/etc/letsencrypt" ]]; then
        if tar -czf "$backup_file" -C /etc letsencrypt/; then
            log "SSL certificates backup completed: $backup_file"
            echo "$backup_file"
        else
            log "WARNING: Failed to backup SSL certificates"
        fi
    else
        log "No SSL certificates found to backup"
    fi
}

# Create system snapshot
create_system_snapshot() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local snapshot_dir="$BACKUP_ROOT/snapshots/system_snapshot_$timestamp"
    
    log "Creating system snapshot"
    
    mkdir -p "$snapshot_dir"
    
    # System information
    {
        echo "=== A2A World System Snapshot ==="
        echo "Created: $(date)"
        echo "Hostname: $(hostname)"
        echo "Uptime: $(uptime)"
        echo ""
        
        echo "=== System Info ==="
        uname -a
        echo ""
        
        echo "=== Disk Usage ==="
        df -h
        echo ""
        
        echo "=== Memory Usage ==="
        free -h
        echo ""
        
        echo "=== Docker Status ==="
        docker ps || echo "Docker not running"
        echo ""
        
        echo "=== Services Status ==="
        systemctl status nginx || true
        systemctl status postgresql || true
        systemctl status redis || true
        echo ""
        
        echo "=== Network Status ==="
        netstat -tuln || ss -tuln
        echo ""
        
        echo "=== Process List ==="
        ps aux | head -20
        
    } > "$snapshot_dir/system_info.txt"
    
    # Configuration checksums
    find /etc -type f -name "*.conf" -o -name "*.cfg" -o -name "*.ini" 2>/dev/null | \
        head -100 | xargs md5sum > "$snapshot_dir/config_checksums.txt" 2>/dev/null || true
    
    # Create archive
    local archive_file="$BACKUP_ROOT/snapshots/system_snapshot_$timestamp.tar.gz"
    if tar -czf "$archive_file" -C "$snapshot_dir" .; then
        rm -rf "$snapshot_dir"
        log "System snapshot created: $archive_file"
        echo "$archive_file"
    else
        log "WARNING: Failed to create system snapshot archive"
    fi
}

# Upload backup to Spaces
upload_to_spaces() {
    local backup_file="$1"
    local backup_type="$2"
    
    if [[ -z "$SPACES_ACCESS_KEY" ]] || [[ -z "$SPACES_SECRET_KEY" ]]; then
        log "Spaces credentials not configured, skipping off-site backup"
        return 0
    fi
    
    if ! command -v s3cmd &> /dev/null; then
        log "s3cmd not available, skipping off-site backup"
        return 0
    fi
    
    log "Uploading backup to DigitalOcean Spaces: $(basename "$backup_file")"
    
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
    
    # Upload file
    local remote_path="system-backups/$backup_type/$(basename "$backup_file")"
    if s3cmd -c "$s3cfg_file" put "$backup_file" "s3://$SPACES_BUCKET/$remote_path"; then
        log "Backup uploaded successfully: $remote_path"
    else
        log "WARNING: Failed to upload backup to Spaces"
    fi
    
    # Cleanup s3cfg file
    rm -f "$s3cfg_file"
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"
    
    local dirs=("application" "docker" "system" "snapshots")
    local total_deleted=0
    
    for dir in "${dirs[@]}"; do
        local backup_dir="$BACKUP_ROOT/$dir"
        if [[ -d "$backup_dir" ]]; then
            local deleted_count=0
            while IFS= read -r -d '' file; do
                log "Deleting old backup: $(basename "$file")"
                rm -f "$file"
                ((deleted_count++))
                ((total_deleted++))
            done < <(find "$backup_dir" -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -print0)
            
            log "Deleted $deleted_count old backups from $dir"
        fi
    done
    
    log "Total deleted backups: $total_deleted"
}

# Generate backup report
generate_backup_report() {
    local backup_files=("$@")
    local report_file="$BACKUP_ROOT/backup_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log "Generating backup report"
    
    {
        echo "A2A World Platform - System Backup Report"
        echo "Generated on: $(date)"
        echo "Hostname: $(hostname)"
        echo ""
        
        echo "Backup Files Created:"
        for backup_file in "${backup_files[@]}"; do
            if [[ -f "$backup_file" ]]; then
                local file_size=$(du -h "$backup_file" | cut -f1)
                echo "  - $(basename "$backup_file") ($file_size)"
            fi
        done
        
        echo ""
        echo "Backup Directory Sizes:"
        du -sh "$BACKUP_ROOT"/* 2>/dev/null || true
        
        echo ""
        echo "Total Backup Storage Used: $(du -sh "$BACKUP_ROOT" | cut -f1)"
        
        echo ""
        echo "System Status at Backup Time:"
        df -h
        echo ""
        free -h
        echo ""
        docker ps 2>/dev/null || echo "Docker not running"
        
    } > "$report_file"
    
    log "Backup report generated: $report_file"
}

# Main backup function
main() {
    local backup_type="${1:-all}"
    
    log "Starting system backup process (type: $backup_type)"
    
    # Setup environment
    setup_backup_env
    
    local backup_files=()
    
    case "$backup_type" in
        "application"|"app")
            backup_files+=($(backup_application_data))
            ;;
        "docker")
            backup_files+=($(backup_docker_data))
            ;;
        "system"|"config")
            backup_files+=($(backup_system_config))
            backup_files+=($(backup_ssl_certificates))
            ;;
        "snapshot")
            backup_files+=($(create_system_snapshot))
            ;;
        "all"|*)
            backup_files+=($(backup_application_data))
            backup_files+=($(backup_docker_data))
            backup_files+=($(backup_system_config))
            backup_files+=($(backup_ssl_certificates))
            backup_files+=($(create_system_snapshot))
            ;;
    esac
    
    # Upload backups to off-site storage
    for backup_file in "${backup_files[@]}"; do
        local file_type=$(echo "$backup_file" | cut -d'/' -f4)
        upload_to_spaces "$backup_file" "$file_type"
    done
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Generate report
    generate_backup_report "${backup_files[@]}"
    
    # Clean up temporary files
    rm -f "$EXCLUDE_FILE"
    
    # Final notification
    local total_size=$(du -sh "$BACKUP_ROOT" | cut -f1)
    notify "System backup completed successfully. Files: ${#backup_files[@]}, Total size: $total_size" "success"
    
    log "System backup process completed"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi