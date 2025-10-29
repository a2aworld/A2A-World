#!/bin/bash
# A2A World Platform - Database Backup Script
# Automated PostgreSQL backup with compression and retention

set -euo pipefail

# Configuration
BACKUP_DIR="/backup/database"
LOG_FILE="/var/log/a2a-world/database-backup.log"
RETENTION_DAYS=30
COMPRESSION_LEVEL=6
MAX_PARALLEL_BACKUPS=2

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-a2a_world}"
DB_USER="${DB_USER:-a2a_user}"
DB_PASSWORD="${DB_PASSWORD:-}"

# DigitalOcean Spaces configuration for off-site backup
SPACES_BUCKET="${SPACES_BUCKET:-a2a-world-backups}"
SPACES_ENDPOINT="${SPACES_ENDPOINT:-https://nyc3.digitaloceanspaces.com}"
SPACES_ACCESS_KEY="${SPACES_ACCESS_KEY:-}"
SPACES_SECRET_KEY="${SPACES_SECRET_KEY:-}"

# Notification configuration
SLACK_WEBHOOK="${SLACK_WEBHOOK_BACKUP:-}"
EMAIL_RECIPIENT="${EMAIL_RECIPIENT:-admin@a2aworld.ai}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    notify "Database backup failed: $1" "error"
    exit 1
}

# Notification function
notify() {
    local message="$1"
    local level="${2:-info}"
    
    log "$message"
    
    # Slack notification
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local color="good"
        [[ "$level" == "error" ]] && color="danger"
        [[ "$level" == "warning" ]] && color="warning"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Database Backup - $level: $message\", \"color\":\"$color\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
    
    # Email notification for errors
    if [[ "$level" == "error" && -n "$EMAIL_RECIPIENT" ]]; then
        echo "$message" | mail -s "A2A World Database Backup Error" "$EMAIL_RECIPIENT" 2>/dev/null || true
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites"
    
    # Check if pg_dump is available
    if ! command -v pg_dump &> /dev/null; then
        error_exit "pg_dump not found. Please install PostgreSQL client tools."
    fi
    
    # Check if gzip is available
    if ! command -v gzip &> /dev/null; then
        error_exit "gzip not found. Please install gzip."
    fi
    
    # Check if s3cmd is available for Spaces backup
    if [[ -n "$SPACES_ACCESS_KEY" ]] && ! command -v s3cmd &> /dev/null; then
        log "WARNING: s3cmd not found. Off-site backup to Spaces will be skipped."
    fi
    
    # Check database connectivity
    if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &>/dev/null; then
        error_exit "Cannot connect to database. Please check connection parameters."
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log "Prerequisites check completed"
}

# Get database size
get_database_size() {
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" | xargs
}

# Create full database backup
create_full_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/full_backup_${DB_NAME}_${timestamp}.sql"
    local compressed_file="${backup_file}.gz"
    
    log "Starting full database backup"
    log "Database size: $(get_database_size)"
    
    # Create backup with progress
    if PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --no-password \
        --format=custom \
        --compress="$COMPRESSION_LEVEL" \
        --file="$backup_file"; then
        
        log "Database dump completed successfully"
        
        # Compress the backup
        if gzip -"$COMPRESSION_LEVEL" "$backup_file"; then
            log "Backup compressed successfully: $compressed_file"
            
            # Get compressed file size
            local file_size=$(du -h "$compressed_file" | cut -f1)
            log "Compressed backup size: $file_size"
            
            # Verify backup integrity
            if verify_backup "$compressed_file"; then
                log "Backup verification successful"
                echo "$compressed_file"
            else
                error_exit "Backup verification failed"
            fi
        else
            error_exit "Failed to compress backup file"
        fi
    else
        error_exit "Database dump failed"
    fi
}

# Create incremental backup (WAL files)
create_incremental_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local wal_backup_dir="$BACKUP_DIR/wal_${timestamp}"
    
    log "Starting WAL backup"
    
    mkdir -p "$wal_backup_dir"
    
    # Archive WAL files
    if PGPASSWORD="$DB_PASSWORD" pg_basebackup \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -D "$wal_backup_dir" \
        --wal-method=stream \
        --compress \
        --progress; then
        
        log "WAL backup completed successfully"
        
        # Compress WAL backup
        local wal_archive="$BACKUP_DIR/wal_${timestamp}.tar.gz"
        if tar -czf "$wal_archive" -C "$BACKUP_DIR" "wal_${timestamp}"; then
            rm -rf "$wal_backup_dir"
            log "WAL backup compressed: $wal_archive"
            echo "$wal_archive"
        else
            error_exit "Failed to compress WAL backup"
        fi
    else
        error_exit "WAL backup failed"
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    log "Verifying backup integrity: $(basename "$backup_file")"
    
    # Check if file exists and is not empty
    if [[ ! -f "$backup_file" ]] || [[ ! -s "$backup_file" ]]; then
        log "ERROR: Backup file is missing or empty"
        return 1
    fi
    
    # Test gzip integrity
    if [[ "$backup_file" == *.gz ]]; then
        if ! gzip -t "$backup_file"; then
            log "ERROR: Backup file compression is corrupted"
            return 1
        fi
    fi
    
    # For custom format backups, use pg_restore to verify
    if [[ "$backup_file" == *.sql.gz ]]; then
        local temp_file="/tmp/backup_verify_$$.sql"
        
        if gunzip -c "$backup_file" > "$temp_file"; then
            # Check if we can list the backup contents
            if PGPASSWORD="$DB_PASSWORD" pg_restore --list "$temp_file" &>/dev/null; then
                rm -f "$temp_file"
                return 0
            else
                log "ERROR: Cannot read backup file structure"
                rm -f "$temp_file"
                return 1
            fi
        else
            log "ERROR: Cannot decompress backup file"
            return 1
        fi
    fi
    
    return 0
}

# Upload backup to DigitalOcean Spaces
upload_to_spaces() {
    local backup_file="$1"
    
    if [[ -z "$SPACES_ACCESS_KEY" ]] || [[ -z "$SPACES_SECRET_KEY" ]]; then
        log "Spaces credentials not configured, skipping off-site backup"
        return 0
    fi
    
    if ! command -v s3cmd &> /dev/null; then
        log "s3cmd not available, skipping off-site backup"
        return 0
    fi
    
    log "Uploading backup to DigitalOcean Spaces"
    
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
    local remote_path="database/$(basename "$backup_file")"
    if s3cmd -c "$s3cfg_file" put "$backup_file" "s3://$SPACES_BUCKET/$remote_path"; then
        log "Backup uploaded successfully to Spaces: $remote_path"
        
        # Set lifecycle policy for automatic cleanup
        s3cmd -c "$s3cfg_file" expire "s3://$SPACES_BUCKET/database/" --expiry-days="$RETENTION_DAYS" &>/dev/null || true
    else
        log "WARNING: Failed to upload backup to Spaces"
    fi
    
    # Cleanup s3cfg file
    rm -f "$s3cfg_file"
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"
    
    # Find and delete old local backups
    local deleted_count=0
    while IFS= read -r -d '' file; do
        log "Deleting old backup: $(basename "$file")"
        rm -f "$file"
        ((deleted_count++))
    done < <(find "$BACKUP_DIR" -name "*.sql.gz" -o -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -print0)
    
    log "Deleted $deleted_count old backup files"
}

# Generate backup report
generate_report() {
    local backup_files=("$@")
    
    log "Generating backup report"
    
    local report_file="$BACKUP_DIR/backup_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "A2A World Platform - Database Backup Report"
        echo "Generated on: $(date)"
        echo "Database: $DB_NAME@$DB_HOST:$DB_PORT"
        echo "Database size: $(get_database_size)"
        echo ""
        echo "Backup files created:"
        
        for backup_file in "${backup_files[@]}"; do
            if [[ -f "$backup_file" ]]; then
                local file_size=$(du -h "$backup_file" | cut -f1)
                echo "  - $(basename "$backup_file") ($file_size)"
            fi
        done
        
        echo ""
        echo "Total backup files in directory: $(find "$BACKUP_DIR" -name "*.gz" -type f | wc -l)"
        echo "Total backup directory size: $(du -sh "$BACKUP_DIR" | cut -f1)"
        
        # Database statistics
        echo ""
        echo "Database Statistics:"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
            "SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables ORDER BY n_tup_ins DESC LIMIT 10;" || true
        
    } > "$report_file"
    
    log "Backup report generated: $report_file"
}

# Main backup function
main() {
    local backup_type="${1:-full}"
    
    log "Starting database backup process (type: $backup_type)"
    
    # Check prerequisites
    check_prerequisites
    
    local backup_files=()
    
    case "$backup_type" in
        "full")
            backup_files+=($(create_full_backup))
            ;;
        "incremental"|"wal")
            backup_files+=($(create_incremental_backup))
            ;;
        "both"|"all")
            backup_files+=($(create_full_backup))
            backup_files+=($(create_incremental_backup))
            ;;
        *)
            error_exit "Invalid backup type: $backup_type. Use 'full', 'incremental', or 'both'"
            ;;
    esac
    
    # Upload to off-site storage
    for backup_file in "${backup_files[@]}"; do
        upload_to_spaces "$backup_file"
    done
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Generate report
    generate_report "${backup_files[@]}"
    
    # Final notification
    local total_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    notify "Database backup completed successfully. Files: ${#backup_files[@]}, Total size: $total_size" "success"
    
    log "Database backup process completed"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi