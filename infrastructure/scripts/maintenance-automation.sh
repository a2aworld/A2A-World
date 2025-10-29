#!/bin/bash
# A2A World Platform - Maintenance Automation Setup
# Configures automated maintenance tasks and cron jobs

set -euo pipefail

# Configuration
LOG_FILE="/var/log/a2a-world/maintenance-setup.log"
SCRIPTS_DIR="/opt/a2a-world/infrastructure/scripts"
BACKUP_DIR="/opt/a2a-world/infrastructure/backup"
SECURITY_DIR="/opt/a2a-world/infrastructure/security"

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

# Setup directories and permissions
setup_directories() {
    log "Setting up maintenance directories and permissions"
    
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p /opt/a2a-world/reports
    mkdir -p /var/log/a2a-world
    
    # Set executable permissions on all scripts
    find "$SCRIPTS_DIR" -name "*.sh" -exec chmod +x {} \;
    find "$BACKUP_DIR" -name "*.sh" -exec chmod +x {} \;
    find "$SECURITY_DIR" -name "*.sh" -exec chmod +x {} \;
    
    log "Directories and permissions configured"
}

# Install required packages for maintenance
install_dependencies() {
    log "Installing maintenance dependencies"
    
    apt-get update
    apt-get install -y \
        bc \
        jq \
        curl \
        wget \
        htop \
        iotop \
        netstat-nat \
        mailutils \
        logrotate \
        cron \
        rsync \
        s3cmd
    
    # Enable and start cron service
    systemctl enable cron
    systemctl start cron
    
    log "Dependencies installed successfully"
}

# Create maintenance scripts
create_maintenance_scripts() {
    log "Creating maintenance utility scripts"
    
    # Weekly health check script
    cat > "$SCRIPTS_DIR/weekly-health-check.sh" << 'EOF'
#!/bin/bash
# A2A World Platform - Weekly Health Check
set -euo pipefail

LOG_FILE="/var/log/a2a-world/weekly-health-check.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting weekly health check"

# Check system resources
echo "=== SYSTEM RESOURCES ==="
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
echo "Memory Usage: $(free -h | grep Mem: | awk '{print $3 "/" $2}')"
echo "Disk Usage: $(df -h / | awk 'NR==2 {print $5}')"

# Check Docker services
echo -e "\n=== DOCKER SERVICES ==="
docker service ls

# Check service health
echo -e "\n=== SERVICE HEALTH ==="
curl -s https://api.a2aworld.ai/health || echo "API health check failed"
curl -s https://a2aworld.ai -o /dev/null || echo "Frontend health check failed"

# Check SSL certificates
echo -e "\n=== SSL CERTIFICATE STATUS ==="
echo | openssl s_client -connect a2aworld.ai:443 -servername a2aworld.ai 2>/dev/null | openssl x509 -noout -dates

# Check recent errors
echo -e "\n=== RECENT ERRORS (last 24 hours) ==="
journalctl --since "24 hours ago" --priority=err | tail -10

log "Weekly health check completed"
EOF

    # Performance monitoring script
    cat > "$SCRIPTS_DIR/performance-monitor.sh" << 'EOF'
#!/bin/bash
# A2A World Platform - Performance Monitoring
set -euo pipefail

LOG_FILE="/var/log/a2a-world/performance-monitor.log"
REPORT_DIR="/opt/a2a-world/reports"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting performance monitoring"

# Create performance report
REPORT_FILE="$REPORT_DIR/performance_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "A2A World Platform - Performance Report"
    echo "Generated: $(date)"
    echo ""
    
    echo "=== SYSTEM OVERVIEW ==="
    uptime
    echo ""
    
    echo "=== CPU USAGE ==="
    top -bn1 | head -20
    echo ""
    
    echo "=== MEMORY USAGE ==="
    free -h
    echo ""
    
    echo "=== DISK USAGE ==="
    df -h
    echo ""
    
    echo "=== NETWORK STATISTICS ==="
    netstat -i
    echo ""
    
    echo "=== DOCKER STATS ==="
    timeout 10 docker stats --no-stream 2>/dev/null || echo "Docker stats unavailable"
    echo ""
    
    echo "=== TOP PROCESSES ==="
    ps aux --sort=-%cpu | head -10
    echo ""
    
    echo "=== API RESPONSE TIME ==="
    curl -w "Total time: %{time_total}s\n" -o /dev/null -s https://api.a2aworld.ai/health 2>/dev/null || echo "API check failed"
    
} > "$REPORT_FILE"

log "Performance report saved to: $REPORT_FILE"
EOF

    # System cleanup script
    cat > "$SCRIPTS_DIR/system-cleanup.sh" << 'EOF'
#!/bin/bash
# A2A World Platform - System Cleanup
set -euo pipefail

LOG_FILE="/var/log/a2a-world/system-cleanup.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting system cleanup"

# Clean package cache
apt-get clean
apt-get autoremove -y

# Clean old log files (older than 30 days)
find /var/log -type f -name "*.log" -mtime +30 -delete 2>/dev/null || true
find /var/log -type f -name "*.gz" -mtime +30 -delete 2>/dev/null || true

# Clean Docker resources
docker system prune -f
docker volume prune -f
docker network prune -f

# Clean temporary files
find /tmp -type f -mtime +7 -delete 2>/dev/null || true

# Clean old reports
find /opt/a2a-world/reports -type f -mtime +30 -delete 2>/dev/null || true

log "System cleanup completed"
EOF

    # Service restart script
    cat > "$SCRIPTS_DIR/restart-services.sh" << 'EOF'
#!/bin/bash
# A2A World Platform - Service Restart Script
set -euo pipefail

LOG_FILE="/var/log/a2a-world/service-restart.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

SERVICE="${1:-all}"

log "Restarting services: $SERVICE"

case "$SERVICE" in
    "api")
        docker service update --force a2a-world_api
        ;;
    "frontend")
        docker service update --force a2a-world_frontend
        ;;
    "agents")
        docker service update --force a2a-world_agent-kml-parser
        docker service update --force a2a-world_agent-pattern-discovery
        docker service update --force a2a-world_agent-monitoring
        ;;
    "database")
        docker service update --force a2a-world_postgres
        ;;
    "all"|*)
        docker service update --force a2a-world_api
        docker service update --force a2a-world_frontend
        docker service update --force a2a-world_agent-kml-parser
        docker service update --force a2a-world_agent-pattern-discovery
        docker service update --force a2a-world_agent-monitoring
        ;;
esac

log "Service restart completed for: $SERVICE"
EOF

    # Make scripts executable
    chmod +x "$SCRIPTS_DIR"/*.sh
    
    log "Maintenance scripts created successfully"
}

# Setup automated cron jobs
setup_cron_jobs() {
    log "Setting up automated cron jobs"
    
    # Create main cron configuration
    cat > /etc/cron.d/a2a-world-maintenance << 'EOF'
# A2A World Platform - Automated Maintenance Tasks
# Generated by maintenance-automation.sh

# Environment variables
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
SHELL=/bin/bash

# Daily tasks
# Database backup at 2:00 AM
0 2 * * * root /opt/a2a-world/infrastructure/backup/database-backup.sh full >> /var/log/a2a-world/cron.log 2>&1

# Performance monitoring every 6 hours
0 */6 * * * root /opt/a2a-world/infrastructure/scripts/performance-monitor.sh >> /var/log/a2a-world/cron.log 2>&1

# Cost optimization analysis daily at 1:00 AM
0 1 * * * root /opt/a2a-world/infrastructure/scripts/cost-optimization.sh analyze >> /var/log/a2a-world/cron.log 2>&1

# Weekly tasks
# Full system backup on Sundays at 3:00 AM
0 3 * * 0 root /opt/a2a-world/infrastructure/backup/system-backup.sh all >> /var/log/a2a-world/cron.log 2>&1

# Weekly health check on Sundays at 4:00 AM
0 4 * * 0 root /opt/a2a-world/infrastructure/scripts/weekly-health-check.sh >> /var/log/a2a-world/cron.log 2>&1

# SSL certificate renewal check on Sundays at 5:00 AM
0 5 * * 0 root /opt/a2a-world/infrastructure/security/ssl/certbot-renewal.sh >> /var/log/a2a-world/cron.log 2>&1

# Monthly tasks
# Full cost optimization on 1st of month at 2:00 AM
0 2 1 * * root /opt/a2a-world/infrastructure/scripts/cost-optimization.sh full >> /var/log/a2a-world/cron.log 2>&1

# System cleanup on 1st and 15th of month at 1:00 AM
0 1 1,15 * * root /opt/a2a-world/infrastructure/scripts/system-cleanup.sh >> /var/log/a2a-world/cron.log 2>&1

# Security audit on 1st of month at 6:00 AM
0 6 1 * * root /opt/a2a-world/infrastructure/security/firewall/ufw-rules.sh >> /var/log/a2a-world/cron.log 2>&1
EOF

    # Set proper permissions
    chmod 644 /etc/cron.d/a2a-world-maintenance
    
    # Restart cron service
    systemctl restart cron
    
    log "Cron jobs configured successfully"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation for A2A World logs"
    
    cat > /etc/logrotate.d/a2a-world << 'EOF'
/var/log/a2a-world/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    create 644 root root
    postrotate
        # Send HUP signal to syslog to reopen log files
        /bin/kill -HUP `cat /var/run/rsyslogd.pid 2> /dev/null` 2> /dev/null || true
    endscript
}

/opt/a2a-world/reports/*.log {
    weekly
    missingok
    rotate 8
    compress
    delaycompress
    notifempty
    copytruncate
    create 644 root root
}
EOF

    # Test logrotate configuration
    logrotate -d /etc/logrotate.d/a2a-world
    
    log "Log rotation configured successfully"
}

# Create monitoring dashboard script
create_monitoring_dashboard() {
    log "Creating monitoring dashboard script"
    
    cat > "$SCRIPTS_DIR/dashboard.sh" << 'EOF'
#!/bin/bash
# A2A World Platform - System Dashboard
# Quick overview of system status

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== A2A World Platform Status Dashboard =====${NC}"
echo -e "Generated: $(date)"
echo ""

# System Overview
echo -e "${BLUE}=== SYSTEM OVERVIEW ===${NC}"
echo -e "Hostname: $(hostname)"
echo -e "Uptime: $(uptime -p)"
echo -e "Load: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

# Resource Usage
echo -e "${BLUE}=== RESOURCE USAGE ===${NC}"
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", ($3/$2) * 100.0}')
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')

echo -e "CPU Usage: ${CPU_USAGE}%"
if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo -e "Memory Usage: ${RED}${MEMORY_USAGE}%${NC} (HIGH)"
else
    echo -e "Memory Usage: ${GREEN}${MEMORY_USAGE}%${NC}"
fi

if [[ $DISK_USAGE -gt 80 ]]; then
    echo -e "Disk Usage: ${RED}${DISK_USAGE}%${NC} (HIGH)"
else
    echo -e "Disk Usage: ${GREEN}${DISK_USAGE}%${NC}"
fi
echo ""

# Docker Services
echo -e "${BLUE}=== DOCKER SERVICES ===${NC}"
docker service ls --format "table {{.Name}}\t{{.Mode}}\t{{.Replicas}}\t{{.Image}}"
echo ""

# Application Health
echo -e "${BLUE}=== APPLICATION HEALTH ===${NC}"
if curl -s https://api.a2aworld.ai/health >/dev/null 2>&1; then
    echo -e "API Health: ${GREEN}UP${NC}"
else
    echo -e "API Health: ${RED}DOWN${NC}"
fi

if curl -s https://a2aworld.ai >/dev/null 2>&1; then
    echo -e "Frontend Health: ${GREEN}UP${NC}"
else
    echo -e "Frontend Health: ${RED}DOWN${NC}"
fi

# SSL Certificate
echo -e "${BLUE}=== SSL CERTIFICATE ===${NC}"
CERT_DAYS=$(echo | openssl s_client -connect a2aworld.ai:443 -servername a2aworld.ai 2>/dev/null | openssl x509 -noout -dates | grep notAfter | cut -d'=' -f2 | xargs -I {} date -d "{}" +%s)
CURRENT_DATE=$(date +%s)
DAYS_UNTIL_EXPIRY=$(( (CERT_DAYS - CURRENT_DATE) / 86400 ))

if [[ $DAYS_UNTIL_EXPIRY -lt 7 ]]; then
    echo -e "SSL Certificate: ${RED}Expires in ${DAYS_UNTIL_EXPIRY} days${NC}"
elif [[ $DAYS_UNTIL_EXPIRY -lt 30 ]]; then
    echo -e "SSL Certificate: ${YELLOW}Expires in ${DAYS_UNTIL_EXPIRY} days${NC}"
else
    echo -e "SSL Certificate: ${GREEN}Valid (${DAYS_UNTIL_EXPIRY} days remaining)${NC}"
fi

echo ""
echo -e "${BLUE}=== RECENT ALERTS ===${NC}"
tail -5 /var/log/a2a-world/maintenance-setup.log 2>/dev/null | grep -E "(ERROR|WARNING|CRITICAL)" || echo "No recent alerts"

echo ""
echo -e "${BLUE}=== QUICK ACTIONS ===${NC}"
echo "View logs: journalctl -fu docker"
echo "Restart services: $0/../restart-services.sh"
echo "Run health check: $0/weekly-health-check.sh"
echo "View cost report: ls -la /opt/a2a-world/reports/"
EOF

    chmod +x "$SCRIPTS_DIR/dashboard.sh"
    
    log "Monitoring dashboard created successfully"
}

# Verify installation
verify_installation() {
    log "Verifying maintenance automation installation"
    
    # Check cron jobs
    if crontab -l 2>/dev/null | grep -q "a2a-world" || [ -f /etc/cron.d/a2a-world-maintenance ]; then
        log "✓ Cron jobs configured"
    else
        log "✗ Cron jobs not found"
    fi
    
    # Check log rotation
    if [ -f /etc/logrotate.d/a2a-world ]; then
        log "✓ Log rotation configured"
    else
        log "✗ Log rotation not configured"
    fi
    
    # Check scripts
    local script_count=$(find "$SCRIPTS_DIR" -name "*.sh" -executable | wc -l)
    log "✓ $script_count executable scripts found"
    
    # Test a simple script
    if "$SCRIPTS_DIR/dashboard.sh" >/dev/null 2>&1; then
        log "✓ Dashboard script test passed"
    else
        log "✗ Dashboard script test failed"
    fi
    
    log "Verification completed"
}

# Main function
main() {
    local action="${1:-install}"
    
    log "Starting maintenance automation setup: $action"
    
    case "$action" in
        "install")
            setup_directories
            install_dependencies
            create_maintenance_scripts
            setup_cron_jobs
            setup_log_rotation
            create_monitoring_dashboard
            verify_installation
            ;;
        "verify")
            verify_installation
            ;;
        "dashboard")
            "$SCRIPTS_DIR/dashboard.sh"
            ;;
        "status")
            echo "Maintenance automation status:"
            echo "- Cron jobs: $(crontab -l 2>/dev/null | grep -c a2a-world || echo 0)"
            echo "- Scripts: $(find "$SCRIPTS_DIR" -name "*.sh" | wc -l)"
            echo "- Log files: $(find /var/log/a2a-world -name "*.log" | wc -l)"
            ;;
        *)
            echo "Usage: $0 {install|verify|dashboard|status}"
            echo "  install   - Install and configure maintenance automation"
            echo "  verify    - Verify installation"
            echo "  dashboard - Show system dashboard"
            echo "  status    - Show automation status"
            ;;
    esac
    
    log "Maintenance automation setup completed"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi