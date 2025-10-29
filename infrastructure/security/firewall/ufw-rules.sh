#!/bin/bash
# A2A World Platform - UFW Firewall Configuration
# Comprehensive firewall rules for production security

set -euo pipefail

# Configuration
LOG_FILE="/var/log/a2a-world/firewall-setup.log"

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

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

log "Starting UFW firewall configuration"

# Reset UFW to defaults
log "Resetting UFW to defaults"
ufw --force reset

# Set default policies
log "Setting default policies"
ufw default deny incoming
ufw default allow outgoing
ufw default deny forward

# Enable logging
log "Enabling UFW logging"
ufw logging on

# SSH access (port 22)
log "Allowing SSH access"
ufw allow ssh
ufw limit ssh comment 'SSH with rate limiting'

# HTTP and HTTPS (ports 80, 443)
log "Allowing HTTP and HTTPS traffic"
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Application specific ports
log "Configuring application ports"

# API service (port 8000) - only from load balancer
ufw allow from 10.0.0.0/16 to any port 8000 comment 'API service - internal'

# Frontend service (port 3000) - only from load balancer
ufw allow from 10.0.0.0/16 to any port 3000 comment 'Frontend service - internal'

# Database (port 5432) - only from application servers
ufw allow from 10.0.0.0/16 to any port 5432 comment 'PostgreSQL - internal'

# Redis (port 6379) - only from application servers
ufw allow from 10.0.0.0/16 to any port 6379 comment 'Redis - internal'

# NATS (port 4222) - only from internal network
ufw allow from 10.0.0.0/16 to any port 4222 comment 'NATS messaging - internal'

# Consul (port 8500) - only from internal network
ufw allow from 10.0.0.0/16 to any port 8500 comment 'Consul - internal'

# Monitoring ports - restricted access
log "Configuring monitoring ports"

# Prometheus (port 9090) - only from monitoring network
ufw allow from 10.0.0.0/16 to any port 9090 comment 'Prometheus - monitoring'

# Grafana (port 3001) - only from admin IPs
# Add your admin IPs here
ADMIN_IPS=(
    "203.0.113.0/24"  # Example admin network
    "198.51.100.0/24" # Example office network
)

for ip in "${ADMIN_IPS[@]}"; do
    ufw allow from "$ip" to any port 3001 comment "Grafana - admin access from $ip"
done

# Node exporter (port 9100) - only from monitoring server
ufw allow from 10.0.0.0/16 to any port 9100 comment 'Node Exporter - monitoring'

# Alert manager (port 9093) - only from monitoring network
ufw allow from 10.0.0.0/16 to any port 9093 comment 'Alertmanager - monitoring'

# Additional security rules
log "Applying additional security rules"

# Block common attack ports
BLOCK_PORTS=(
    21    # FTP
    23    # Telnet
    135   # RPC
    139   # NetBIOS
    445   # SMB
    1433  # MSSQL
    3389  # RDP
    5432  # PostgreSQL (allow only internal above)
    5984  # CouchDB
    6379  # Redis (allow only internal above)
    27017 # MongoDB
)

for port in "${BLOCK_PORTS[@]}"; do
    ufw deny "$port" comment "Block common attack port $port"
done

# Rate limiting for common services
log "Applying rate limiting rules"

# Limit HTTP connections
ufw limit 80/tcp comment 'HTTP rate limiting'
ufw limit 443/tcp comment 'HTTPS rate limiting'

# Custom application rules
log "Applying custom application rules"

# Allow internal Docker network communication
ufw allow from 172.16.0.0/12 comment 'Docker internal networks'
ufw allow from 192.168.0.0/16 comment 'Private networks'

# Allow loopback
ufw allow in on lo comment 'Allow loopback'

# ICMP rules for ping and network diagnostics
log "Configuring ICMP rules"
ufw allow out on any to any port 53 comment 'DNS lookups'

# Block IP forwarding
log "Disabling IP forwarding"
echo 'net.ipv4.ip_forward=0' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv6.conf.all.forwarding=0' >> /etc/sysctl.d/99-a2a-security.conf

# Additional security settings
log "Applying additional security settings"

# Protect against SYN flood attacks
echo 'net.ipv4.tcp_syncookies=1' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv4.tcp_max_syn_backlog=2048' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv4.tcp_synack_retries=2' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv4.tcp_syn_retries=5' >> /etc/sysctl.d/99-a2a-security.conf

# Protect against IP spoofing
echo 'net.ipv4.conf.default.rp_filter=1' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv4.conf.all.rp_filter=1' >> /etc/sysctl.d/99-a2a-security.conf

# Ignore ICMP ping requests
echo 'net.ipv4.icmp_echo_ignore_all=0' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv4.icmp_echo_ignore_broadcasts=1' >> /etc/sysctl.d/99-a2a-security.conf

# Disable source packet routing
echo 'net.ipv4.conf.all.accept_source_route=0' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv6.conf.all.accept_source_route=0' >> /etc/sysctl.d/99-a2a-security.conf

# Disable ICMP redirects
echo 'net.ipv4.conf.all.send_redirects=0' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv4.conf.default.send_redirects=0' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv4.conf.all.accept_redirects=0' >> /etc/sysctl.d/99-a2a-security.conf
echo 'net.ipv6.conf.all.accept_redirects=0' >> /etc/sysctl.d/99-a2a-security.conf

# Apply sysctl settings
sysctl -p /etc/sysctl.d/99-a2a-security.conf

# Create UFW application profiles
log "Creating UFW application profiles"

# A2A World API profile
cat > /etc/ufw/applications.d/a2a-world << 'EOF'
[A2A-World-API]
title=A2A World API
description=A2A World Platform API service
ports=8000/tcp

[A2A-World-Frontend]
title=A2A World Frontend
description=A2A World Platform frontend service
ports=3000/tcp

[A2A-World-Monitoring]
title=A2A World Monitoring
description=A2A World Platform monitoring services
ports=9090,3001,9093,9100/tcp
EOF

# Reload application profiles
ufw app update A2A-World-API
ufw app update A2A-World-Frontend
ufw app update A2A-World-Monitoring

# Geo-blocking (basic implementation)
log "Setting up geo-blocking"
# Block traffic from known hostile countries (adjust as needed)
# This is a basic implementation - consider using GeoIP databases for production

# Create custom chains for geo-blocking
iptables -N GEO_BLOCK || true
iptables -F GEO_BLOCK

# Add rules to log and drop traffic from specific countries
# Example: Block traffic from specific IP ranges (use actual country IP ranges)
# iptables -A GEO_BLOCK -s 1.2.3.0/24 -j LOG --log-prefix "GEO_BLOCK: "
# iptables -A GEO_BLOCK -s 1.2.3.0/24 -j DROP

# Insert geo-blocking chain at the beginning
iptables -I INPUT 1 -j GEO_BLOCK || true

# DDoS protection rules
log "Setting up DDoS protection"

# Limit new connections per second
iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT || true
iptables -A INPUT -p tcp --dport 443 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT || true

# Create fail2ban jail for A2A World
log "Creating fail2ban configuration"

cat > /etc/fail2ban/jail.d/a2a-world.conf << 'EOF'
[a2a-api-auth]
enabled = true
port = http,https,8000
filter = a2a-api-auth
logpath = /var/log/a2a-world/security.log
maxretry = 5
bantime = 3600
findtime = 300

[a2a-api-dos]
enabled = true
port = http,https,8000
filter = a2a-api-dos
logpath = /var/log/a2a-world/application.log
maxretry = 100
bantime = 600
findtime = 60

[nginx-limit-req]
enabled = true
port = http,https
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 10
bantime = 600
findtime = 60
EOF

# Create fail2ban filters
mkdir -p /etc/fail2ban/filter.d

cat > /etc/fail2ban/filter.d/a2a-api-auth.conf << 'EOF'
[Definition]
failregex = ^.*"message":"Authentication failed".*"remote_addr":"<HOST>".*$
            ^.*"message":"Invalid API key".*"remote_addr":"<HOST>".*$
            ^.*"message":"Access denied".*"remote_addr":"<HOST>".*$
ignoreregex =
EOF

cat > /etc/fail2ban/filter.d/a2a-api-dos.conf << 'EOF'
[Definition]
failregex = ^.*"message":"Rate limit exceeded".*"remote_addr":"<HOST>".*$
            ^.*"status":429.*"remote_addr":"<HOST>".*$
ignoreregex =
EOF

# Restart fail2ban
systemctl restart fail2ban || log "Warning: Could not restart fail2ban"

# Enable UFW
log "Enabling UFW firewall"
ufw --force enable

# Verify firewall status
log "Current UFW status:"
ufw status verbose | tee -a "$LOG_FILE"

# Create monitoring script
log "Creating firewall monitoring script"

cat > /usr/local/bin/firewall-monitor.sh << 'EOF'
#!/bin/bash
# A2A World Firewall Monitoring Script

LOG_FILE="/var/log/a2a-world/firewall-monitor.log"
ALERT_THRESHOLD=100

# Count blocked connections in the last 5 minutes
BLOCKED_COUNT=$(journalctl -u ufw --since "5 minutes ago" | grep -c "BLOCK" || echo "0")

if [[ $BLOCKED_COUNT -gt $ALERT_THRESHOLD ]]; then
    echo "[$(date)] WARNING: High number of blocked connections: $BLOCKED_COUNT" >> "$LOG_FILE"
    # Send alert (implement notification logic here)
fi

# Log current status
echo "[$(date)] Blocked connections in last 5 minutes: $BLOCKED_COUNT" >> "$LOG_FILE"
EOF

chmod +x /usr/local/bin/firewall-monitor.sh

# Add cron job for monitoring
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/firewall-monitor.sh") | crontab -

log "UFW firewall configuration completed successfully"
log "Important: Remember to update ADMIN_IPS in this script with your actual admin IP addresses"
log "Log files location: /var/log/a2a-world/"