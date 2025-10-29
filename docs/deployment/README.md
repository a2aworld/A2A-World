# A2A World Platform - Deployment Guide

## Overview

This guide provides complete instructions for deploying the A2A World Platform to production using DigitalOcean infrastructure. The deployment includes:

- FastAPI backend with PostgreSQL + PostGIS database
- Next.js frontend with SSR
- Multi-agent system with NATS messaging
- Monitoring stack (Prometheus + Grafana + Loki)
- Automated backups and disaster recovery
- SSL/TLS encryption with automatic renewal
- Comprehensive security hardening

**Estimated Monthly Cost**: $75-150 USD (depending on configuration)

## Prerequisites

Before starting the deployment, ensure you have:

### Required Accounts
- [DigitalOcean Account](https://digitalocean.com) with API token
- [Cloudflare Account](https://cloudflare.com) (for DNS and CDN)
- [GitHub Account](https://github.com) (for CI/CD)
- Domain name (preferably `a2aworld.ai`)

### Required Tools
- [Terraform](https://terraform.io) >= 1.5.0
- [Docker](https://docker.com) >= 24.0.0
- [Docker Compose](https://docs.docker.com/compose/) >= 2.20.0
- [GitHub CLI](https://cli.github.com/) (optional but recommended)
- SSH client
- `curl`, `jq`, `git` command-line tools

### Local Environment
- Linux/macOS/WSL2 environment
- At least 8GB RAM for local development
- 50GB free disk space

## Quick Start (Production Ready in 30 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/A2A-World.git
cd A2A-World
```

### 2. Configure Environment Variables
```bash
# Copy environment template
cp .env.example .env

# Edit with your values
nano .env
```

Required variables:
```bash
# DigitalOcean
DO_TOKEN=your_digitalocean_token
DOMAIN_NAME=a2aworld.ai

# Cloudflare
CLOUDFLARE_API_TOKEN=your_cloudflare_token

# GitHub (for CI/CD)
GITHUB_TOKEN=your_github_token

# Notification
SLACK_WEBHOOK=your_slack_webhook_url
EMAIL_RECIPIENT=admin@a2aworld.ai
```

### 3. Initialize Infrastructure
```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars"

# Deploy infrastructure (takes 10-15 minutes)
terraform apply -var-file="production.tfvars"
```

### 4. Configure DNS
Point your domain's nameservers to Cloudflare and create the following DNS records:
```
A     @              -> [Load Balancer IP]
A     www            -> [Load Balancer IP]  
A     api            -> [Load Balancer IP]
A     monitoring     -> [Monitoring Server IP]
```

### 5. Deploy Application
```bash
# Set up secrets management
sudo ./infrastructure/security/secrets/secrets-manager.sh init

# Deploy using Docker Swarm
docker swarm init
docker stack deploy --compose-file infrastructure/docker/production/docker-compose.production.yml a2a-world
```

### 6. Verify Deployment
```bash
# Check service status
docker service ls

# Verify endpoints
curl https://api.a2aworld.ai/health
curl https://a2aworld.ai

# Check monitoring
open https://monitoring.a2aworld.ai:3000
```

## Detailed Deployment Steps

### Phase 1: Infrastructure Setup (15-20 minutes)

#### 1.1 Terraform Configuration
```bash
cd infrastructure/terraform

# Create production variables file
cat > production.tfvars << EOF
do_token = "your_digitalocean_token"
cloudflare_api_token = "your_cloudflare_token"
environment = "production"
domain_name = "a2aworld.ai"
droplet_size = "s-2vcpu-4gb"
droplet_count = 2
db_size = "db-s-1vcpu-1gb"
enable_monitoring = true
EOF

# Initialize and deploy
terraform init
terraform plan -var-file="production.tfvars"
terraform apply -var-file="production.tfvars"
```

#### 1.2 Post-Infrastructure Setup
```bash
# Get server IPs from Terraform output
terraform output

# Add SSH keys to servers
ssh-copy-id root@[SERVER_IP]

# Run security hardening
ssh root@[SERVER_IP] 'bash -s' < infrastructure/security/firewall/ufw-rules.sh
```

### Phase 2: Application Deployment (10-15 minutes)

#### 2.1 Secrets Management
```bash
# Initialize secrets on each server
ssh root@[SERVER_IP] << 'EOF'
cd /opt/a2a-world
./infrastructure/security/secrets/secrets-manager.sh init
./infrastructure/security/secrets/secrets-manager.sh env
EOF
```

#### 2.2 Docker Swarm Setup
```bash
# Initialize swarm on manager node
ssh root@[MANAGER_IP] 'docker swarm init'

# Get join token
JOIN_TOKEN=$(ssh root@[MANAGER_IP] 'docker swarm join-token worker -q')

# Join worker nodes
ssh root@[WORKER_IP] "docker swarm join --token $JOIN_TOKEN [MANAGER_IP]:2377"
```

#### 2.3 Deploy Application Stack
```bash
# Create external networks and volumes
ssh root@[MANAGER_IP] << 'EOF'
docker network create -d overlay a2a-network
docker network create -d overlay monitoring
docker volume create postgres_data
docker volume create redis_data
docker volume create app_data
docker volume create logs
EOF

# Deploy the stack
scp infrastructure/docker/production/docker-compose.production.yml root@[MANAGER_IP]:/opt/a2a-world/
ssh root@[MANAGER_IP] 'cd /opt/a2a-world && docker stack deploy --compose-file docker-compose.production.yml a2a-world'
```

### Phase 3: SSL and Security Setup (5-10 minutes)

#### 3.1 SSL Certificate Setup
```bash
# Run SSL setup on each server
ssh root@[SERVER_IP] 'bash -s' < infrastructure/security/ssl/certbot-renewal.sh

# Set up automatic renewal
ssh root@[SERVER_IP] << 'EOF'
echo "0 3 * * 0 /opt/a2a-world/infrastructure/security/ssl/certbot-renewal.sh" | crontab -
EOF
```

#### 3.2 Security Hardening
```bash
# Apply security configurations
ssh root@[SERVER_IP] << 'EOF'
# Set up fail2ban
systemctl enable fail2ban
systemctl start fail2ban

# Configure logrotate
systemctl enable logrotate.timer
systemctl start logrotate.timer
EOF
```

### Phase 4: Monitoring Setup (5 minutes)

#### 4.1 Deploy Monitoring Stack
```bash
# Copy monitoring configs
scp -r infrastructure/monitoring/ root@[MONITORING_IP]:/opt/monitoring/

# Deploy monitoring
ssh root@[MONITORING_IP] << 'EOF'
cd /opt/monitoring
docker-compose up -d
EOF
```

#### 4.2 Configure Grafana
1. Access Grafana at `https://monitoring.a2aworld.ai:3000`
2. Login with `admin/A2AWorld2024!`
3. Datasources are auto-configured via provisioning
4. Dashboards are available in the A2A World folder

### Phase 5: Backup Configuration (5 minutes)

#### 5.1 Setup Automated Backups
```bash
# Configure backup scripts
ssh root@[SERVER_IP] << 'EOF'
# Set executable permissions
chmod +x /opt/a2a-world/infrastructure/backup/*.sh

# Setup cron jobs
cat > /etc/cron.d/a2a-backups << 'EOC'
# Database backup - daily at 2 AM
0 2 * * * root /opt/a2a-world/infrastructure/backup/database-backup.sh full

# System backup - weekly on Sundays at 3 AM  
0 3 * * 0 root /opt/a2a-world/infrastructure/backup/system-backup.sh all

# SSL renewal check - weekly on Sundays at 4 AM
0 4 * * 0 root /opt/a2a-world/infrastructure/security/ssl/certbot-renewal.sh
EOC
EOF
```

## CI/CD Pipeline Setup

### 1. Configure GitHub Secrets
In your GitHub repository, go to Settings > Secrets and add:

```bash
DIGITALOCEAN_ACCESS_TOKEN=your_do_token
SPACES_ACCESS_KEY=your_spaces_key  
SPACES_SECRET_KEY=your_spaces_secret
SSH_PRIVATE_KEY=your_ssh_private_key
STAGING_HOST=your_staging_server_ip
PROD_HOSTS=server1_ip,server2_ip
SLACK_WEBHOOK=your_slack_webhook
POSTGRES_PASSWORD=your_db_password
SECRET_KEY=your_app_secret_key
```

### 2. Enable GitHub Actions
The CI/CD pipeline automatically:
- Runs tests on every push/PR
- Builds and pushes Docker images
- Deploys to staging on `develop` branch
- Deploys to production on `main` branch
- Performs security scans
- Sends notifications

### 3. Branch Protection
Configure branch protection rules:
```bash
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci-cd"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1}'
```

## Verification and Testing

### 1. Application Health Checks
```bash
# API health
curl https://api.a2aworld.ai/health

# Frontend
curl https://a2aworld.ai

# Database connectivity
curl https://api.a2aworld.ai/api/v1/health/db
```

### 2. Monitoring Verification
- **Prometheus**: `https://monitoring.a2aworld.ai:9090`
- **Grafana**: `https://monitoring.a2aworld.ai:3000`
- **Alertmanager**: `https://monitoring.a2aworld.ai:9093`

### 3. Security Testing
```bash
# SSL/TLS check
echo | openssl s_client -connect a2aworld.ai:443 -servername a2aworld.ai 2>/dev/null | openssl x509 -noout -dates

# Port scanning (should show minimal open ports)
nmap -sS a2aworld.ai

# HTTP security headers
curl -I https://a2aworld.ai
```

### 4. Performance Testing
```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 https://api.a2aworld.ai/health

# Response time testing
curl -w "@curl-format.txt" -o /dev/null -s https://a2aworld.ai
```

## Cost Optimization

The deployment is configured for cost-effectiveness:

**Base Infrastructure** (~$75/month):
- 2x App servers (s-2vcpu-4gb): $48/month
- 1x Database (managed): $15/month  
- 1x Load balancer: $12/month
- 1x Monitoring server: $12/month
- Container registry: $5/month
- Spaces storage: $5/month
- Bandwidth: ~$10/month

**Optional Additions**:
- Backup storage: $5-10/month
- Additional monitoring: $10-20/month
- CDN bandwidth: $5-15/month

## Troubleshooting

See [Troubleshooting Guide](./troubleshooting.md) for common issues and solutions.

## Maintenance

See [Maintenance Guide](./maintenance.md) for ongoing maintenance procedures.

## Support

For deployment support:
- Check the troubleshooting guide
- Review logs in `/var/log/a2a-world/`
- Contact: admin@a2aworld.ai
- Documentation: https://docs.a2aworld.ai

---

**Next Steps**: After successful deployment, see [Post-Deployment Configuration](./post-deployment.md) for additional optimizations and configurations.