# A2A World Platform - Production Deployment Guide

This guide provides step-by-step instructions for deploying the A2A World Platform to production on DigitalOcean.

## 🏗️ Infrastructure Overview

The production deployment creates:

- **2x Application Droplets** (s-2vcpu-4gb) - $48/month
- **1x Monitoring Droplet** (s-1vcpu-2gb) - $12/month
- **1x PostgreSQL Database** (db-s-1vcpu-1gb) - $15/month
- **1x Load Balancer** (lb-small) - $12/month
- **1x Container Registry** - $5/month
- **1x Spaces Storage** - $5/month + usage
- **Backup Storage** - $5-10/month estimated

**Total Estimated Cost: $102-112/month**

## 🔑 Prerequisites

### 1. Required Tools

Install the following tools on your deployment machine:

- **Terraform** (>= 1.0): [Download](https://www.terraform.io/downloads)
- **Docker**: [Download](https://www.docker.com/get-started)
- **DigitalOcean CLI (doctl)**: [Download](https://docs.digitalocean.com/reference/doctl/how-to/install/)

### 2. API Tokens

You'll need the following API tokens:

- **DigitalOcean API Token**: [Create here](https://cloud.digitalocean.com/account/api/tokens)
- **Cloudflare API Token**: [Create here](https://dash.cloudflare.com/profile/api-tokens)

### 3. Domain Configuration

Ensure your domain (`a2aworld.ai`) is:
- Added to your Cloudflare account
- DNS managed by Cloudflare
- Ready for DNS record updates

### 4. SSH Key Setup

Add your SSH public key to DigitalOcean:
```bash
doctl compute ssh-key create "a2a-world-key" --public-key-file ~/.ssh/id_rsa.pub
```

Get your SSH key fingerprint:
```bash
doctl compute ssh-key list
```

Update `infrastructure/deployment/terraform.tfvars` with your SSH key fingerprint.

## 🚀 Deployment Process

### Step 1: Environment Setup

1. **Set API tokens**:
   ```bash
   # Linux/Mac
   export DO_TOKEN="your_digitalocean_api_token"
   export CLOUDFLARE_API_TOKEN="your_cloudflare_api_token"
   
   # Windows PowerShell
   $env:DO_TOKEN = "your_digitalocean_api_token"
   $env:CLOUDFLARE_API_TOKEN = "your_cloudflare_api_token"
   ```

2. **Update configuration**:
   - Edit `infrastructure/deployment/terraform.tfvars`
   - Add your SSH key fingerprint
   - Update allowed IP addresses for SSH access

### Step 2: Infrastructure Deployment

Choose your deployment method:

#### Option A: Automated Deployment (Linux/Mac)
```bash
./infrastructure/deployment/deploy-production.sh
```

#### Option B: PowerShell Deployment (Windows)
```powershell
./infrastructure/deployment/deploy-production.ps1
```

#### Option C: Manual Step-by-Step

1. **Initialize Terraform**:
   ```bash
   cd infrastructure/terraform
   terraform init \
     -backend-config="access_key=$DO_TOKEN" \
     -backend-config="secret_key=$DO_TOKEN"
   ```

2. **Plan deployment**:
   ```bash
   terraform plan \
     -var="do_token=$DO_TOKEN" \
     -var="cloudflare_api_token=$CLOUDFLARE_API_TOKEN" \
     -var-file="../deployment/terraform.tfvars" \
     -out=tfplan
   ```

3. **Apply deployment**:
   ```bash
   terraform apply tfplan
   ```

### Step 3: Application Deployment

1. **Build and push Docker images**:
   ```bash
   # Login to container registry
   doctl registry login
   
   # Build and push images
   docker build -f infrastructure/docker/production/Dockerfile.api -t registry.digitalocean.com/a2aworldregistry/a2a-world/api:latest .
   docker push registry.digitalocean.com/a2aworldregistry/a2a-world/api:latest
   
   docker build -f infrastructure/docker/production/Dockerfile.frontend -t registry.digitalocean.com/a2aworldregistry/a2a-world/frontend:latest .
   docker push registry.digitalocean.com/a2aworldregistry/a2a-world/frontend:latest
   
   docker build -f infrastructure/docker/production/Dockerfile.agents -t registry.digitalocean.com/a2aworldregistry/a2a-world/agents:latest .
   docker push registry.digitalocean.com/a2aworldregistry/a2a-world/agents:latest
   ```

2. **Deploy to application servers**:
   ```bash
   # Get server IPs from Terraform output
   terraform output app_server_ips
   
   # Copy configuration to each server
   scp docker-compose.production.yml .env.production deploy@<SERVER_IP>:/opt/a2a-world/
   
   # SSH to each server and start services
   ssh deploy@<SERVER_IP>
   cd /opt/a2a-world
   docker-compose -f docker-compose.production.yml up -d
   ```

### Step 4: Database Initialization

1. **SSH to first application server**:
   ```bash
   ssh deploy@<FIRST_SERVER_IP>
   ```

2. **Initialize database schema**:
   ```bash
   cd /opt/a2a-world
   docker-compose -f docker-compose.production.yml exec api python database/scripts/init_database.py
   ```

3. **Verify database setup**:
   ```bash
   docker-compose -f docker-compose.production.yml exec postgres psql -U a2a_user -d a2a_world -c "\dt"
   ```

### Step 5: DNS Configuration

1. **Update DNS records** in Cloudflare (or your DNS provider):
   - `a2aworld.ai` → `<LOAD_BALANCER_IP>`
   - `www.a2aworld.ai` → `<LOAD_BALANCER_IP>`
   - `api.a2aworld.ai` → `<LOAD_BALANCER_IP>`
   - `monitoring.a2aworld.ai` → `<MONITORING_SERVER_IP>`

2. **Wait for DNS propagation** (5-10 minutes)

### Step 6: SSL Certificate Setup

SSL certificates are automatically provisioned by Let's Encrypt through the Terraform configuration. Wait 5-10 minutes after DNS propagation for certificates to be issued.

## 🔍 Validation & Testing

### Health Checks

1. **API Health**:
   ```bash
   curl https://api.a2aworld.ai/health
   ```

2. **Frontend**:
   ```bash
   curl https://a2aworld.ai
   ```

3. **Database Connection**:
   ```bash
   # SSH to app server
   docker-compose -f docker-compose.production.yml exec api python -c "
   from database.connection import get_db_connection
   conn = get_db_connection()
   print('Database connection successful!' if conn else 'Failed')
   "
   ```

### Monitoring Access

- **Grafana**: `http://<MONITORING_IP>:3000`
  - Username: `admin`
  - Password: `A2AWorld2024!`

- **Prometheus**: `http://<MONITORING_IP>:9090`

## 🔐 Security Checklist

- [ ] Update default passwords
- [ ] Restrict SSH access to specific IP addresses
- [ ] Configure firewall rules
- [ ] Set up automated security updates
- [ ] Enable audit logging
- [ ] Configure backup encryption
- [ ] Set up intrusion detection

## 📊 Monitoring & Maintenance

### Daily Checks
- Service health status
- Resource utilization
- Error logs
- Backup status

### Weekly Checks
- Security updates
- Performance metrics
- Cost optimization
- Capacity planning

### Monthly Tasks
- Security audit
- Backup testing
- Cost review
- Performance optimization

## 🆘 Troubleshooting

### Common Issues

1. **SSL Certificate Issues**:
   - Check DNS propagation
   - Verify Cloudflare proxy status
   - Check Let's Encrypt rate limits

2. **Database Connection Issues**:
   - Verify network connectivity
   - Check credentials
   - Review firewall rules

3. **Docker Image Issues**:
   - Check registry authentication
   - Verify image tags
   - Review build logs

### Emergency Contacts

- **Infrastructure Issues**: Check DigitalOcean status page
- **DNS Issues**: Check Cloudflare status page  
- **Application Issues**: Check application logs

## 🔄 Backup & Recovery

### Automated Backups

- **Database**: Automatic daily backups (7-day retention)
- **Application Data**: Daily volume snapshots
- **Configuration**: Git repository backups

### Recovery Procedures

1. **Database Recovery**:
   ```bash
   # Restore from backup
   doctl databases backup restore <database-id> <backup-id>
   ```

2. **Full System Recovery**:
   ```bash
   # Redeploy from infrastructure code
   terraform apply
   ```

## 📞 Support

For deployment support:
- Review logs: `/var/log/a2a-world/`
- Check monitoring: Grafana dashboard
- Contact: DevOps team

---

## 📋 Post-Deployment Checklist

- [ ] Infrastructure deployed successfully
- [ ] DNS records configured
- [ ] SSL certificates active
- [ ] All services running
- [ ] Database initialized
- [ ] Monitoring configured
- [ ] Backups enabled
- [ ] Security hardened
- [ ] Documentation updated
- [ ] Team notified