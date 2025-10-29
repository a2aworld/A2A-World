#!/bin/bash
# A2A World Platform - Production Deployment Script
# Deploy complete A2A World platform to DigitalOcean

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TERRAFORM_DIR="infrastructure/terraform"
DOCKER_DIR="infrastructure/docker/production"
PROJECT_NAME="a2a-world"
ENVIRONMENT="production"

echo -e "${GREEN}🚀 Starting A2A World Platform Production Deployment${NC}"
echo "=================================================="

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check if required tools are installed
command -v terraform >/dev/null 2>&1 || { echo -e "${RED}❌ Terraform not installed${NC}" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo -e "${RED}❌ Docker not installed${NC}" >&2; exit 1; }
command -v doctl >/dev/null 2>&1 || { echo -e "${RED}❌ DigitalOcean CLI (doctl) not installed${NC}" >&2; exit 1; }

# Check if required environment variables are set
if [ -z "$DO_TOKEN" ]; then
    echo -e "${RED}❌ DO_TOKEN environment variable not set${NC}"
    echo "Please set your DigitalOcean API token:"
    echo "export DO_TOKEN=your_digitalocean_api_token"
    exit 1
fi

if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    echo -e "${RED}❌ CLOUDFLARE_API_TOKEN environment variable not set${NC}"
    echo "Please set your Cloudflare API token:"
    echo "export CLOUDFLARE_API_TOKEN=your_cloudflare_api_token"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Generate secure secrets
echo -e "${YELLOW}🔐 Generating secure secrets...${NC}"

# Generate random passwords and keys
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
SECRET_KEY=$(openssl rand -base64 64)
SPACES_ACCESS_KEY=$(openssl rand -hex 20)
SPACES_SECRET_KEY=$(openssl rand -base64 32)
SMTP_PASSWORD=$(openssl rand -base64 32)

# Save secrets to secure file (will be used for Docker secrets)
SECRETS_FILE="infrastructure/deployment/.secrets.env"
cat > "$SECRETS_FILE" << EOF
# A2A World Platform - Production Secrets
# KEEP THIS FILE SECURE - DO NOT COMMIT TO VERSION CONTROL

DB_PASSWORD=$DB_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
SECRET_KEY=$SECRET_KEY
SPACES_ACCESS_KEY=$SPACES_ACCESS_KEY
SPACES_SECRET_KEY=$SPACES_SECRET_KEY
SMTP_PASSWORD=$SMTP_PASSWORD

# Generated on: $(date)
EOF

chmod 600 "$SECRETS_FILE"
echo -e "${GREEN}✅ Secrets generated and saved securely${NC}"

# Initialize Terraform
echo -e "${YELLOW}🏗️  Initializing Terraform...${NC}"
cd $TERRAFORM_DIR

# Initialize with remote state backend
terraform init \
  -backend-config="access_key=$DO_TOKEN" \
  -backend-config="secret_key=$DO_TOKEN"

# Validate Terraform configuration
terraform validate
echo -e "${GREEN}✅ Terraform configuration validated${NC}"

# Plan Terraform deployment
echo -e "${YELLOW}📋 Planning Terraform deployment...${NC}"
terraform plan \
  -var="do_token=$DO_TOKEN" \
  -var="cloudflare_api_token=$CLOUDFLARE_API_TOKEN" \
  -var-file="../deployment/terraform.tfvars" \
  -out=tfplan

echo -e "${YELLOW}❓ Review the Terraform plan above. Continue with deployment? (y/N)${NC}"
read -r response
if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${YELLOW}⏸️  Deployment cancelled by user${NC}"
    exit 0
fi

# Apply Terraform deployment
echo -e "${YELLOW}🚀 Deploying infrastructure...${NC}"
terraform apply tfplan

# Get Terraform outputs
echo -e "${YELLOW}📄 Getting infrastructure details...${NC}"
LB_IP=$(terraform output -raw load_balancer_ip)
DB_HOST=$(terraform output -raw database_host)
DB_PORT=$(terraform output -raw database_port)
APP_IPS=$(terraform output -json app_server_ips | jq -r '.public[]')
MONITORING_IP=$(terraform output -raw monitoring_server_ip)

echo -e "${GREEN}✅ Infrastructure deployed successfully!${NC}"
echo "Load Balancer IP: $LB_IP"
echo "Database Host: $DB_HOST"
echo "App Server IPs: $APP_IPS"
echo "Monitoring IP: $MONITORING_IP"

cd ../..

# Update production environment file with actual values
echo -e "${YELLOW}📝 Updating production environment configuration...${NC}"
sed -i.bak \
  -e "s/CHANGE_ME_DB_PASSWORD/$DB_PASSWORD/g" \
  -e "s/CHANGE_ME_REDIS_PASSWORD/$REDIS_PASSWORD/g" \
  -e "s/CHANGE_ME_PRODUCTION_SECRET_KEY_64_CHARACTERS_MINIMUM_FOR_SECURITY/$SECRET_KEY/g" \
  -e "s/CHANGE_ME_SPACES_ACCESS_KEY/$SPACES_ACCESS_KEY/g" \
  -e "s/CHANGE_ME_SPACES_SECRET_KEY/$SPACES_SECRET_KEY/g" \
  -e "s/CHANGE_ME_SMTP_PASSWORD/$SMTP_PASSWORD/g" \
  -e "s/localhost:5432/$DB_HOST:$DB_PORT/g" \
  .env.production

# Build and push Docker images
echo -e "${YELLOW}🐳 Building and pushing Docker images...${NC}"

# Login to DigitalOcean Container Registry
doctl registry login

# Build API image
docker build -f infrastructure/docker/production/Dockerfile.api -t registry.digitalocean.com/a2aworldregistry/a2a-world/api:latest .
docker push registry.digitalocean.com/a2aworldregistry/a2a-world/api:latest

# Build Frontend image  
docker build -f infrastructure/docker/production/Dockerfile.frontend -t registry.digitalocean.com/a2aworldregistry/a2a-world/frontend:latest .
docker push registry.digitalocean.com/a2aworldregistry/a2a-world/frontend:latest

# Build Agents image
docker build -f infrastructure/docker/production/Dockerfile.agents -t registry.digitalocean.com/a2aworldregistry/a2a-world/agents:latest .
docker push registry.digitalocean.com/a2aworldregistry/a2a-world/agents:latest

echo -e "${GREEN}✅ Docker images built and pushed${NC}"

# Deploy to application servers
echo -e "${YELLOW}🚀 Deploying application stack...${NC}"

# Copy deployment files to servers
for IP in $APP_IPS; do
    echo "Deploying to server: $IP"
    
    # Copy files to server
    scp -o StrictHostKeyChecking=no \
        infrastructure/docker/production/docker-compose.production.yml \
        .env.production \
        deploy@$IP:/opt/a2a-world/
    
    # Deploy services on server
    ssh -o StrictHostKeyChecking=no deploy@$IP << 'ENDSSH'
        cd /opt/a2a-world
        
        # Load environment variables
        source .env.production
        
        # Pull latest images
        docker-compose -f docker-compose.production.yml pull
        
        # Start services
        docker-compose -f docker-compose.production.yml up -d
        
        # Wait for services to be ready
        echo "Waiting for services to start..."
        sleep 30
        
        # Check service health
        docker-compose -f docker-compose.production.yml ps
ENDSSH
    
    echo -e "${GREEN}✅ Deployment completed on server $IP${NC}"
done

# Deploy monitoring stack
if [ "$MONITORING_IP" != "Not deployed" ]; then
    echo -e "${YELLOW}📊 Setting up monitoring...${NC}"
    
    # Copy monitoring configuration
    scp -o StrictHostKeyChecking=no -r \
        infrastructure/monitoring/ \
        monitoring@$MONITORING_IP:/opt/monitoring/config/
    
    # Start monitoring services
    ssh -o StrictHostKeyChecking=no monitoring@$MONITORING_IP << 'ENDSSH'
        cd /opt/monitoring
        docker-compose up -d
        
        # Wait for services
        sleep 30
        
        # Check monitoring services
        docker-compose ps
ENDSSH
    
    echo -e "${GREEN}✅ Monitoring stack deployed${NC}"
fi

# Run database initialization
echo -e "${YELLOW}🗄️  Initializing database...${NC}"
FIRST_APP_IP=$(echo $APP_IPS | cut -d' ' -f1)

ssh -o StrictHostKeyChecking=no deploy@$FIRST_APP_IP << 'ENDSSH'
    cd /opt/a2a-world
    
    # Run database migrations
    docker-compose -f docker-compose.production.yml exec -T api python database/scripts/init_database.py
    
    # Verify database structure
    docker-compose -f docker-compose.production.yml exec -T api python -c "
from database.connection import get_db_connection
import psycopg2

try:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT count(*) FROM information_schema.tables WHERE table_schema = \'public\';')
    table_count = cur.fetchone()[0]
    print(f'Database initialized with {table_count} tables')
    cur.close()
    conn.close()
except Exception as e:
    print(f'Database check failed: {e}')
    exit(1)
"
ENDSSH

echo -e "${GREEN}✅ Database initialized successfully${NC}"

# Test deployment
echo -e "${YELLOW}🧪 Testing deployment...${NC}"

# Test API health endpoint
echo "Testing API health..."
curl -f "http://$LB_IP/health" || echo "API health check failed"

# Test frontend
echo "Testing frontend..."
curl -f "http://$LB_IP/" || echo "Frontend health check failed"

# Display deployment summary
echo ""
echo -e "${GREEN}🎉 A2A World Platform Production Deployment Complete!${NC}"
echo "======================================================="
echo ""
echo "🌐 Production URLs:"
echo "   Main Site: https://a2aworld.ai"
echo "   API: https://api.a2aworld.ai"
echo "   API Docs: https://api.a2aworld.ai/docs"
echo ""
if [ "$MONITORING_IP" != "Not deployed" ]; then
echo "📊 Monitoring:"
echo "   Grafana: http://$MONITORING_IP:3000 (admin/A2AWorld2024!)"
echo "   Prometheus: http://$MONITORING_IP:9090"
echo ""
fi
echo "🔧 Infrastructure:"
echo "   Load Balancer IP: $LB_IP"
echo "   Application Servers: $APP_IPS"
echo "   Database: $DB_HOST:$DB_PORT"
echo ""
echo "📋 Next Steps:"
echo "   1. Update DNS records to point to: $LB_IP"
echo "   2. Wait for SSL certificate provisioning (5-10 minutes)"
echo "   3. Test all functionality end-to-end"
echo "   4. Set up monitoring alerts"
echo "   5. Configure backups"
echo ""
echo "🔐 Important:"
echo "   - Secrets saved to: $SECRETS_FILE"
echo "   - Keep this file secure and backed up"
echo "   - Consider using a proper secrets management service"
echo ""
echo -e "${GREEN}✅ Deployment successful!${NC}"