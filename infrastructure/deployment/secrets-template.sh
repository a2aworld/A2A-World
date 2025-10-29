#!/bin/bash
# A2A World Platform - Secrets Management Template
# Generate secure passwords and configure production secrets

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🔐 A2A World Platform - Production Secrets Generator${NC}"
echo "============================================================"

# Check if OpenSSL is available
if ! command -v openssl &> /dev/null; then
    echo "OpenSSL is required but not installed. Please install OpenSSL."
    exit 1
fi

echo -e "${YELLOW}Generating secure production secrets...${NC}"

# Generate secure random passwords and keys
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-64)
SPACES_ACCESS_KEY=$(openssl rand -hex 20)
SPACES_SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
SMTP_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)

# Create secrets file
SECRETS_FILE=".secrets.production"

cat > "$SECRETS_FILE" << EOF
# A2A World Platform - Production Secrets
# Generated: $(date)
# KEEP THIS FILE SECURE - DO NOT COMMIT TO VERSION CONTROL

# Database Configuration
export DB_PASSWORD="$DB_PASSWORD"
export POSTGRES_PASSWORD="$DB_PASSWORD"

# Redis Configuration  
export REDIS_PASSWORD="$REDIS_PASSWORD"

# Application Security
export SECRET_KEY="$SECRET_KEY"

# DigitalOcean Spaces
export SPACES_ACCESS_KEY="$SPACES_ACCESS_KEY"
export SPACES_SECRET_KEY="$SPACES_SECRET_KEY"

# Email Configuration (if using SMTP)
export SMTP_PASSWORD="$SMTP_PASSWORD"

# DigitalOcean & Cloudflare API Tokens (set these manually)
# export DO_TOKEN="your_digitalocean_api_token_here"
# export CLOUDFLARE_API_TOKEN="your_cloudflare_api_token_here"
EOF

chmod 600 "$SECRETS_FILE"

echo -e "${GREEN}✅ Production secrets generated successfully!${NC}"
echo ""
echo "📁 Secrets saved to: $SECRETS_FILE"
echo ""
echo "🔧 Next steps:"
echo "1. Edit $SECRETS_FILE and add your DigitalOcean and Cloudflare API tokens"
echo "2. Source the secrets file: source $SECRETS_FILE"
echo "3. Run the deployment script: ./infrastructure/deployment/deploy-production.sh"
echo ""
echo "⚠️  Important Security Notes:"
echo "   • Keep the secrets file secure and never commit it to version control"
echo "   • Consider using a proper secrets management service for production"
echo "   • Rotate these secrets regularly"
echo "   • Limit access to production systems"

echo ""
echo -e "${YELLOW}Generated Secrets Preview:${NC}"
echo "Database Password: $DB_PASSWORD"
echo "Redis Password: $REDIS_PASSWORD"  
echo "Secret Key: ${SECRET_KEY:0:20}... (truncated)"
echo "Spaces Access Key: $SPACES_ACCESS_KEY"