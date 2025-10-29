# A2A World Platform - Production Deployment Script (PowerShell)
# Deploy complete A2A World platform to DigitalOcean

param(
    [string]$DoToken = $env:DO_TOKEN,
    [string]$CloudflareToken = $env:CLOUDFLARE_API_TOKEN
)

# Colors for output
$RED = "Red"
$GREEN = "Green" 
$YELLOW = "Yellow"

# Configuration
$TERRAFORM_DIR = "infrastructure/terraform"
$PROJECT_NAME = "a2a-world"
$ENVIRONMENT = "production"

Write-Host "🚀 Starting A2A World Platform Production Deployment" -ForegroundColor $GREEN
Write-Host "==================================================="

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor $YELLOW

# Check if required tools are installed
$tools = @("terraform", "docker", "doctl")
foreach ($tool in $tools) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "❌ $tool not installed" -ForegroundColor $RED
        exit 1
    }
}

# Check if required environment variables are set
if ([string]::IsNullOrEmpty($DoToken)) {
    Write-Host "❌ DO_TOKEN environment variable not set" -ForegroundColor $RED
    Write-Host "Please set your DigitalOcean API token:"
    Write-Host '$env:DO_TOKEN = "your_digitalocean_api_token"'
    exit 1
}

if ([string]::IsNullOrEmpty($CloudflareToken)) {
    Write-Host "❌ CLOUDFLARE_API_TOKEN environment variable not set" -ForegroundColor $RED
    Write-Host "Please set your Cloudflare API token:"
    Write-Host '$env:CLOUDFLARE_API_TOKEN = "your_cloudflare_api_token"'
    exit 1
}

Write-Host "✅ Prerequisites check passed" -ForegroundColor $GREEN

# Generate secure secrets
Write-Host "🔐 Generating secure secrets..." -ForegroundColor $YELLOW

# Generate random passwords and keys using .NET crypto
Add-Type -AssemblyName System.Security
$rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::new()

function Generate-SecurePassword {
    param([int]$length = 32)
    $bytes = New-Object byte[] $length
    $rng.GetBytes($bytes)
    return [Convert]::ToBase64String($bytes).Replace('+','').Replace('/','').Replace('=','').Substring(0, $length)
}

$DB_PASSWORD = Generate-SecurePassword
$REDIS_PASSWORD = Generate-SecurePassword  
$SECRET_KEY = Generate-SecurePassword -length 64
$SPACES_ACCESS_KEY = Generate-SecurePassword -length 20
$SPACES_SECRET_KEY = Generate-SecurePassword
$SMTP_PASSWORD = Generate-SecurePassword -length 16

# Save secrets to secure file
$SECRETS_FILE = "infrastructure/deployment/.secrets.env"
$secretsContent = @"
# A2A World Platform - Production Secrets
# KEEP THIS FILE SECURE - DO NOT COMMIT TO VERSION CONTROL

DB_PASSWORD=$DB_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
SECRET_KEY=$SECRET_KEY
SPACES_ACCESS_KEY=$SPACES_ACCESS_KEY
SPACES_SECRET_KEY=$SPACES_SECRET_KEY
SMTP_PASSWORD=$SMTP_PASSWORD

# Generated on: $(Get-Date)
"@

$secretsContent | Out-File -FilePath $SECRETS_FILE -Encoding utf8
Write-Host "✅ Secrets generated and saved securely" -ForegroundColor $GREEN

# Initialize Terraform
Write-Host "🏗️  Initializing Terraform..." -ForegroundColor $YELLOW
Set-Location $TERRAFORM_DIR

# Initialize with remote state backend
terraform init `
  -backend-config="access_key=$DoToken" `
  -backend-config="secret_key=$DoToken"

# Validate Terraform configuration
terraform validate
Write-Host "✅ Terraform configuration validated" -ForegroundColor $GREEN

# Plan Terraform deployment
Write-Host "📋 Planning Terraform deployment..." -ForegroundColor $YELLOW
terraform plan `
  -var="do_token=$DoToken" `
  -var="cloudflare_api_token=$CloudflareToken" `
  -var-file="../deployment/terraform.tfvars" `
  -out=tfplan

Write-Host "❓ Review the Terraform plan above. Continue with deployment? (y/N)" -ForegroundColor $YELLOW
$response = Read-Host
if ($response -notmatch '^[yY]([eE][sS])?$') {
    Write-Host "⏸️  Deployment cancelled by user" -ForegroundColor $YELLOW
    exit 0
}

# Apply Terraform deployment
Write-Host "🚀 Deploying infrastructure..." -ForegroundColor $YELLOW
terraform apply tfplan

# Get Terraform outputs
Write-Host "📄 Getting infrastructure details..." -ForegroundColor $YELLOW
$LB_IP = terraform output -raw load_balancer_ip
$DB_HOST = terraform output -raw database_host
$DB_PORT = terraform output -raw database_port
$APP_IPS_JSON = terraform output -json app_server_ips
$APP_IPS = ($APP_IPS_JSON | ConvertFrom-Json).public
$MONITORING_IP = terraform output -raw monitoring_server_ip

Write-Host "✅ Infrastructure deployed successfully!" -ForegroundColor $GREEN
Write-Host "Load Balancer IP: $LB_IP"
Write-Host "Database Host: $DB_HOST"
Write-Host "App Server IPs: $($APP_IPS -join ', ')"
Write-Host "Monitoring IP: $MONITORING_IP"

Set-Location "../.."

# Update production environment file with actual values
Write-Host "📝 Updating production environment configuration..." -ForegroundColor $YELLOW
$envContent = Get-Content .env.production
$envContent = $envContent.Replace("CHANGE_ME_DB_PASSWORD", $DB_PASSWORD)
$envContent = $envContent.Replace("CHANGE_ME_REDIS_PASSWORD", $REDIS_PASSWORD)  
$envContent = $envContent.Replace("CHANGE_ME_PRODUCTION_SECRET_KEY_64_CHARACTERS_MINIMUM_FOR_SECURITY", $SECRET_KEY)
$envContent = $envContent.Replace("CHANGE_ME_SPACES_ACCESS_KEY", $SPACES_ACCESS_KEY)
$envContent = $envContent.Replace("CHANGE_ME_SPACES_SECRET_KEY", $SPACES_SECRET_KEY)
$envContent = $envContent.Replace("CHANGE_ME_SMTP_PASSWORD", $SMTP_PASSWORD)
$envContent = $envContent.Replace("localhost:5432", "$DB_HOST`:$DB_PORT")
$envContent | Out-File .env.production -Encoding utf8

# Build and push Docker images
Write-Host "🐳 Building and pushing Docker images..." -ForegroundColor $YELLOW

# Login to DigitalOcean Container Registry
doctl registry login

# Build and push images
docker build -f infrastructure/docker/production/Dockerfile.api -t registry.digitalocean.com/a2aworldregistry/a2a-world/api:latest .
docker push registry.digitalocean.com/a2aworldregistry/a2a-world/api:latest

docker build -f infrastructure/docker/production/Dockerfile.frontend -t registry.digitalocean.com/a2aworldregistry/a2a-world/frontend:latest .
docker push registry.digitalocean.com/a2aworldregistry/a2a-world/frontend:latest

docker build -f infrastructure/docker/production/Dockerfile.agents -t registry.digitalocean.com/a2aworldregistry/a2a-world/agents:latest .
docker push registry.digitalocean.com/a2aworldregistry/a2a-world/agents:latest

Write-Host "✅ Docker images built and pushed" -ForegroundColor $GREEN

# Display deployment summary
Write-Host ""
Write-Host "🎉 A2A World Platform Production Deployment Complete!" -ForegroundColor $GREEN
Write-Host "======================================================="
Write-Host ""
Write-Host "🌐 Production URLs:"
Write-Host "   Main Site: https://a2aworld.ai"
Write-Host "   API: https://api.a2aworld.ai" 
Write-Host "   API Docs: https://api.a2aworld.ai/docs"
Write-Host ""
if ($MONITORING_IP -ne "Not deployed") {
    Write-Host "📊 Monitoring:"
    Write-Host "   Grafana: http://$MONITORING_IP`:3000 (admin/A2AWorld2024!)"
    Write-Host "   Prometheus: http://$MONITORING_IP`:9090"
    Write-Host ""
}
Write-Host "🔧 Infrastructure:"
Write-Host "   Load Balancer IP: $LB_IP"
Write-Host "   Application Servers: $($APP_IPS -join ', ')"
Write-Host "   Database: $DB_HOST`:$DB_PORT"
Write-Host ""
Write-Host "📋 Next Steps:"
Write-Host "   1. Update DNS records to point to: $LB_IP"
Write-Host "   2. Wait for SSL certificate provisioning (5-10 minutes)"
Write-Host "   3. SSH to servers and deploy application stack"
Write-Host "   4. Test all functionality end-to-end"
Write-Host "   5. Set up monitoring alerts"
Write-Host ""
Write-Host "🔐 Important:"
Write-Host "   - Secrets saved to: $SECRETS_FILE"
Write-Host "   - Keep this file secure and backed up"
Write-Host ""
Write-Host "✅ Infrastructure deployment successful!" -ForegroundColor $GREEN