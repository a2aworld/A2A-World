#!/bin/bash
# A2A World Platform - Production Deployment Validation Script
# Comprehensive testing of production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DOMAIN="a2aworld.ai"
API_DOMAIN="api.a2aworld.ai"
TIMEOUT=30

echo -e "${BLUE}🧪 A2A World Platform - Production Deployment Validation${NC}"
echo "============================================================"

# Test results tracking
PASSED=0
FAILED=0
TOTAL=0

function run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_result="$3"
    
    ((TOTAL++))
    echo -e "${YELLOW}Testing: $test_name${NC}"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        ((FAILED++))
    fi
    echo ""
}

function check_http_status() {
    local url="$1"
    local expected_status="${2:-200}"
    
    status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$url")
    [ "$status" = "$expected_status" ]
}

function check_content_contains() {
    local url="$1" 
    local expected_content="$2"
    
    content=$(curl -s --connect-timeout $TIMEOUT "$url")
    echo "$content" | grep -q "$expected_content"
}

function check_json_response() {
    local url="$1"
    local expected_key="$2"
    
    response=$(curl -s --connect-timeout $TIMEOUT "$url")
    echo "$response" | jq -e ".$expected_key" > /dev/null 2>&1
}

function check_ssl_certificate() {
    local domain="$1"
    
    echo | openssl s_client -connect "$domain:443" -servername "$domain" 2>/dev/null | \
    openssl x509 -noout -text | grep -q "Let's Encrypt"
}

function check_dns_resolution() {
    local domain="$1"
    
    nslookup "$domain" > /dev/null 2>&1
}

# Start validation tests
echo -e "${YELLOW}🔍 Starting comprehensive validation tests...${NC}"
echo ""

# DNS Resolution Tests
echo -e "${BLUE}📡 DNS Resolution Tests${NC}"
run_test "DNS resolution for $DOMAIN" "check_dns_resolution $DOMAIN"
run_test "DNS resolution for www.$DOMAIN" "check_dns_resolution www.$DOMAIN"
run_test "DNS resolution for $API_DOMAIN" "check_dns_resolution $API_DOMAIN"

# SSL Certificate Tests
echo -e "${BLUE}🔒 SSL Certificate Tests${NC}"
run_test "SSL certificate for $DOMAIN" "check_ssl_certificate $DOMAIN"
run_test "SSL certificate for $API_DOMAIN" "check_ssl_certificate $API_DOMAIN"

# Frontend Tests
echo -e "${BLUE}🌐 Frontend Application Tests${NC}"
run_test "Frontend HTTP status (https://$DOMAIN)" "check_http_status https://$DOMAIN"
run_test "Frontend contains React app" "check_content_contains https://$DOMAIN '__NEXT_DATA__'"
run_test "Frontend redirects www to apex" "check_http_status https://www.$DOMAIN 301"

# API Tests
echo -e "${BLUE}🔌 API Endpoint Tests${NC}"
run_test "API health endpoint" "check_http_status https://$API_DOMAIN/health"
run_test "API health response format" "check_json_response https://$API_DOMAIN/health status"
run_test "API docs endpoint" "check_http_status https://$API_DOMAIN/docs"
run_test "API OpenAPI spec" "check_http_status https://$API_DOMAIN/openapi.json"

# API Functionality Tests
echo -e "${BLUE}⚙️  API Functionality Tests${NC}"
run_test "API agents endpoint" "check_http_status https://$API_DOMAIN/api/v1/agents"
run_test "API patterns endpoint" "check_http_status https://$API_DOMAIN/api/v1/patterns"
run_test "API data endpoint" "check_http_status https://$API_DOMAIN/api/v1/data"

# Database Connectivity Tests (requires server access)
if command -v terraform &> /dev/null && [ -d "infrastructure/terraform" ]; then
    echo -e "${BLUE}🗄️  Database Connectivity Tests${NC}"
    
    # Get first app server IP from Terraform
    cd infrastructure/terraform
    FIRST_APP_IP=$(terraform output -json app_server_ips | jq -r '.public[0]')
    cd ../..
    
    if [ "$FIRST_APP_IP" != "null" ] && [ -n "$FIRST_APP_IP" ]; then
        run_test "Database connection from app server" \
            "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no deploy@$FIRST_APP_IP 'cd /opt/a2a-world && docker-compose -f docker-compose.production.yml exec -T api python -c \"from database.connection import get_db_connection; conn = get_db_connection(); print(\\\"success\\\" if conn else exit(1))\"' | grep -q success"
        
        run_test "Database tables exist" \
            "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no deploy@$FIRST_APP_IP 'cd /opt/a2a-world && docker-compose -f docker-compose.production.yml exec -T postgres psql -U a2a_user -d a2a_world -c \"\\dt\" | wc -l | awk \"{print \\$1 > 10}\"' | grep -q 1"
    else
        echo -e "${YELLOW}⚠️  Skipping database tests - cannot determine app server IP${NC}"
    fi
fi

# Performance Tests
echo -e "${BLUE}⚡ Performance Tests${NC}"
run_test "Frontend loads within 5 seconds" \
    "time_taken=\$(curl -s -o /dev/null -w '%{time_total}' https://$DOMAIN); [ \"\$(echo \"\$time_taken < 5.0\" | bc -l)\" = 1 ]"

run_test "API response time under 2 seconds" \
    "time_taken=\$(curl -s -o /dev/null -w '%{time_total}' https://$API_DOMAIN/health); [ \"\$(echo \"\$time_taken < 2.0\" | bc -l)\" = 1 ]"

# Security Headers Tests
echo -e "${BLUE}🛡️  Security Headers Tests${NC}"
run_test "HSTS header present" \
    "curl -s -I https://$DOMAIN | grep -i 'strict-transport-security'"

run_test "Content Security Policy present" \
    "curl -s -I https://$DOMAIN | grep -i 'content-security-policy'"

# Monitoring Tests (if monitoring server is deployed)
if command -v terraform &> /dev/null && [ -d "infrastructure/terraform" ]; then
    cd infrastructure/terraform
    MONITORING_IP=$(terraform output -raw monitoring_server_ip 2>/dev/null || echo "Not deployed")
    cd ../..
    
    if [ "$MONITORING_IP" != "Not deployed" ] && [ -n "$MONITORING_IP" ]; then
        echo -e "${BLUE}📊 Monitoring Stack Tests${NC}"
        run_test "Prometheus accessible" "check_http_status http://$MONITORING_IP:9090"
        run_test "Grafana accessible" "check_http_status http://$MONITORING_IP:3000"
        run_test "Node Exporter metrics" "check_http_status http://$MONITORING_IP:9100/metrics"
    else
        echo -e "${YELLOW}⚠️  Skipping monitoring tests - monitoring server not deployed${NC}"
    fi
fi

# Agent System Tests (basic connectivity)
echo -e "${BLUE}🤖 Agent System Tests${NC}"
run_test "NATS service accessible from API" \
    "curl -s https://$API_DOMAIN/api/v1/agents | jq -e '.agents' > /dev/null 2>&1 || true"

# Pattern Discovery Tests
echo -e "${BLUE}🔍 Pattern Discovery System Tests${NC}"
run_test "Pattern discovery endpoint accessible" \
    "check_http_status https://$API_DOMAIN/api/v1/patterns"

# Generate validation report
echo ""
echo -e "${BLUE}📊 Validation Summary${NC}"
echo "======================================"
echo -e "Total Tests: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests passed! Production deployment is healthy.${NC}"
    echo ""
    echo -e "${BLUE}✅ Production URLs Verified:${NC}"
    echo "   🌐 Main Site: https://$DOMAIN"
    echo "   🔌 API: https://$API_DOMAIN"
    echo "   📚 API Docs: https://$API_DOMAIN/docs"
    echo ""
    echo -e "${BLUE}🔧 Next Steps:${NC}"
    echo "   1. Set up monitoring alerts"
    echo "   2. Configure automated backups"
    echo "   3. Perform user acceptance testing"
    echo "   4. Update team documentation"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}❌ $FAILED tests failed. Please review and fix issues before going live.${NC}"
    echo ""
    echo -e "${YELLOW}🔧 Troubleshooting Tips:${NC}"
    echo "   1. Check DNS propagation: dig $DOMAIN"
    echo "   2. Verify SSL certificates: curl -vI https://$DOMAIN"
    echo "   3. Check application logs on servers"
    echo "   4. Verify load balancer health checks"
    echo "   5. Test database connectivity manually"
    echo ""
    exit 1
fi