# A2A World Platform - Troubleshooting Guide

## Common Issues and Solutions

### 1. Deployment Issues

#### Terraform Deployment Fails
**Error**: `Error creating droplet: POST https://api.digitalocean.com/v2/droplets: 402 Payment Required`

**Solution**:
```bash
# Check your DigitalOcean account billing
# Ensure you have sufficient credits or payment method
# Verify API token has correct permissions

# Check token permissions
curl -X GET -H "Authorization: Bearer $DO_TOKEN" "https://api.digitalocean.com/v2/account"
```

#### Docker Services Won't Start
**Error**: `service a2a-world_api has no replicas running`

**Solution**:
```bash
# Check service logs
docker service logs a2a-world_api

# Check secrets and environment variables
docker secret ls
docker config ls

# Verify database connectivity
docker exec -it $(docker ps -q --filter name=postgres) psql -U a2a_user -d a2a_world -c "SELECT 1;"

# Restart service
docker service update --force a2a-world_api
```

#### SSL Certificate Issues
**Error**: `Failed to obtain SSL certificate from Let's Encrypt`

**Solution**:
```bash
# Check DNS propagation
dig a2aworld.ai
dig www.a2aworld.ai
dig api.a2aworld.ai

# Verify domain ownership
curl -I http://a2aworld.ai/.well-known/acme-challenge/test

# Manual certificate request
sudo certbot certonly --manual -d a2aworld.ai -d www.a2aworld.ai -d api.a2aworld.ai

# Check certificate renewal
sudo /opt/a2a-world/infrastructure/security/ssl/certbot-renewal.sh
```

### 2. Database Issues

#### Database Connection Errors
**Error**: `could not connect to server: Connection refused`

**Solution**:
```bash
# Check PostgreSQL service status
docker service ls | grep postgres
docker service logs a2a-world_postgres

# Verify database credentials
docker secret inspect db_password

# Check database connectivity from app container
docker exec -it $(docker ps -q --filter name=api) \
  psql postgresql://a2a_user:password@postgres:5432/a2a_world -c "SELECT 1;"

# Restart database service
docker service update --force a2a-world_postgres
```

#### Database Performance Issues
**Symptoms**: Slow queries, high CPU usage

**Solution**:
```bash
# Connect to database and check performance
docker exec -it $(docker ps -q --filter name=postgres) psql -U a2a_user -d a2a_world

-- Check slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

-- Check database size
SELECT pg_size_pretty(pg_database_size('a2a_world'));

-- Check table sizes
SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size 
FROM pg_tables 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
LIMIT 10;

# Optimize database
VACUUM ANALYZE;
REINDEX DATABASE a2a_world;
```

### 3. Application Issues

#### API Returning 500 Errors
**Error**: Internal server errors in API responses

**Solution**:
```bash
# Check API logs
docker service logs -f a2a-world_api

# Check application health endpoint
curl -v https://api.a2aworld.ai/health

# Verify environment variables
docker exec -it $(docker ps -q --filter name=api) env | grep -E "(DATABASE|REDIS|NATS)"

# Check database connectivity from API
curl https://api.a2aworld.ai/api/v1/health/db

# Restart API service
docker service update --force a2a-world_api
```

#### Frontend Not Loading
**Symptoms**: White screen, JavaScript errors

**Solution**:
```bash
# Check frontend service
docker service logs a2a-world_frontend

# Verify API connectivity
curl https://api.a2aworld.ai/health

# Check browser console for errors
# Verify CORS settings
curl -H "Origin: https://a2aworld.ai" -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: X-Requested-With" \
  -X OPTIONS https://api.a2aworld.ai/api/v1/health

# Restart frontend service
docker service update --force a2a-world_frontend
```

### 4. Agent System Issues

#### Agents Not Processing Tasks
**Symptoms**: Tasks stuck in queue, no agent activity

**Solution**:
```bash
# Check agent services
docker service ls | grep agent
docker service logs a2a-world_agent-kml-parser
docker service logs a2a-world_agent-pattern-discovery

# Check NATS connectivity
docker service logs a2a-world_nats

# Verify agent registration
curl https://api.a2aworld.ai/api/v1/agents

# Check agent health endpoints
docker exec -it $(docker ps -q --filter name=agent-kml-parser) \
  curl http://localhost:9200/health

# Restart agent services
docker service update --force a2a-world_agent-kml-parser
docker service update --force a2a-world_agent-pattern-discovery
```

### 5. Monitoring Issues

#### Prometheus Not Scraping Metrics
**Symptoms**: No data in Grafana dashboards

**Solution**:
```bash
# Check Prometheus configuration
curl http://monitoring.a2aworld.ai:9090/config

# Verify targets
curl http://monitoring.a2aworld.ai:9090/targets

# Check service discovery
docker exec -it $(docker ps -q --filter name=prometheus) \
  wget -qO- http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# Restart Prometheus
docker service update --force monitoring_prometheus
```

#### Grafana Dashboards Not Working
**Symptoms**: No data displayed, connection errors

**Solution**:
```bash
# Check Grafana logs
docker service logs monitoring_grafana

# Verify datasource connectivity
curl -u admin:A2AWorld2024! http://monitoring.a2aworld.ai:3000/api/datasources

# Test Prometheus connectivity from Grafana container
docker exec -it $(docker ps -q --filter name=grafana) \
  wget -qO- http://prometheus:9090/api/v1/query?query=up

# Restart Grafana
docker service update --force monitoring_grafana
```

### 6. SSL/Security Issues

#### SSL Certificate Not Renewing
**Error**: Certificate expiration warnings

**Solution**:
```bash
# Check certificate status
echo | openssl s_client -connect a2aworld.ai:443 -servername a2aworld.ai 2>/dev/null | \
  openssl x509 -noout -dates

# Check certbot logs
sudo tail -f /var/log/letsencrypt/letsencrypt.log

# Run manual renewal
sudo /opt/a2a-world/infrastructure/security/ssl/certbot-renewal.sh

# Verify auto-renewal cron job
sudo crontab -l | grep certbot
```

#### Firewall Blocking Connections
**Symptoms**: Connection timeouts, services unreachable

**Solution**:
```bash
# Check UFW status
sudo ufw status verbose

# Check iptables rules
sudo iptables -L -n

# Check service ports
sudo netstat -tuln | grep :8000
sudo ss -tuln | grep :8000

# Temporarily disable firewall for testing
sudo ufw disable
# Test connectivity
curl http://server-ip:8000/health
# Re-enable firewall
sudo ufw enable

# Add specific rules if needed
sudo ufw allow from 10.0.0.0/16 to any port 8000
```

### 7. Performance Issues

#### High Server Load
**Symptoms**: Slow response times, high CPU usage

**Solution**:
```bash
# Check system resources
htop
iostat -x 1 5
free -h
df -h

# Check Docker resource usage
docker stats

# Check service resource limits
docker service inspect a2a-world_api | jq '.[0].Spec.TaskTemplate.Resources'

# Scale services if needed
docker service scale a2a-world_api=3

# Check for resource leaks
docker exec -it $(docker ps -q --filter name=api) \
  ps aux | head -20
```

#### Database Performance Issues
**Symptoms**: Slow queries, connection pool exhaustion

**Solution**:
```bash
# Check active connections
docker exec -it $(docker ps -q --filter name=postgres) \
  psql -U a2a_user -d a2a_world -c "SELECT count(*) FROM pg_stat_activity;"

# Check long-running queries
docker exec -it $(docker ps -q --filter name=postgres) \
  psql -U a2a_user -d a2a_world -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"

# Optimize queries and add indexes as needed
# Consider upgrading database instance size
```

### 8. Backup and Recovery Issues

#### Backup Script Failures
**Error**: Backup job not completing successfully

**Solution**:
```bash
# Check backup logs
sudo tail -f /var/log/a2a-world/database-backup.log
sudo tail -f /var/log/a2a-world/system-backup.log

# Run backup manually to debug
sudo /opt/a2a-world/infrastructure/backup/database-backup.sh full

# Check disk space
df -h /backup

# Verify credentials and permissions
sudo ls -la /backup/
sudo ls -la /opt/a2a-world/secrets/
```

#### Recovery Process Failures
**Error**: Cannot restore from backup

**Solution**:
```bash
# Verify backup integrity
sudo /opt/a2a-world/infrastructure/backup/disaster-recovery.sh list

# Test backup file
gunzip -t /backup/database/full_backup_*.sql.gz

# Check database connectivity for restore
PGPASSWORD=password psql -h localhost -U a2a_user -d postgres -c "SELECT 1;"

# Run recovery interactively for debugging
sudo /opt/a2a-world/infrastructure/backup/disaster-recovery.sh interactive
```

## Log Locations

### Application Logs
- **API**: `docker service logs a2a-world_api`
- **Frontend**: `docker service logs a2a-world_frontend`
- **Agents**: `docker service logs a2a-world_agent-*`

### System Logs
- **Deployment**: `/var/log/a2a-world/`
- **Nginx**: `/var/log/nginx/`
- **SSL**: `/var/log/letsencrypt/`
- **UFW**: `/var/log/ufw.log`

### Database Logs
- **PostgreSQL**: `docker service logs a2a-world_postgres`

### Monitoring Logs
- **Prometheus**: `docker service logs monitoring_prometheus`
- **Grafana**: `docker service logs monitoring_grafana`
- **Alertmanager**: `docker service logs monitoring_alertmanager`

## Debug Commands

### Quick Health Check
```bash
#!/bin/bash
# Quick system health check

echo "=== Docker Swarm Status ==="
docker node ls

echo "=== Service Status ==="
docker service ls

echo "=== System Resources ==="
free -h
df -h

echo "=== Network Connectivity ==="
curl -I https://a2aworld.ai
curl -I https://api.a2aworld.ai/health

echo "=== Recent Logs ==="
docker service logs --tail=10 a2a-world_api
```

### Performance Check
```bash
#!/bin/bash
# Performance monitoring

echo "=== Load Average ==="
uptime

echo "=== Docker Stats ==="
docker stats --no-stream

echo "=== Database Performance ==="
docker exec -it $(docker ps -q --filter name=postgres) \
  psql -U a2a_user -d a2a_world -c "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"

echo "=== Response Times ==="
curl -w "Total: %{time_total}s\n" -o /dev/null -s https://api.a2aworld.ai/health
```

## Emergency Procedures

### Service Restart Sequence
```bash
# 1. Stop all services
docker stack rm a2a-world

# 2. Wait for cleanup
sleep 30

# 3. Verify cleanup
docker service ls
docker network ls | grep a2a
docker volume ls | grep a2a

# 4. Redeploy
docker stack deploy --compose-file docker-compose.production.yml a2a-world
```

### Emergency Database Recovery
```bash
# 1. Stop application services
docker service scale a2a-world_api=0
docker service scale a2a-world_frontend=0

# 2. Create database backup
sudo /opt/a2a-world/infrastructure/backup/database-backup.sh full

# 3. Perform recovery
sudo /opt/a2a-world/infrastructure/backup/disaster-recovery.sh restore-db /path/to/backup

# 4. Restart services
docker service scale a2a-world_api=2
docker service scale a2a-world_frontend=1
```

## Getting Help

1. **Check Documentation**: Review deployment and maintenance guides
2. **Search Logs**: Look for specific error messages in relevant logs
3. **Community Support**: GitHub Issues and Discussions
4. **Professional Support**: Contact admin@a2aworld.ai

For urgent production issues, include:
- Current system status (`docker service ls`)
- Recent error logs
- Steps to reproduce the issue
- Timeline of when the issue started