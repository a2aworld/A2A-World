# A2A World Platform - Production Maintenance & Operations Guide

## 🛠️ Daily Operations

### Morning Health Checks (5 minutes)
```bash
# Check all services status
curl -f https://api.a2aworld.ai/health
curl -f https://a2aworld.ai

# Monitor resource usage
ssh deploy@<app-server-ip> "docker stats --no-stream"

# Check system load
ssh deploy@<app-server-ip> "uptime && free -h && df -h"
```

### Service Management
```bash
# View running services
docker-compose -f docker-compose.production.yml ps

# Check service logs
docker-compose -f docker-compose.production.yml logs -f api
docker-compose -f docker-compose.production.yml logs -f frontend
docker-compose -f docker-compose.production.yml logs -f agent-pattern-discovery

# Restart a service
docker-compose -f docker-compose.production.yml restart api

# Scale services
docker-compose -f docker-compose.production.yml up -d --scale api=4
```

## 📊 Monitoring & Alerting

### Grafana Dashboards
Access: `http://monitoring-server-ip:3000`
- **System Overview**: CPU, memory, disk usage across all servers
- **Application Metrics**: API response times, request rates, error rates
- **Database Performance**: Connection counts, query performance, disk usage
- **Agent System**: Task processing rates, pattern discovery metrics

### Key Metrics to Monitor
```bash
# API Response Time (should be < 500ms)
curl -w "@curl-format.txt" -o /dev/null -s https://api.a2aworld.ai/health

# Database Connections (should be < 80% of max)
docker-compose exec postgres psql -U a2a_user -d a2a_world -c "SELECT count(*) FROM pg_stat_activity;"

# Memory Usage (should be < 80%)
free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'

# Disk Usage (should be < 85%)
df -h | awk '$NF=="/"{printf "Disk Usage: %d/%dGB (%s)\n", $3,$2,$5}'
```

### Alert Thresholds
- **CPU Usage**: > 80% for 5 minutes
- **Memory Usage**: > 85% for 3 minutes  
- **Disk Usage**: > 90%
- **API Response Time**: > 2 seconds
- **Database Connections**: > 100 concurrent
- **SSL Certificate**: < 30 days to expiry

## 🔄 Deployment Updates

### Rolling Updates
```bash
# Update API service
docker build -f infrastructure/docker/production/Dockerfile.api -t registry.digitalocean.com/a2aworldregistry/a2a-world/api:v1.1.0 .
docker push registry.digitalocean.com/a2aworldregistry/a2a-world/api:v1.1.0

# Update environment to use new version
sed -i 's/VERSION=latest/VERSION=v1.1.0/' .env.production

# Rolling update with zero downtime
docker-compose -f docker-compose.production.yml up -d api
```

### Database Migrations
```bash
# Backup database before migration
docker-compose exec postgres pg_dump -U a2a_user a2a_world > backup_$(date +%Y%m%d_%H%M%S).sql

# Run migration
docker-compose exec api python database/scripts/migration_manager.py --apply

# Verify migration
docker-compose exec api python -c "
from database.connection import get_db_connection
conn = get_db_connection()
cur = conn.cursor()
cur.execute('SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;')
print(f'Current schema version: {cur.fetchone()[0]}')
"
```

## 🔒 Security Operations

### SSL Certificate Management
```bash
# Check certificate expiry
echo | openssl s_client -connect a2aworld.ai:443 2>/dev/null | openssl x509 -noout -dates

# Force certificate renewal (if needed)
ssh deploy@<app-server> "sudo certbot renew --force-renewal"

# Test certificate renewal
ssh deploy@<app-server> "sudo certbot renew --dry-run"
```

### Security Updates
```bash
# Update system packages (monthly)
ssh deploy@<app-server> "sudo apt update && sudo apt upgrade -y"

# Update Docker images (weekly)
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d

# Security audit
ssh deploy@<app-server> "sudo lynis audit system"
```

## 💾 Backup Operations

### Database Backups
```bash
# Create manual backup
ssh deploy@<app-server> "cd /opt/a2a-world && docker-compose exec -T postgres pg_dump -U a2a_user a2a_world | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz"

# Restore from backup
gunzip backup_20241029_120000.sql.gz
ssh deploy@<app-server> "cd /opt/a2a-world && docker-compose exec -T postgres psql -U a2a_user -d a2a_world < backup_20241029_120000.sql"

# List available backups
doctl databases backup list <database-id>
```

### Application Data Backup
```bash
# Backup application volumes
ssh deploy@<app-server> "sudo tar czf app_data_$(date +%Y%m%d).tar.gz /opt/a2a-world/data"

# Backup configuration
tar czf config_backup_$(date +%Y%m%d).tar.gz infrastructure/ .env.production
```

## 📈 Performance Optimization

### Database Optimization
```bash
# Analyze query performance
docker-compose exec postgres psql -U a2a_user -d a2a_world -c "
SELECT query, calls, total_time, rows, mean_time
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"

# Optimize database
docker-compose exec postgres psql -U a2a_user -d a2a_world -c "VACUUM ANALYZE;"

# Check index usage
docker-compose exec postgres psql -U a2a_user -d a2a_world -c "
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
"
```

### Application Performance
```bash
# Monitor API performance
curl -w "@curl-format.txt" -o /dev/null -s https://api.a2aworld.ai/api/v1/patterns

# Check application memory usage
docker-compose exec api ps aux --sort=-%mem | head

# Profile Python application
docker-compose exec api python -m cProfile -o profile.stats your_script.py
```

## 🚨 Incident Response

### Service Outage Response
1. **Immediate Assessment**:
   ```bash
   # Check service status
   curl -I https://a2aworld.ai
   curl -I https://api.a2aworld.ai/health
   
   # Check load balancer health
   doctl compute load-balancer get <lb-id>
   ```

2. **Log Analysis**:
   ```bash
   # Application logs
   docker-compose logs --tail=100 api
   docker-compose logs --tail=100 frontend
   
   # System logs
   ssh deploy@<app-server> "sudo journalctl -u docker -f"
   ```

3. **Recovery Actions**:
   ```bash
   # Restart services
   docker-compose -f docker-compose.production.yml restart
   
   # Scale up if needed
   docker-compose -f docker-compose.production.yml up -d --scale api=4
   ```

### Database Issues
```bash
# Check database status
docker-compose exec postgres pg_isready -U a2a_user

# Check connections
docker-compose exec postgres psql -U a2a_user -d a2a_world -c "SELECT count(*) FROM pg_stat_activity;"

# Kill long-running queries
docker-compose exec postgres psql -U a2a_user -d a2a_world -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'active' AND query_start < now() - interval '5 minutes';
"
```

## 💰 Cost Monitoring

### Monthly Cost Review
```bash
# Get DigitalOcean usage
doctl account get
doctl billing balance get
doctl billing history list

# Analyze resource usage
doctl compute droplet list --format "Name,Memory,VCPUs,Disk,Region,Status"
doctl databases list --format "Name,Engine,Size,Region,Status"
```

### Cost Optimization Actions
1. **Right-size Resources**: Monitor CPU/memory usage and downsize if consistently low
2. **Clean Up Unused Resources**: Remove old snapshots, unused images
3. **Optimize Storage**: Clean up logs, compress backups, optimize database
4. **Review Backup Retention**: Adjust retention policies based on requirements

## 🔧 Troubleshooting Guide

### Common Issues

**Issue**: API returns 502 Bad Gateway
```bash
# Check API service status
docker-compose ps api
docker-compose logs api

# Check load balancer health checks
curl -I http://app-server-ip:8000/health
```

**Issue**: Frontend not loading
```bash
# Check frontend service
docker-compose ps frontend
docker-compose logs frontend

# Check nginx configuration
docker-compose exec frontend nginx -t
```

**Issue**: Pattern discovery not working
```bash
# Check agent status
docker-compose ps agent-pattern-discovery
docker-compose logs agent-pattern-discovery

# Check NATS connectivity
docker-compose exec agent-pattern-discovery python -c "import nats; print('NATS OK')"
```

**Issue**: Database connection errors
```bash
# Check database service
docker-compose ps postgres
docker-compose logs postgres

# Test connection
docker-compose exec api python -c "
from database.connection import get_db_connection
try:
    conn = get_db_connection()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

## 📋 Weekly Maintenance Checklist

### Every Monday
- [ ] Review monitoring alerts from previous week
- [ ] Check SSL certificate expiry dates
- [ ] Review backup success/failure status
- [ ] Update Docker images if security updates available
- [ ] Check disk space usage on all servers

### Every Wednesday  
- [ ] Review application performance metrics
- [ ] Check database query performance
- [ ] Review error logs for patterns
- [ ] Test backup restoration process
- [ ] Update documentation if needed

### Every Friday
- [ ] Review weekly cost report
- [ ] Plan maintenance windows for following week
- [ ] Update team on any issues or improvements
- [ ] Check for infrastructure updates
- [ ] Review security scan results

## 📞 Emergency Contacts

**Infrastructure Issues**: DevOps Team  
**Application Issues**: Development Team  
**Database Issues**: DBA Team  
**Security Issues**: Security Team  

**Escalation Path**: 
1. Team Lead
2. Engineering Manager  
3. VP Engineering

---

*This maintenance guide should be reviewed and updated monthly to reflect operational experience and changing requirements.*