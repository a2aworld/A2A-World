# A2A World Platform - Maintenance Guide

## Overview

This guide covers ongoing maintenance procedures for the A2A World Platform to ensure optimal performance, security, and reliability.

## Daily Tasks (Automated)

These tasks are handled automatically by the deployed system:

### Automated Monitoring
- **Health checks**: Every 30 seconds for all services
- **Metrics collection**: Every 15 seconds via Prometheus
- **Log aggregation**: Real-time via Loki/Promtail
- **Alert evaluation**: Every 15 seconds
- **SSL certificate monitoring**: Daily checks

### Automated Backups
- **Database backup**: Daily at 2:00 AM
- **Incremental backups**: Every 6 hours
- **Log rotation**: Daily
- **Backup cleanup**: Removes backups older than 30 days

## Weekly Tasks

### 1. System Health Review
```bash
# Run the weekly health check script
sudo /opt/a2a-world/infrastructure/scripts/weekly-health-check.sh
```

**Review checklist**:
- [ ] All services running normally
- [ ] No critical alerts in past 7 days
- [ ] SSL certificates valid (>30 days remaining)
- [ ] Backup completion status
- [ ] Disk space usage (<80%)
- [ ] Memory usage patterns
- [ ] Database performance metrics

### 2. Security Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
docker service update --image registry.digitalocean.com/a2aregistry/a2a-world/api:latest a2a-world_api
docker service update --image registry.digitalocean.com/a2aregistry/a2a-world/frontend:latest a2a-world_frontend
docker service update --image registry.digitalocean.com/a2aregistry/a2a-world/agents:latest a2a-world_agent-kml-parser

# Restart services if needed
docker service update --force a2a-world_api
```

### 3. Log Analysis
```bash
# Check for security incidents
sudo grep -i "attack\|intrusion\|failed login" /var/log/auth.log | tail -20

# Review application errors
docker service logs a2a-world_api 2>&1 | grep -i error | tail -20

# Check fail2ban status
sudo fail2ban-client status
sudo fail2ban-client status a2a-api-auth
```

### 4. Database Maintenance
```bash
# Connect to database
docker exec -it $(docker ps -q --filter name=postgres) psql -U a2a_user -d a2a_world

-- Analyze database performance
ANALYZE;

-- Check table sizes
SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
LIMIT 10;

-- Check slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

-- Vacuum if needed
VACUUM ANALYZE;
```

## Monthly Tasks

### 1. Complete System Backup
```bash
# Run full system backup
sudo /opt/a2a-world/infrastructure/backup/system-backup.sh all

# Verify backup integrity
sudo /opt/a2a-world/infrastructure/backup/disaster-recovery.sh list
```

### 2. Security Audit
```bash
# Run security scan
sudo /opt/a2a-world/infrastructure/security/security-audit.sh

# Update firewall rules if needed
sudo /opt/a2a-world/infrastructure/security/firewall/ufw-rules.sh

# Rotate secrets (if needed)
sudo /opt/a2a-world/infrastructure/security/secrets/secrets-manager.sh rotate
```

### 3. Performance Review
- Review Grafana dashboards for trends
- Analyze resource usage patterns
- Check for performance degradation
- Plan capacity upgrades if needed

### 4. SSL Certificate Management
```bash
# Check certificate expiration
sudo /opt/a2a-world/infrastructure/security/ssl/certbot-renewal.sh

# Test certificate renewal
sudo certbot renew --dry-run
```

## Quarterly Tasks

### 1. Disaster Recovery Testing
```bash
# Test backup restoration (on staging environment)
sudo /opt/a2a-world/infrastructure/backup/disaster-recovery.sh restore-db /backup/latest-backup.sql.gz staging_db

# Verify application functionality after restore
curl https://staging.a2aworld.ai/health
```

### 2. Security Updates and Hardening
- Review and update security policies
- Audit user access and permissions
- Update security patches
- Review firewall rules and access logs

### 3. Performance Optimization
- Database query optimization
- Resource allocation review
- CDN and caching optimization
- Infrastructure scaling assessment

### 4. Cost Optimization Review
```bash
# Run cost analysis
sudo /opt/a2a-world/infrastructure/scripts/cost-analysis.sh

# Review resource utilization
docker stats --no-stream
htop
```

## Monitoring and Alerting

### Key Metrics to Monitor

#### System Metrics
- **CPU Usage**: Target <70% average
- **Memory Usage**: Target <85%
- **Disk Usage**: Target <80%
- **Network I/O**: Monitor for unusual patterns
- **Load Average**: Target <2.0 on 4-core systems

#### Application Metrics
- **Response Time**: Target <500ms for API endpoints
- **Error Rate**: Target <1%
- **Throughput**: Monitor requests per second
- **Active Users**: Track user engagement
- **Agent Processing Time**: Monitor task completion rates

#### Database Metrics
- **Connection Count**: Monitor against max connections
- **Query Performance**: Watch for slow queries >5 seconds
- **Database Size**: Monitor growth rate
- **Backup Success**: Ensure daily backups complete

### Alert Thresholds

#### Critical Alerts (Immediate Response)
- Service down (>1 minute)
- Error rate >5%
- Database connection failures
- SSL certificate expires <7 days
- Disk space >95%

#### Warning Alerts (Response within 4 hours)
- CPU usage >80% for >10 minutes
- Memory usage >90% for >5 minutes
- Response time >2 seconds
- Failed login attempts >100 per hour

#### Info Alerts (Response within 24 hours)
- Backup completion status
- SSL certificate expires <30 days
- Unusual traffic patterns

## Troubleshooting Common Issues

### High CPU Usage
1. Identify process: `htop`, `docker stats`
2. Check for runaway processes
3. Scale services if needed: `docker service scale a2a-world_api=3`
4. Review application code for optimization opportunities

### Memory Leaks
1. Monitor memory usage over time
2. Restart affected services: `docker service update --force service_name`
3. Check application logs for memory-related errors
4. Consider upgrading instance size if persistent

### Database Performance
1. Identify slow queries: Check `pg_stat_statements`
2. Add database indexes as needed
3. Optimize query performance
4. Consider read replicas for read-heavy workloads

### Network Issues
1. Check firewall rules: `sudo ufw status`
2. Verify DNS resolution: `dig a2aworld.ai`
3. Test connectivity: `curl -v https://api.a2aworld.ai/health`
4. Review load balancer configuration

## Capacity Planning

### Scaling Triggers
**Scale UP when**:
- CPU usage >70% for >30 minutes
- Memory usage >85% consistently
- Response times >1 second consistently
- Error rates increase due to resource constraints

**Scale DOWN when**:
- CPU usage <30% for >2 hours during peak times
- Memory usage <50% consistently
- Over-provisioned resources identified

### Scaling Procedures
```bash
# Scale application services
docker service scale a2a-world_api=3
docker service scale a2a-world_frontend=2

# Scale agent services
docker service scale a2a-world_agent-kml-parser=2
docker service scale a2a-world_agent-pattern-discovery=2

# Monitor scaling results
docker service ls
docker stats --no-stream
```

## Maintenance Windows

### Planned Maintenance (Monthly)
- **Duration**: 2-4 hours
- **Time**: Sunday 2:00 AM - 6:00 AM UTC
- **Notification**: 48 hours advance notice
- **Activities**: 
  - System updates
  - Database maintenance
  - Security patches
  - Performance optimizations

### Emergency Maintenance
- **Response Time**: <1 hour for critical issues
- **Communication**: Real-time updates via Slack/Email
- **Rollback Plan**: Always available
- **Post-Incident Review**: Within 24 hours

## Documentation Updates

### Keep Updated
- API documentation
- Configuration changes
- New monitoring alerts
- Troubleshooting procedures
- Performance benchmarks

### Version Control
- All configuration changes in Git
- Infrastructure as Code updates
- Deployment procedure updates
- Runbook versioning

## Contact Information

### On-Call Procedures
1. **Primary**: admin@a2aworld.ai
2. **Escalation**: Check #ops-alerts Slack channel
3. **Emergency**: Use incident management system

### External Vendors
- **DigitalOcean Support**: For infrastructure issues
- **Cloudflare Support**: For DNS/CDN issues  
- **GitHub Support**: For CI/CD pipeline issues

## Maintenance Scripts

All maintenance scripts are located in `/opt/a2a-world/infrastructure/scripts/`:

### Daily Scripts
- `daily-health-check.sh` - Automated health monitoring
- `backup-verification.sh` - Verify backup integrity
- `log-cleanup.sh` - Clean up old log files

### Weekly Scripts  
- `weekly-health-check.sh` - Comprehensive system review
- `security-update.sh` - Apply security updates
- `performance-report.sh` - Generate performance report

### Monthly Scripts
- `monthly-maintenance.sh` - Full maintenance routine
- `security-audit.sh` - Security assessment
- `cost-analysis.sh` - Resource utilization review

### On-Demand Scripts
- `emergency-restart.sh` - Emergency service restart
- `scale-services.sh` - Quick service scaling
- `diagnostic-report.sh` - System diagnostic collection

## Maintenance Calendar

### Weekly Schedule
- **Monday**: Review weekend alerts and logs
- **Tuesday**: Security update check
- **Wednesday**: Performance review  
- **Thursday**: Backup verification
- **Friday**: Weekly health check
- **Saturday**: Optional maintenance window
- **Sunday**: Planned maintenance window (monthly)

### Monthly Schedule
- **Week 1**: Security audit and updates
- **Week 2**: Performance optimization
- **Week 3**: Backup and disaster recovery testing
- **Week 4**: Capacity planning and cost optimization

Remember: Always test changes in staging environment before applying to production!