#!/bin/bash
# A2A World Platform - Cost Optimization Script
# Automated resource optimization and cost monitoring

set -euo pipefail

# Configuration
LOG_FILE="/var/log/a2a-world/cost-optimization.log"
REPORT_DIR="/opt/a2a-world/reports"
THRESHOLD_CPU_LOW=20  # Scale down if CPU < 20% for extended period
THRESHOLD_CPU_HIGH=80 # Scale up if CPU > 80%
THRESHOLD_MEMORY_HIGH=85 # Alert if memory > 85%
COST_BUDGET_MONTHLY=200 # Monthly budget in USD

# DigitalOcean API configuration
DO_TOKEN="${DO_TOKEN:-}"
SPACES_BUCKET="${SPACES_BUCKET:-a2a-world-backups}"

# Notification configuration
SLACK_WEBHOOK="${SLACK_WEBHOOK_COST:-}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    notify "Cost optimization error: $1" "error"
    exit 1
}

# Notification function
notify() {
    local message="$1"
    local level="${2:-info}"
    
    log "$message"
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local color="good"
        [[ "$level" == "error" ]] && color="danger"
        [[ "$level" == "warning" ]] && color="warning"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"💰 Cost Optimization - $level: $message\", \"color\":\"$color\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

# Setup directories
setup_directories() {
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$REPORT_DIR"
}

# Get current resource usage
get_resource_usage() {
    local usage_file="$REPORT_DIR/resource_usage_$(date +%Y%m%d_%H%M%S).json"
    
    log "Collecting resource usage metrics"
    
    {
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"cpu\": {"
        
        # CPU usage per service
        local cpu_data=""
        for service in $(docker service ls --format "{{.Name}}"); do
            local cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" $(docker service ps $service --format "{{.Name}}" | head -1) 2>/dev/null | sed 's/%//' || echo "0")
            cpu_data+="{\"service\":\"$service\",\"usage\":$cpu_usage},"
        done
        echo "    \"services\": [${cpu_data%,}],"
        
        # Overall system CPU
        local sys_cpu=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
        echo "    \"system\": $sys_cpu"
        echo "  },"
        
        echo "  \"memory\": {"
        
        # Memory usage per service
        local mem_data=""
        for service in $(docker service ls --format "{{.Name}}"); do
            local mem_usage=$(docker stats --no-stream --format "{{.MemPerc}}" $(docker service ps $service --format "{{.Name}}" | head -1) 2>/dev/null | sed 's/%//' || echo "0")
            mem_data+="{\"service\":\"$service\",\"usage\":$mem_usage},"
        done
        echo "    \"services\": [${mem_data%,}],"
        
        # Overall system memory
        local mem_total=$(free -m | awk 'NR==2{printf "%.1f", $2}')
        local mem_used=$(free -m | awk 'NR==2{printf "%.1f", $3}')
        local mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used/$mem_total)*100}")
        echo "    \"system\": {\"total\":$mem_total,\"used\":$mem_used,\"percent\":$mem_percent}"
        echo "  },"
        
        echo "  \"disk\": {"
        local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
        local disk_available=$(df -h / | awk 'NR==2 {print $4}')
        echo "    \"usage_percent\": $disk_usage,"
        echo "    \"available\": \"$disk_available\""
        echo "  },"
        
        echo "  \"services\": {"
        local service_count=$(docker service ls -q | wc -l)
        local running_tasks=$(docker service ls --format "{{.Replicas}}" | grep -o '^[0-9]*' | paste -sd+ | bc)
        echo "    \"total_services\": $service_count,"
        echo "    \"total_tasks\": $running_tasks"
        echo "  }"
        echo "}"
        
    } > "$usage_file"
    
    log "Resource usage data saved to: $usage_file"
    echo "$usage_file"
}

# Analyze cost efficiency
analyze_cost_efficiency() {
    local usage_file="$1"
    
    log "Analyzing cost efficiency"
    
    # Extract key metrics
    local avg_cpu=$(jq -r '.cpu.services[] | .usage' "$usage_file" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
    local avg_memory=$(jq -r '.memory.services[] | .usage' "$usage_file" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
    local total_services=$(jq -r '.services.total_services' "$usage_file")
    local total_tasks=$(jq -r '.services.total_tasks' "$usage_file")
    
    log "Average CPU usage: ${avg_cpu}%"
    log "Average Memory usage: ${avg_memory}%"
    log "Total services: $total_services"
    log "Total tasks: $total_tasks"
    
    # Generate recommendations
    local recommendations=""
    
    if (( $(echo "$avg_cpu < $THRESHOLD_CPU_LOW" | bc -l) )); then
        recommendations+="- Consider scaling down services (avg CPU: ${avg_cpu}%)\n"
    fi
    
    if (( $(echo "$avg_cpu > $THRESHOLD_CPU_HIGH" | bc -l) )); then
        recommendations+="- Consider scaling up services (avg CPU: ${avg_cpu}%)\n"
    fi
    
    if (( $(echo "$avg_memory > $THRESHOLD_MEMORY_HIGH" | bc -l) )); then
        recommendations+="- High memory usage detected (avg: ${avg_memory}%)\n"
    fi
    
    # Check for over-provisioned services
    local over_provisioned=""
    for service in $(docker service ls --format "{{.Name}}"); do
        local replicas=$(docker service ls --filter "name=$service" --format "{{.Replicas}}" | cut -d'/' -f1)
        if [[ $replicas -gt 1 ]]; then
            local cpu_usage=$(jq -r ".cpu.services[] | select(.service==\"$service\") | .usage" "$usage_file")
            if (( $(echo "$cpu_usage < $THRESHOLD_CPU_LOW" | bc -l) )); then
                over_provisioned+="- $service: $replicas replicas, ${cpu_usage}% CPU\n"
            fi
        fi
    done
    
    if [[ -n "$over_provisioned" ]]; then
        recommendations+="Over-provisioned services:\n$over_provisioned"
    fi
    
    # Save recommendations
    local rec_file="$REPORT_DIR/cost_recommendations_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "A2A World Platform - Cost Optimization Recommendations"
        echo "Generated: $(date)"
        echo ""
        echo "Current Resource Usage:"
        echo "- Average CPU: ${avg_cpu}%"
        echo "- Average Memory: ${avg_memory}%"
        echo "- Total Services: $total_services"
        echo "- Total Tasks: $total_tasks"
        echo ""
        echo "Recommendations:"
        echo -e "$recommendations"
    } > "$rec_file"
    
    log "Cost recommendations saved to: $rec_file"
    
    # Send notification if significant issues found
    if [[ -n "$recommendations" ]]; then
        notify "Cost optimization opportunities identified. Check report: $(basename "$rec_file")" "warning"
    fi
    
    echo "$rec_file"
}

# Get DigitalOcean billing information
get_do_billing() {
    if [[ -z "$DO_TOKEN" ]]; then
        log "DigitalOcean token not configured, skipping billing analysis"
        return
    fi
    
    log "Retrieving DigitalOcean billing information"
    
    local billing_file="$REPORT_DIR/do_billing_$(date +%Y%m%d_%H%M%S).json"
    
    # Get current month's usage
    curl -X GET -H "Authorization: Bearer $DO_TOKEN" \
         "https://api.digitalocean.com/v2/customers/my/billing_history" \
         > "$billing_file" 2>/dev/null || {
        log "WARNING: Could not retrieve billing information"
        return
    }
    
    # Parse billing data
    local current_month_cost=$(jq -r '.billing_history[] | select(.date | startswith("'$(date +%Y-%m)'")) | .amount' "$billing_file" | head -1)
    
    if [[ -n "$current_month_cost" && "$current_month_cost" != "null" ]]; then
        log "Current month cost: \$${current_month_cost}"
        
        # Check against budget
        if (( $(echo "$current_month_cost > $COST_BUDGET_MONTHLY" | bc -l) )); then
            notify "Monthly cost (\$${current_month_cost}) exceeds budget (\$${COST_BUDGET_MONTHLY})" "warning"
        fi
    else
        log "Could not determine current month cost"
    fi
}

# Optimize Docker resources
optimize_docker_resources() {
    log "Optimizing Docker resources"
    
    # Clean up unused images
    local removed_images=$(docker image prune -f 2>&1 | grep "Total reclaimed space" | awk '{print $4}' || echo "0B")
    log "Reclaimed space from unused images: $removed_images"
    
    # Clean up unused volumes
    local removed_volumes=$(docker volume prune -f 2>&1 | grep "Total reclaimed space" | awk '{print $4}' || echo "0B")
    log "Reclaimed space from unused volumes: $removed_volumes"
    
    # Clean up build cache
    local removed_cache=$(docker builder prune -f 2>&1 | grep "Total reclaimed space" | awk '{print $4}' || echo "0B")
    log "Reclaimed space from build cache: $removed_cache"
    
    # Clean up unused networks
    docker network prune -f >/dev/null 2>&1
    
    log "Docker cleanup completed"
}

# Auto-scale services based on metrics
auto_scale_services() {
    local usage_file="$1"
    
    log "Analyzing services for auto-scaling opportunities"
    
    # Check each service for scaling opportunities
    for service in $(docker service ls --format "{{.Name}}"); do
        # Skip infrastructure services
        if [[ "$service" =~ ^(postgres|redis|nats|consul|traefik)$ ]]; then
            continue
        fi
        
        local current_replicas=$(docker service ls --filter "name=$service" --format "{{.Replicas}}" | cut -d'/' -f1)
        local cpu_usage=$(jq -r ".cpu.services[] | select(.service==\"$service\") | .usage" "$usage_file" 2>/dev/null || echo "0")
        local mem_usage=$(jq -r ".memory.services[] | select(.service==\"$service\") | .usage" "$usage_file" 2>/dev/null || echo "0")
        
        log "Service $service: ${current_replicas} replicas, CPU: ${cpu_usage}%, Memory: ${mem_usage}%"
        
        # Scale up if high resource usage
        if (( $(echo "$cpu_usage > $THRESHOLD_CPU_HIGH" | bc -l) )) && [[ $current_replicas -lt 5 ]]; then
            local new_replicas=$((current_replicas + 1))
            log "Scaling UP $service from $current_replicas to $new_replicas replicas (CPU: ${cpu_usage}%)"
            docker service scale "$service=$new_replicas"
            notify "Auto-scaled UP $service to $new_replicas replicas due to high CPU usage (${cpu_usage}%)"
        
        # Scale down if low resource usage (but keep at least 1 replica)
        elif (( $(echo "$cpu_usage < $THRESHOLD_CPU_LOW" | bc -l) )) && [[ $current_replicas -gt 1 ]]; then
            # Only scale down if consistently low for some time (check previous measurements)
            local new_replicas=$((current_replicas - 1))
            log "Scaling DOWN $service from $current_replicas to $new_replicas replicas (CPU: ${cpu_usage}%)"
            docker service scale "$service=$new_replicas"
            notify "Auto-scaled DOWN $service to $new_replicas replicas due to low CPU usage (${cpu_usage}%)"
        fi
    done
}

# Generate cost report
generate_cost_report() {
    local usage_file="$1"
    local recommendations_file="$2"
    
    log "Generating comprehensive cost report"
    
    local report_file="$REPORT_DIR/cost_report_$(date +%Y%m%d_%H%M%S).html"
    
    {
        echo "<!DOCTYPE html>"
        echo "<html><head><title>A2A World - Cost Optimization Report</title>"
        echo "<style>"
        echo "body { font-family: Arial, sans-serif; margin: 20px; }"
        echo "table { border-collapse: collapse; width: 100%; margin: 20px 0; }"
        echo "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
        echo "th { background-color: #f2f2f2; }"
        echo ".warning { color: #ff6600; }"
        echo ".good { color: #009900; }"
        echo ".metric { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }"
        echo "</style></head><body>"
        
        echo "<h1>A2A World Platform - Cost Optimization Report</h1>"
        echo "<p><strong>Generated:</strong> $(date)</p>"
        
        # Resource Usage Summary
        echo "<h2>Resource Usage Summary</h2>"
        echo "<div class='metric'>"
        local avg_cpu=$(jq -r '.cpu.services[] | .usage' "$usage_file" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        local avg_memory=$(jq -r '.memory.services[] | .usage' "$usage_file" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        local disk_usage=$(jq -r '.disk.usage_percent' "$usage_file")
        
        echo "<p><strong>Average CPU Usage:</strong> <span class='$([ $(echo "$avg_cpu < 50" | bc) -eq 1 ] && echo "good" || echo "warning")'>${avg_cpu}%</span></p>"
        echo "<p><strong>Average Memory Usage:</strong> <span class='$([ $(echo "$avg_memory < 70" | bc) -eq 1 ] && echo "good" || echo "warning")'>${avg_memory}%</span></p>"
        echo "<p><strong>Disk Usage:</strong> <span class='$([ $disk_usage -lt 80 ] && echo "good" || echo "warning")'>${disk_usage}%</span></p>"
        echo "</div>"
        
        # Service Details
        echo "<h2>Service Resource Usage</h2>"
        echo "<table>"
        echo "<tr><th>Service</th><th>CPU Usage</th><th>Memory Usage</th><th>Replicas</th><th>Status</th></tr>"
        
        for service in $(docker service ls --format "{{.Name}}"); do
            local cpu_usage=$(jq -r ".cpu.services[] | select(.service==\"$service\") | .usage" "$usage_file" 2>/dev/null || echo "0")
            local mem_usage=$(jq -r ".memory.services[] | select(.service==\"$service\") | .usage" "$usage_file" 2>/dev/null || echo "0")
            local replicas=$(docker service ls --filter "name=$service" --format "{{.Replicas}}")
            
            local status="Normal"
            local status_class="good"
            if (( $(echo "$cpu_usage > 80" | bc -l) )); then
                status="High CPU"
                status_class="warning"
            elif (( $(echo "$cpu_usage < 20" | bc -l) )) && [[ $(echo $replicas | cut -d'/' -f1) -gt 1 ]]; then
                status="Over-provisioned"
                status_class="warning"
            fi
            
            echo "<tr>"
            echo "<td>$service</td>"
            echo "<td>${cpu_usage}%</td>"
            echo "<td>${mem_usage}%</td>"
            echo "<td>$replicas</td>"
            echo "<td class='$status_class'>$status</td>"
            echo "</tr>"
        done
        
        echo "</table>"
        
        # Recommendations
        if [[ -f "$recommendations_file" ]]; then
            echo "<h2>Cost Optimization Recommendations</h2>"
            echo "<div class='metric'>"
            echo "<pre>$(cat "$recommendations_file")</pre>"
            echo "</div>"
        fi
        
        # Estimated Costs
        echo "<h2>Estimated Monthly Costs</h2>"
        echo "<table>"
        echo "<tr><th>Resource</th><th>Quantity</th><th>Unit Cost</th><th>Monthly Cost</th></tr>"
        
        local droplet_count=$(docker node ls -q | wc -l)
        local droplet_cost=$((droplet_count * 24))
        echo "<tr><td>Application Servers (2vcpu-4gb)</td><td>$droplet_count</td><td>\$24</td><td>\$${droplet_cost}</td></tr>"
        echo "<tr><td>Managed Database</td><td>1</td><td>\$15</td><td>\$15</td></tr>"
        echo "<tr><td>Load Balancer</td><td>1</td><td>\$12</td><td>\$12</td></tr>"
        echo "<tr><td>Container Registry</td><td>1</td><td>\$5</td><td>\$5</td></tr>"
        echo "<tr><td>Object Storage</td><td>~50GB</td><td>\$0.02/GB</td><td>\$5</td></tr>"
        echo "<tr><td>Backup Storage</td><td>~100GB</td><td>\$0.02/GB</td><td>\$5</td></tr>"
        echo "<tr><td>Bandwidth</td><td>~1TB</td><td>\$0.01/GB</td><td>\$10</td></tr>"
        
        local total_cost=$((droplet_cost + 15 + 12 + 5 + 5 + 5 + 10))
        echo "<tr><td><strong>TOTAL</strong></td><td></td><td></td><td><strong>\$${total_cost}</strong></td></tr>"
        echo "</table>"
        
        echo "</body></html>"
        
    } > "$report_file"
    
    log "Cost report generated: $report_file"
    echo "$report_file"
}

# Cleanup old reports
cleanup_old_reports() {
    log "Cleaning up old reports"
    
    # Keep reports for 30 days
    find "$REPORT_DIR" -name "*.json" -type f -mtime +30 -delete
    find "$REPORT_DIR" -name "*.txt" -type f -mtime +30 -delete  
    find "$REPORT_DIR" -name "*.html" -type f -mtime +30 -delete
    
    # Keep only 10 most recent reports of each type
    ls -t "$REPORT_DIR"/cost_report_*.html 2>/dev/null | tail -n +11 | xargs rm -f
    ls -t "$REPORT_DIR"/resource_usage_*.json 2>/dev/null | tail -n +11 | xargs rm -f
    ls -t "$REPORT_DIR"/cost_recommendations_*.txt 2>/dev/null | tail -n +11 | xargs rm -f
    
    log "Old reports cleanup completed"
}

# Main cost optimization function
main() {
    local action="${1:-analyze}"
    
    log "Starting cost optimization process: $action"
    
    setup_directories
    
    case "$action" in
        "analyze"|"report")
            local usage_file=$(get_resource_usage)
            local recommendations_file=$(analyze_cost_efficiency "$usage_file")
            get_do_billing
            local report_file=$(generate_cost_report "$usage_file" "$recommendations_file")
            
            notify "Cost optimization analysis completed. Report: $(basename "$report_file")"
            ;;
            
        "optimize")
            local usage_file=$(get_resource_usage)
            analyze_cost_efficiency "$usage_file"
            optimize_docker_resources
            notify "Cost optimization completed"
            ;;
            
        "auto-scale")
            local usage_file=$(get_resource_usage)
            auto_scale_services "$usage_file"
            ;;
            
        "cleanup")
            optimize_docker_resources
            cleanup_old_reports
            ;;
            
        "full"|*)
            local usage_file=$(get_resource_usage)
            local recommendations_file=$(analyze_cost_efficiency "$usage_file")
            get_do_billing
            optimize_docker_resources
            auto_scale_services "$usage_file"
            local report_file=$(generate_cost_report "$usage_file" "$recommendations_file")
            cleanup_old_reports
            
            notify "Full cost optimization completed. Report: $(basename "$report_file")"
            ;;
    esac
    
    log "Cost optimization process completed"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi