#!/bin/bash
# A2A World Platform - Start Agents Script
# Starts the multi-agent system with all required components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${PROJECT_ROOT}/pids"

# Create directories
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Configuration
DEFAULT_AGENTS=("monitoring" "validation" "parser" "discovery")
LOG_LEVEL="INFO"
CONFIG_DIR="${PROJECT_ROOT}/config"

# Functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if required services are running
    if ! curl -s http://localhost:4222 > /dev/null; then
        print_warning "NATS server not responding at localhost:4222"
        print_status "Please ensure NATS is running: docker-compose up nats"
    fi
    
    if ! curl -s http://localhost:8500 > /dev/null; then
        print_warning "Consul server not responding at localhost:8500"
        print_status "Please ensure Consul is running: docker-compose up consul"
    fi
    
    print_success "Dependencies checked"
}

start_agent() {
    local agent_type=$1
    local agent_id=$2
    local config_file=$3
    
    print_status "Starting ${agent_type} agent (ID: ${agent_id})"
    
    local log_file="${LOG_DIR}/${agent_id}.log"
    local pid_file="${PID_DIR}/${agent_id}.pid"
    local cmd_args=""
    
    # Add configuration file if provided
    if [[ -n "$config_file" && -f "$config_file" ]]; then
        cmd_args="--config $config_file"
    fi
    
    # Start agent in background
    cd "$PROJECT_ROOT"
    nohup python3 -m agents.scripts.agent_launcher \
        "$agent_type" \
        --agent-id "$agent_id" \
        --log-level "$LOG_LEVEL" \
        $cmd_args \
        >> "$log_file" 2>&1 &
    
    local pid=$!
    echo $pid > "$pid_file"
    
    # Wait a moment and check if process started successfully
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        print_success "${agent_type} agent started (PID: $pid)"
        echo "  Log file: $log_file"
        echo "  PID file: $pid_file"
    else
        print_error "Failed to start ${agent_type} agent"
        return 1
    fi
}

stop_agent() {
    local agent_id=$1
    local pid_file="${PID_DIR}/${agent_id}.pid"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            print_status "Stopping agent ${agent_id} (PID: $pid)"
            kill -TERM $pid
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 $pid 2>/dev/null && [[ $count -lt 30 ]]; do
                sleep 1
                ((count++))
            done
            
            if kill -0 $pid 2>/dev/null; then
                print_warning "Agent did not stop gracefully, forcing shutdown"
                kill -KILL $pid
            fi
            
            rm -f "$pid_file"
            print_success "Agent ${agent_id} stopped"
        else
            print_warning "Agent ${agent_id} is not running"
            rm -f "$pid_file"
        fi
    else
        print_warning "No PID file found for agent ${agent_id}"
    fi
}

list_agents() {
    print_status "Listing running agents:"
    
    local found_agents=0
    for pid_file in "${PID_DIR}"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            local agent_id=$(basename "$pid_file" .pid)
            local pid=$(cat "$pid_file")
            
            if kill -0 $pid 2>/dev/null; then
                echo "  ${agent_id} (PID: $pid) - RUNNING"
                ((found_agents++))
            else
                echo "  ${agent_id} (PID: $pid) - STOPPED"
                rm -f "$pid_file"
            fi
        fi
    done
    
    if [[ $found_agents -eq 0 ]]; then
        print_warning "No agents are currently running"
    else
        print_success "Found $found_agents running agents"
    fi
}

show_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start [AGENT_TYPE] [AGENT_ID]  Start specific agent or all agents"
    echo "  stop [AGENT_ID]                Stop specific agent or all agents"
    echo "  restart [AGENT_ID]             Restart specific agent or all agents"
    echo "  status                         Show status of all agents"
    echo "  logs [AGENT_ID]               Show logs for specific agent"
    echo "  clean                          Clean up PID and log files"
    echo ""
    echo "Options:"
    echo "  -c, --config DIR              Configuration directory (default: ./config)"
    echo "  -l, --log-level LEVEL         Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Agent Types:"
    echo "  monitoring                     System monitoring agent"
    echo "  validation                     Pattern validation agent"
    echo "  parser                         KML/GeoJSON parser agent"
    echo "  discovery                      Pattern discovery agent"
    echo ""
    echo "Examples:"
    echo "  $0 start                      # Start all default agents"
    echo "  $0 start monitoring           # Start monitoring agent with auto-generated ID"
    echo "  $0 start validation val-001   # Start validation agent with specific ID"
    echo "  $0 stop                       # Stop all agents"
    echo "  $0 stop val-001               # Stop specific agent"
    echo "  $0 status                     # Show agent status"
    echo "  $0 logs monitor-001           # Show logs for specific agent"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_DIR="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        start)
            COMMAND="start"
            AGENT_TYPE="$2"
            AGENT_ID="$3"
            break
            ;;
        stop)
            COMMAND="stop"
            AGENT_ID="$2"
            break
            ;;
        restart)
            COMMAND="restart"
            AGENT_ID="$2"
            break
            ;;
        status)
            COMMAND="status"
            break
            ;;
        logs)
            COMMAND="logs"
            AGENT_ID="$2"
            break
            ;;
        clean)
            COMMAND="clean"
            break
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Default command is start
if [[ -z "$COMMAND" ]]; then
    COMMAND="start"
fi

# Main execution
print_status "A2A World Agent Management Script"
print_status "=================================="

case $COMMAND in
    start)
        check_dependencies
        
        if [[ -n "$AGENT_TYPE" ]]; then
            # Start specific agent
            if [[ -z "$AGENT_ID" ]]; then
                AGENT_ID="${AGENT_TYPE}-$(date +%s)"
            fi
            
            config_file=""
            if [[ -f "${CONFIG_DIR}/${AGENT_TYPE}.yaml" ]]; then
                config_file="${CONFIG_DIR}/${AGENT_TYPE}.yaml"
            fi
            
            start_agent "$AGENT_TYPE" "$AGENT_ID" "$config_file"
        else
            # Start all default agents
            print_status "Starting all default agents..."
            for agent_type in "${DEFAULT_AGENTS[@]}"; do
                agent_id="${agent_type}-$(date +%s)"
                
                config_file=""
                if [[ -f "${CONFIG_DIR}/${agent_type}.yaml" ]]; then
                    config_file="${CONFIG_DIR}/${agent_type}.yaml"
                fi
                
                start_agent "$agent_type" "$agent_id" "$config_file"
                sleep 1  # Brief delay between starts
            done
            
            print_success "All agents started"
            print_status "Use '$0 status' to check agent status"
        fi
        ;;
        
    stop)
        if [[ -n "$AGENT_ID" ]]; then
            # Stop specific agent
            stop_agent "$AGENT_ID"
        else
            # Stop all agents
            print_status "Stopping all agents..."
            for pid_file in "${PID_DIR}"/*.pid; do
                if [[ -f "$pid_file" ]]; then
                    agent_id=$(basename "$pid_file" .pid)
                    stop_agent "$agent_id"
                fi
            done
            print_success "All agents stopped"
        fi
        ;;
        
    restart)
        if [[ -n "$AGENT_ID" ]]; then
            # Restart specific agent
            stop_agent "$AGENT_ID"
            sleep 2
            # Note: This is simplified - in practice you'd need to remember the agent type
            print_status "Manual restart required - use start command with appropriate parameters"
        else
            # Restart all agents
            print_status "Restarting all agents..."
            "$0" stop
            sleep 5
            "$0" start
        fi
        ;;
        
    status)
        list_agents
        ;;
        
    logs)
        if [[ -n "$AGENT_ID" ]]; then
            log_file="${LOG_DIR}/${AGENT_ID}.log"
            if [[ -f "$log_file" ]]; then
                print_status "Showing logs for agent ${AGENT_ID}:"
                echo "=============================================="
                tail -f "$log_file"
            else
                print_error "Log file not found: $log_file"
                exit 1
            fi
        else
            print_error "Agent ID required for logs command"
            show_usage
            exit 1
        fi
        ;;
        
    clean)
        print_status "Cleaning up PID and log files..."
        
        # Remove stale PID files
        for pid_file in "${PID_DIR}"/*.pid; do
            if [[ -f "$pid_file" ]]; then
                pid=$(cat "$pid_file")
                if ! kill -0 $pid 2>/dev/null; then
                    agent_id=$(basename "$pid_file" .pid)
                    print_status "Removing stale PID file for ${agent_id}"
                    rm -f "$pid_file"
                fi
            fi
        done
        
        # Optionally clean old log files (older than 7 days)
        find "${LOG_DIR}" -name "*.log" -mtime +7 -delete 2>/dev/null || true
        
        print_success "Cleanup completed"
        ;;
        
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

print_status "Script completed"