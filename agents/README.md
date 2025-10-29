# A2A World Platform - Multi-Agent System

## Overview

The A2A World multi-agent system is a distributed architecture for autonomous pattern discovery across geospatial data, cultural mythology, and environmental phenomena. The system uses NATS for messaging, Consul for service discovery, and provides multiple specialized agent types for different tasks.

## Architecture

### Core Components

- **Base Agent Framework** (`agents/core/`) - Common functionality for all agents
- **NATS Messaging** - Pub/sub communication between agents  
- **Consul Service Discovery** - Agent registration and health monitoring
- **Task Queue System** - Distributed job processing
- **Configuration Management** - Environment-based configuration with Consul KV
- **Health Check System** - HTTP endpoints for monitoring

### Agent Types

- **ValidationAgent** - Statistical validation of discovered patterns using spatial autocorrelation
- **MonitorAgent** - System health monitoring and alerting
- **KMLParserAgent** - Parse KML/GeoJSON files and extract geospatial features  
- **PatternDiscoveryAgent** - Discover patterns using HDBSCAN clustering

## Quick Start

### Prerequisites

1. **Docker & Docker Compose** - For running infrastructure services
2. **Python 3.8+** - For running agents
3. **Required Python packages** - Install from `requirements.txt`

```bash
# Install dependencies
pip install -r requirements.txt

# Add required packages for full functionality
pip install tabulate aiohttp aiohttp-cors
```

### 1. Start Infrastructure Services

```bash
# Start NATS, Consul, PostgreSQL, and Redis
docker-compose up -d nats consul postgres redis

# Verify services are running
curl http://localhost:8222/  # NATS monitoring
curl http://localhost:8500/  # Consul UI
```

### 2. Start Agents

#### Using the CLI Tool

```bash
# Start all default agents
python -m agents.cli.agent_cli agents start

# Start specific agent type
python -m agents.cli.agent_cli agents start validation

# Monitor system status
python -m agents.cli.agent_cli monitor --watch
```

#### Using Shell Scripts

```bash
# Linux/Mac
chmod +x agents/scripts/start-agents.sh
./agents/scripts/start-agents.sh start

# Windows  
agents\scripts\start-agents.bat start
```

#### Manual Agent Startup

```bash
# Start individual agents
python -m agents.scripts.agent_launcher validation
python -m agents.scripts.agent_launcher monitoring  
python -m agents.scripts.agent_launcher parser
python -m agents.scripts.agent_launcher discovery
```

### 3. Verify System Health

```bash
# Check agent status
python -m agents.cli.agent_cli agents list

# Check system health
python -m agents.cli.agent_cli system health

# View system status
python -m agents.cli.agent_cli system status
```

## Usage Examples

### Submit Tasks via CLI

```bash
# Submit a KML parsing task
python -m agents.cli.agent_cli tasks submit parse_kml_file \
    --param file_path=/path/to/data.kml

# Submit a pattern discovery task  
python -m agents.cli.agent_cli tasks submit discover_patterns \
    --param dataset_id=my_dataset \
    --param algorithm=hdbscan

# Check task status
python -m agents.cli.agent_cli tasks status <task_id>
```

### Using Python API

```python
import asyncio
from agents.validation.validation_agent import ValidationAgent
from agents.core.task_queue import Task, create_parse_task

async def example_usage():
    # Create and start a validation agent
    agent = ValidationAgent()
    
    # Start agent in background
    agent_task = asyncio.create_task(agent.start())
    
    # Create a parsing task
    task = create_parse_task("/data/sample.kml", "kml")
    
    # Submit task to queue (requires NATS/Consul)
    # await task_queue.submit_task(task)
    
    # Shutdown agent
    agent.shutdown_event.set()
    await agent_task

# Run example
asyncio.run(example_usage())
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Core Services  
NATS_URL=nats://localhost:4222
CONSUL_HOST=localhost
CONSUL_PORT=8500

# Database
DATABASE_URL=postgresql://a2a_user:a2a_password@localhost:5432/a2a_world

# Agent Settings
AGENT_HEARTBEAT_INTERVAL=30
MAX_CONCURRENT_AGENTS=10
LOG_LEVEL=INFO
```

### Agent-Specific Configuration

Create YAML configuration files in `config/` directory:

**config/validation.yaml**
```yaml
agent_type: validation
significance_level: 0.05
min_sample_size: 30
bootstrap_iterations: 1000
enable_morans_i: true
enable_getis_ord: true
```

**config/discovery.yaml**  
```yaml
agent_type: discovery
default_algorithm: hdbscan
min_cluster_size: 5
confidence_threshold: 0.7
search_radius_km: 50.0
enable_parallel_processing: true
```

## Health Monitoring

### Agent Health Endpoints

Each agent can expose HTTP health endpoints:

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/status  

# Metrics
curl http://localhost:8080/metrics

# Kubernetes-style probes
curl http://localhost:8080/ready
curl http://localhost:8080/alive
```

### System Monitoring

The MonitorAgent provides system-wide monitoring:

- Resource usage tracking (CPU, memory, disk)
- Agent health monitoring  
- Infrastructure service health checks
- Alerting and notifications
- Performance metrics collection

## Testing

### Run System Tests

```bash
# Run comprehensive test suite
python -m agents.tests.test_agent_system

# Run specific tests with pytest (if available)
pytest agents/tests/ -v
```

### Test Results

The test suite validates:
- ✅ Configuration management
- ✅ NATS messaging components  
- ✅ Consul registry functionality
- ✅ Task queue system
- ✅ Base agent framework
- ✅ Health check system
- ✅ All agent implementations
- ✅ System integration
- ✅ Performance characteristics

## Development

### Adding New Agent Types

1. Create agent class inheriting from `BaseAgent`:

```python
from agents.core.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(agent_type="custom", **kwargs)
    
    async def process(self):
        # Implement agent logic
        pass
    
    def _get_capabilities(self):
        return ["custom", "my_capability"]
```

2. Add configuration class:

```python
from agents.core.config import AgentConfig

class CustomAgentConfig(AgentConfig):
    agent_type: str = "custom"
    custom_setting: int = 42
```

3. Register in CLI and launcher tools.

### Adding Custom Tasks

```python
from agents.core.task_queue import Task

def create_custom_task(param1: str, param2: int) -> Task:
    return Task(
        task_type="custom_task",
        parameters={"param1": param1, "param2": param2},
        priority=5
    )
```

## API Integration

The multi-agent system integrates with the FastAPI backend:

- Agent status endpoints in `/api/v1/agents/`
- Task management endpoints  
- Pattern discovery results
- Health monitoring integration

## Deployment

### Docker Deployment

The system is designed to run in Docker containers:

```bash
# Build agent container
docker build -f infrastructure/docker/Dockerfile.agents -t a2a-agents .

# Run with docker-compose
docker-compose up -d
```

### Kubernetes Deployment

Agents support Kubernetes-style health probes and can be deployed as pods with proper service discovery.

## Troubleshooting

### Common Issues

1. **NATS Connection Failed**
   ```bash
   # Check NATS is running
   docker-compose ps nats
   curl http://localhost:8222/
   ```

2. **Consul Registration Failed**  
   ```bash
   # Check Consul is running
   docker-compose ps consul
   curl http://localhost:8500/v1/status/leader
   ```

3. **Agent Not Starting**
   ```bash
   # Check logs
   python -m agents.scripts.agent_launcher validation --log-level DEBUG
   ```

4. **Health Check Failures**
   ```bash
   # Check agent health endpoint
   curl http://localhost:8080/health
   
   # Check system resources
   python -m agents.cli.agent_cli system status
   ```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m agents.scripts.agent_launcher monitoring --log-level DEBUG
```

## Performance

### Resource Requirements

- **Minimum**: 2GB RAM, 2 CPU cores
- **Recommended**: 4GB RAM, 4 CPU cores  
- **Production**: 8GB RAM, 8 CPU cores

### Scaling

- Horizontal scaling: Run multiple instances of each agent type
- Load balancing: NATS queue groups distribute work
- Service discovery: Consul manages agent registration
- Health monitoring: Automatic failover and recovery

## Security

- **Network**: Use TLS for NATS and Consul in production
- **Authentication**: Configure Consul ACLs and NATS auth
- **Configuration**: Use secrets management for sensitive data
- **Health endpoints**: Restrict access in production environments

## Contributing

1. Follow the established agent patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure health checks work properly
5. Test with full system integration

## License

See the main project LICENSE file.

---

For more information, see the main project documentation in `/docs/`.