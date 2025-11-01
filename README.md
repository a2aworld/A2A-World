# A2A World Platform

An AI-driven system for discovering meaningful patterns across geospatial data, cultural mythology, and environmental phenomena using autonomous agents.

## Overview

The A2A World Platform is a comprehensive multi-agent system designed to identify, validate, and explain patterns across diverse datasets including geospatial information, cultural and mythological references, and environmental measurements. The platform uses advanced clustering algorithms, statistical validation, and AI-powered narrative generation to provide scientifically rigorous pattern discovery.

## Key Features

- **Multi-Agent Architecture**: Autonomous agents for parsing, discovery, validation, and monitoring
- **Geospatial Analysis**: PostGIS-powered geographic data processing
- **Pattern Discovery**: HDBSCAN clustering and statistical validation
- **Cultural Integration**: Mythology and folklore correlation with geographic patterns
- **AI Narratives**: LLM-powered pattern explanations and insights
- **Real-time Processing**: NATS messaging for scalable agent communication
- **Interactive Dashboard**: React/Next.js frontend for pattern exploration

## Technology Stack

### Backend
- **FastAPI** (Python) - REST API server
- **PostgreSQL + PostGIS** - Geospatial database
- **Redis** - Caching and session storage
- **NATS** - Message bus for agent communication
- **Consul** - Service discovery and configuration

### Frontend
- **Next.js** (React/TypeScript) - Web dashboard
- **Tailwind CSS** - Styling and components
- **Leaflet** - Interactive mapping
- **Recharts** - Data visualizations

### AI/ML
- **scikit-learn** - Machine learning algorithms
- **HDBSCAN** - Density-based clustering
- **sentence-transformers** - Text embeddings
- **Ollama** - Local LLM inference
- **GeoPandas/Shapely** - Geospatial processing

### Infrastructure
- **Docker** - Containerization
- **Prometheus + Grafana** - Monitoring
- **GitHub Actions** - CI/CD

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Git

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/a2aworld/A2A-World.git
cd A2A-World
```

2. Start all services:
```bash
docker-compose up -d
```

3. Initialize the database:
```bash
docker-compose exec postgres psql -U a2a_user -d a2a_world -f /docker-entrypoint-initdb.d/001_initial_schema.sql
```

4. Access the platform:
    - **Web Dashboard**: http://localhost:3000
    - **API Documentation**: http://localhost:8000/docs
    - **API Health**: http://localhost:8000/health

## Usage

### Quick Start Guide

Once the platform is running, follow these steps to start discovering patterns:

#### 1. Upload Data
- Navigate to the **Data** section in the web dashboard
- Upload KML, GeoJSON, or CSV files containing geospatial data
- The system will automatically parse and store your data in the PostGIS database

#### 2. Configure Agents
- Go to the **Agents** section to view available agent types
- Start pattern discovery agents for your uploaded datasets
- Monitor agent status and health through the dashboard

#### 3. Run Pattern Discovery
- Select datasets from the **Patterns** section
- Choose clustering parameters (HDBSCAN recommended)
- Execute pattern discovery and wait for results
- View discovered patterns on interactive maps

#### 4. Validate Results
- Use the validation dashboard to assess pattern significance
- Review statistical metrics and confidence scores
- Generate AI-powered narratives explaining discovered patterns

### API Usage Examples

#### List Available Agents
```bash
curl -X GET "http://localhost:8000/api/v1/agents/" \
  -H "accept: application/json"
```

#### Start a Pattern Discovery Agent
```bash
curl -X POST "http://localhost:8000/api/v1/agents/pattern-discovery-001/start" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "pattern_discovery",
    "configuration": {
      "clustering_algorithm": "hdbscan",
      "min_cluster_size": 5
    }
  }'
```

#### Upload Data File
```bash
curl -X POST "http://localhost:8000/api/v1/data/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_data.kml"
```

#### Get Pattern Discovery Results
```bash
curl -X GET "http://localhost:8000/api/v1/patterns/?dataset_id=your_dataset_id" \
  -H "accept: application/json"
```

### Agent Management

#### Starting Agents
- Use the web interface or API to start agents
- Specify agent type and configuration parameters
- Monitor startup progress through agent logs

#### Monitoring Agents
- View real-time metrics in the dashboard
- Check agent health status and resource usage
- Review task completion rates and error logs

#### Scaling Agents
- Start multiple instances of the same agent type
- Distribute workload across available agents
- Use Consul for automatic service discovery

### Data Processing Workflow

1. **Data Ingestion**: Upload files through web interface or API
2. **Preprocessing**: Agents automatically parse and validate data
3. **Pattern Discovery**: Apply clustering algorithms to find patterns
4. **Validation**: Statistical analysis confirms pattern significance
5. **Narrative Generation**: AI creates human-readable explanations
6. **Visualization**: Interactive maps and charts display results

### Configuration

Create a `.env` file based on `.env.example` with your specific settings:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/a2a_world

# Redis
REDIS_URL=redis://localhost:6379

# NATS
NATS_URL=nats://localhost:4222

# Consul
CONSUL_URL=http://localhost:8500
CONSUL_DC=dc1

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway    │    │  Agent System   │
│   (Next.js)     │◄──►│   (FastAPI)      │◄──►│   (Python)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │      Redis       │    │      NATS       │
│   + PostGIS     │    │    (Cache)       │    │   (Messaging)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Agent Types

- **Parser Agents**: Process KML/GeoJSON files and extract features
- **Discovery Agents**: Apply clustering algorithms to find patterns
- **Validation Agents**: Perform statistical validation of discovered patterns
- **Narrative Agents**: Generate human-readable explanations using LLMs
- **Monitor Agents**: System health monitoring and alerting

## Development

### Local Development
See [Development Setup Guide](docs/development/setup.md) for detailed instructions.

### Project Structure
```
A2A-World/
├── api/                 # FastAPI backend
├── frontend/           # Next.js frontend
├── agents/             # Autonomous agent system
├── database/           # Database schemas and migrations
├── infrastructure/     # Docker and deployment files
├── docs/              # Documentation
├── docker-compose.yml # Local development environment
└── requirements.txt   # Python dependencies
```

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run tests: `pytest` (backend) or `npm test` (frontend)
5. Commit changes: `git commit -am 'Add your feature'`
6. Push to branch: `git push origin feature/your-feature`
7. Create a Pull Request

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [API Reference](docs/api/overview.md)
- [Development Setup](docs/development/setup.md)
- [Deployment Guide](docs/deployment/local.md)
- [User Guide](docs/user/getting-started.md)

## Roadmap

### Phase 1: Foundation (Months 1-3)
- [x] Core infrastructure setup
- [x] Multi-agent system framework
- [ ] Basic pattern discovery
- [ ] Web dashboard

### Phase 2: Data Ingestion (Months 4-6)
- [ ] KML/GeoJSON parsing
- [ ] Database integration
- [ ] File upload system
- [ ] Basic visualizations

### Phase 3: Pattern Discovery (Months 7-9)
- [ ] HDBSCAN clustering implementation
- [ ] Statistical validation
- [ ] Pattern exploration interface
- [ ] Narrative generation

### Phase 4: Advanced Features (Months 10-12)
- [ ] Cultural data integration
- [ ] Multi-modal analysis
- [ ] Advanced visualizations
- [ ] Email notifications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/A2A-World/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/A2A-World/discussions)

## Acknowledgments

This project builds upon research in geospatial analysis, cultural studies, and multi-agent systems. Special thanks to the open-source communities behind PostGIS, HDBSCAN, and the broader geospatial data science ecosystem.
