# A2A World Platform - Documentation

Welcome to the A2A World Platform documentation. This directory contains comprehensive documentation for developers, users, and system administrators.

## Documentation Structure

- `api/` - API reference and endpoint documentation
- `architecture/` - System architecture and design decisions
- `deployment/` - Deployment guides and infrastructure setup
- `development/` - Development setup and contribution guidelines
- `user/` - User guides and tutorials

## Quick Start

1. [Development Setup](development/setup.md) - Get the development environment running
2. [Architecture Overview](architecture/overview.md) - Understand the system design
3. [API Reference](api/overview.md) - Learn about the REST API
4. [Deployment Guide](deployment/local.md) - Deploy locally with Docker

## Key Features

The A2A World Platform is an AI-driven system for discovering meaningful patterns across:

- **Geospatial Data** - KML/GeoJSON files with geographic information
- **Cultural Mythology** - Cultural and mythological references with geographic context
- **Environmental Phenomena** - Environmental measurements and observations

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Next.js (React/TypeScript)
- **Database**: PostgreSQL with PostGIS
- **Agents**: Custom Python agents with asyncio
- **Messaging**: NATS
- **Service Discovery**: Consul
- **Caching**: Redis
- **Containerization**: Docker
- **ML/AI**: scikit-learn, HDBSCAN, sentence-transformers

## Contributing

Please read our [contributing guidelines](development/contributing.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.