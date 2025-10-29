# A2A World Platform - Development Setup

This guide will help you set up the A2A World Platform for local development.

## Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **Python 3.11+** (for local development)
- **Node.js 18+** (for frontend development)
- **PostgreSQL 14+** with **PostGIS** extension
- **Redis** (for caching)

## Quick Start with Docker

1. Clone the repository:
```bash
git clone <repository-url>
cd A2A-World
```

2. Start all services with Docker Compose:
```bash
docker-compose up -d
```

3. Initialize the database:
```bash
docker-compose exec postgres psql -U a2a_user -d a2a_world -f /docker-entrypoint-initdb.d/001_initial_schema.sql
```

4. Access the services:
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Local Development Setup

### Backend API

1. Navigate to the API directory:
```bash
cd api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the API server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

### Agent System

1. Navigate to the agents directory:
```bash
cd agents
```

2. Install dependencies (same as API):
```bash
pip install -r ../requirements.txt
```

3. Run individual agents:
```bash
python -m agents.parsers.kml_parser
python -m agents.discovery.pattern_discovery
```

## Database Setup

### Using Docker (Recommended)

PostgreSQL with PostGIS is automatically set up when using Docker Compose.

### Manual Setup

1. Install PostgreSQL and PostGIS
2. Create database:
```sql
CREATE DATABASE a2a_world;
CREATE USER a2a_user WITH PASSWORD 'a2a_password';
GRANT ALL PRIVILEGES ON DATABASE a2a_world TO a2a_user;
```

3. Run schema migrations:
```bash
psql -U a2a_user -d a2a_world -f database/schemas/001_initial_schema.sql
```

## Environment Variables

Create `.env` files in the root directory and configure:

```env
# Database
DATABASE_URL=postgresql://a2a_user:a2a_password@localhost:5432/a2a_world

# Redis
REDIS_URL=redis://localhost:6379

# NATS
NATS_URL=nats://localhost:4222

# API Configuration
SECRET_KEY=your-secret-key-here
API_V1_STR=/api/v1

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development Workflow

1. Make changes to code
2. Run tests: `pytest` (backend) or `npm test` (frontend)
3. Check code quality: `flake8` (backend) or `npm run lint` (frontend)
4. Commit changes with descriptive messages
5. Push to feature branch and create pull request

## Troubleshooting

### Common Issues

1. **Port already in use**: Change ports in `docker-compose.yml`
2. **Database connection failed**: Check PostgreSQL is running and credentials are correct
3. **Agent startup errors**: Ensure NATS and Consul are running

### Getting Help

- Check the [FAQ](../user/faq.md)
- Review [architecture documentation](../architecture/overview.md)
- Create an issue on GitHub