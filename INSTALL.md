# A2A World Platform - Windows Installation Guide

This guide provides step-by-step instructions for installing and setting up the A2A World Platform on a Windows machine.

## Prerequisites

Before installing the A2A World Platform, ensure your system meets the following requirements:

- **Windows 10 or 11** (64-bit)
- **Docker Desktop for Windows** (latest version)
  - Download from: https://www.docker.com/products/docker-desktop
  - Enable WSL 2 backend during installation
- **Python 3.11 or higher**
  - Download from: https://www.python.org/downloads/
  - Ensure Python is added to PATH during installation
- **Node.js 18 or higher**
  - Download from: https://nodejs.org/
  - Includes npm package manager
- **Git**
  - Download from: https://git-scm.com/downloads
- **At least 8GB RAM** (16GB recommended)
- **10GB free disk space**

## Quick Start with Docker (Recommended)

The easiest way to get started is using Docker, which handles all dependencies automatically.

### Step 1: Clone the Repository

Open PowerShell or Command Prompt and run:

```powershell
git clone https://github.com/your-org/A2A-World.git
cd A2A-World
```

### Step 2: Start All Services

```powershell
docker-compose up -d
```

This will start all required services:
- PostgreSQL database with PostGIS
- Redis cache
- NATS messaging server
- API server
- Frontend application
- Agent system

### Step 3: Initialize the Database

```powershell
docker-compose exec postgres psql -U a2a_user -d a2a_world -f /docker-entrypoint-initdb.d/001_initial_schema.sql
```

### Step 4: Access the Application

- **Web Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Manual Installation (Alternative)

If you prefer to run components individually without Docker:

### Backend Setup

1. **Navigate to the API directory**:
   ```powershell
   cd api
   ```

2. **Create a Python virtual environment**:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```powershell
   pip install -r ..\requirements.txt
   ```

4. **Set up environment variables**:
   ```powershell
   copy .env.example .env
   # Edit .env with your configuration (see Environment Variables section below)
   ```

5. **Run the API server**:
   ```powershell
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```powershell
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```powershell
   npm install
   ```

3. **Start the development server**:
   ```powershell
   npm run dev
   ```

### Database Setup

1. **Install PostgreSQL and PostGIS** (if not using Docker):
   - Download PostgreSQL from: https://www.postgresql.org/download/windows/
   - Ensure PostGIS extension is installed

2. **Create the database**:
   ```sql
   CREATE DATABASE a2a_world;
   CREATE USER a2a_user WITH PASSWORD 'a2a_password';
   GRANT ALL PRIVILEGES ON DATABASE a2a_world TO a2a_user;
   ```

3. **Run database migrations**:
   ```powershell
   psql -U a2a_user -d a2a_world -f database\schemas\001_initial_schema.sql
   ```

### Agent System Setup

1. **Navigate to the agents directory**:
   ```powershell
   cd agents
   ```

2. **Install dependencies** (same as API):
   ```powershell
   pip install -r ..\requirements.txt
   ```

3. **Start agents using the Windows script**:
   ```powershell
   .\scripts\start-agents.bat start
   ```

## Environment Variables

Create a `.env` file in the project root with the following configuration:

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

# Ollama (for AI features)
OLLAMA_BASE_URL=http://localhost:11434
```

## Starting Services

### Using Docker Compose (Recommended)

```powershell
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Startup

1. **Start database services** (if not using Docker):
   - PostgreSQL service should start automatically
   - Start Redis server
   - Start NATS server

2. **Start the API server**:
   ```powershell
   cd api
   venv\Scripts\activate
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Start the frontend**:
   ```powershell
   cd frontend
   npm run dev
   ```

4. **Start agents**:
   ```powershell
   cd agents
   ..\venv\Scripts\activate
   .\scripts\start-agents.bat start
   ```

## Verification

After installation, verify everything is working:

1. **Check service health**:
   - Visit http://localhost:8000/health
   - Should return `{"status": "healthy"}`

2. **Check API documentation**:
   - Visit http://localhost:8000/docs
   - Should show FastAPI interactive documentation

3. **Check web dashboard**:
   - Visit http://localhost:3000
   - Should load the A2A World dashboard

4. **Check agent status**:
   ```powershell
   cd agents
   .\scripts\start-agents.bat status
   ```

## Troubleshooting

### Common Issues

1. **Docker Desktop not starting**:
   - Ensure WSL 2 is enabled: `wsl --set-default-version 2`
   - Restart Docker Desktop
   - Check Windows features for Virtual Machine Platform and Windows Subsystem for Linux

2. **Port already in use**:
   - Change ports in `docker-compose.yml`
   - Stop conflicting services

3. **Python virtual environment issues**:
   - Ensure Python is in PATH
   - Use `python` instead of `python3`
   - Delete venv folder and recreate if corrupted

4. **Database connection failed**:
   - Verify PostgreSQL is running
   - Check credentials in `.env` file
   - Ensure PostGIS extension is installed

5. **Agent startup errors**:
   - Ensure NATS and Consul are running
   - Check agent logs in `logs/` directory
   - Verify Python dependencies are installed

### Getting Help

- Check the [main README](../README.md) for additional documentation
- Review [troubleshooting guide](../docs/deployment/troubleshooting.md)
- Create an issue on GitHub if you encounter problems

## Next Steps

Once installed, you can:

1. **Upload data** through the web interface
2. **Configure agents** for pattern discovery
3. **Run pattern analysis** on your datasets
4. **Explore results** on interactive maps and charts

For detailed usage instructions, see the [User Guide](../docs/user/getting-started.md).