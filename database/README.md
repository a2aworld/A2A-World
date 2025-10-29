# A2A World Platform - Database System

This directory contains the complete PostgreSQL + PostGIS database schema and models for the A2A World platform's geospatial data storage system.

## 🗃️ Structure

```
database/
├── models/              # SQLAlchemy ORM models
│   ├── __init__.py     # Model exports
│   ├── base.py         # Base model class
│   ├── users.py        # User authentication
│   ├── datasets.py     # Data management
│   ├── geospatial.py   # Spatial data models
│   ├── patterns.py     # Pattern discovery models
│   ├── agents.py       # Agent system models
│   ├── cultural.py     # Cultural/mythological data
│   └── system.py       # System logging
├── schemas/            # SQL schema definitions
│   ├── 001_initial_schema.sql      # Core tables & PostGIS
│   ├── 002_geospatial_data.sql     # Spatial data tables
│   ├── 003_pattern_discovery.sql   # Pattern & analysis tables
│   ├── 004_agent_system.sql        # Agent management tables
│   └── 005_cultural_data.sql       # Cultural data tables
├── migrations/         # Database migration system
│   └── migration_manager.py
├── scripts/           # Database utilities
│   └── init_database.py
├── connection.py      # Database connection utilities
└── README.md         # This file
```

## 🚀 Quick Start

### 1. Initialize Database

```bash
# Start PostgreSQL with PostGIS
docker-compose up -d postgres

# Initialize database schema
python database/scripts/init_database.py
```

### 2. Connect from FastAPI

```python
from database.connection import init_database, get_db
from database.models import User, Pattern, SacredSite

# Initialize database connection
init_database()

# Use in FastAPI endpoints
from fastapi import Depends
from sqlalchemy.orm import Session

@app.get("/patterns")
def get_patterns(db: Session = Depends(get_db)):
    return db.query(Pattern).all()
```

## 📊 Database Schema

### Core Tables

- **`users`** - User authentication and authorization
- **`datasets`** - Uploaded data files and metadata
- **`system_logs`** - Application logging and monitoring

### Geospatial Data

- **`sacred_sites`** - Cultural landmarks with geographic coordinates
- **`geospatial_features`** - KML/GeoJSON imported features
- **`geographic_regions`** - Administrative boundaries and regions
- **`environmental_data`** - Environmental measurements and sensors
- **`ley_lines`** - Energy grid and spiritual alignment data
- **`geological_features`** - Mountains, caves, formations
- **`astronomical_alignments`** - Celestial correlations with sites

### Pattern Discovery

- **`patterns`** - Discovered patterns with statistical validation
- **`pattern_components`** - Individual data points in patterns  
- **`clustering_results`** - Machine learning clustering analysis
- **`spatial_analysis`** - Geospatial statistical analysis
- **`cross_correlations`** - Multi-modal data correlations
- **`pattern_validations`** - Expert and peer review results
- **`pattern_relationships`** - How patterns relate to each other

### Agent System

- **`agents`** - Registered autonomous agents
- **`agent_tasks`** - Task queue and coordination
- **`agent_metrics`** - Performance monitoring
- **`agent_communications`** - Inter-agent messaging
- **`resource_locks`** - Shared resource management
- **`system_health`** - Infrastructure monitoring

### Cultural Data

- **`cultural_traditions`** - Mythologies and cultural systems
- **`mythological_narratives`** - Stories and texts
- **`mythological_entities`** - Gods, spirits, beings
- **`cultural_patterns`** - Cross-cultural archetypal patterns
- **`cultural_relationships`** - Knowledge graph connections
- **`linguistic_analysis`** - Etymology and language analysis

## 🗂️ Key Features

### PostGIS Integration
- Full spatial data support with geometry columns
- Efficient spatial indexing (GIST indexes)
- Support for points, lines, polygons, and complex geometries
- Spatial analysis functions (ST_DWithin, ST_Intersects, etc.)

### Performance Optimization
- Strategic indexing on frequently queried columns
- JSONB indexes for flexible metadata searches
- Partitioning for time-series data
- Connection pooling and optimization

### Data Integrity
- Foreign key constraints and referential integrity
- Check constraints for data validation
- Proper CASCADE and SET NULL behaviors
- Transaction safety and ACID compliance

### Scalability Design
- Prepared for horizontal scaling
- Efficient query patterns
- Partitioning strategies for large datasets
- Resource usage optimization

## 🛠️ Development

### Adding New Models

1. Create model in appropriate file under `models/`
2. Add corresponding SQL schema to `schemas/`
3. Update `models/__init__.py` exports
4. Run migrations to update database

### Running Migrations

```python
from database.migrations.migration_manager import MigrationManager
from api.app.core.config import settings

manager = MigrationManager(settings.DATABASE_URL)
manager.run_pending_migrations([
    "001_initial_schema.sql",
    "002_geospatial_data.sql", 
    "003_pattern_discovery.sql",
    "004_agent_system.sql",
    "005_cultural_data.sql"
])
```

### Database Health Check

```python
from database.connection import health_check

status = health_check()
print(f"Database status: {status['status']}")
print(f"PostGIS version: {status.get('postgis_version')}")
```

## 🔧 Configuration

### Environment Variables

```bash
# Database Connection
DATABASE_URL=postgresql://a2a_user:a2a_password@localhost:5432/a2a_world
POSTGRES_SERVER=localhost
POSTGRES_USER=a2a_user
POSTGRES_PASSWORD=a2a_password
POSTGRES_DB=a2a_world
POSTGRES_PORT=5432
```

### Docker Compose

The database runs in a PostGIS container with automatic schema initialization:

```yaml
postgres:
  image: postgis/postgis:15-3.3
  environment:
    POSTGRES_DB: a2a_world
    POSTGRES_USER: a2a_user  
    POSTGRES_PASSWORD: a2a_password
  volumes:
    - postgres_data:/var/lib/postgresql/data
    - ./database/schemas:/docker-entrypoint-initdb.d
```

## 📈 Monitoring

### Database Metrics
- Connection pool status
- Query performance statistics  
- Storage usage and growth
- Index effectiveness

### Health Checks
- PostGIS extension status
- Table schema validation
- Connection availability
- Query response times

## 🔒 Security

- Password hashing with bcrypt
- SQL injection prevention via parameterized queries
- Connection encryption support
- Role-based access control ready
- Audit logging capability

## 🧪 Testing

Run database tests with:

```bash
# Start test database
docker-compose -f docker-compose.test.yml up -d postgres-test

# Run tests
pytest tests/database/ -v

# Clean up
docker-compose -f docker-compose.test.yml down -v
```

## 📚 Resources

- [PostGIS Documentation](https://postgis.net/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [GeoAlchemy2 Documentation](https://geoalchemy-2.readthedocs.io/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## 🤝 Contributing

1. Follow existing naming conventions
2. Add appropriate indexes for new queries
3. Include foreign key relationships
4. Update this README for schema changes
5. Write tests for new functionality

---

The A2A World database system is designed to support the platform's multi-agent pattern discovery across cultural, geospatial, and temporal dimensions while maintaining high performance and data integrity.