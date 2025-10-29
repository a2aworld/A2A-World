"""
A2A World Platform - Database Initialization Script

Script to initialize the PostgreSQL + PostGIS database with all schemas.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Database initialization and migration manager."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.schemas_dir = Path(__file__).parent.parent / "schemas"
        self.engine = None
        
        # Schema files in execution order
        self.schema_files = [
            "001_initial_schema.sql",
            "002_geospatial_data.sql", 
            "003_pattern_discovery.sql",
            "004_agent_system.sql",
            "005_cultural_data.sql"
        ]
    
    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist."""
        # Parse database URL to get components
        from urllib.parse import urlparse
        parsed = urlparse(self.database_url)
        
        # Connect to postgres database to create the target database
        admin_db_url = f"postgresql://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}/postgres"
        target_db_name = parsed.path.lstrip('/')
        
        logger.info(f"Checking if database '{target_db_name}' exists...")
        
        try:
            # Connect to admin database
            conn = psycopg2.connect(admin_db_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (target_db_name,)
            )
            
            if cursor.fetchone():
                logger.info(f"Database '{target_db_name}' already exists")
            else:
                logger.info(f"Creating database '{target_db_name}'...")
                cursor.execute(f'CREATE DATABASE "{target_db_name}"')
                logger.info(f"Database '{target_db_name}' created successfully")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    def connect_to_database(self):
        """Connect to the target database."""
        try:
            self.engine = create_engine(
                self.database_url,
                echo=False,  # Set to True for SQL logging
                isolation_level="AUTOCOMMIT"
            )
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def check_postgis_extension(self) -> bool:
        """Check if PostGIS extension is available and enabled."""
        try:
            with self.engine.connect() as conn:
                # Check if PostGIS extension exists
                result = conn.execute(text(
                    "SELECT name FROM pg_available_extensions WHERE name = 'postgis'"
                ))
                
                if not result.fetchone():
                    logger.error("PostGIS extension is not available. Please install PostGIS.")
                    return False
                
                # Check if PostGIS is already enabled
                result = conn.execute(text(
                    "SELECT extname FROM pg_extension WHERE extname = 'postgis'"
                ))
                
                if result.fetchone():
                    logger.info("PostGIS extension is already enabled")
                    
                    # Get PostGIS version
                    result = conn.execute(text("SELECT PostGIS_Version()"))
                    version = result.fetchone()[0]
                    logger.info(f"PostGIS version: {version}")
                else:
                    logger.info("PostGIS extension not yet enabled")
                
                return True
                
        except Exception as e:
            logger.error(f"Error checking PostGIS extension: {e}")
            return False
    
    def execute_sql_file(self, filepath: Path) -> bool:
        """Execute a SQL file."""
        try:
            logger.info(f"Executing SQL file: {filepath.name}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Split by semicolons and execute each statement
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            with self.engine.connect() as conn:
                for i, statement in enumerate(statements):
                    if statement and not statement.startswith('--'):
                        try:
                            conn.execute(text(statement))
                            logger.debug(f"Executed statement {i+1}/{len(statements)}")
                        except Exception as e:
                            logger.warning(f"Error in statement {i+1}: {e}")
                            logger.debug(f"Statement was: {statement[:100]}...")
                            # Continue with other statements
            
            logger.info(f"Successfully executed {filepath.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing SQL file {filepath.name}: {e}")
            return False
    
    def run_migrations(self) -> bool:
        """Run all migration files in order."""
        logger.info("Starting database migrations...")
        
        success = True
        for schema_file in self.schema_files:
            filepath = self.schemas_dir / schema_file
            
            if not filepath.exists():
                logger.error(f"Schema file not found: {filepath}")
                success = False
                continue
            
            if not self.execute_sql_file(filepath):
                success = False
                # Continue with other files for now
        
        if success:
            logger.info("All migrations completed successfully")
        else:
            logger.error("Some migrations failed - check logs above")
        
        return success
    
    def verify_schema(self) -> bool:
        """Verify that all expected tables exist."""
        logger.info("Verifying database schema...")
        
        expected_tables = [
            # Core tables
            'users', 'datasets', 'system_logs',
            # Geospatial tables
            'sacred_sites', 'geospatial_features', 'geographic_regions', 
            'environmental_data', 'environmental_time_series', 'ley_lines',
            'geological_features', 'astronomical_alignments',
            # Pattern discovery tables
            'patterns', 'pattern_components', 'clustering_results',
            'spatial_analysis', 'cross_correlations', 'pattern_validations',
            'pattern_relationships', 'pattern_evolution',
            # Agent system tables
            'agents', 'agent_tasks', 'agent_metrics', 'agent_communications',
            'agent_collaborations', 'resource_locks', 'system_health', 'agent_profiles',
            # Cultural data tables
            'cultural_traditions', 'mythological_narratives', 'mythological_entities',
            'cultural_patterns', 'narrative_patterns', 'cultural_relationships',
            'cultural_interpretations', 'cultural_relevance', 'linguistic_analysis'
        ]
        
        try:
            with self.engine.connect() as conn:
                # Check if schema exists
                result = conn.execute(text(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'a2a_world'"
                ))
                
                if not result.fetchone():
                    logger.error("Schema 'a2a_world' not found")
                    return False
                
                # Check if tables exist
                missing_tables = []
                for table in expected_tables:
                    result = conn.execute(text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'a2a_world' AND table_name = :table_name"
                    ), {"table_name": table})
                    
                    if not result.fetchone():
                        missing_tables.append(table)
                
                if missing_tables:
                    logger.error(f"Missing tables: {missing_tables}")
                    return False
                
                # Check PostGIS functions
                result = conn.execute(text("SELECT PostGIS_Version()"))
                postgis_version = result.fetchone()[0]
                logger.info(f"PostGIS version verified: {postgis_version}")
                
                logger.info(f"Schema verification successful - found {len(expected_tables)} tables")
                return True
                
        except Exception as e:
            logger.error(f"Error verifying schema: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Complete database initialization process."""
        logger.info("Starting A2A World database initialization...")
        
        try:
            # Step 1: Create database if needed
            self.create_database_if_not_exists()
            
            # Step 2: Connect to database
            self.connect_to_database()
            
            # Step 3: Check PostGIS
            if not self.check_postgis_extension():
                return False
            
            # Step 4: Run migrations
            if not self.run_migrations():
                return False
            
            # Step 5: Verify schema
            if not self.verify_schema():
                return False
            
            logger.info("Database initialization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
        
        finally:
            if self.engine:
                self.engine.dispose()


def main():
    """Main entry point for database initialization."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('database_init.log')
        ]
    )
    
    # Initialize database
    initializer = DatabaseInitializer()
    success = initializer.initialize_database()
    
    if success:
        print("\n✅ Database initialization completed successfully!")
        print("You can now start the A2A World application.")
        sys.exit(0)
    else:
        print("\n❌ Database initialization failed!")
        print("Check the logs above and database_init.log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()