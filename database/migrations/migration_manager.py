"""
A2A World Platform - Migration Manager

Simple migration system for tracking database schema changes.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Boolean, Integer

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.migrations_dir = Path(__file__).parent
        self.migration_table_name = "schema_migrations"
        
    def ensure_migrations_table(self):
        """Ensure the migrations tracking table exists."""
        try:
            with self.engine.connect() as conn:
                # Create migrations table if it doesn't exist
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS a2a_world.{self.migration_table_name} (
                        id SERIAL PRIMARY KEY,
                        migration_name VARCHAR(255) UNIQUE NOT NULL,
                        executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        success BOOLEAN NOT NULL DEFAULT FALSE,
                        error_message TEXT,
                        execution_time_ms INTEGER
                    )
                """))
                
                logger.info("Migrations tracking table ready")
                
        except Exception as e:
            logger.error(f"Error creating migrations table: {e}")
            raise
    
    def get_executed_migrations(self) -> List[str]:
        """Get list of already executed migrations."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT migration_name 
                    FROM a2a_world.{self.migration_table_name} 
                    WHERE success = TRUE
                    ORDER BY executed_at
                """))
                
                return [row[0] for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting executed migrations: {e}")
            return []
    
    def record_migration(self, migration_name: str, success: bool, 
                        error_message: str = None, execution_time_ms: int = None):
        """Record a migration execution."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO a2a_world.{self.migration_table_name} 
                    (migration_name, success, error_message, execution_time_ms)
                    VALUES (:name, :success, :error, :time_ms)
                """), {
                    "name": migration_name,
                    "success": success,
                    "error": error_message,
                    "time_ms": execution_time_ms
                })
                
        except Exception as e:
            logger.error(f"Error recording migration: {e}")
    
    def execute_migration_file(self, filepath: Path) -> bool:
        """Execute a single migration file."""
        migration_name = filepath.stem
        start_time = datetime.now()
        
        logger.info(f"Executing migration: {migration_name}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Execute the migration
            with self.engine.connect() as conn:
                # Split by semicolons and execute each statement
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement and not statement.startswith('--'):
                        conn.execute(text(statement))
            
            # Record successful migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            self.record_migration(migration_name, True, execution_time_ms=execution_time)
            
            logger.info(f"Migration {migration_name} completed successfully in {execution_time}ms")
            return True
            
        except Exception as e:
            # Record failed migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = str(e)
            self.record_migration(migration_name, False, error_msg, execution_time)
            
            logger.error(f"Migration {migration_name} failed: {error_msg}")
            return False
    
    def run_pending_migrations(self, migration_files: List[str]) -> bool:
        """Run all pending migrations."""
        # Ensure migrations table exists
        self.ensure_migrations_table()
        
        # Get already executed migrations
        executed = set(self.get_executed_migrations())
        
        # Find pending migrations
        pending = []
        for migration_file in sorted(migration_files):
            migration_name = Path(migration_file).stem
            if migration_name not in executed:
                pending.append(migration_file)
        
        if not pending:
            logger.info("No pending migrations found")
            return True
        
        logger.info(f"Found {len(pending)} pending migrations")
        
        # Execute pending migrations
        success = True
        for migration_file in pending:
            filepath = self.migrations_dir.parent / "schemas" / migration_file
            
            if not filepath.exists():
                logger.error(f"Migration file not found: {filepath}")
                success = False
                continue
            
            if not self.execute_migration_file(filepath):
                success = False
                break  # Stop on first failure
        
        return success
    
    def rollback_migration(self, migration_name: str) -> bool:
        """Rollback a specific migration (if rollback file exists)."""
        rollback_file = self.migrations_dir / f"rollback_{migration_name}.sql"
        
        if not rollback_file.exists():
            logger.error(f"Rollback file not found: {rollback_file}")
            return False
        
        logger.info(f"Rolling back migration: {migration_name}")
        
        try:
            with open(rollback_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            with self.engine.connect() as conn:
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement and not statement.startswith('--'):
                        conn.execute(text(statement))
            
            # Remove from migrations table
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    DELETE FROM a2a_world.{self.migration_table_name} 
                    WHERE migration_name = :name
                """), {"name": migration_name})
            
            logger.info(f"Migration {migration_name} rolled back successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed for {migration_name}: {e}")
            return False
    
    def get_migration_status(self) -> Dict:
        """Get current migration status."""
        try:
            with self.engine.connect() as conn:
                # Get total migrations
                result = conn.execute(text(f"""
                    SELECT COUNT(*) as total, 
                           COUNT(CASE WHEN success = TRUE THEN 1 END) as successful,
                           COUNT(CASE WHEN success = FALSE THEN 1 END) as failed
                    FROM a2a_world.{self.migration_table_name}
                """))
                
                row = result.fetchone()
                total = row[0] if row else 0
                successful = row[1] if row else 0
                failed = row[2] if row else 0
                
                # Get recent migrations
                result = conn.execute(text(f"""
                    SELECT migration_name, executed_at, success, error_message
                    FROM a2a_world.{self.migration_table_name}
                    ORDER BY executed_at DESC
                    LIMIT 10
                """))
                
                recent_migrations = [
                    {
                        "name": row[0],
                        "executed_at": row[1].isoformat() if row[1] else None,
                        "success": row[2],
                        "error": row[3]
                    }
                    for row in result.fetchall()
                ]
                
                return {
                    "total_migrations": total,
                    "successful_migrations": successful,
                    "failed_migrations": failed,
                    "recent_migrations": recent_migrations
                }
                
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return {
                "total_migrations": 0,
                "successful_migrations": 0,
                "failed_migrations": 0,
                "recent_migrations": [],
                "error": str(e)
            }


def create_new_migration(name: str, content: str = "") -> Path:
    """Create a new migration file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{name}.sql"
    filepath = Path(__file__).parent / filename
    
    if not content:
        content = f"""-- Migration: {name}
-- Created: {datetime.now().isoformat()}
-- Description: Add description here

SET search_path TO a2a_world, public;

-- Add your migration SQL here

-- Example:
-- CREATE TABLE example_table (
--     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
--     name VARCHAR(255) NOT NULL,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
-- );
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Created new migration file: {filepath}")
    return filepath