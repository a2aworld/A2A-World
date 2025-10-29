"""
A2A World Platform - Database Connection

Database connection utilities for PostgreSQL with PostGIS.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from api.app.core.config import settings

logger = logging.getLogger(__name__)

# Database engine
engine = None
SessionLocal = None


def create_database_engine(database_url: str = None) -> Engine:
    """Create database engine with optimal configuration."""
    global engine
    
    if database_url is None:
        database_url = settings.DATABASE_URL
    
    logger.info(f"Creating database engine for: {database_url.split('@')[1] if '@' in database_url else database_url}")
    
    engine = create_engine(
        database_url,
        # Connection pool settings
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections after 1 hour
        
        # SQLAlchemy settings
        echo=False,  # Set to True for SQL logging in development
        echo_pool=False,
        future=True,
        
        # PostgreSQL specific settings
        connect_args={
            "options": "-c default_transaction_isolation=read_committed"
        }
    )
    
    # Add event listeners
    setup_engine_events(engine)
    
    return engine


def setup_engine_events(engine: Engine):
    """Setup database engine event listeners."""
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set up connection-specific settings."""
        # This would be for PostgreSQL-specific settings if needed
        pass
    
    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_connection, connection_record, connection_proxy):
        """Handle connection checkout."""
        logger.debug("Database connection checked out from pool")
    
    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_connection, connection_record):
        """Handle connection checkin."""
        logger.debug("Database connection returned to pool")


def create_session_factory() -> sessionmaker:
    """Create session factory."""
    global SessionLocal
    
    if engine is None:
        raise RuntimeError("Database engine not initialized. Call create_database_engine() first.")
    
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False
    )
    
    return SessionLocal


def get_database_session() -> Generator[Session, None, None]:
    """Get database session dependency for FastAPI."""
    if SessionLocal is None:
        raise RuntimeError("Session factory not initialized. Call create_session_factory() first.")
    
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    if SessionLocal is None:
        raise RuntimeError("Session factory not initialized. Call create_session_factory() first.")
    
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_database():
    """Initialize database connection."""
    global engine, SessionLocal
    
    try:
        # Create engine
        engine = create_database_engine()
        
        # Create session factory
        SessionLocal = create_session_factory()
        
        # Test connection
        with engine.connect() as connection:
            result = connection.execute("SELECT 1")
            logger.info("Database connection successful")
            
        # Test PostGIS extension
        with engine.connect() as connection:
            result = connection.execute("SELECT PostGIS_Version()")
            postgis_version = result.fetchone()[0]
            logger.info(f"PostGIS version: {postgis_version}")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def close_database():
    """Close database connections."""
    global engine
    
    if engine:
        engine.dispose()
        logger.info("Database connections closed")


def health_check() -> dict:
    """Perform database health check."""
    try:
        with engine.connect() as connection:
            # Basic connectivity test
            result = connection.execute("SELECT 1")
            
            # PostGIS test
            postgis_result = connection.execute("SELECT PostGIS_Version()")
            postgis_version = postgis_result.fetchone()[0]
            
            # Get connection pool stats
            pool = engine.pool
            pool_status = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
            
            return {
                "status": "healthy",
                "postgis_version": postgis_version,
                "pool_status": pool_status
            }
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Database session dependency for FastAPI
def get_db():
    """FastAPI dependency for database sessions."""
    return get_database_session()