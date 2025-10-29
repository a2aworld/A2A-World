"""
A2A World Platform - Base Database Model

Base SQLAlchemy model with common functionality.
"""

import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import Session


class BaseModel:
    """Base model class with common attributes and methods."""
    
    @declared_attr
    def __tablename__(cls):
        """Generate table name from class name."""
        return cls.__name__.lower()
    
    # Common primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Timestamp fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model instance from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def create(cls, db: Session, **kwargs) -> "BaseModel":
        """Create new instance and save to database."""
        instance = cls(**kwargs)
        db.add(instance)
        db.commit()
        db.refresh(instance)
        return instance
    
    def save(self, db: Session) -> "BaseModel":
        """Save instance to database."""
        db.add(self)
        db.commit()
        db.refresh(self)
        return self
    
    def delete(self, db: Session) -> None:
        """Delete instance from database."""
        db.delete(self)
        db.commit()


# Base declarative class
Base = declarative_base(cls=BaseModel)

# Set schema for all tables
Base.metadata.schema = "a2a_world"