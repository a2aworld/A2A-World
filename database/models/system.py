"""
A2A World Platform - System Models

SQLAlchemy models for system logging and monitoring.
"""

from sqlalchemy import Column, ForeignKey, String, Text, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base


class SystemLog(Base):
    """System log model for monitoring and debugging."""
    
    __tablename__ = "system_logs"
    
    log_level = Column(String(20), nullable=False)
    component = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    metadata = Column(JSONB)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # Relationships
    user = relationship("User", back_populates="system_logs")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name="check_log_level"),
    )
    
    def __repr__(self):
        return f"<SystemLog(level='{self.log_level}', component='{self.component}', created_at='{self.created_at}')>"
    
    @classmethod
    def log_info(cls, component: str, message: str, metadata: dict = None, user_id: str = None):
        """Helper method to create info log entry."""
        return cls(
            log_level="INFO",
            component=component,
            message=message,
            metadata=metadata,
            user_id=user_id
        )
    
    @classmethod
    def log_error(cls, component: str, message: str, metadata: dict = None, user_id: str = None):
        """Helper method to create error log entry."""
        return cls(
            log_level="ERROR",
            component=component,
            message=message,
            metadata=metadata,
            user_id=user_id
        )
    
    @classmethod
    def log_warning(cls, component: str, message: str, metadata: dict = None, user_id: str = None):
        """Helper method to create warning log entry."""
        return cls(
            log_level="WARNING",
            component=component,
            message=message,
            metadata=metadata,
            user_id=user_id
        )