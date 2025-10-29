"""
A2A World Platform - User Models

SQLAlchemy models for user authentication and management.
"""

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.orm import relationship

from .base import Base


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True, index=True)
    is_superuser = Column(Boolean, default=False)
    email_verified = Column(Boolean, default=False)
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    datasets = relationship("Dataset", back_populates="uploaded_by_user")
    system_logs = relationship("SystemLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return True
    
    @property
    def is_anonymous(self) -> bool:
        """Check if user is anonymous."""
        return False
    
    def get_id(self) -> str:
        """Get user ID for session management."""
        return str(self.id)