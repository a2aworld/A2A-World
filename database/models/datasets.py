"""
A2A World Platform - Dataset Models

SQLAlchemy models for dataset management and file uploads.
"""

from sqlalchemy import BigInteger, Column, ForeignKey, String, Text, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base


class Dataset(Base):
    """Dataset model for uploaded data files."""
    
    __tablename__ = "datasets"
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    file_path = Column(String(500))
    file_type = Column(String(50))
    file_size = Column(BigInteger)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    status = Column(String(50), default="pending")
    processing_log = Column(Text)
    metadata = Column(JSONB)
    
    # Relationships
    uploaded_by_user = relationship("User", back_populates="datasets")
    geospatial_features = relationship("GeospatialFeature", back_populates="dataset", cascade="all, delete-orphan")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("file_type IN ('kml', 'geojson', 'csv', 'gpx', 'shp')", name="check_file_type"),
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')", name="check_status"),
    )
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', file_type='{self.file_type}', status='{self.status}')>"
    
    @property
    def is_processed(self) -> bool:
        """Check if dataset has been successfully processed."""
        return self.status == "completed"
    
    @property
    def is_processing(self) -> bool:
        """Check if dataset is currently being processed."""
        return self.status in ["pending", "processing"]
    
    @property
    def has_failed(self) -> bool:
        """Check if dataset processing has failed."""
        return self.status == "failed"