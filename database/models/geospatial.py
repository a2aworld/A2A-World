"""
A2A World Platform - Geospatial Models

SQLAlchemy models for geospatial data with PostGIS geometry support.
"""

from sqlalchemy import (
    Boolean, Column, ForeignKey, Integer, String, Text, 
    DateTime, Numeric, CheckConstraint, Date
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from .base import Base


class SacredSite(Base):
    """Sacred sites and cultural landmarks with geospatial data."""
    
    __tablename__ = "sacred_sites"
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    site_type = Column(String(100))
    culture = Column(String(100), index=True)
    time_period = Column(String(100))
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    elevation_meters = Column(Numeric(10, 2))
    significance_level = Column(Integer)
    verified = Column(Boolean, default=False)
    source_reference = Column(Text)
    metadata = Column(JSONB)
    
    # Relationships
    astronomical_alignments = relationship("AstronomicalAlignment", back_populates="site", cascade="all, delete-orphan")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("site_type IN ('temple', 'shrine', 'monument', 'burial_ground', 'ceremonial_site', 'pilgrimage_site', 'natural_sacred', 'archaeological', 'historical')", name="check_site_type"),
        CheckConstraint("significance_level BETWEEN 1 AND 5", name="check_significance_level"),
    )
    
    def __repr__(self):
        return f"<SacredSite(name='{self.name}', culture='{self.culture}', site_type='{self.site_type}')>"


class GeospatialFeature(Base):
    """Geospatial features from imported data (KML, GeoJSON, etc.)."""
    
    __tablename__ = "geospatial_features"
    
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"))
    name = Column(String(255))
    description = Column(Text)
    geometry = Column(Geometry('GEOMETRY', srid=4326), nullable=False)
    properties = Column(JSONB)
    feature_type = Column(String(100), index=True)
    source_layer = Column(String(255))
    style_info = Column(JSONB)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="geospatial_features")
    pattern_components = relationship("PatternComponent", back_populates="geospatial_feature")
    
    def __repr__(self):
        return f"<GeospatialFeature(name='{self.name}', feature_type='{self.feature_type}')>"


class GeographicRegion(Base):
    """Geographic regions and administrative boundaries."""
    
    __tablename__ = "geographic_regions"
    
    name = Column(String(255), nullable=False)
    region_type = Column(String(100), index=True)
    boundary = Column(Geometry('MULTIPOLYGON', srid=4326))
    center_point = Column(Geometry('POINT', srid=4326))
    area_sqkm = Column(Numeric(15, 2))
    population = Column(Integer)
    administrative_level = Column(Integer)
    parent_region_id = Column(UUID(as_uuid=True), ForeignKey("geographic_regions.id", ondelete="SET NULL"))
    metadata = Column(JSONB)
    
    # Self-referencing relationship
    parent_region = relationship("GeographicRegion", remote_side=[Base.id], back_populates="child_regions")
    child_regions = relationship("GeographicRegion", back_populates="parent_region")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("region_type IN ('country', 'state', 'province', 'city', 'cultural_region', 'watershed', 'mountain_range', 'desert', 'forest', 'coastline')", name="check_region_type"),
    )
    
    def __repr__(self):
        return f"<GeographicRegion(name='{self.name}', region_type='{self.region_type}')>"


class EnvironmentalData(Base):
    """Environmental data points (climate, seismic, etc.)."""
    
    __tablename__ = "environmental_data"
    
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    data_type = Column(String(100), index=True)
    measurement_value = Column(Numeric(15, 6))
    measurement_unit = Column(String(50))
    measurement_date = Column(DateTime(timezone=True), nullable=False, index=True)
    measurement_duration_hours = Column(Integer, default=0)
    quality_score = Column(Numeric(3, 2))
    source = Column(String(255), index=True)
    sensor_id = Column(String(100))
    metadata = Column(JSONB)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("data_type IN ('temperature', 'precipitation', 'humidity', 'wind_speed', 'atmospheric_pressure', 'seismic_activity', 'magnetic_field', 'solar_radiation', 'air_quality', 'vegetation_index', 'soil_composition', 'water_quality')", name="check_data_type"),
        CheckConstraint("quality_score BETWEEN 0 AND 1", name="check_quality_score"),
    )
    
    def __repr__(self):
        return f"<EnvironmentalData(data_type='{self.data_type}', measurement_date='{self.measurement_date}')>"


class EnvironmentalTimeSeries(Base):
    """Environmental time series data (partitioned by month)."""
    
    __tablename__ = "environmental_time_series"
    
    location_id = Column(UUID(as_uuid=True))
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    data_type = Column(String(100), nullable=False)
    timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    value = Column(Numeric(15, 6))
    unit = Column(String(50))
    source = Column(String(255))
    metadata = Column(JSONB)
    
    def __repr__(self):
        return f"<EnvironmentalTimeSeries(data_type='{self.data_type}', timestamp='{self.timestamp_utc}')>"


class LeyLine(Base):
    """Ley lines and energy grid data."""
    
    __tablename__ = "ley_lines"
    
    name = Column(String(255))
    description = Column(Text)
    line_geometry = Column(Geometry('LINESTRING', srid=4326), nullable=False)
    strength_rating = Column(Numeric(3, 2))
    discovery_method = Column(String(100))
    validation_status = Column(String(50), default="unverified")
    connected_sites = Column(ARRAY(UUID(as_uuid=True)))
    researcher_notes = Column(Text)
    metadata = Column(JSONB)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("strength_rating BETWEEN 0 AND 10", name="check_strength_rating"),
        CheckConstraint("validation_status IN ('unverified', 'pending_validation', 'validated', 'disputed', 'rejected')", name="check_validation_status"),
    )
    
    def __repr__(self):
        return f"<LeyLine(name='{self.name}', validation_status='{self.validation_status}')>"


class GeologicalFeature(Base):
    """Geological formations and features."""
    
    __tablename__ = "geological_features"
    
    name = Column(String(255), nullable=False)
    feature_type = Column(String(100), index=True)
    geometry = Column(Geometry('GEOMETRY', srid=4326), nullable=False)
    elevation_meters = Column(Numeric(10, 2))
    geological_age = Column(String(100))
    rock_type = Column(String(100))
    formation_process = Column(String(255))
    cultural_significance = Column(Text)
    metadata = Column(JSONB)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("feature_type IN ('mountain', 'volcano', 'cave', 'spring', 'fault_line', 'crater', 'canyon', 'plateau', 'ridge', 'valley', 'cliff', 'rock_formation')", name="check_feature_type"),
    )
    
    def __repr__(self):
        return f"<GeologicalFeature(name='{self.name}', feature_type='{self.feature_type}')>"


class AstronomicalAlignment(Base):
    """Astronomical alignments and celestial correlations."""
    
    __tablename__ = "astronomical_alignments"
    
    site_id = Column(UUID(as_uuid=True), ForeignKey("sacred_sites.id", ondelete="CASCADE"))
    alignment_type = Column(String(100))
    celestial_body = Column(String(100))
    alignment_direction = Column(Geometry('LINESTRING', srid=4326))
    azimuth_degrees = Column(Numeric(6, 3))
    elevation_degrees = Column(Numeric(5, 2))
    alignment_date = Column(Date)
    precision_arc_seconds = Column(Numeric(8, 2))
    verification_status = Column(String(50), default="unverified")
    researcher_notes = Column(Text)
    metadata = Column(JSONB)
    
    # Relationships
    site = relationship("SacredSite", back_populates="astronomical_alignments")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("alignment_type IN ('solar_solstice', 'solar_equinox', 'lunar_standstill', 'star_alignment', 'constellation_alignment', 'planetary_alignment', 'eclipse_alignment')", name="check_alignment_type"),
        CheckConstraint("azimuth_degrees >= 0 AND azimuth_degrees < 360", name="check_azimuth"),
        CheckConstraint("elevation_degrees >= -90 AND elevation_degrees <= 90", name="check_elevation"),
    )
    
    def __repr__(self):
        return f"<AstronomicalAlignment(alignment_type='{self.alignment_type}', celestial_body='{self.celestial_body}')>"