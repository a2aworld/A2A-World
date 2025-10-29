"""
A2A World Platform - Database Models for Pattern Discovery

SQLAlchemy models for pattern discovery database tables.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, ForeignKey
from sqlalchemy import DECIMAL, ARRAY, JSON, CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

Base = declarative_base()


class Pattern(Base):
    """Discovered patterns with statistical validation."""
    
    __tablename__ = "patterns"
    __table_args__ = {"schema": "a2a_world"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    pattern_type = Column(String(100), CheckConstraint(
        "pattern_type IN ('spatial_clustering', 'temporal_correlation', 'cultural_alignment', "
        "'astronomical_correlation', 'geometric_pattern', 'energy_grid', "
        "'environmental_correlation', 'mythological_correlation')"
    ))
    confidence_score = Column(DECIMAL(5,4), CheckConstraint("confidence_score >= 0 AND confidence_score <= 1"))
    statistical_significance = Column(DECIMAL(10,8))  # p-value
    effect_size = Column(DECIMAL(8,4))  # Cohen's d or similar
    sample_size = Column(Integer)
    algorithm_used = Column(String(100))
    algorithm_version = Column(String(50))
    parameters = Column(JSONB)
    discovery_region = Column(Geometry('POLYGON', srid=4326))
    validation_status = Column(String(50), default='pending')
    validation_consensus_score = Column(DECIMAL(5,4))
    reproducibility_score = Column(DECIMAL(5,4))
    discovered_by_agent = Column(String(255), nullable=False)
    discovery_timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_validated = Column(DateTime(timezone=True))
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    components = relationship("PatternComponent", back_populates="pattern", cascade="all, delete-orphan")
    clustering_results = relationship("ClusteringResult", back_populates="pattern", cascade="all, delete-orphan")
    spatial_analyses = relationship("SpatialAnalysis", back_populates="pattern", cascade="all, delete-orphan")
    validations = relationship("PatternValidation", back_populates="pattern", cascade="all, delete-orphan")


class PatternComponent(Base):
    """Pattern components - the actual data points that make up a pattern."""
    
    __tablename__ = "pattern_components"
    __table_args__ = {"schema": "a2a_world"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('a2a_world.patterns.id', ondelete='CASCADE'), nullable=False)
    component_type = Column(String(100), CheckConstraint(
        "component_type IN ('sacred_site', 'geospatial_feature', 'environmental_data', 'cultural_data', "
        "'geological_feature', 'astronomical_alignment', 'ley_line')"
    ))
    component_id = Column(UUID(as_uuid=True), nullable=False)
    relevance_score = Column(DECIMAL(5,4), CheckConstraint("relevance_score >= 0 AND relevance_score <= 1"))
    component_role = Column(String(100))  # 'anchor_point', 'connector', 'outlier', 'center'
    distance_to_center = Column(DECIMAL(15,2))  # Distance in meters
    contribution_weight = Column(DECIMAL(5,4))
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="components")


class ClusteringResult(Base):
    """Clustering analysis results."""
    
    __tablename__ = "clustering_results"
    __table_args__ = {"schema": "a2a_world"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('a2a_world.patterns.id', ondelete='CASCADE'), nullable=False)
    clustering_algorithm = Column(String(100), CheckConstraint(
        "clustering_algorithm IN ('kmeans', 'dbscan', 'hierarchical', 'gaussian_mixture', 'spectral', "
        "'affinity_propagation', 'mean_shift', 'optics')"
    ))
    num_clusters = Column(Integer)
    silhouette_score = Column(DECIMAL(5,4))
    calinski_harabasz_score = Column(DECIMAL(10,4))
    davies_bouldin_score = Column(DECIMAL(8,4))
    inertia = Column(DECIMAL(15,4))
    cluster_centers = Column(JSONB)  # Array of cluster center coordinates
    cluster_labels = Column(ARRAY(Integer))  # Array of cluster labels
    outliers_detected = Column(Integer, default=0)
    algorithm_parameters = Column(JSONB)
    execution_time_ms = Column(Integer)
    data_dimensions = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="clustering_results")


class SpatialAnalysis(Base):
    """Spatial analysis results for geographic patterns."""
    
    __tablename__ = "spatial_analysis"
    __table_args__ = {"schema": "a2a_world"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('a2a_world.patterns.id', ondelete='CASCADE'), nullable=False)
    analysis_type = Column(String(100), CheckConstraint(
        "analysis_type IN ('nearest_neighbor', 'moran_i', 'getis_ord', 'ripley_k', 'g_function', "
        "'hotspot_analysis', 'spatial_autocorrelation', 'spatial_regression')"
    ))
    test_statistic = Column(DECIMAL(15,8))
    p_value = Column(DECIMAL(15,12))
    z_score = Column(DECIMAL(10,6))
    expected_value = Column(DECIMAL(15,8))
    variance = Column(DECIMAL(15,8))
    analysis_result = Column(String(100))  # 'clustered', 'dispersed', 'random'
    significance_level = Column(DECIMAL(3,2), default=0.05)
    analysis_parameters = Column(JSONB)
    spatial_weights_matrix = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="spatial_analyses")


class PatternValidation(Base):
    """Pattern validation results from different validators."""
    
    __tablename__ = "pattern_validations"
    __table_args__ = {"schema": "a2a_world"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('a2a_world.patterns.id', ondelete='CASCADE'), nullable=False)
    validation_type = Column(String(100), CheckConstraint(
        "validation_type IN ('statistical_test', 'cross_validation', 'bootstrap', 'permutation_test', "
        "'expert_review', 'peer_review', 'replication_study', 'sensitivity_analysis')"
    ))
    validator_type = Column(String(50), CheckConstraint(
        "validator_type IN ('agent', 'human_expert', 'peer_reviewer', 'automated_system')"
    ))
    validator_id = Column(String(255))
    validation_result = Column(String(50), CheckConstraint(
        "validation_result IN ('approved', 'rejected', 'needs_revision', 'inconclusive')"
    ))
    validation_score = Column(DECIMAL(5,4), CheckConstraint("validation_score >= 0 AND validation_score <= 1"))
    confidence_level = Column(DECIMAL(5,4))
    validation_method = Column(Text)
    test_statistics = Column(JSONB)
    validation_notes = Column(Text)
    replication_successful = Column(Boolean)
    limitations_noted = Column(Text)
    recommendations = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="validations")


class SacredSite(Base):
    """Sacred sites for pattern discovery."""
    
    __tablename__ = "sacred_sites"
    __table_args__ = {"schema": "a2a_world"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    site_type = Column(String(100))
    location = Column(Geometry('POINT', srid=4326))
    latitude = Column(DECIMAL(10,8))
    longitude = Column(DECIMAL(11,8))
    elevation = Column(DECIMAL(10,2))
    cultural_context = Column(String(255))
    historical_period = Column(String(100))
    significance_level = Column(Integer, CheckConstraint("significance_level >= 1 AND significance_level <= 5"))
    verification_status = Column(String(50), default='unverified')
    data_source = Column(String(255))
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)