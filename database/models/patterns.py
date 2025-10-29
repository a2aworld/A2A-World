"""
A2A World Platform - Pattern Discovery Models

SQLAlchemy models for pattern discovery, clustering, and statistical validation.
"""

from sqlalchemy import (
    Boolean, Column, ForeignKey, Integer, String, Text, 
    DateTime, Numeric, CheckConstraint
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from .base import Base


class Pattern(Base):
    """Discovered patterns with statistical validation."""
    
    __tablename__ = "patterns"
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    pattern_type = Column(String(100), index=True)
    confidence_score = Column(Numeric(5, 4))
    statistical_significance = Column(Numeric(10, 8))  # p-value
    effect_size = Column(Numeric(8, 4))  # Cohen's d or similar
    sample_size = Column(Integer)
    algorithm_used = Column(String(100))
    algorithm_version = Column(String(50))
    parameters = Column(JSONB)
    discovery_region = Column(Geometry('POLYGON', srid=4326))
    validation_status = Column(String(50), default="pending")
    validation_consensus_score = Column(Numeric(5, 4))
    reproducibility_score = Column(Numeric(5, 4))
    discovered_by_agent = Column(String(255), nullable=False)
    discovery_timestamp = Column(DateTime(timezone=True))
    last_validated = Column(DateTime(timezone=True))
    metadata = Column(JSONB)
    
    # Relationships
    pattern_components = relationship("PatternComponent", back_populates="pattern", cascade="all, delete-orphan")
    clustering_results = relationship("ClusteringResult", back_populates="pattern", cascade="all, delete-orphan")
    spatial_analyses = relationship("SpatialAnalysis", back_populates="pattern", cascade="all, delete-orphan")
    cross_correlations = relationship("CrossCorrelation", back_populates="pattern", cascade="all, delete-orphan")
    validations = relationship("PatternValidation", back_populates="pattern", cascade="all, delete-orphan")
    primary_relationships = relationship("PatternRelationship", foreign_keys="PatternRelationship.primary_pattern_id", back_populates="primary_pattern")
    related_relationships = relationship("PatternRelationship", foreign_keys="PatternRelationship.related_pattern_id", back_populates="related_pattern")
    evolution_history = relationship("PatternEvolution", back_populates="pattern", cascade="all, delete-orphan")
    cultural_relevance = relationship("CulturalRelevance", back_populates="pattern")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("pattern_type IN ('spatial_clustering', 'temporal_correlation', 'cultural_alignment', 'astronomical_correlation', 'geometric_pattern', 'energy_grid', 'environmental_correlation', 'mythological_correlation')", name="check_pattern_type"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_score"),
        CheckConstraint("validation_status IN ('pending', 'validated', 'rejected', 'needs_review', 'partially_validated')", name="check_validation_status"),
    )
    
    def __repr__(self):
        return f"<Pattern(name='{self.name}', pattern_type='{self.pattern_type}', confidence={self.confidence_score})>"


class PatternComponent(Base):
    """Pattern components - the actual data points that make up a pattern."""
    
    __tablename__ = "pattern_components"
    
    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    component_type = Column(String(100))
    component_id = Column(UUID(as_uuid=True), nullable=False)
    relevance_score = Column(Numeric(5, 4))
    component_role = Column(String(100))  # 'anchor_point', 'connector', 'outlier', 'center'
    distance_to_center = Column(Numeric(15, 2))  # Distance in meters
    contribution_weight = Column(Numeric(5, 4))
    metadata = Column(JSONB)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="pattern_components")
    sacred_site = relationship("SacredSite", foreign_keys=[component_id], primaryjoin="and_(PatternComponent.component_id==SacredSite.id, PatternComponent.component_type=='sacred_site')", viewonly=True)
    geospatial_feature = relationship("GeospatialFeature", foreign_keys=[component_id], primaryjoin="and_(PatternComponent.component_id==GeospatialFeature.id, PatternComponent.component_type=='geospatial_feature')", viewonly=True)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("component_type IN ('sacred_site', 'geospatial_feature', 'environmental_data', 'cultural_data', 'geological_feature', 'astronomical_alignment', 'ley_line')", name="check_component_type"),
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="check_relevance_score"),
    )
    
    def __repr__(self):
        return f"<PatternComponent(pattern_id='{self.pattern_id}', component_type='{self.component_type}')>"


class ClusteringResult(Base):
    """Clustering analysis results."""
    
    __tablename__ = "clustering_results"
    
    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    clustering_algorithm = Column(String(100))
    num_clusters = Column(Integer)
    silhouette_score = Column(Numeric(5, 4))  # Clustering quality metric
    calinski_harabasz_score = Column(Numeric(10, 4))
    davies_bouldin_score = Column(Numeric(8, 4))
    inertia = Column(Numeric(15, 4))
    cluster_centers = Column(JSONB)  # Array of cluster center coordinates
    cluster_labels = Column(ARRAY(Integer))  # Array of cluster labels
    outliers_detected = Column(Integer, default=0)
    algorithm_parameters = Column(JSONB)
    execution_time_ms = Column(Integer)
    data_dimensions = Column(Integer)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="clustering_results")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("clustering_algorithm IN ('kmeans', 'dbscan', 'hierarchical', 'gaussian_mixture', 'spectral', 'affinity_propagation', 'mean_shift', 'optics')", name="check_clustering_algorithm"),
    )
    
    def __repr__(self):
        return f"<ClusteringResult(algorithm='{self.clustering_algorithm}', num_clusters={self.num_clusters})>"


class SpatialAnalysis(Base):
    """Spatial analysis results for geographic patterns."""
    
    __tablename__ = "spatial_analysis"
    
    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    analysis_type = Column(String(100))
    test_statistic = Column(Numeric(15, 8))
    p_value = Column(Numeric(15, 12))
    z_score = Column(Numeric(10, 6))
    expected_value = Column(Numeric(15, 8))
    variance = Column(Numeric(15, 8))
    analysis_result = Column(String(100))  # 'clustered', 'dispersed', 'random'
    significance_level = Column(Numeric(3, 2), default=0.05)
    analysis_parameters = Column(JSONB)
    spatial_weights_matrix = Column(JSONB)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="spatial_analyses")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("analysis_type IN ('nearest_neighbor', 'moran_i', 'getis_ord', 'ripley_k', 'g_function', 'hotspot_analysis', 'spatial_autocorrelation', 'spatial_regression')", name="check_analysis_type"),
    )
    
    def __repr__(self):
        return f"<SpatialAnalysis(analysis_type='{self.analysis_type}', p_value={self.p_value})>"


class CrossCorrelation(Base):
    """Cross-correlation analysis between different data types."""
    
    __tablename__ = "cross_correlations"
    
    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    primary_data_type = Column(String(100))
    secondary_data_type = Column(String(100))
    correlation_coefficient = Column(Numeric(8, 6))
    correlation_type = Column(String(50))
    p_value = Column(Numeric(15, 12))
    confidence_interval_lower = Column(Numeric(8, 6))
    confidence_interval_upper = Column(Numeric(8, 6))
    sample_size = Column(Integer)
    lag_periods = Column(Integer, default=0)
    temporal_window_days = Column(Integer)
    spatial_window_km = Column(Numeric(10, 2))
    correlation_strength = Column(String(50))  # 'weak', 'moderate', 'strong'
    statistical_significance = Column(Boolean)
    metadata = Column(JSONB)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="cross_correlations")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("correlation_coefficient >= -1 AND correlation_coefficient <= 1", name="check_correlation_coefficient"),
        CheckConstraint("correlation_type IN ('pearson', 'spearman', 'kendall', 'partial', 'cross_lagged')", name="check_correlation_type"),
    )
    
    def __repr__(self):
        return f"<CrossCorrelation(primary='{self.primary_data_type}', secondary='{self.secondary_data_type}', r={self.correlation_coefficient})>"


class PatternValidation(Base):
    """Pattern validation results from different validators."""
    
    __tablename__ = "pattern_validations"
    
    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    validation_type = Column(String(100))
    validator_type = Column(String(50))
    validator_id = Column(String(255))  # Agent ID or user ID
    validation_result = Column(String(50))
    validation_score = Column(Numeric(5, 4))
    confidence_level = Column(Numeric(5, 4))
    validation_method = Column(Text)
    test_statistics = Column(JSONB)
    validation_notes = Column(Text)
    replication_successful = Column(Boolean)
    limitations_noted = Column(Text)
    recommendations = Column(Text)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="validations")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("validation_type IN ('statistical_test', 'cross_validation', 'bootstrap', 'permutation_test', 'expert_review', 'peer_review', 'replication_study', 'sensitivity_analysis')", name="check_validation_type"),
        CheckConstraint("validator_type IN ('agent', 'human_expert', 'peer_reviewer', 'automated_system')", name="check_validator_type"),
        CheckConstraint("validation_result IN ('approved', 'rejected', 'needs_revision', 'inconclusive')", name="check_validation_result"),
        CheckConstraint("validation_score >= 0 AND validation_score <= 1", name="check_validation_score"),
    )
    
    def __repr__(self):
        return f"<PatternValidation(validation_type='{self.validation_type}', result='{self.validation_result}')>"


class PatternRelationship(Base):
    """Pattern relationships - how patterns relate to each other."""
    
    __tablename__ = "pattern_relationships"
    
    primary_pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    related_pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    relationship_type = Column(String(100))
    relationship_strength = Column(Numeric(5, 4))
    spatial_overlap_area = Column(Numeric(15, 2))  # Square meters
    temporal_overlap_days = Column(Integer)
    statistical_correlation = Column(Numeric(8, 6))
    dependency_score = Column(Numeric(5, 4))
    validation_status = Column(String(50), default="unverified")
    discovered_by_agent = Column(String(255))
    notes = Column(Text)
    metadata = Column(JSONB)
    
    # Relationships
    primary_pattern = relationship("Pattern", foreign_keys=[primary_pattern_id], back_populates="primary_relationships")
    related_pattern = relationship("Pattern", foreign_keys=[related_pattern_id], back_populates="related_relationships")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("relationship_type IN ('contains', 'overlaps', 'adjacent', 'similar', 'opposite', 'causal_predecessor', 'causal_successor', 'correlated', 'independent')", name="check_relationship_type"),
        CheckConstraint("relationship_strength >= 0 AND relationship_strength <= 1", name="check_relationship_strength"),
        CheckConstraint("primary_pattern_id != related_pattern_id", name="check_no_self_reference"),
    )
    
    def __repr__(self):
        return f"<PatternRelationship(type='{self.relationship_type}', strength={self.relationship_strength})>"


class PatternEvolution(Base):
    """Pattern evolution tracking - how patterns change over time."""
    
    __tablename__ = "pattern_evolution"
    
    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    version_number = Column(Integer, nullable=False)
    change_type = Column(String(100))
    changes_summary = Column(Text)
    previous_confidence = Column(Numeric(5, 4))
    new_confidence = Column(Numeric(5, 4))
    previous_parameters = Column(JSONB)
    new_parameters = Column(JSONB)
    change_significance = Column(Numeric(5, 4))
    changed_by_agent = Column(String(255))
    change_reason = Column(Text)
    validation_impact = Column(Text)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="evolution_history")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("change_type IN ('refinement', 'extension', 'contraction', 'split', 'merge', 'parameter_update', 'validation_update', 'correction')", name="check_change_type"),
    )
    
    def __repr__(self):
        return f"<PatternEvolution(pattern_id='{self.pattern_id}', version={self.version_number}, change_type='{self.change_type}')>"