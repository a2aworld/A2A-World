"""
A2A World Platform - Multidisciplinary Protocols Models

SQLAlchemy models for multidisciplinary research protocol results and cross-disciplinary correlations.
Supports storage and retrieval of results from archaeoastronomy, cognitive psychology,
environmental mythology, artistic motif diffusion, and mythological geography agents.
"""

from sqlalchemy import (
    Boolean, Column, ForeignKey, Integer, String, Text,
    DateTime, Numeric, CheckConstraint, Index
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from .base import Base


class MultidisciplinaryProtocol(Base):
    """Multidisciplinary research protocol results."""

    __tablename__ = "multidisciplinary_protocols"

    protocol_type = Column(String(100), nullable=False, index=True)
    protocol_name = Column(String(255), nullable=False)
    agent_id = Column(String(255), nullable=False, index=True)
    task_id = Column(String(255), index=True)
    execution_timestamp = Column(DateTime(timezone=True), nullable=False)
    completion_status = Column(String(50), default="completed")
    protocol_version = Column(String(20), default="1.0.0")

    # Protocol results and metadata
    results = Column(JSONB)
    metadata = Column(JSONB)
    performance_metrics = Column(JSONB)

    # Cross-disciplinary correlations
    cross_disciplinary_links = Column(JSONB)
    integration_score = Column(Numeric(5, 4))

    # Validation and quality
    validation_status = Column(String(50), default="pending")
    quality_score = Column(Numeric(5, 4))
    reproducibility_score = Column(Numeric(5, 4))

    # Relationships
    protocol_components = relationship("ProtocolComponent", back_populates="protocol", cascade="all, delete-orphan")
    cross_correlations = relationship("ProtocolCrossCorrelation", back_populates="protocol", cascade="all, delete-orphan")
    protocol_evaluations = relationship("ProtocolEvaluation", back_populates="protocol", cascade="all, delete-orphan")

    # Table constraints
    __table_args__ = (
        CheckConstraint("protocol_type IN ('archaeoastronomy_celestial_alignment', 'cognitive_psychology_sacred_landscapes', 'environmental_mythology_determinants', 'artistic_motif_diffusion_cultural_contact', 'mythological_geography_classical_literature')", name="check_protocol_type"),
        CheckConstraint("completion_status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name="check_completion_status"),
        CheckConstraint("validation_status IN ('pending', 'validated', 'rejected', 'needs_review')", name="check_validation_status"),
        Index("idx_protocol_agent_timestamp", "agent_id", "execution_timestamp"),
        Index("idx_protocol_type_status", "protocol_type", "completion_status"),
    )

    def __repr__(self):
        return f"<MultidisciplinaryProtocol(protocol_type='{self.protocol_type}', agent_id='{self.agent_id}')>"


class ProtocolComponent(Base):
    """Components of multidisciplinary protocol results."""

    __tablename__ = "protocol_components"

    protocol_id = Column(UUID(as_uuid=True), ForeignKey("multidisciplinary_protocols.id", ondelete="CASCADE"))
    component_type = Column(String(100), nullable=False)
    component_name = Column(String(255))
    component_data = Column(JSONB)
    significance_score = Column(Numeric(5, 4))
    confidence_level = Column(Numeric(5, 4))

    # Spatial data for geographical components
    geometry = Column(Geometry('GEOMETRY', srid=4326))

    # Relationships
    protocol = relationship("MultidisciplinaryProtocol", back_populates="protocol_components")

    # Table constraints
    __table_args__ = (
        CheckConstraint("component_type IN ('celestial_alignment', 'cognitive_pattern', 'environmental_correlation', 'artistic_motif', 'geographical_reference', 'validation_result', 'cross_correlation')", name="check_component_type"),
        CheckConstraint("significance_score >= 0 AND significance_score <= 1", name="check_significance_score"),
    )

    def __repr__(self):
        return f"<ProtocolComponent(protocol_id='{self.protocol_id}', component_type='{self.component_type}')>"


class ProtocolCrossCorrelation(Base):
    """Cross-disciplinary correlations between protocol results."""

    __tablename__ = "protocol_cross_correlations"

    protocol_id = Column(UUID(as_uuid=True), ForeignKey("multidisciplinary_protocols.id", ondelete="CASCADE"))
    source_discipline = Column(String(100), nullable=False)
    target_discipline = Column(String(100), nullable=False)
    correlation_type = Column(String(100))
    correlation_strength = Column(Numeric(8, 6))
    correlation_method = Column(String(50))
    correlation_data = Column(JSONB)

    # Statistical significance
    p_value = Column(Numeric(15, 12))
    effect_size = Column(Numeric(8, 4))
    sample_size = Column(Integer)

    # Spatial correlation data
    spatial_overlap = Column(Geometry('POLYGON', srid=4326))
    temporal_overlap_start = Column(DateTime(timezone=True))
    temporal_overlap_end = Column(DateTime(timezone=True))

    # Relationships
    protocol = relationship("MultidisciplinaryProtocol", back_populates="cross_correlations")

    # Table constraints
    __table_args__ = (
        CheckConstraint("source_discipline IN ('archaeoastronomy', 'cognitive_psychology', 'environmental_mythology', 'artistic_motif', 'mythological_geography')", name="check_source_discipline"),
        CheckConstraint("target_discipline IN ('archaeoastronomy', 'cognitive_psychology', 'environmental_mythology', 'artistic_motif', 'mythological_geography')", name="check_target_discipline"),
        CheckConstraint("correlation_type IN ('spatial_alignment', 'temporal_synchrony', 'symbolic_resonance', 'functional_similarity', 'causal_relationship')", name="check_correlation_type"),
        CheckConstraint("correlation_method IN ('statistical', 'symbolic', 'narrative', 'spatial', 'temporal')", name="check_correlation_method"),
        CheckConstraint("source_discipline != target_discipline", name="check_no_self_correlation"),
    )

    def __repr__(self):
        return f"<ProtocolCrossCorrelation(source='{self.source_discipline}', target='{self.target_discipline}', strength={self.correlation_strength})>"


class ProtocolEvaluation(Base):
    """Evaluations and assessments of multidisciplinary protocols."""

    __tablename__ = "protocol_evaluations"

    protocol_id = Column(UUID(as_uuid=True), ForeignKey("multidisciplinary_protocols.id", ondelete="CASCADE"))
    evaluation_type = Column(String(50), nullable=False)
    evaluator_type = Column(String(50))
    evaluator_id = Column(String(255))
    evaluation_score = Column(Numeric(5, 4))
    evaluation_criteria = Column(JSONB)
    evaluation_notes = Column(Text)
    evaluation_timestamp = Column(DateTime(timezone=True), nullable=False)

    # Detailed evaluation metrics
    methodological_rigor = Column(Numeric(5, 4))
    interdisciplinary_integration = Column(Numeric(5, 4))
    cultural_sensitivity = Column(Numeric(5, 4))
    scientific_validity = Column(Numeric(5, 4))

    # Recommendations
    recommendations = Column(JSONB)
    improvement_suggestions = Column(Text)

    # Relationships
    protocol = relationship("MultidisciplinaryProtocol", back_populates="protocol_evaluations")

    # Table constraints
    __table_args__ = (
        CheckConstraint("evaluation_type IN ('peer_review', 'expert_assessment', 'automated_evaluation', 'community_feedback')", name="check_evaluation_type"),
        CheckConstraint("evaluator_type IN ('human_expert', 'ai_agent', 'peer_reviewer', 'community_member')", name="check_evaluator_type"),
        CheckConstraint("evaluation_score >= 0 AND evaluation_score <= 1", name="check_evaluation_score"),
    )

    def __repr__(self):
        return f"<ProtocolEvaluation(protocol_id='{self.protocol_id}', type='{self.evaluation_type}', score={self.evaluation_score})>"


class DisciplineIntegrationIndex(Base):
    """Index of cross-disciplinary integrations and correlations."""

    __tablename__ = "discipline_integration_index"

    integration_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    primary_discipline = Column(String(100), nullable=False)
    secondary_discipline = Column(String(100), nullable=False)
    integration_type = Column(String(100))
    integration_strength = Column(Numeric(5, 4))
    integration_evidence = Column(JSONB)

    # Integration metadata
    protocols_involved = Column(ARRAY(String(255)))
    key_findings = Column(JSONB)
    methodological_approaches = Column(ARRAY(String(100)))

    # Temporal and spatial scope
    temporal_scope_start = Column(DateTime(timezone=True))
    temporal_scope_end = Column(DateTime(timezone=True))
    spatial_scope = Column(Geometry('POLYGON', srid=4326))

    # Quality and validation
    integration_confidence = Column(Numeric(5, 4))
    validation_status = Column(String(50), default="unvalidated")
    last_updated = Column(DateTime(timezone=True), nullable=False)

    # Table constraints
    __table_args__ = (
        CheckConstraint("primary_discipline IN ('archaeoastronomy', 'cognitive_psychology', 'environmental_mythology', 'artistic_motif', 'mythological_geography')", name="check_primary_discipline"),
        CheckConstraint("secondary_discipline IN ('archaeoastronomy', 'cognitive_psychology', 'environmental_mythology', 'artistic_motif', 'mythological_geography')", name="check_secondary_discipline"),
        CheckConstraint("integration_type IN ('spatial_correlation', 'temporal_synchrony', 'symbolic_connection', 'functional_linkage', 'causal_relationship', 'methodological_synergy')", name="check_integration_type"),
        CheckConstraint("validation_status IN ('unvalidated', 'validated', 'rejected', 'needs_review')", name="check_validation_status"),
        CheckConstraint("primary_discipline != secondary_discipline", name="check_no_self_integration"),
        Index("idx_integration_disciplines", "primary_discipline", "secondary_discipline"),
        Index("idx_integration_strength", "integration_strength"),
    )

    def __repr__(self):
        return f"<DisciplineIntegrationIndex(primary='{self.primary_discipline}', secondary='{self.secondary_discipline}', strength={self.integration_strength})>"


# Specialized models for each protocol type

class ArchaeoastronomyResult(Base):
    """Specialized storage for archaeoastronomy protocol results."""

    __tablename__ = "archaeoastronomy_results"

    protocol_id = Column(UUID(as_uuid=True), ForeignKey("multidisciplinary_protocols.id", ondelete="CASCADE"), primary_key=True)

    # Celestial alignment data
    alignment_type = Column(String(50))
    azimuth = Column(Numeric(8, 4))
    altitude = Column(Numeric(8, 4))
    precision_degrees = Column(Numeric(8, 4))
    significance_score = Column(Numeric(5, 4))

    # Site information
    site1_coordinates = Column(Geometry('POINT', srid=4326))
    site2_coordinates = Column(Geometry('POINT', srid=4326))
    distance_km = Column(Numeric(10, 2))

    # Astronomical context
    celestial_body = Column(String(50))
    astronomical_event = Column(String(100))
    seasonal_timing = Column(String(50))

    # Cultural integration
    mythological_context = Column(JSONB)
    cultural_significance = Column(Text)

    # Table constraints
    __table_args__ = (
        CheckConstraint("alignment_type IN ('solstice', 'equinox', 'lunar_standstill', 'planetary', 'stellar')", name="check_alignment_type"),
        CheckConstraint("celestial_body IN ('sun', 'moon', 'venus', 'mars', 'jupiter', 'saturn', 'stars', 'milky_way')", name="check_celestial_body"),
    )

    def __repr__(self):
        return f"<ArchaeoastronomyResult(alignment_type='{self.alignment_type}', significance={self.significance_score})>"


class CognitivePsychologyResult(Base):
    """Specialized storage for cognitive psychology protocol results."""

    __tablename__ = "cognitive_psychology_results"

    protocol_id = Column(UUID(as_uuid=True), ForeignKey("multidisciplinary_protocols.id", ondelete="CASCADE"), primary_key=True)

    # Cognitive pattern data
    cognitive_process = Column(String(100))
    perception_mechanism = Column(String(50))
    memory_system = Column(String(50))
    emotional_response = Column(String(50))

    # Pattern characteristics
    pattern_complexity = Column(Numeric(5, 4))
    cultural_resonance = Column(Numeric(5, 4))
    neurological_plausibility = Column(Numeric(5, 4))

    # Sacred landscape cognition
    landscape_perception = Column(JSONB)
    sacred_associations = Column(JSONB)
    cognitive_load_metrics = Column(JSONB)

    # Table constraints
    __table_args__ = (
        CheckConstraint("cognitive_process IN ('perception', 'memory', 'emotion', 'attention', 'schema_activation', 'cultural_cognition')", name="check_cognitive_process"),
        CheckConstraint("perception_mechanism IN ('gestalt', 'bottom_up', 'top_down', 'feature_based', 'spatial_attention')", name="check_perception_mechanism"),
        CheckConstraint("memory_system IN ('sensory', 'working', 'episodic', 'semantic', 'procedural')", name="check_memory_system"),
    )

    def __repr__(self):
        return f"<CognitivePsychologyResult(process='{self.cognitive_process}', complexity={self.pattern_complexity})>"


class EnvironmentalMythologyResult(Base):
    """Specialized storage for environmental mythology protocol results."""

    __tablename__ = "environmental_mythology_results"

    protocol_id = Column(UUID(as_uuid=True), ForeignKey("multidisciplinary_protocols.id", ondelete="CASCADE"), primary_key=True)

    # Environmental factor data
    environmental_category = Column(String(50))
    climatic_variable = Column(String(50))
    geological_feature = Column(String(50))
    ecological_factor = Column(String(50))

    # Mythological correlation
    mythological_motif = Column(String(100))
    narrative_function = Column(String(100))
    symbolic_meaning = Column(Text)

    # Correlation metrics
    correlation_coefficient = Column(Numeric(8, 6))
    causal_strength = Column(Numeric(5, 4))
    temporal_synchrony = Column(Numeric(5, 4))

    # Cultural context
    cultural_adaptation = Column(JSONB)
    environmental_determinism = Column(JSONB)

    # Table constraints
    __table_args__ = (
        CheckConstraint("environmental_category IN ('climate', 'geology', 'ecology', 'hydrology', 'atmospheric')", name="check_environmental_category"),
        CheckConstraint("narrative_function IN ('creation', 'destruction', 'transformation', 'preservation', 'explanation')", name="check_narrative_function"),
    )

    def __repr__(self):
        return f"<EnvironmentalMythologyResult(category='{self.environmental_category}', correlation={self.correlation_coefficient})>"


class ArtisticMotifResult(Base):
    """Specialized storage for artistic motif diffusion protocol results."""

    __tablename__ = "artistic_motif_results"

    protocol_id = Column(UUID(as_uuid=True), ForeignKey("multidisciplinary_protocols.id", ondelete="CASCADE"), primary_key=True)

    # Motif characteristics
    motif_category = Column(String(50))
    stylistic_elements = Column(ARRAY(String(100)))
    diffusion_mechanism = Column(String(50))
    transmission_route = Column(String(100))

    # Cultural contact data
    source_culture = Column(String(100))
    target_culture = Column(String(100))
    contact_type = Column(String(50))
    temporal_sequence = Column(String(50))

    # Diffusion metrics
    diffusion_velocity = Column(Numeric(8, 4))  # motifs per time unit
    cultural_adoption_rate = Column(Numeric(5, 4))
    stylistic_convergence = Column(Numeric(5, 4))

    # Iconographic data
    symbolic_meaning = Column(JSONB)
    functional_purpose = Column(JSONB)

    # Table constraints
    __table_args__ = (
        CheckConstraint("motif_category IN ('geometric', 'figurative', 'symbolic', 'ornamental', 'abstract')", name="check_motif_category"),
        CheckConstraint("diffusion_mechanism IN ('migration', 'trade', 'conquest', 'cultural_exchange', 'religious_spread')", name="check_diffusion_mechanism"),
        CheckConstraint("contact_type IN ('direct', 'indirect', 'mediated', 'acculturative', 'syncretic')", name="check_contact_type"),
    )

    def __repr__(self):
        return f"<ArtisticMotifResult(category='{self.motif_category}', source='{self.source_culture}', target='{self.target_culture}')>"


class MythologicalGeographyResult(Base):
    """Specialized storage for mythological geography protocol results."""

    __tablename__ = "mythological_geography_results"

    protocol_id = Column(UUID(as_uuid=True), ForeignKey("multidisciplinary_protocols.id", ondelete="CASCADE"), primary_key=True)

    # Geographical feature data
    geographical_category = Column(String(50))
    literary_period = Column(String(50))
    spatial_configuration = Column(String(100))

    # Mythological interpretation
    mythological_function = Column(String(100))
    symbolic_significance = Column(Text)
    narrative_role = Column(String(50))

    # Literary analysis
    textual_density = Column(Numeric(8, 4))  # references per 1000 words
    spatial_symbolism = Column(JSONB)
    geographical_imagery = Column(JSONB)

    # Sacred geography
    sacred_associations = Column(JSONB)
    ritual_landscape = Column(JSONB)

    # Table constraints
    __table_args__ = (
        CheckConstraint("geographical_category IN ('mountains', 'rivers', 'seas', 'islands', 'forests', 'caves', 'cities', 'deserts')", name="check_geographical_category"),
        CheckConstraint("literary_period IN ('ancient_greek', 'ancient_roman', 'medieval', 'renaissance', 'modern')", name="check_literary_period"),
        CheckConstraint("mythological_function IN ('sacred_realm', 'underworld_entrance', 'divine_residence', 'hero_journey', 'cosmic_center')", name="check_mythological_function"),
    )

    def __repr__(self):
        return f"<MythologicalGeographyResult(category='{self.geographical_category}', period='{self.literary_period}')>"