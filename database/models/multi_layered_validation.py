"""
A2A World Platform - Multi-Layered Validation Models

SQLAlchemy models for multi-layered validation results including
cultural relevance, ethical assessments, and multidisciplinary scores.
"""

from sqlalchemy import (
    Boolean, Column, ForeignKey, Integer, String, Text,
    DateTime, Numeric, CheckConstraint
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base


class MultiLayeredValidation(Base):
    """Multi-layered validation results combining statistical, cultural, and ethical assessments."""

    __tablename__ = "multi_layered_validations"

    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    validation_agent_id = Column(String(255), nullable=False)
    validation_timestamp = Column(DateTime(timezone=True))

    # Overall assessment
    overall_validation_success = Column(Boolean, default=False)
    multidisciplinary_score = Column(Numeric(5, 4))
    integrated_classification = Column(String(100))

    # Layer-specific results
    statistical_validation_id = Column(UUID(as_uuid=True), ForeignKey("pattern_validations.id"))
    cultural_validation_id = Column(UUID(as_uuid=True), ForeignKey("cultural_relevance.id"))
    ethical_validation_id = Column(UUID(as_uuid=True), ForeignKey("ethical_assessments.id"))
    consensus_validation_id = Column(UUID(as_uuid=True), ForeignKey("consensus_validations.id"))

    # Validation metadata
    validation_layers_executed = Column(ARRAY(String(50)))
    execution_time_seconds = Column(Numeric(10, 2))
    validation_version = Column(String(20))

    # Recommendations and concerns
    multidisciplinary_recommendations = Column(ARRAY(Text))
    critical_concerns = Column(ARRAY(Text))

    # Relationships
    pattern = relationship("Pattern", back_populates="multi_layered_validations")
    statistical_validation = relationship("PatternValidation")
    cultural_validation = relationship("CulturalRelevance")
    ethical_validation = relationship("EthicalAssessment")
    consensus_validation = relationship("ConsensusValidation")

    # Table constraints
    __table_args__ = (
        CheckConstraint("multidisciplinary_score >= 0 AND multidisciplinary_score <= 1", name="check_multidisciplinary_score"),
        CheckConstraint("integrated_classification IN ('excellent_multidisciplinary_validation', 'good_multidisciplinary_validation', 'moderate_multidisciplinary_validation', 'weak_multidisciplinary_validation', 'poor_multidisciplinary_validation')", name="check_integrated_classification"),
    )

    def __repr__(self):
        return f"<MultiLayeredValidation(pattern_id='{self.pattern_id}', score={self.multidisciplinary_score})>"


class CulturalValidation(Base):
    """Cultural validation results for patterns."""

    __tablename__ = "cultural_validations"

    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    assessment_timestamp = Column(DateTime(timezone=True))

    # Cultural relevance scores
    cultural_relevance_score = Column(Numeric(5, 4))
    mythological_alignment_score = Column(Numeric(5, 4))

    # Cultural sensitivity
    sensitivity_level = Column(String(50))
    sensitivity_score = Column(Numeric(5, 4))

    # Cultural contexts and associations
    identified_traditions = Column(ARRAY(String(100)))
    cultural_contexts = Column(ARRAY(String(255)))
    cross_cultural_connections = Column(ARRAY(Text))

    # Archetypal and symbolic analysis
    identified_archetypes = Column(JSONB)
    symbolic_elements = Column(JSONB)

    # Recommendations and concerns
    cultural_recommendations = Column(ARRAY(Text))
    sensitivity_concerns = Column(ARRAY(Text))

    # Assessment metadata
    assessment_method = Column(String(100))
    cultural_expert_consulted = Column(Boolean, default=False)

    # Relationships
    pattern = relationship("Pattern", back_populates="cultural_validations")

    # Table constraints
    __table_args__ = (
        CheckConstraint("cultural_relevance_score >= 0 AND cultural_relevance_score <= 1", name="check_cultural_relevance_score"),
        CheckConstraint("mythological_alignment_score >= 0 AND mythological_alignment_score <= 1", name="check_mythological_alignment_score"),
        CheckConstraint("sensitivity_score >= 0 AND sensitivity_score <= 1", name="check_sensitivity_score"),
        CheckConstraint("sensitivity_level IN ('highly_sensitive', 'moderately_sensitive', 'low_sensitivity', 'culturally_neutral')", name="check_sensitivity_level"),
    )

    def __repr__(self):
        return f"<CulturalValidation(pattern_id='{self.pattern_id}', relevance_score={self.cultural_relevance_score})>"


class EthicalAssessment(Base):
    """Ethical assessment results for patterns."""

    __tablename__ = "ethical_assessments"

    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    assessment_timestamp = Column(DateTime(timezone=True))

    # Human flourishing scores
    flourishing_score = Column(Numeric(5, 4))
    emotional_wellbeing_score = Column(Numeric(5, 4))
    psychological_wellbeing_score = Column(Numeric(5, 4))
    social_wellbeing_score = Column(Numeric(5, 4))
    physical_health_score = Column(Numeric(5, 4))
    spiritual_growth_score = Column(Numeric(5, 4))
    environmental_harmony_score = Column(Numeric(5, 4))

    # Bias and diversity scores
    diversity_score = Column(Numeric(5, 4))
    bias_score = Column(Numeric(5, 4))

    # Ethical compliance
    ethical_compliance = Column(Boolean, default=True)
    ethical_concerns_count = Column(Integer, default=0)

    # Identified biases and concerns
    identified_biases = Column(JSONB)
    ethical_concerns = Column(ARRAY(Text))

    # Recommendations
    ethical_recommendations = Column(ARRAY(Text))
    diversity_recommendations = Column(ARRAY(Text))

    # Assessment metadata
    assessment_method = Column(String(100))
    stakeholder_impact_assessed = Column(Boolean, default=False)

    # Relationships
    pattern = relationship("Pattern", back_populates="ethical_assessments")

    # Table constraints
    __table_args__ = (
        CheckConstraint("flourishing_score >= 0 AND flourishing_score <= 1", name="check_flourishing_score"),
        CheckConstraint("emotional_wellbeing_score >= 0 AND emotional_wellbeing_score <= 1", name="check_emotional_wellbeing"),
        CheckConstraint("psychological_wellbeing_score >= 0 AND psychological_wellbeing_score <= 1", name="check_psychological_wellbeing"),
        CheckConstraint("social_wellbeing_score >= 0 AND social_wellbeing_score <= 1", name="check_social_wellbeing"),
        CheckConstraint("physical_health_score >= 0 AND physical_health_score <= 1", name="check_physical_health"),
        CheckConstraint("spiritual_growth_score >= 0 AND spiritual_growth_score <= 1", name="check_spiritual_growth"),
        CheckConstraint("environmental_harmony_score >= 0 AND environmental_harmony_score <= 1", name="check_environmental_harmony"),
        CheckConstraint("diversity_score >= 0 AND diversity_score <= 1", name="check_diversity_score"),
        CheckConstraint("bias_score >= 0 AND bias_score <= 1", name="check_bias_score"),
    )

    def __repr__(self):
        return f"<EthicalAssessment(pattern_id='{self.pattern_id}', flourishing_score={self.flourishing_score})>"


class ConsensusValidation(Base):
    """Consensus validation results from peer agents."""

    __tablename__ = "consensus_validations"

    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))
    consensus_request_id = Column(UUID(as_uuid=True))
    consensus_timestamp = Column(DateTime(timezone=True))

    # Consensus results
    consensus_achieved = Column(Boolean, default=False)
    consensus_decision = Column(String(50))
    consensus_confidence = Column(Numeric(5, 4))
    peer_agreement_score = Column(Numeric(5, 4))

    # Participation details
    participating_agents = Column(ARRAY(String(255)))
    total_participants = Column(Integer)
    votes_significant = Column(Integer, default=0)
    votes_not_significant = Column(Integer, default=0)
    votes_uncertain = Column(Integer, default=0)

    # Voting mechanism
    voting_mechanism = Column(String(50))
    minimum_participants = Column(Integer)
    timeout_seconds = Column(Integer)

    # Consensus metadata
    execution_time_seconds = Column(Numeric(10, 2))
    consensus_method = Column(String(100))

    # Relationships
    pattern = relationship("Pattern", back_populates="consensus_validations")

    # Table constraints
    __table_args__ = (
        CheckConstraint("consensus_decision IN ('significant', 'not_significant', 'uncertain', 'inconclusive')", name="check_consensus_decision"),
        CheckConstraint("consensus_confidence >= 0 AND consensus_confidence <= 1", name="check_consensus_confidence"),
        CheckConstraint("peer_agreement_score >= 0 AND peer_agreement_score <= 1", name="check_peer_agreement_score"),
        CheckConstraint("voting_mechanism IN ('adaptive', 'majority', 'weighted', 'supermajority')", name="check_voting_mechanism"),
    )

    def __repr__(self):
        return f"<ConsensusValidation(pattern_id='{self.pattern_id}', consensus_achieved={self.consensus_achieved})>"


class ValidationLayerResult(Base):
    """Individual validation layer results within multi-layered validation."""

    __tablename__ = "validation_layer_results"

    multi_layered_validation_id = Column(UUID(as_uuid=True), ForeignKey("multi_layered_validations.id", ondelete="CASCADE"))
    layer_name = Column(String(50))
    layer_type = Column(String(50))

    # Layer results
    validation_success = Column(Boolean, default=False)
    layer_score = Column(Numeric(5, 4))
    execution_time_seconds = Column(Numeric(8, 2))

    # Layer-specific data
    layer_results = Column(JSONB)
    layer_errors = Column(ARRAY(Text))

    # Relationships
    multi_layered_validation = relationship("MultiLayeredValidation", back_populates="layer_results")

    # Table constraints
    __table_args__ = (
        CheckConstraint("layer_score >= 0 AND layer_score <= 1", name="check_layer_score"),
        CheckConstraint("layer_type IN ('statistical', 'cultural', 'ethical', 'consensus', 'integrated')", name="check_layer_type"),
    )

    def __repr__(self):
        return f"<ValidationLayerResult(layer_name='{self.layer_name}', success={self.validation_success})>"


# Update Pattern model to include new relationships
# This would be done in the patterns.py file, but shown here for reference:

# In Pattern class, add these relationships:
"""
multi_layered_validations = relationship("MultiLayeredValidation", back_populates="pattern", cascade="all, delete-orphan")
cultural_validations = relationship("CulturalValidation", back_populates="pattern", cascade="all, delete-orphan")
ethical_assessments = relationship("EthicalAssessment", back_populates="pattern", cascade="all, delete-orphan")
consensus_validations = relationship("ConsensusValidation", back_populates="pattern", cascade="all, delete-orphan")
"""

# Update MultiLayeredValidation to include layer_results relationship:
"""
layer_results = relationship("ValidationLayerResult", back_populates="multi_layered_validation", cascade="all, delete-orphan")
"""