"""
A2A World Platform - Cultural Data Models

SQLAlchemy models for mythological texts, cultural narratives, and cross-cultural patterns.
"""

from sqlalchemy import (
    Boolean, Column, Integer, String, Text, Date,
    Numeric, CheckConstraint, ForeignKey
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base


class CulturalTradition(Base):
    """Cultural traditions and mythologies."""
    
    __tablename__ = "cultural_traditions"
    
    name = Column(String(255), nullable=False)
    culture = Column(String(100), nullable=False, index=True)
    tradition_type = Column(String(100))
    region = Column(String(255))
    time_period_start = Column(Integer)  # Year (negative for BCE)
    time_period_end = Column(Integer)
    language = Column(String(100))
    source_reliability = Column(String(50))
    description = Column(Text)
    cultural_context = Column(Text)
    historical_significance = Column(Text)
    preservation_status = Column(String(50))
    metadata = Column(JSONB)
    
    # Relationships
    mythological_narratives = relationship("MythologicalNarrative", back_populates="tradition", cascade="all, delete-orphan")
    mythological_entities = relationship("MythologicalEntity", back_populates="tradition", cascade="all, delete-orphan")
    cultural_relevance = relationship("CulturalRelevance", back_populates="cultural_tradition")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("tradition_type IN ('mythology', 'folklore', 'oral_tradition', 'ritual', 'ceremony', 'legend', 'creation_story', 'historical_account', 'spiritual_practice')", name="check_tradition_type"),
        CheckConstraint("source_reliability IN ('primary_source', 'secondary_source', 'oral_tradition', 'reconstructed', 'speculative')", name="check_source_reliability"),
        CheckConstraint("preservation_status IN ('well_preserved', 'partially_preserved', 'fragmentary', 'reconstructed', 'lost')", name="check_preservation_status"),
        CheckConstraint("time_period_end IS NULL OR time_period_end >= time_period_start", name="check_time_period"),
    )
    
    def __repr__(self):
        return f"<CulturalTradition(name='{self.name}', culture='{self.culture}', tradition_type='{self.tradition_type}')>"


class MythologicalNarrative(Base):
    """Mythological narratives and stories."""
    
    __tablename__ = "mythological_narratives"
    
    title = Column(String(255), nullable=False)
    tradition_id = Column(UUID(as_uuid=True), ForeignKey("cultural_traditions.id", ondelete="CASCADE"))
    narrative_type = Column(String(100))
    summary = Column(Text)
    full_text = Column(Text)
    text_language = Column(String(100))
    translation_notes = Column(Text)
    cultural_themes = Column(ARRAY(Text))
    archetypal_elements = Column(ARRAY(Text))
    symbolic_meanings = Column(JSONB)
    geographical_references = Column(ARRAY(Text))
    character_entities = Column(ARRAY(UUID(as_uuid=True)))  # References to mythological_entities
    location_references = Column(ARRAY(UUID(as_uuid=True)))  # References to sacred_sites
    source_references = Column(Text)
    scholarly_interpretations = Column(Text)
    cross_cultural_parallels = Column(ARRAY(Text))
    metadata = Column(JSONB)
    
    # Relationships
    tradition = relationship("CulturalTradition", back_populates="mythological_narratives")
    narrative_patterns = relationship("NarrativePattern", back_populates="narrative", cascade="all, delete-orphan")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("narrative_type IN ('creation_myth', 'hero_journey', 'origin_story', 'flood_myth', 'apocalypse_myth', 'underworld_journey', 'divine_intervention', 'transformation_story', 'trickster_tale', 'foundation_myth')", name="check_narrative_type"),
    )
    
    def __repr__(self):
        return f"<MythologicalNarrative(title='{self.title}', narrative_type='{self.narrative_type}')>"


class MythologicalEntity(Base):
    """Mythological entities (gods, spirits, beings)."""
    
    __tablename__ = "mythological_entities"
    
    name = Column(String(255), nullable=False)
    alternative_names = Column(ARRAY(Text))
    tradition_id = Column(UUID(as_uuid=True), ForeignKey("cultural_traditions.id", ondelete="CASCADE"))
    entity_type = Column(String(100))
    gender = Column(String(50))
    domain_of_influence = Column(ARRAY(Text))  # What they rule over or influence
    attributes = Column(JSONB)  # Physical and spiritual attributes
    powers_abilities = Column(ARRAY(Text))
    symbols = Column(JSONB)  # Sacred symbols and representations
    associations = Column(JSONB)  # Animals, plants, objects, concepts
    family_relationships = Column(JSONB)  # Parent, spouse, children relationships
    location_associations = Column(ARRAY(UUID(as_uuid=True)))  # Sacred sites or geographical features
    cultural_role = Column(Text)
    worship_practices = Column(Text)
    seasonal_associations = Column(ARRAY(Text))
    astronomical_connections = Column(ARRAY(Text))
    modern_interpretations = Column(Text)
    metadata = Column(JSONB)
    
    # Relationships
    tradition = relationship("CulturalTradition", back_populates="mythological_entities")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("entity_type IN ('deity', 'spirit', 'ancestor', 'hero', 'monster', 'trickster', 'nature_spirit', 'guardian', 'creator_being', 'underworld_deity')", name="check_entity_type"),
    )
    
    def __repr__(self):
        return f"<MythologicalEntity(name='{self.name}', entity_type='{self.entity_type}')>"


class CulturalPattern(Base):
    """Cross-cultural pattern associations."""
    
    __tablename__ = "cultural_patterns"
    
    pattern_name = Column(String(255), nullable=False)
    pattern_type = Column(String(100))
    description = Column(Text)
    universal_themes = Column(ARRAY(Text))
    cultural_variations = Column(JSONB)  # How pattern varies across cultures
    psychological_basis = Column(Text)
    symbolic_meaning = Column(Text)
    frequency_score = Column(Numeric(5, 4))  # How common this pattern is across cultures
    significance_rating = Column(Integer)
    scholarly_consensus = Column(String(50))
    research_references = Column(ARRAY(Text))
    metadata = Column(JSONB)
    
    # Relationships
    narrative_patterns = relationship("NarrativePattern", back_populates="cultural_pattern", cascade="all, delete-orphan")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("pattern_type IN ('archetypal_motif', 'symbolic_pattern', 'ritual_structure', 'narrative_pattern', 'cosmological_concept', 'seasonal_cycle', 'hero_journey_stage', 'creation_sequence', 'transformation_cycle')", name="check_pattern_type"),
        CheckConstraint("significance_rating BETWEEN 1 AND 10", name="check_significance_rating"),
        CheckConstraint("scholarly_consensus IN ('strong_consensus', 'moderate_consensus', 'debated', 'emerging_theory')", name="check_scholarly_consensus"),
    )
    
    def __repr__(self):
        return f"<CulturalPattern(pattern_name='{self.pattern_name}', pattern_type='{self.pattern_type}')>"


class NarrativePattern(Base):
    """Junction table linking narratives to cultural patterns."""
    
    __tablename__ = "narrative_patterns"
    
    narrative_id = Column(UUID(as_uuid=True), ForeignKey("mythological_narratives.id", ondelete="CASCADE"))
    pattern_id = Column(UUID(as_uuid=True), ForeignKey("cultural_patterns.id", ondelete="CASCADE"))
    relevance_score = Column(Numeric(5, 4))
    manifestation_description = Column(Text)
    scholarly_notes = Column(Text)
    
    # Relationships
    narrative = relationship("MythologicalNarrative", back_populates="narrative_patterns")
    cultural_pattern = relationship("CulturalPattern", back_populates="narrative_patterns")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="check_relevance_score"),
    )
    
    def __repr__(self):
        return f"<NarrativePattern(narrative_id='{self.narrative_id}', pattern_id='{self.pattern_id}')>"


class CulturalRelationship(Base):
    """Cultural knowledge graphs - relationships between entities."""
    
    __tablename__ = "cultural_relationships"
    
    source_entity_type = Column(String(100))
    source_entity_id = Column(UUID(as_uuid=True), nullable=False)
    target_entity_type = Column(String(100))
    target_entity_id = Column(UUID(as_uuid=True), nullable=False)
    relationship_type = Column(String(100))
    relationship_strength = Column(Numeric(5, 4))
    confidence_level = Column(Numeric(5, 4))
    evidence_type = Column(String(100))
    evidence_description = Column(Text)
    scholarly_support = Column(String(50))
    temporal_relationship = Column(String(100))  # contemporaneous, predecessor, successor
    geographical_context = Column(Text)
    metadata = Column(JSONB)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("source_entity_type IN ('tradition', 'narrative', 'entity', 'pattern', 'sacred_site')", name="check_source_entity_type"),
        CheckConstraint("target_entity_type IN ('tradition', 'narrative', 'entity', 'pattern', 'sacred_site')", name="check_target_entity_type"),
        CheckConstraint("relationship_type IN ('influences', 'derives_from', 'opposes', 'parallels', 'contains', 'associated_with', 'located_at', 'worshipped_at', 'transforms_into', 'precedes', 'follows', 'variant_of', 'synthesizes')", name="check_relationship_type"),
        CheckConstraint("relationship_strength >= 0 AND relationship_strength <= 1", name="check_relationship_strength"),
        CheckConstraint("confidence_level >= 0 AND confidence_level <= 1", name="check_confidence_level"),
        CheckConstraint("evidence_type IN ('textual_reference', 'archaeological', 'linguistic', 'comparative', 'geographical', 'astronomical', 'anthropological', 'psychological')", name="check_evidence_type"),
        CheckConstraint("NOT (source_entity_type = target_entity_type AND source_entity_id = target_entity_id)", name="check_no_self_reference"),
    )
    
    def __repr__(self):
        return f"<CulturalRelationship(relationship_type='{self.relationship_type}', strength={self.relationship_strength})>"


class CulturalInterpretation(Base):
    """Cultural interpretations and scholarly analyses."""
    
    __tablename__ = "cultural_interpretations"
    
    subject_type = Column(String(100))
    subject_id = Column(UUID(as_uuid=True), nullable=False)
    interpretation_type = Column(String(100))
    scholar_name = Column(String(255))
    institution = Column(String(255))
    publication_year = Column(Integer)
    interpretation_text = Column(Text)
    theoretical_framework = Column(String(255))
    methodology = Column(String(255))
    key_findings = Column(Text)
    criticisms_limitations = Column(Text)
    cultural_sensitivity_notes = Column(Text)
    academic_reception = Column(String(100))
    citation_count = Column(Integer, default=0)
    peer_review_status = Column(String(50))
    source_reference = Column(Text)
    metadata = Column(JSONB)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("subject_type IN ('tradition', 'narrative', 'entity', 'pattern', 'relationship')", name="check_subject_type"),
        CheckConstraint("interpretation_type IN ('psychological_analysis', 'anthropological_study', 'comparative_mythology', 'linguistic_analysis', 'archaeological_interpretation', 'historical_analysis', 'symbolic_interpretation', 'feminist_analysis', 'postcolonial_analysis')", name="check_interpretation_type"),
    )
    
    def __repr__(self):
        return f"<CulturalInterpretation(interpretation_type='{self.interpretation_type}', scholar_name='{self.scholar_name}')>"


class CulturalRelevance(Base):
    """Cultural relevance scores and validation."""
    
    __tablename__ = "cultural_relevance"
    
    pattern_id = Column(UUID(as_uuid=True), ForeignKey("patterns.id", ondelete="CASCADE"))  # From pattern discovery
    cultural_tradition_id = Column(UUID(as_uuid=True), ForeignKey("cultural_traditions.id", ondelete="CASCADE"))
    relevance_score = Column(Numeric(5, 4))
    relevance_type = Column(String(100))
    evidence_strength = Column(String(50))
    validation_method = Column(String(100))
    validator_type = Column(String(50))
    validator_credentials = Column(Text)
    validation_notes = Column(Text)
    cultural_sensitivity_review = Column(Text)
    community_validation = Column(Boolean)
    publication_status = Column(String(50))
    metadata = Column(JSONB)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="cultural_relevance")
    cultural_tradition = relationship("CulturalTradition", back_populates="cultural_relevance")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="check_relevance_score"),
        CheckConstraint("relevance_type IN ('direct_correlation', 'symbolic_alignment', 'geographical_overlap', 'temporal_correlation', 'thematic_similarity', 'structural_parallel')", name="check_relevance_type"),
        CheckConstraint("evidence_strength IN ('strong', 'moderate', 'weak', 'speculative')", name="check_evidence_strength"),
        CheckConstraint("validator_type IN ('cultural_expert', 'anthropologist', 'archaeologist', 'linguist', 'comparative_mythologist', 'indigenous_knowledge_keeper')", name="check_validator_type"),
    )
    
    def __repr__(self):
        return f"<CulturalRelevance(relevance_score={self.relevance_score}, relevance_type='{self.relevance_type}')>"


class LinguisticAnalysis(Base):
    """Linguistic analysis and etymology."""
    
    __tablename__ = "linguistic_analysis"
    
    term = Column(String(255), nullable=False)
    language = Column(String(100), nullable=False)
    language_family = Column(String(100))
    etymology = Column(Text)
    semantic_field = Column(String(255))
    cognates = Column(JSONB)  # Related words in other languages
    phonetic_changes = Column(Text)
    historical_evolution = Column(Text)
    cultural_connotations = Column(Text)
    ritual_usage = Column(Text)
    taboo_restrictions = Column(Text)
    regional_variations = Column(JSONB)
    related_entities = Column(ARRAY(UUID(as_uuid=True)))  # Related mythological entities
    source_references = Column(ARRAY(Text))
    linguistic_classification = Column(JSONB)
    metadata = Column(JSONB)
    
    def __repr__(self):
        return f"<LinguisticAnalysis(term='{self.term}', language='{self.language}')>"