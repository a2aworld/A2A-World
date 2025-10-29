-- A2A World Platform - Cultural Data Schema
-- Tables for mythological texts, cultural narratives, and cross-cultural patterns

SET search_path TO a2a_world, public;

-- Cultural traditions and mythologies
CREATE TABLE cultural_traditions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    culture VARCHAR(100) NOT NULL,
    tradition_type VARCHAR(100) CHECK (tradition_type IN (
        'mythology', 'folklore', 'oral_tradition', 'ritual', 'ceremony',
        'legend', 'creation_story', 'historical_account', 'spiritual_practice'
    )),
    region VARCHAR(255),
    time_period_start INTEGER, -- Year (negative for BCE)
    time_period_end INTEGER,
    language VARCHAR(100),
    source_reliability VARCHAR(50) CHECK (source_reliability IN (
        'primary_source', 'secondary_source', 'oral_tradition', 'reconstructed', 'speculative'
    )),
    description TEXT,
    cultural_context TEXT,
    historical_significance TEXT,
    preservation_status VARCHAR(50) CHECK (preservation_status IN (
        'well_preserved', 'partially_preserved', 'fragmentary', 'reconstructed', 'lost'
    )),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mythological narratives and stories
CREATE TABLE mythological_narratives (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    tradition_id UUID REFERENCES cultural_traditions(id) ON DELETE CASCADE,
    narrative_type VARCHAR(100) CHECK (narrative_type IN (
        'creation_myth', 'hero_journey', 'origin_story', 'flood_myth',
        'apocalypse_myth', 'underworld_journey', 'divine_intervention',
        'transformation_story', 'trickster_tale', 'foundation_myth'
    )),
    summary TEXT,
    full_text TEXT,
    text_language VARCHAR(100),
    translation_notes TEXT,
    cultural_themes TEXT[],
    archetypal_elements TEXT[],
    symbolic_meanings JSONB,
    geographical_references TEXT[],
    character_entities UUID[], -- References to mythological_entities
    location_references UUID[], -- References to sacred_sites or geospatial_features
    source_references TEXT,
    scholarly_interpretations TEXT,
    cross_cultural_parallels TEXT[],
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mythological entities (gods, spirits, beings)
CREATE TABLE mythological_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    alternative_names TEXT[],
    tradition_id UUID REFERENCES cultural_traditions(id) ON DELETE CASCADE,
    entity_type VARCHAR(100) CHECK (entity_type IN (
        'deity', 'spirit', 'ancestor', 'hero', 'monster', 'trickster',
        'nature_spirit', 'guardian', 'creator_being', 'underworld_deity'
    )),
    gender VARCHAR(50),
    domain_of_influence TEXT[], -- What they rule over or influence
    attributes JSONB, -- Physical and spiritual attributes
    powers_abilities TEXT[],
    symbols JSONB, -- Sacred symbols and representations
    associations JSONB, -- Animals, plants, objects, concepts
    family_relationships JSONB, -- Parent, spouse, children relationships
    location_associations UUID[], -- Sacred sites or geographical features
    cultural_role TEXT,
    worship_practices TEXT,
    seasonal_associations TEXT[],
    astronomical_connections TEXT[],
    modern_interpretations TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cross-cultural pattern associations
CREATE TABLE cultural_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_name VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100) CHECK (pattern_type IN (
        'archetypal_motif', 'symbolic_pattern', 'ritual_structure',
        'narrative_pattern', 'cosmological_concept', 'seasonal_cycle',
        'hero_journey_stage', 'creation_sequence', 'transformation_cycle'
    )),
    description TEXT,
    universal_themes TEXT[],
    cultural_variations JSONB, -- How pattern varies across cultures
    psychological_basis TEXT,
    symbolic_meaning TEXT,
    frequency_score DECIMAL(5,4), -- How common this pattern is across cultures
    significance_rating INTEGER CHECK (significance_rating BETWEEN 1 AND 10),
    scholarly_consensus VARCHAR(50) CHECK (scholarly_consensus IN (
        'strong_consensus', 'moderate_consensus', 'debated', 'emerging_theory'
    )),
    research_references TEXT[],
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Junction table linking narratives to cultural patterns
CREATE TABLE narrative_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    narrative_id UUID REFERENCES mythological_narratives(id) ON DELETE CASCADE,
    pattern_id UUID REFERENCES cultural_patterns(id) ON DELETE CASCADE,
    relevance_score DECIMAL(5,4) CHECK (relevance_score >= 0 AND relevance_score <= 1),
    manifestation_description TEXT,
    scholarly_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cultural knowledge graphs - relationships between entities
CREATE TABLE cultural_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_type VARCHAR(100) CHECK (source_entity_type IN (
        'tradition', 'narrative', 'entity', 'pattern', 'sacred_site'
    )),
    source_entity_id UUID NOT NULL,
    target_entity_type VARCHAR(100) CHECK (target_entity_type IN (
        'tradition', 'narrative', 'entity', 'pattern', 'sacred_site'
    )),
    target_entity_id UUID NOT NULL,
    relationship_type VARCHAR(100) CHECK (relationship_type IN (
        'influences', 'derives_from', 'opposes', 'parallels', 'contains',
        'associated_with', 'located_at', 'worshipped_at', 'transforms_into',
        'precedes', 'follows', 'variant_of', 'synthesizes'
    )),
    relationship_strength DECIMAL(5,4) CHECK (relationship_strength >= 0 AND relationship_strength <= 1),
    confidence_level DECIMAL(5,4) CHECK (confidence_level >= 0 AND confidence_level <= 1),
    evidence_type VARCHAR(100) CHECK (evidence_type IN (
        'textual_reference', 'archaeological', 'linguistic', 'comparative',
        'geographical', 'astronomical', 'anthropological', 'psychological'
    )),
    evidence_description TEXT,
    scholarly_support VARCHAR(50),
    temporal_relationship VARCHAR(100), -- contemporaneous, predecessor, successor
    geographical_context TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cultural interpretations and scholarly analyses
CREATE TABLE cultural_interpretations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    subject_type VARCHAR(100) CHECK (subject_type IN (
        'tradition', 'narrative', 'entity', 'pattern', 'relationship'
    )),
    subject_id UUID NOT NULL,
    interpretation_type VARCHAR(100) CHECK (interpretation_type IN (
        'psychological_analysis', 'anthropological_study', 'comparative_mythology',
        'linguistic_analysis', 'archaeological_interpretation', 'historical_analysis',
        'symbolic_interpretation', 'feminist_analysis', 'postcolonial_analysis'
    )),
    scholar_name VARCHAR(255),
    institution VARCHAR(255),
    publication_year INTEGER,
    interpretation_text TEXT,
    theoretical_framework VARCHAR(255),
    methodology VARCHAR(255),
    key_findings TEXT,
    criticisms_limitations TEXT,
    cultural_sensitivity_notes TEXT,
    academic_reception VARCHAR(100),
    citation_count INTEGER DEFAULT 0,
    peer_review_status VARCHAR(50),
    source_reference TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cultural relevance scores and validation
CREATE TABLE cultural_relevance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE, -- From pattern discovery
    cultural_tradition_id UUID REFERENCES cultural_traditions(id) ON DELETE CASCADE,
    relevance_score DECIMAL(5,4) CHECK (relevance_score >= 0 AND relevance_score <= 1),
    relevance_type VARCHAR(100) CHECK (relevance_type IN (
        'direct_correlation', 'symbolic_alignment', 'geographical_overlap',
        'temporal_correlation', 'thematic_similarity', 'structural_parallel'
    )),
    evidence_strength VARCHAR(50) CHECK (evidence_strength IN (
        'strong', 'moderate', 'weak', 'speculative'
    )),
    validation_method VARCHAR(100),
    validator_type VARCHAR(50) CHECK (validator_type IN (
        'cultural_expert', 'anthropologist', 'archaeologist', 'linguist',
        'comparative_mythologist', 'indigenous_knowledge_keeper'
    )),
    validator_credentials TEXT,
    validation_notes TEXT,
    cultural_sensitivity_review TEXT,
    community_validation BOOLEAN,
    publication_status VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Linguistic analysis and etymology
CREATE TABLE linguistic_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    term VARCHAR(255) NOT NULL,
    language VARCHAR(100) NOT NULL,
    language_family VARCHAR(100),
    etymology TEXT,
    semantic_field VARCHAR(255),
    cognates JSONB, -- Related words in other languages
    phonetic_changes TEXT,
    historical_evolution TEXT,
    cultural_connotations TEXT,
    ritual_usage TEXT,
    taboo_restrictions TEXT,
    regional_variations JSONB,
    related_entities UUID[], -- Related mythological entities
    source_references TEXT[],
    linguistic_classification JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for cultural data tables
CREATE INDEX idx_cultural_traditions_culture ON cultural_traditions(culture);
CREATE INDEX idx_cultural_traditions_type ON cultural_traditions(tradition_type);
CREATE INDEX idx_cultural_traditions_region ON cultural_traditions(region);
CREATE INDEX idx_cultural_traditions_time_period ON cultural_traditions(time_period_start, time_period_end);
CREATE INDEX idx_cultural_traditions_reliability ON cultural_traditions(source_reliability);

CREATE INDEX idx_mythological_narratives_tradition ON mythological_narratives(tradition_id);
CREATE INDEX idx_mythological_narratives_type ON mythological_narratives(narrative_type);
CREATE INDEX idx_mythological_narratives_title ON mythological_narratives(title);
CREATE INDEX idx_mythological_narratives_language ON mythological_narratives(text_language);

CREATE INDEX idx_mythological_entities_tradition ON mythological_entities(tradition_id);
CREATE INDEX idx_mythological_entities_type ON mythological_entities(entity_type);
CREATE INDEX idx_mythological_entities_name ON mythological_entities(name);
CREATE INDEX idx_mythological_entities_gender ON mythological_entities(gender);

CREATE INDEX idx_cultural_patterns_type ON cultural_patterns(pattern_type);
CREATE INDEX idx_cultural_patterns_frequency ON cultural_patterns(frequency_score DESC);
CREATE INDEX idx_cultural_patterns_significance ON cultural_patterns(significance_rating DESC);
CREATE INDEX idx_cultural_patterns_consensus ON cultural_patterns(scholarly_consensus);

CREATE INDEX idx_narrative_patterns_narrative ON narrative_patterns(narrative_id);
CREATE INDEX idx_narrative_patterns_pattern ON narrative_patterns(pattern_id);
CREATE INDEX idx_narrative_patterns_relevance ON narrative_patterns(relevance_score DESC);

CREATE INDEX idx_cultural_relationships_source ON cultural_relationships(source_entity_type, source_entity_id);
CREATE INDEX idx_cultural_relationships_target ON cultural_relationships(target_entity_type, target_entity_id);
CREATE INDEX idx_cultural_relationships_type ON cultural_relationships(relationship_type);
CREATE INDEX idx_cultural_relationships_strength ON cultural_relationships(relationship_strength DESC);

CREATE INDEX idx_cultural_interpretations_subject ON cultural_interpretations(subject_type, subject_id);
CREATE INDEX idx_cultural_interpretations_type ON cultural_interpretations(interpretation_type);
CREATE INDEX idx_cultural_interpretations_scholar ON cultural_interpretations(scholar_name);
CREATE INDEX idx_cultural_interpretations_year ON cultural_interpretations(publication_year);

CREATE INDEX idx_cultural_relevance_pattern ON cultural_relevance(pattern_id);
CREATE INDEX idx_cultural_relevance_tradition ON cultural_relevance(cultural_tradition_id);
CREATE INDEX idx_cultural_relevance_score ON cultural_relevance(relevance_score DESC);
CREATE INDEX idx_cultural_relevance_strength ON cultural_relevance(evidence_strength);

CREATE INDEX idx_linguistic_analysis_term ON linguistic_analysis(term);
CREATE INDEX idx_linguistic_analysis_language ON linguistic_analysis(language);
CREATE INDEX idx_linguistic_analysis_family ON linguistic_analysis(language_family);

-- Create GIN indexes for array and JSONB fields
CREATE INDEX idx_mythological_narratives_themes_gin ON mythological_narratives USING gin(cultural_themes);
CREATE INDEX idx_mythological_narratives_elements_gin ON mythological_narratives USING gin(archetypal_elements);
CREATE INDEX idx_mythological_narratives_meanings_gin ON mythological_narratives USING gin(symbolic_meanings);
CREATE INDEX idx_mythological_narratives_geo_refs_gin ON mythological_narratives USING gin(geographical_references);
CREATE INDEX idx_mythological_narratives_parallels_gin ON mythological_narratives USING gin(cross_cultural_parallels);
CREATE INDEX idx_mythological_narratives_characters_gin ON mythological_narratives USING gin(character_entities);
CREATE INDEX idx_mythological_narratives_locations_gin ON mythological_narratives USING gin(location_references);

CREATE INDEX idx_mythological_entities_names_gin ON mythological_entities USING gin(alternative_names);
CREATE INDEX idx_mythological_entities_domain_gin ON mythological_entities USING gin(domain_of_influence);
CREATE INDEX idx_mythological_entities_abilities_gin ON mythological_entities USING gin(powers_abilities);
CREATE INDEX idx_mythological_entities_symbols_gin ON mythological_entities USING gin(symbols);
CREATE INDEX idx_mythological_entities_associations_gin ON mythological_entities USING gin(associations);
CREATE INDEX idx_mythological_entities_relationships_gin ON mythological_entities USING gin(family_relationships);
CREATE INDEX idx_mythological_entities_locations_gin ON mythological_entities USING gin(location_associations);
CREATE INDEX idx_mythological_entities_seasonal_gin ON mythological_entities USING gin(seasonal_associations);
CREATE INDEX idx_mythological_entities_astronomical_gin ON mythological_entities USING gin(astronomical_connections);

CREATE INDEX idx_cultural_patterns_themes_gin ON cultural_patterns USING gin(universal_themes);
CREATE INDEX idx_cultural_patterns_variations_gin ON cultural_patterns USING gin(cultural_variations);
CREATE INDEX idx_cultural_patterns_references_gin ON cultural_patterns USING gin(research_references);

CREATE INDEX idx_linguistic_analysis_cognates_gin ON linguistic_analysis USING gin(cognates);
CREATE INDEX idx_linguistic_analysis_variations_gin ON linguistic_analysis USING gin(regional_variations);
CREATE INDEX idx_linguistic_analysis_entities_gin ON linguistic_analysis USING gin(related_entities);
CREATE INDEX idx_linguistic_analysis_sources_gin ON linguistic_analysis USING gin(source_references);
CREATE INDEX idx_linguistic_analysis_classification_gin ON linguistic_analysis USING gin(linguistic_classification);

-- Create text search indexes
CREATE INDEX idx_mythological_narratives_text_search ON mythological_narratives USING gin(to_tsvector('english', title || ' ' || summary || ' ' || COALESCE(full_text, '')));
CREATE INDEX idx_cultural_traditions_text_search ON cultural_traditions USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));
CREATE INDEX idx_mythological_entities_text_search ON mythological_entities USING gin(to_tsvector('english', name || ' ' || COALESCE(cultural_role, '')));

-- Apply updated_at triggers
CREATE TRIGGER update_cultural_traditions_updated_at 
    BEFORE UPDATE ON cultural_traditions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mythological_narratives_updated_at 
    BEFORE UPDATE ON mythological_narratives 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mythological_entities_updated_at 
    BEFORE UPDATE ON mythological_entities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cultural_patterns_updated_at 
    BEFORE UPDATE ON cultural_patterns 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cultural_relationships_updated_at 
    BEFORE UPDATE ON cultural_relationships 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cultural_interpretations_updated_at 
    BEFORE UPDATE ON cultural_interpretations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cultural_relevance_updated_at 
    BEFORE UPDATE ON cultural_relevance 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_linguistic_analysis_updated_at 
    BEFORE UPDATE ON linguistic_analysis 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create unique constraint for narrative-pattern relationships
CREATE UNIQUE INDEX idx_narrative_patterns_unique 
ON narrative_patterns(narrative_id, pattern_id);

-- Create constraint to prevent self-referencing cultural relationships
ALTER TABLE cultural_relationships 
ADD CONSTRAINT cultural_relationships_no_self_reference 
CHECK (NOT (source_entity_type = target_entity_type AND source_entity_id = target_entity_id));

-- Create constraint for valid time periods
ALTER TABLE cultural_traditions 
ADD CONSTRAINT cultural_traditions_valid_time_period 
CHECK (time_period_end IS NULL OR time_period_end >= time_period_start);