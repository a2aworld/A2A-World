-- A2A World Platform - Pattern Discovery Schema
-- Tables for storing discovered patterns, clustering results, and statistical validation

SET search_path TO a2a_world, public;

-- Discovered patterns with statistical validation
CREATE TABLE patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    pattern_type VARCHAR(100) CHECK (pattern_type IN (
        'spatial_clustering', 'temporal_correlation', 'cultural_alignment',
        'astronomical_correlation', 'geometric_pattern', 'energy_grid',
        'environmental_correlation', 'mythological_correlation'
    )),
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    statistical_significance DECIMAL(10,8), -- p-value
    effect_size DECIMAL(8,4), -- Cohen's d or similar
    sample_size INTEGER,
    algorithm_used VARCHAR(100),
    algorithm_version VARCHAR(50),
    parameters JSONB,
    discovery_region GEOMETRY(POLYGON, 4326),
    validation_status VARCHAR(50) DEFAULT 'pending' CHECK (validation_status IN (
        'pending', 'validated', 'rejected', 'needs_review', 'partially_validated'
    )),
    validation_consensus_score DECIMAL(5,4), -- Average of all validation scores
    reproducibility_score DECIMAL(5,4), -- How well pattern reproduces in different datasets
    discovered_by_agent VARCHAR(255) NOT NULL,
    discovery_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_validated TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pattern components - the actual data points that make up a pattern
CREATE TABLE pattern_components (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    component_type VARCHAR(100) CHECK (component_type IN (
        'sacred_site', 'geospatial_feature', 'environmental_data', 'cultural_data',
        'geological_feature', 'astronomical_alignment', 'ley_line'
    )),
    component_id UUID NOT NULL, -- Reference to actual component (sacred_sites, etc.)
    relevance_score DECIMAL(5,4) CHECK (relevance_score >= 0 AND relevance_score <= 1),
    component_role VARCHAR(100), -- 'anchor_point', 'connector', 'outlier', 'center'
    distance_to_center DECIMAL(15,2), -- Distance in meters to pattern center
    contribution_weight DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Clustering analysis results
CREATE TABLE clustering_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    clustering_algorithm VARCHAR(100) CHECK (clustering_algorithm IN (
        'kmeans', 'dbscan', 'hierarchical', 'gaussian_mixture', 'spectral',
        'affinity_propagation', 'mean_shift', 'optics'
    )),
    num_clusters INTEGER,
    silhouette_score DECIMAL(5,4), -- Clustering quality metric
    calinski_harabasz_score DECIMAL(10,4),
    davies_bouldin_score DECIMAL(8,4),
    inertia DECIMAL(15,4),
    cluster_centers JSONB, -- Array of cluster center coordinates
    cluster_labels INTEGER[], -- Array of cluster labels for each data point
    outliers_detected INTEGER DEFAULT 0,
    algorithm_parameters JSONB,
    execution_time_ms INTEGER,
    data_dimensions INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Spatial analysis results for geographic patterns
CREATE TABLE spatial_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    analysis_type VARCHAR(100) CHECK (analysis_type IN (
        'nearest_neighbor', 'moran_i', 'getis_ord', 'ripley_k', 'g_function',
        'hotspot_analysis', 'spatial_autocorrelation', 'spatial_regression'
    )),
    test_statistic DECIMAL(15,8),
    p_value DECIMAL(15,12),
    z_score DECIMAL(10,6),
    expected_value DECIMAL(15,8),
    variance DECIMAL(15,8),
    analysis_result VARCHAR(100), -- 'clustered', 'dispersed', 'random'
    significance_level DECIMAL(3,2) DEFAULT 0.05,
    analysis_parameters JSONB,
    spatial_weights_matrix JSONB, -- For spatial analysis algorithms that need it
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cross-correlation analysis between different data types
CREATE TABLE cross_correlations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    primary_data_type VARCHAR(100),
    secondary_data_type VARCHAR(100),
    correlation_coefficient DECIMAL(8,6) CHECK (correlation_coefficient >= -1 AND correlation_coefficient <= 1),
    correlation_type VARCHAR(50) CHECK (correlation_type IN (
        'pearson', 'spearman', 'kendall', 'partial', 'cross_lagged'
    )),
    p_value DECIMAL(15,12),
    confidence_interval_lower DECIMAL(8,6),
    confidence_interval_upper DECIMAL(8,6),
    sample_size INTEGER,
    lag_periods INTEGER DEFAULT 0, -- For time-lagged correlations
    temporal_window_days INTEGER,
    spatial_window_km DECIMAL(10,2),
    correlation_strength VARCHAR(50), -- 'weak', 'moderate', 'strong'
    statistical_significance BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pattern validation results from different validators (agents, experts, peer review)
CREATE TABLE pattern_validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    validation_type VARCHAR(100) CHECK (validation_type IN (
        'statistical_test', 'cross_validation', 'bootstrap', 'permutation_test',
        'expert_review', 'peer_review', 'replication_study', 'sensitivity_analysis'
    )),
    validator_type VARCHAR(50) CHECK (validator_type IN (
        'agent', 'human_expert', 'peer_reviewer', 'automated_system'
    )),
    validator_id VARCHAR(255), -- Agent ID or user ID
    validation_result VARCHAR(50) CHECK (validation_result IN (
        'approved', 'rejected', 'needs_revision', 'inconclusive'
    )),
    validation_score DECIMAL(5,4) CHECK (validation_score >= 0 AND validation_score <= 1),
    confidence_level DECIMAL(5,4),
    validation_method TEXT,
    test_statistics JSONB, -- Store various test statistics
    validation_notes TEXT,
    replication_successful BOOLEAN,
    limitations_noted TEXT,
    recommendations TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pattern relationships - how patterns relate to or depend on each other
CREATE TABLE pattern_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    primary_pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    related_pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) CHECK (relationship_type IN (
        'contains', 'overlaps', 'adjacent', 'similar', 'opposite',
        'causal_predecessor', 'causal_successor', 'correlated', 'independent'
    )),
    relationship_strength DECIMAL(5,4) CHECK (relationship_strength >= 0 AND relationship_strength <= 1),
    spatial_overlap_area DECIMAL(15,2), -- Square meters of overlap
    temporal_overlap_days INTEGER,
    statistical_correlation DECIMAL(8,6),
    dependency_score DECIMAL(5,4),
    validation_status VARCHAR(50) DEFAULT 'unverified',
    discovered_by_agent VARCHAR(255),
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pattern evolution tracking - how patterns change over time
CREATE TABLE pattern_evolution (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    change_type VARCHAR(100) CHECK (change_type IN (
        'refinement', 'extension', 'contraction', 'split', 'merge',
        'parameter_update', 'validation_update', 'correction'
    )),
    changes_summary TEXT,
    previous_confidence DECIMAL(5,4),
    new_confidence DECIMAL(5,4),
    previous_parameters JSONB,
    new_parameters JSONB,
    change_significance DECIMAL(5,4),
    changed_by_agent VARCHAR(255),
    change_reason TEXT,
    validation_impact TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for pattern discovery tables
CREATE INDEX idx_patterns_type ON patterns(pattern_type);
CREATE INDEX idx_patterns_confidence ON patterns(confidence_score DESC);
CREATE INDEX idx_patterns_validation_status ON patterns(validation_status);
CREATE INDEX idx_patterns_discovery_region ON patterns USING GIST (discovery_region);
CREATE INDEX idx_patterns_discovered_by ON patterns(discovered_by_agent);
CREATE INDEX idx_patterns_discovery_timestamp ON patterns(discovery_timestamp);
CREATE INDEX idx_patterns_statistical_significance ON patterns(statistical_significance);

CREATE INDEX idx_pattern_components_pattern ON pattern_components(pattern_id);
CREATE INDEX idx_pattern_components_type ON pattern_components(component_type);
CREATE INDEX idx_pattern_components_component ON pattern_components(component_id);
CREATE INDEX idx_pattern_components_relevance ON pattern_components(relevance_score DESC);

CREATE INDEX idx_clustering_results_pattern ON clustering_results(pattern_id);
CREATE INDEX idx_clustering_results_algorithm ON clustering_results(clustering_algorithm);
CREATE INDEX idx_clustering_results_silhouette ON clustering_results(silhouette_score DESC);
CREATE INDEX idx_clustering_results_clusters ON clustering_results(num_clusters);

CREATE INDEX idx_spatial_analysis_pattern ON spatial_analysis(pattern_id);
CREATE INDEX idx_spatial_analysis_type ON spatial_analysis(analysis_type);
CREATE INDEX idx_spatial_analysis_p_value ON spatial_analysis(p_value);
CREATE INDEX idx_spatial_analysis_result ON spatial_analysis(analysis_result);

CREATE INDEX idx_cross_correlations_pattern ON cross_correlations(pattern_id);
CREATE INDEX idx_cross_correlations_types ON cross_correlations(primary_data_type, secondary_data_type);
CREATE INDEX idx_cross_correlations_coefficient ON cross_correlations(abs(correlation_coefficient) DESC);
CREATE INDEX idx_cross_correlations_significance ON cross_correlations(statistical_significance);

CREATE INDEX idx_pattern_validations_pattern ON pattern_validations(pattern_id);
CREATE INDEX idx_pattern_validations_validator ON pattern_validations(validator_id);
CREATE INDEX idx_pattern_validations_type ON pattern_validations(validation_type);
CREATE INDEX idx_pattern_validations_result ON pattern_validations(validation_result);
CREATE INDEX idx_pattern_validations_score ON pattern_validations(validation_score DESC);

CREATE INDEX idx_pattern_relationships_primary ON pattern_relationships(primary_pattern_id);
CREATE INDEX idx_pattern_relationships_related ON pattern_relationships(related_pattern_id);
CREATE INDEX idx_pattern_relationships_type ON pattern_relationships(relationship_type);
CREATE INDEX idx_pattern_relationships_strength ON pattern_relationships(relationship_strength DESC);

CREATE INDEX idx_pattern_evolution_pattern ON pattern_evolution(pattern_id);
CREATE INDEX idx_pattern_evolution_version ON pattern_evolution(pattern_id, version_number);
CREATE INDEX idx_pattern_evolution_type ON pattern_evolution(change_type);
CREATE INDEX idx_pattern_evolution_created ON pattern_evolution(created_at);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_patterns_parameters_gin ON patterns USING gin(parameters);
CREATE INDEX idx_patterns_metadata_gin ON patterns USING gin(metadata);
CREATE INDEX idx_pattern_components_metadata_gin ON pattern_components USING gin(metadata);
CREATE INDEX idx_clustering_results_parameters_gin ON clustering_results USING gin(algorithm_parameters);
CREATE INDEX idx_clustering_results_centers_gin ON clustering_results USING gin(cluster_centers);
CREATE INDEX idx_spatial_analysis_parameters_gin ON spatial_analysis USING gin(analysis_parameters);
CREATE INDEX idx_cross_correlations_metadata_gin ON cross_correlations USING gin(metadata);
CREATE INDEX idx_pattern_validations_statistics_gin ON pattern_validations USING gin(test_statistics);
CREATE INDEX idx_pattern_relationships_metadata_gin ON pattern_relationships USING gin(metadata);
CREATE INDEX idx_pattern_evolution_prev_params_gin ON pattern_evolution USING gin(previous_parameters);
CREATE INDEX idx_pattern_evolution_new_params_gin ON pattern_evolution USING gin(new_parameters);

-- Create array indexes
CREATE INDEX idx_clustering_results_labels_gin ON clustering_results USING gin(cluster_labels);

-- Apply updated_at triggers
CREATE TRIGGER update_patterns_updated_at 
    BEFORE UPDATE ON patterns 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pattern_relationships_updated_at 
    BEFORE UPDATE ON pattern_relationships 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create constraint to prevent self-referencing pattern relationships
ALTER TABLE pattern_relationships 
ADD CONSTRAINT pattern_relationships_no_self_reference 
CHECK (primary_pattern_id != related_pattern_id);

-- Create unique constraint for pattern evolution versions
CREATE UNIQUE INDEX idx_pattern_evolution_unique_version 
ON pattern_evolution(pattern_id, version_number);