-- A2A World Platform - Enhanced Statistical Validation Schema
-- Tables for comprehensive statistical validation including Moran's I analysis,
-- null hypothesis testing, spatial statistics, and significance classification

SET search_path TO a2a_world, public;

-- Enhanced statistical validation results table
-- Stores comprehensive results from statistical validation framework
CREATE TABLE enhanced_statistical_validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES patterns(id) ON DELETE CASCADE,
    validation_session_id UUID DEFAULT uuid_generate_v4(), -- Groups related validations
    validation_framework_version VARCHAR(50) DEFAULT '1.0',
    validation_methods TEXT[] NOT NULL, -- Array of methods used
    total_statistical_tests INTEGER DEFAULT 0,
    significant_tests INTEGER DEFAULT 0,
    highly_significant_tests INTEGER DEFAULT 0, -- p < 0.001
    overall_significance_classification VARCHAR(50) CHECK (overall_significance_classification IN (
        'very_high', 'high', 'moderate', 'low', 'not_significant'
    )),
    reliability_score DECIMAL(5,4) CHECK (reliability_score >= 0 AND reliability_score <= 1),
    confidence_level DECIMAL(5,4) DEFAULT 0.95,
    sample_size INTEGER,
    validation_success BOOLEAN DEFAULT true,
    processing_time_ms INTEGER,
    performed_by_agent VARCHAR(255) NOT NULL,
    validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    recommendations TEXT[],
    validation_summary JSONB,
    raw_results JSONB, -- Store complete validation results
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Moran's I analysis results (both global and local)
CREATE TABLE morans_i_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_id UUID REFERENCES enhanced_statistical_validations(id) ON DELETE CASCADE,
    analysis_type VARCHAR(20) CHECK (analysis_type IN ('global', 'local')),
    weights_method VARCHAR(50) DEFAULT 'knn', -- 'knn', 'distance', 'queen', 'rook'
    k_neighbors INTEGER DEFAULT 8,
    distance_threshold DECIMAL(15,4),
    
    -- Global Moran's I fields
    morans_i_statistic DECIMAL(15,8),
    expected_value DECIMAL(15,8),
    variance DECIMAL(15,8),
    z_score DECIMAL(10,6),
    p_value DECIMAL(15,12),
    monte_carlo_p_value DECIMAL(15,12),
    n_permutations INTEGER DEFAULT 999,
    confidence_interval_lower DECIMAL(15,8),
    confidence_interval_upper DECIMAL(15,8),
    
    -- Local Moran's I fields (for LISA analysis)
    local_statistics JSONB, -- Array of local statistics for each location
    significant_locations INTEGER DEFAULT 0,
    bonferroni_significant INTEGER DEFAULT 0,
    cluster_counts JSONB, -- HH, LL, HL, LH counts
    bonferroni_alpha DECIMAL(15,12),
    
    significant BOOLEAN DEFAULT false,
    significance_level DECIMAL(5,4) DEFAULT 0.05,
    interpretation TEXT,
    spatial_pattern VARCHAR(50), -- 'clustered', 'dispersed', 'random'
    effect_size DECIMAL(8,4),
    
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Null hypothesis test results (Monte Carlo, Bootstrap, CSR)
CREATE TABLE null_hypothesis_tests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_id UUID REFERENCES enhanced_statistical_validations(id) ON DELETE CASCADE,
    test_type VARCHAR(50) CHECK (test_type IN (
        'monte_carlo_permutation', 'bootstrap_validation', 'csr_ripley_k', 
        'nearest_neighbor_analysis', 'complete_spatial_randomness'
    )),
    
    -- Common fields
    observed_statistic DECIMAL(15,8),
    test_statistic_name VARCHAR(100),
    p_value DECIMAL(15,12),
    significance_level DECIMAL(5,4) DEFAULT 0.05,
    significant BOOLEAN DEFAULT false,
    effect_size DECIMAL(8,4),
    confidence_interval_lower DECIMAL(15,8),
    confidence_interval_upper DECIMAL(15,8),
    
    -- Monte Carlo specific fields
    n_permutations INTEGER,
    n_more_extreme INTEGER,
    null_distribution_mean DECIMAL(15,8),
    null_distribution_std DECIMAL(15,8),
    
    -- Bootstrap specific fields
    n_bootstrap INTEGER,
    bootstrap_mean DECIMAL(15,8),
    bootstrap_std DECIMAL(15,8),
    bootstrap_se DECIMAL(15,8),
    bias_estimate DECIMAL(15,8),
    confidence_level DECIMAL(5,4),
    bootstrap_method VARCHAR(50), -- 'percentile', 'bias-corrected', 'bca'
    
    -- CSR/Ripley's K specific fields
    study_area_bounds DECIMAL(10,6)[4], -- [min_x, min_y, max_x, max_y]
    point_intensity DECIMAL(15,8),
    ripley_k_values DECIMAL(15,8)[],
    expected_k_values DECIMAL(15,8)[],
    l_function_values DECIMAL(15,8)[],
    distance_bands DECIMAL(10,4)[],
    pattern_classification VARCHAR(50), -- 'clustered', 'dispersed', 'random'
    max_deviation DECIMAL(15,8),
    
    -- Nearest neighbor specific fields
    observed_mean_distance DECIMAL(15,8),
    expected_mean_distance DECIMAL(15,8),
    nearest_neighbor_ratio DECIMAL(8,4),
    standard_error DECIMAL(15,8),
    z_score DECIMAL(10,6),
    
    interpretation TEXT,
    test_parameters JSONB,
    raw_results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Getis-Ord Gi* hotspot analysis results
CREATE TABLE getis_ord_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_id UUID REFERENCES enhanced_statistical_validations(id) ON DELETE CASCADE,
    weights_method VARCHAR(50) DEFAULT 'distance',
    distance_threshold DECIMAL(15,4),
    total_locations INTEGER,
    significant_hotspots INTEGER DEFAULT 0,
    significant_coldspots INTEGER DEFAULT 0,
    bonferroni_significant INTEGER DEFAULT 0,
    
    -- Hotspot locations and details
    hotspot_locations JSONB, -- Array of hotspot coordinates and statistics
    coldspot_locations JSONB, -- Array of coldspot coordinates and statistics
    
    -- Statistical results for each location
    gi_star_statistics JSONB, -- Complete Gi* statistics for all locations
    
    -- Multiple testing correction
    original_significant INTEGER DEFAULT 0,
    corrected_alpha DECIMAL(15,12),
    correction_method VARCHAR(50) DEFAULT 'bonferroni',
    
    interpretation TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Spatial concentration and advanced spatial metrics
CREATE TABLE spatial_concentration_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_id UUID REFERENCES enhanced_statistical_validations(id) ON DELETE CASCADE,
    
    -- Gini coefficient for spatial inequality
    gini_coefficient DECIMAL(8,6) CHECK (gini_coefficient >= 0 AND gini_coefficient <= 1),
    gini_interpretation VARCHAR(100),
    
    -- Location quotients
    location_quotients DECIMAL(8,4)[],
    lq_categories TEXT[],
    significant_concentrations INTEGER DEFAULT 0,
    
    -- Spatial association matrix results
    association_matrix DECIMAL(6,4)[][], -- For categorical spatial data
    association_indices JSONB, -- Category pair association indices
    strong_associations TEXT[],
    
    -- Silhouette analysis for cluster quality
    overall_silhouette_score DECIMAL(5,4),
    cluster_silhouette_scores JSONB, -- Per-cluster analysis
    cluster_quality_assessment VARCHAR(50), -- 'excellent', 'good', 'fair', 'poor'
    n_clusters INTEGER,
    n_points_analyzed INTEGER,
    
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pattern significance classification results
CREATE TABLE pattern_significance_classification (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_id UUID REFERENCES enhanced_statistical_validations(id) ON DELETE CASCADE,
    
    -- Overall classification
    overall_classification VARCHAR(50) CHECK (overall_classification IN (
        'very_high', 'high', 'moderate', 'low', 'not_significant'
    )),
    reliability_score DECIMAL(5,4) CHECK (reliability_score >= 0 AND reliability_score <= 1),
    
    -- P-value summary statistics
    min_p_value DECIMAL(15,12),
    mean_p_value DECIMAL(15,12),
    median_p_value DECIMAL(15,12),
    geometric_mean_p_value DECIMAL(15,12),
    
    -- Significance breakdown by levels
    very_high_significant INTEGER DEFAULT 0, -- p < 0.001
    high_significant INTEGER DEFAULT 0,      -- p < 0.01
    moderate_significant INTEGER DEFAULT 0,   -- p < 0.05
    low_significant INTEGER DEFAULT 0,        -- p < 0.10
    
    -- Component scores for reliability calculation
    statistical_significance_score DECIMAL(5,4),
    effect_size_score DECIMAL(5,4),
    consistency_score DECIMAL(5,4),
    sample_size_score DECIMAL(5,4),
    multiple_testing_score DECIMAL(5,4),
    
    -- Confidence metrics
    mean_ci_width DECIMAL(15,8),
    median_ci_width DECIMAL(15,8),
    mean_z_score DECIMAL(10,6),
    max_z_score DECIMAL(10,6),
    
    interpretation TEXT,
    recommendations TEXT[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Multiple comparison corrections
CREATE TABLE multiple_comparison_corrections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_id UUID REFERENCES enhanced_statistical_validations(id) ON DELETE CASCADE,
    correction_method VARCHAR(50) CHECK (correction_method IN (
        'bonferroni', 'holm', 'fdr_bh', 'fdr_by', 'none'
    )),
    original_alpha DECIMAL(5,4) DEFAULT 0.05,
    corrected_alpha DECIMAL(15,12),
    original_p_values DECIMAL(15,12)[],
    corrected_p_values DECIMAL(15,12)[],
    test_names TEXT[],
    
    -- Results
    n_tests INTEGER,
    n_significant_original INTEGER,
    n_significant_corrected INTEGER,
    family_wise_error_rate DECIMAL(5,4),
    false_discovery_rate DECIMAL(5,4),
    
    significant_after_correction BOOLEAN[],
    correction_impact_summary TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Statistical validation reports
CREATE TABLE statistical_validation_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_id UUID REFERENCES enhanced_statistical_validations(id) ON DELETE CASCADE,
    report_type VARCHAR(50) CHECK (report_type IN (
        'comprehensive', 'summary', 'executive', 'technical', 'dashboard'
    )),
    report_format VARCHAR(20) CHECK (report_format IN ('json', 'html', 'pdf', 'csv')),
    
    -- Report content
    report_title TEXT,
    executive_summary TEXT,
    detailed_findings JSONB,
    statistical_tables JSONB,
    visualization_data JSONB,
    conclusions TEXT[],
    recommendations TEXT[],
    limitations TEXT[],
    
    -- Report metadata
    generated_by_agent VARCHAR(255),
    generation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    report_version VARCHAR(20) DEFAULT '1.0',
    include_visualizations BOOLEAN DEFAULT true,
    
    -- File storage (if applicable)
    report_file_path TEXT,
    report_file_size INTEGER,
    report_checksum VARCHAR(64),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Batch validation results for multiple patterns
CREATE TABLE batch_validation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id UUID DEFAULT uuid_generate_v4(),
    initiated_by_agent VARCHAR(255),
    batch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Batch configuration
    validation_methods TEXT[],
    pattern_ids UUID[],
    max_parallel INTEGER DEFAULT 4,
    
    -- Batch results summary
    total_patterns INTEGER,
    successful_validations INTEGER,
    failed_validations INTEGER,
    success_rate DECIMAL(5,4),
    highly_significant_patterns INTEGER,
    significance_rate DECIMAL(5,4),
    
    -- Timing and performance
    total_processing_time_ms INTEGER,
    avg_processing_time_per_pattern_ms INTEGER,
    
    -- Individual pattern results (references to enhanced_statistical_validations)
    validation_ids UUID[],
    failed_pattern_ids UUID[],
    failure_reasons TEXT[],
    
    batch_summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Validation performance metrics and monitoring
CREATE TABLE validation_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    agent_id VARCHAR(255),
    
    -- Performance counters
    validations_performed INTEGER DEFAULT 0,
    statistical_tests_executed INTEGER DEFAULT 0,
    highly_significant_patterns INTEGER DEFAULT 0,
    validation_reports_generated INTEGER DEFAULT 0,
    
    -- Performance rates
    enhanced_significance_rate DECIMAL(5,4),
    avg_tests_per_validation DECIMAL(8,2),
    validation_success_rate DECIMAL(5,4),
    
    -- Resource utilization
    avg_processing_time_ms INTEGER,
    max_processing_time_ms INTEGER,
    cache_hit_rate DECIMAL(5,4),
    memory_usage_mb INTEGER,
    
    -- Quality metrics
    avg_reliability_score DECIMAL(5,4),
    statistical_framework_errors INTEGER DEFAULT 0,
    
    metric_period_start TIMESTAMP WITH TIME ZONE,
    metric_period_end TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create comprehensive indexes for enhanced statistical validation tables
CREATE INDEX idx_enhanced_validations_pattern ON enhanced_statistical_validations(pattern_id);
CREATE INDEX idx_enhanced_validations_session ON enhanced_statistical_validations(validation_session_id);
CREATE INDEX idx_enhanced_validations_classification ON enhanced_statistical_validations(overall_significance_classification);
CREATE INDEX idx_enhanced_validations_reliability ON enhanced_statistical_validations(reliability_score DESC);
CREATE INDEX idx_enhanced_validations_agent ON enhanced_statistical_validations(performed_by_agent);
CREATE INDEX idx_enhanced_validations_timestamp ON enhanced_statistical_validations(validation_timestamp);
CREATE INDEX idx_enhanced_validations_success ON enhanced_statistical_validations(validation_success);

CREATE INDEX idx_morans_i_validation ON morans_i_analysis(validation_id);
CREATE INDEX idx_morans_i_type ON morans_i_analysis(analysis_type);
CREATE INDEX idx_morans_i_significant ON morans_i_analysis(significant);
CREATE INDEX idx_morans_i_p_value ON morans_i_analysis(p_value);
CREATE INDEX idx_morans_i_statistic ON morans_i_analysis(morans_i_statistic);

CREATE INDEX idx_null_hypothesis_validation ON null_hypothesis_tests(validation_id);
CREATE INDEX idx_null_hypothesis_type ON null_hypothesis_tests(test_type);
CREATE INDEX idx_null_hypothesis_significant ON null_hypothesis_tests(significant);
CREATE INDEX idx_null_hypothesis_p_value ON null_hypothesis_tests(p_value);

CREATE INDEX idx_getis_ord_validation ON getis_ord_analysis(validation_id);
CREATE INDEX idx_getis_ord_hotspots ON getis_ord_analysis(significant_hotspots);
CREATE INDEX idx_getis_ord_coldspots ON getis_ord_analysis(significant_coldspots);

CREATE INDEX idx_spatial_concentration_validation ON spatial_concentration_metrics(validation_id);
CREATE INDEX idx_spatial_concentration_gini ON spatial_concentration_metrics(gini_coefficient);
CREATE INDEX idx_spatial_concentration_silhouette ON spatial_concentration_metrics(overall_silhouette_score DESC);
CREATE INDEX idx_spatial_concentration_quality ON spatial_concentration_metrics(cluster_quality_assessment);

CREATE INDEX idx_significance_classification_validation ON pattern_significance_classification(validation_id);
CREATE INDEX idx_significance_classification_overall ON pattern_significance_classification(overall_classification);
CREATE INDEX idx_significance_classification_reliability ON pattern_significance_classification(reliability_score DESC);
CREATE INDEX idx_significance_classification_min_p ON pattern_significance_classification(min_p_value);

CREATE INDEX idx_multiple_comparison_validation ON multiple_comparison_corrections(validation_id);
CREATE INDEX idx_multiple_comparison_method ON multiple_comparison_corrections(correction_method);
CREATE INDEX idx_multiple_comparison_significant ON multiple_comparison_corrections(n_significant_corrected);

CREATE INDEX idx_validation_reports_validation ON statistical_validation_reports(validation_id);
CREATE INDEX idx_validation_reports_type ON statistical_validation_reports(report_type);
CREATE INDEX idx_validation_reports_generated ON statistical_validation_reports(generation_timestamp);
CREATE INDEX idx_validation_reports_agent ON statistical_validation_reports(generated_by_agent);

CREATE INDEX idx_batch_validation_batch_id ON batch_validation_results(batch_id);
CREATE INDEX idx_batch_validation_timestamp ON batch_validation_results(batch_timestamp);
CREATE INDEX idx_batch_validation_agent ON batch_validation_results(initiated_by_agent);
CREATE INDEX idx_batch_validation_success_rate ON batch_validation_results(success_rate DESC);

CREATE INDEX idx_validation_performance_timestamp ON validation_performance_metrics(metric_timestamp);
CREATE INDEX idx_validation_performance_agent ON validation_performance_metrics(agent_id);
CREATE INDEX idx_validation_performance_significance ON validation_performance_metrics(enhanced_significance_rate DESC);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_enhanced_validations_methods_gin ON enhanced_statistical_validations USING gin(to_tsvector('english', array_to_string(validation_methods, ' ')));
CREATE INDEX idx_enhanced_validations_summary_gin ON enhanced_statistical_validations USING gin(validation_summary);
CREATE INDEX idx_enhanced_validations_results_gin ON enhanced_statistical_validations USING gin(raw_results);
CREATE INDEX idx_enhanced_validations_recommendations_gin ON enhanced_statistical_validations USING gin(to_tsvector('english', array_to_string(recommendations, ' ')));

CREATE INDEX idx_morans_i_local_stats_gin ON morans_i_analysis USING gin(local_statistics);
CREATE INDEX idx_morans_i_cluster_counts_gin ON morans_i_analysis USING gin(cluster_counts);
CREATE INDEX idx_morans_i_metadata_gin ON morans_i_analysis USING gin(metadata);

CREATE INDEX idx_null_hypothesis_parameters_gin ON null_hypothesis_tests USING gin(test_parameters);
CREATE INDEX idx_null_hypothesis_results_gin ON null_hypothesis_tests USING gin(raw_results);

CREATE INDEX idx_getis_ord_hotspots_gin ON getis_ord_analysis USING gin(hotspot_locations);
CREATE INDEX idx_getis_ord_coldspots_gin ON getis_ord_analysis USING gin(coldspot_locations);
CREATE INDEX idx_getis_ord_stats_gin ON getis_ord_analysis USING gin(gi_star_statistics);

CREATE INDEX idx_spatial_concentration_metadata_gin ON spatial_concentration_metrics USING gin(metadata);
CREATE INDEX idx_spatial_concentration_silhouette_gin ON spatial_concentration_metrics USING gin(cluster_silhouette_scores);

CREATE INDEX idx_validation_reports_findings_gin ON statistical_validation_reports USING gin(detailed_findings);
CREATE INDEX idx_validation_reports_tables_gin ON statistical_validation_reports USING gin(statistical_tables);
CREATE INDEX idx_validation_reports_viz_gin ON statistical_validation_reports USING gin(visualization_data);

-- Create array indexes for better query performance
CREATE INDEX idx_multiple_comparison_p_values_gin ON multiple_comparison_corrections USING gin(original_p_values);
CREATE INDEX idx_multiple_comparison_corrected_gin ON multiple_comparison_corrections USING gin(corrected_p_values);
CREATE INDEX idx_multiple_comparison_names_gin ON multiple_comparison_corrections USING gin(to_tsvector('english', array_to_string(test_names, ' ')));

CREATE INDEX idx_batch_validation_pattern_ids_gin ON batch_validation_results USING gin(pattern_ids);
CREATE INDEX idx_batch_validation_methods_gin ON batch_validation_results USING gin(to_tsvector('english', array_to_string(validation_methods, ' ')));

-- Apply updated_at triggers where needed
CREATE TRIGGER update_enhanced_validations_updated_at 
    BEFORE UPDATE ON enhanced_statistical_validations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries

-- View for validation summary with pattern details
CREATE VIEW validation_summary_view AS
SELECT 
    esv.id AS validation_id,
    esv.pattern_id,
    p.name AS pattern_name,
    p.pattern_type,
    esv.overall_significance_classification,
    esv.reliability_score,
    esv.total_statistical_tests,
    esv.significant_tests,
    esv.highly_significant_tests,
    esv.sample_size,
    esv.validation_timestamp,
    esv.performed_by_agent,
    p.discovery_timestamp,
    p.discovered_by_agent AS pattern_discovered_by
FROM enhanced_statistical_validations esv
JOIN patterns p ON esv.pattern_id = p.id
WHERE esv.validation_success = true;

-- View for highly significant patterns with detailed metrics
CREATE VIEW highly_significant_patterns_view AS
SELECT 
    esv.pattern_id,
    p.name AS pattern_name,
    p.pattern_type,
    esv.overall_significance_classification,
    esv.reliability_score,
    psc.min_p_value,
    psc.mean_p_value,
    psc.very_high_significant,
    psc.high_significant,
    esv.validation_timestamp,
    mi.morans_i_statistic,
    mi.p_value AS morans_i_p_value,
    go.significant_hotspots,
    go.significant_coldspots
FROM enhanced_statistical_validations esv
JOIN patterns p ON esv.pattern_id = p.id
LEFT JOIN pattern_significance_classification psc ON esv.id = psc.validation_id
LEFT JOIN morans_i_analysis mi ON esv.id = mi.validation_id AND mi.analysis_type = 'global'
LEFT JOIN getis_ord_analysis go ON esv.id = go.validation_id
WHERE esv.overall_significance_classification IN ('very_high', 'high')
    AND esv.validation_success = true;

-- View for validation performance dashboard
CREATE VIEW validation_performance_dashboard AS
SELECT 
    date_trunc('day', esv.validation_timestamp) AS validation_date,
    COUNT(*) AS total_validations,
    COUNT(*) FILTER (WHERE esv.overall_significance_classification IN ('very_high', 'high')) AS highly_significant_count,
    AVG(esv.reliability_score) AS avg_reliability_score,
    AVG(esv.total_statistical_tests) AS avg_tests_per_validation,
    AVG(esv.processing_time_ms) AS avg_processing_time_ms,
    COUNT(DISTINCT esv.performed_by_agent) AS active_agents
FROM enhanced_statistical_validations esv
WHERE esv.validation_success = true
    AND esv.validation_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY date_trunc('day', esv.validation_timestamp)
ORDER BY validation_date DESC;

-- Create materialized view for pattern validation statistics (for performance)
CREATE MATERIALIZED VIEW pattern_validation_statistics AS
SELECT 
    p.pattern_type,
    COUNT(*) AS total_patterns,
    COUNT(*) FILTER (WHERE esv.id IS NOT NULL) AS validated_patterns,
    COUNT(*) FILTER (WHERE esv.overall_significance_classification IN ('very_high', 'high')) AS highly_significant_patterns,
    AVG(esv.reliability_score) AS avg_reliability_score,
    AVG(psc.min_p_value) AS avg_min_p_value,
    COUNT(*) FILTER (WHERE mi.significant = true) AS morans_i_significant_count,
    AVG(mi.morans_i_statistic) AS avg_morans_i_statistic,
    SUM(go.significant_hotspots) AS total_hotspots,
    SUM(go.significant_coldspots) AS total_coldspots
FROM patterns p
LEFT JOIN enhanced_statistical_validations esv ON p.id = esv.pattern_id
LEFT JOIN pattern_significance_classification psc ON esv.id = psc.validation_id
LEFT JOIN morans_i_analysis mi ON esv.id = mi.validation_id AND mi.analysis_type = 'global'
LEFT JOIN getis_ord_analysis go ON esv.id = go.validation_id
WHERE esv.validation_success IS NULL OR esv.validation_success = true
GROUP BY p.pattern_type;

-- Create index on materialized view
CREATE INDEX idx_pattern_validation_stats_type ON pattern_validation_statistics(pattern_type);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_pattern_validation_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW pattern_validation_statistics;
END;
$$ LANGUAGE plpgsql;

-- Add comments for documentation
COMMENT ON TABLE enhanced_statistical_validations IS 'Comprehensive statistical validation results using enhanced validation framework';
COMMENT ON TABLE morans_i_analysis IS 'Global and Local Moran''s I spatial autocorrelation analysis results';
COMMENT ON TABLE null_hypothesis_tests IS 'Monte Carlo, Bootstrap, and Complete Spatial Randomness test results';
COMMENT ON TABLE getis_ord_analysis IS 'Getis-Ord Gi* hotspot and coldspot analysis results';
COMMENT ON TABLE spatial_concentration_metrics IS 'Spatial concentration indices including Gini coefficient and location quotients';
COMMENT ON TABLE pattern_significance_classification IS 'Multi-tier significance classification with reliability scoring';
COMMENT ON TABLE multiple_comparison_corrections IS 'Multiple comparison correction results (Bonferroni, FDR, etc.)';
COMMENT ON TABLE statistical_validation_reports IS 'Generated statistical validation reports and documentation';
COMMENT ON TABLE batch_validation_results IS 'Results from batch validation of multiple patterns';
COMMENT ON TABLE validation_performance_metrics IS 'Performance monitoring metrics for validation agents';

-- Grant appropriate permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA a2a_world TO a2a_validation_agents;
-- GRANT SELECT ON ALL TABLES IN SCHEMA a2a_world TO a2a_readonly_users;
-- GRANT EXECUTE ON FUNCTION refresh_pattern_validation_statistics() TO a2a_validation_agents;