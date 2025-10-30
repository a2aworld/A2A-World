/**
 * A2A World Platform - MARL Optimization Schema
 *
 * Database schema for Multi-Agent Reinforcement Learning (MARL) optimization
 * of HDBSCAN clustering parameters and performance tracking.
 */

-- MARL-learned HDBSCAN parameters for different datasets
CREATE TABLE IF NOT EXISTS marl_parameters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id VARCHAR(255) NOT NULL,
    algorithm VARCHAR(100) DEFAULT 'hdbscan',
    min_samples INTEGER NOT NULL CHECK (min_samples > 0),
    min_cluster_size INTEGER NOT NULL CHECK (min_cluster_size > 0),
    cluster_selection_epsilon NUMERIC(5, 3) DEFAULT 0.0 CHECK (cluster_selection_epsilon >= 0),
    performance_score NUMERIC(5, 4) CHECK (performance_score >= 0 AND performance_score <= 1),
    confidence_level NUMERIC(5, 4) CHECK (confidence_level >= 0 AND confidence_level <= 1),
    training_episodes INTEGER,
    convergence_achieved BOOLEAN DEFAULT FALSE,
    learned_by_agent VARCHAR(255) NOT NULL,
    learning_timestamp TIMESTAMPTZ,
    last_used TIMESTAMPTZ,
    usage_count INTEGER DEFAULT 0,
    parameters_metadata JSONB,
    validation_metrics JSONB,

    -- Indexes
    INDEX idx_marl_parameters_dataset (dataset_id),
    INDEX idx_marl_parameters_performance (performance_score DESC),
    INDEX idx_marl_parameters_agent (learned_by_agent),
    INDEX idx_marl_parameters_timestamp (learning_timestamp)
);

-- Evaluation results for MARL-learned parameters
CREATE TABLE IF NOT EXISTS marl_parameter_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parameters_id UUID NOT NULL REFERENCES marl_parameters(id) ON DELETE CASCADE,
    evaluation_dataset VARCHAR(255),
    silhouette_score NUMERIC(5, 4) CHECK (silhouette_score >= -1 AND silhouette_score <= 1),
    calinski_harabasz_score NUMERIC(10, 4),
    davies_bouldin_score NUMERIC(8, 4),
    num_clusters_found INTEGER,
    noise_ratio NUMERIC(5, 4) CHECK (noise_ratio >= 0 AND noise_ratio <= 1),
    pattern_quality_score NUMERIC(5, 4) CHECK (pattern_quality_score >= 0 AND pattern_quality_score <= 1),
    spatial_coherence NUMERIC(5, 4) CHECK (spatial_coherence >= 0 AND spatial_coherence <= 1),
    significance_score NUMERIC(5, 4) CHECK (significance_score >= 0 AND significance_score <= 1),
    evaluation_timestamp TIMESTAMPTZ,
    evaluation_method VARCHAR(100),
    evaluation_metadata JSONB,

    -- Indexes
    INDEX idx_marl_evaluations_parameters (parameters_id),
    INDEX idx_marl_evaluations_dataset (evaluation_dataset),
    INDEX idx_marl_evaluations_timestamp (evaluation_timestamp)
);

-- MARL training session records
CREATE TABLE IF NOT EXISTS marl_training_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    dataset_id VARCHAR(255),
    num_agents INTEGER DEFAULT 1,
    total_timesteps INTEGER,
    episodes_completed INTEGER,
    best_reward NUMERIC(10, 4),
    average_reward NUMERIC(10, 4),
    convergence_achieved BOOLEAN DEFAULT FALSE,
    convergence_episode INTEGER,
    training_start_time TIMESTAMPTZ,
    training_end_time TIMESTAMPTZ,
    training_duration_seconds INTEGER,
    model_saved BOOLEAN DEFAULT FALSE,
    model_path VARCHAR(500),
    training_config JSONB,
    performance_metrics JSONB,
    collaboration_metrics JSONB,

    -- Indexes
    INDEX idx_marl_sessions_agent (agent_id),
    INDEX idx_marl_sessions_dataset (dataset_id),
    INDEX idx_marl_sessions_start_time (training_start_time)
);

-- Parameters learned during MARL training sessions
CREATE TABLE IF NOT EXISTS marl_training_parameters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL REFERENCES marl_training_sessions(session_id) ON DELETE CASCADE,
    parameter_name VARCHAR(100),
    final_value NUMERIC(10, 4),
    best_value NUMERIC(10, 4),
    parameter_range_min NUMERIC(10, 4),
    parameter_range_max NUMERIC(10, 4),
    optimization_history JSONB,
    convergence_value NUMERIC(10, 4),
    parameter_metadata JSONB,

    -- Indexes
    INDEX idx_marl_training_params_session (session_id),
    INDEX idx_marl_training_params_name (parameter_name)
);

-- Performance tracking for individual MARL agents
CREATE TABLE IF NOT EXISTS marl_agent_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) REFERENCES marl_training_sessions(session_id) ON DELETE CASCADE,
    episode_number INTEGER,
    episode_reward NUMERIC(10, 4) CHECK (episode_reward >= -1000 AND episode_reward <= 1000),
    episode_length INTEGER,
    parameters_used JSONB,
    clustering_metrics JSONB,
    collaboration_contribution NUMERIC(5, 4) CHECK (collaboration_contribution >= 0 AND collaboration_contribution <= 1),
    knowledge_shared BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMPTZ,

    -- Indexes
    INDEX idx_marl_agent_perf_agent (agent_id),
    INDEX idx_marl_agent_perf_session (session_id),
    INDEX idx_marl_agent_perf_episode (episode_number),
    INDEX idx_marl_agent_perf_timestamp (timestamp)
);

-- Comments for documentation
COMMENT ON TABLE marl_parameters IS 'MARL-learned HDBSCAN parameters optimized for different datasets';
COMMENT ON TABLE marl_parameter_evaluations IS 'Performance evaluations of MARL-learned parameters on various datasets';
COMMENT ON TABLE marl_training_sessions IS 'Records of MARL training sessions with performance metrics';
COMMENT ON TABLE marl_training_parameters IS 'Detailed parameter optimization history during training';
COMMENT ON TABLE marl_agent_performance IS 'Individual agent performance tracking during collaborative learning';