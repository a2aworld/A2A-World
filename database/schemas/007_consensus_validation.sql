-- A2A World Platform - Consensus Validation Database Schema
-- Database schema extensions for peer-to-peer consensus mechanism
-- Supports Byzantine Fault Tolerant and RAFT consensus protocols

-- =====================================================
-- CONSENSUS VALIDATION TABLES
-- =====================================================

-- Table: consensus_validation_requests
-- Stores consensus validation requests and their metadata
CREATE TABLE IF NOT EXISTS consensus_validation_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    pattern_id UUID NOT NULL,
    requester_agent_id VARCHAR(255) NOT NULL,
    consensus_protocol VARCHAR(50) NOT NULL DEFAULT 'adaptive', -- 'bft', 'raft', 'voting_only', 'adaptive'
    voting_mechanism VARCHAR(50) NOT NULL DEFAULT 'adaptive', -- 'majority', 'weighted', 'threshold', 'quorum', 'adaptive'
    min_participants INTEGER NOT NULL DEFAULT 3,
    timeout_seconds INTEGER NOT NULL DEFAULT 60,
    require_statistical_evidence BOOLEAN NOT NULL DEFAULT true,
    validation_methods TEXT[], -- Array of validation methods requested
    pattern_data JSONB, -- Pattern data for validation
    statistical_results JSONB, -- Statistical validation results
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed', 'timeout'
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    timeout_at TIMESTAMP WITH TIME ZONE,
    
    -- Foreign key relationships
    FOREIGN KEY (pattern_id) REFERENCES discovered_patterns(id) ON DELETE CASCADE
);

-- Table: consensus_validation_results
-- Stores the results of consensus validation processes
CREATE TABLE IF NOT EXISTS consensus_validation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) NOT NULL,
    pattern_id UUID NOT NULL,
    consensus_protocol_used VARCHAR(50),
    voting_mechanism_used VARCHAR(50),
    decision VARCHAR(50), -- 'significant', 'not_significant', 'uncertain', 'abstain'
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    consensus_achieved BOOLEAN NOT NULL DEFAULT false,
    participating_agents TEXT[] NOT NULL DEFAULT '{}',
    total_votes INTEGER NOT NULL DEFAULT 0,
    vote_breakdown JSONB, -- Vote counts by type
    weighted_breakdown JSONB, -- Weighted vote counts
    execution_time_seconds NUMERIC(10,4),
    error_message TEXT,
    statistical_summary JSONB,
    reputation_adjustments JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Foreign key relationships
    FOREIGN KEY (pattern_id) REFERENCES discovered_patterns(id) ON DELETE CASCADE
);

-- Table: consensus_votes
-- Stores individual votes cast by agents in consensus processes
CREATE TABLE IF NOT EXISTS consensus_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vote_id VARCHAR(255) UNIQUE NOT NULL,
    request_id VARCHAR(255) NOT NULL,
    pattern_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    vote VARCHAR(50) NOT NULL, -- 'significant', 'not_significant', 'uncertain', 'abstain'
    confidence NUMERIC(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    voting_weight NUMERIC(8,4) NOT NULL DEFAULT 1.0,
    reputation_score NUMERIC(5,4) NOT NULL DEFAULT 0.5,
    effective_weight NUMERIC(8,4) NOT NULL DEFAULT 1.0,
    statistical_evidence JSONB,
    reasoning TEXT,
    response_time_seconds NUMERIC(8,4),
    vote_signature TEXT, -- Cryptographic signature (simplified)
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Foreign key relationships
    FOREIGN KEY (pattern_id) REFERENCES discovered_patterns(id) ON DELETE CASCADE,
    FOREIGN KEY (request_id) REFERENCES consensus_validation_requests(request_id) ON DELETE CASCADE
);

-- =====================================================
-- AGENT REPUTATION SYSTEM TABLES
-- =====================================================

-- Table: agent_reputation_scores
-- Tracks comprehensive reputation scores for consensus agents
CREATE TABLE IF NOT EXISTS agent_reputation_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) UNIQUE NOT NULL,
    overall_score NUMERIC(5,4) NOT NULL DEFAULT 0.5 CHECK (overall_score >= 0 AND overall_score <= 1),
    accuracy_score NUMERIC(5,4) NOT NULL DEFAULT 0.5 CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    reliability_score NUMERIC(5,4) NOT NULL DEFAULT 0.5 CHECK (reliability_score >= 0 AND reliability_score <= 1),
    timeliness_score NUMERIC(5,4) NOT NULL DEFAULT 0.5 CHECK (timeliness_score >= 0 AND timeliness_score <= 1),
    participation_score NUMERIC(5,4) NOT NULL DEFAULT 0.5 CHECK (participation_score >= 0 AND participation_score <= 1),
    quality_score NUMERIC(5,4) NOT NULL DEFAULT 0.5 CHECK (quality_score >= 0 AND quality_score <= 1),
    peer_score NUMERIC(5,4) NOT NULL DEFAULT 0.5 CHECK (peer_score >= 0 AND peer_score <= 1),
    
    -- Performance statistics
    total_validations INTEGER NOT NULL DEFAULT 0,
    correct_predictions INTEGER NOT NULL DEFAULT 0,
    consensus_agreements INTEGER NOT NULL DEFAULT 0,
    average_response_time NUMERIC(8,4) NOT NULL DEFAULT 0.0,
    uptime_percentage NUMERIC(5,2) NOT NULL DEFAULT 100.0,
    
    -- Metadata
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Table: validation_outcomes
-- Records individual validation outcomes for reputation tracking
CREATE TABLE IF NOT EXISTS validation_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    pattern_id UUID NOT NULL,
    prediction VARCHAR(50) NOT NULL, -- 'significant', 'not_significant', 'uncertain'
    confidence NUMERIC(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    actual_outcome VARCHAR(50), -- Ground truth if available
    peer_consensus VARCHAR(50), -- What consensus decided
    is_correct BOOLEAN, -- Whether prediction matched outcome/consensus
    statistical_evidence_quality NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    response_time_seconds NUMERIC(8,4) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Foreign key relationships
    FOREIGN KEY (pattern_id) REFERENCES discovered_patterns(id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES agent_reputation_scores(agent_id) ON DELETE CASCADE
);

-- Table: peer_ratings
-- Stores peer-to-peer agent ratings
CREATE TABLE IF NOT EXISTS peer_ratings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rated_agent_id VARCHAR(255) NOT NULL,
    rating_agent_id VARCHAR(255) NOT NULL,
    rating NUMERIC(5,4) NOT NULL CHECK (rating >= 0 AND rating <= 1),
    rating_context VARCHAR(100), -- Context of rating (e.g., 'consensus_participation', 'statistical_quality')
    reasoning TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Foreign key relationships
    FOREIGN KEY (rated_agent_id) REFERENCES agent_reputation_scores(agent_id) ON DELETE CASCADE,
    FOREIGN KEY (rating_agent_id) REFERENCES agent_reputation_scores(agent_id) ON DELETE CASCADE,
    
    -- Prevent self-rating and duplicate ratings
    CHECK (rated_agent_id != rating_agent_id),
    UNIQUE (rated_agent_id, rating_agent_id, rating_context)
);

-- Table: agent_availability_events
-- Tracks agent availability for uptime calculations
CREATE TABLE IF NOT EXISTS agent_availability_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(20) NOT NULL, -- 'online', 'offline', 'heartbeat'
    available BOOLEAN NOT NULL,
    event_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB,
    
    -- Foreign key relationships
    FOREIGN KEY (agent_id) REFERENCES agent_reputation_scores(agent_id) ON DELETE CASCADE
);

-- =====================================================
-- CONSENSUS NETWORK TOPOLOGY TABLES
-- =====================================================

-- Table: consensus_network_nodes
-- Information about nodes in the consensus network
CREATE TABLE IF NOT EXISTS consensus_network_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id VARCHAR(255) UNIQUE NOT NULL,
    node_type VARCHAR(50) NOT NULL, -- 'consensus_coordinator', 'validation_agent', 'raft_node', 'bft_node'
    agent_id VARCHAR(255), -- Associated agent ID if applicable
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    protocols_supported TEXT[] NOT NULL DEFAULT '{}', -- 'bft', 'raft', 'voting'
    current_status VARCHAR(50) NOT NULL DEFAULT 'unknown', -- 'online', 'offline', 'degraded'
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    network_address INET,
    port INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Table: consensus_network_connections
-- Tracks connections between nodes in the consensus network
CREATE TABLE IF NOT EXISTS consensus_network_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id VARCHAR(255) NOT NULL,
    target_node_id VARCHAR(255) NOT NULL,
    connection_type VARCHAR(50) NOT NULL, -- 'peer', 'coordinator', 'backup', 'observer'
    connection_status VARCHAR(50) NOT NULL DEFAULT 'active', -- 'active', 'inactive', 'failed'
    latency_ms NUMERIC(8,2),
    bandwidth_mbps NUMERIC(10,2),
    reliability_score NUMERIC(5,4) DEFAULT 1.0,
    last_communication TIMESTAMP WITH TIME ZONE,
    established_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Foreign key relationships
    FOREIGN KEY (source_node_id) REFERENCES consensus_network_nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES consensus_network_nodes(node_id) ON DELETE CASCADE,
    
    -- Prevent self-connections
    CHECK (source_node_id != target_node_id),
    UNIQUE (source_node_id, target_node_id)
);

-- =====================================================
-- CONSENSUS PERFORMANCE AND ANALYTICS TABLES
-- =====================================================

-- Table: consensus_performance_metrics
-- Tracks performance metrics for consensus operations
CREATE TABLE IF NOT EXISTS consensus_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_type VARCHAR(100) NOT NULL, -- 'consensus_time', 'agreement_rate', 'participation_rate', etc.
    metric_scope VARCHAR(50) NOT NULL, -- 'system', 'coordinator', 'agent', 'pattern_type'
    scope_identifier VARCHAR(255), -- ID of specific coordinator, agent, etc.
    metric_value NUMERIC(15,6) NOT NULL,
    measurement_unit VARCHAR(50), -- 'seconds', 'percentage', 'count', etc.
    measurement_period VARCHAR(50), -- 'hourly', 'daily', 'weekly', 'monthly'
    measurement_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB,
    
    -- Composite index for efficient queries
    INDEX idx_consensus_metrics_scope_time (metric_scope, scope_identifier, measurement_timestamp)
);

-- Table: consensus_protocol_statistics
-- Statistics for different consensus protocols
CREATE TABLE IF NOT EXISTS consensus_protocol_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    protocol_type VARCHAR(50) NOT NULL, -- 'bft', 'raft', 'voting_only', 'adaptive'
    coordinator_id VARCHAR(255),
    total_requests INTEGER NOT NULL DEFAULT 0,
    successful_requests INTEGER NOT NULL DEFAULT 0,
    failed_requests INTEGER NOT NULL DEFAULT 0,
    timeout_requests INTEGER NOT NULL DEFAULT 0,
    average_response_time NUMERIC(8,4) NOT NULL DEFAULT 0.0,
    average_participants NUMERIC(6,2) NOT NULL DEFAULT 0.0,
    average_confidence NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    statistical_period VARCHAR(20) NOT NULL, -- 'hourly', 'daily', 'weekly', 'monthly'
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE (protocol_type, coordinator_id, statistical_period, period_start)
);

-- Table: pattern_consensus_history
-- Historical consensus decisions for patterns
CREATE TABLE IF NOT EXISTS pattern_consensus_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id UUID NOT NULL,
    consensus_request_id VARCHAR(255) NOT NULL,
    decision VARCHAR(50) NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    participating_agents TEXT[] NOT NULL,
    vote_breakdown JSONB NOT NULL,
    consensus_protocol VARCHAR(50) NOT NULL,
    voting_mechanism VARCHAR(50) NOT NULL,
    execution_time_seconds NUMERIC(10,4),
    decision_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Foreign key relationships
    FOREIGN KEY (pattern_id) REFERENCES discovered_patterns(id) ON DELETE CASCADE
);

-- =====================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Consensus validation requests indexes
CREATE INDEX IF NOT EXISTS idx_consensus_requests_pattern_id ON consensus_validation_requests(pattern_id);
CREATE INDEX IF NOT EXISTS idx_consensus_requests_requester ON consensus_validation_requests(requester_agent_id);
CREATE INDEX IF NOT EXISTS idx_consensus_requests_status ON consensus_validation_requests(status);
CREATE INDEX IF NOT EXISTS idx_consensus_requests_created_at ON consensus_validation_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_consensus_requests_protocol ON consensus_validation_requests(consensus_protocol);

-- Consensus validation results indexes
CREATE INDEX IF NOT EXISTS idx_consensus_results_request_id ON consensus_validation_results(request_id);
CREATE INDEX IF NOT EXISTS idx_consensus_results_pattern_id ON consensus_validation_results(pattern_id);
CREATE INDEX IF NOT EXISTS idx_consensus_results_decision ON consensus_validation_results(decision);
CREATE INDEX IF NOT EXISTS idx_consensus_results_consensus_achieved ON consensus_validation_results(consensus_achieved);
CREATE INDEX IF NOT EXISTS idx_consensus_results_created_at ON consensus_validation_results(created_at);

-- Consensus votes indexes
CREATE INDEX IF NOT EXISTS idx_consensus_votes_request_id ON consensus_votes(request_id);
CREATE INDEX IF NOT EXISTS idx_consensus_votes_pattern_id ON consensus_votes(pattern_id);
CREATE INDEX IF NOT EXISTS idx_consensus_votes_agent_id ON consensus_votes(agent_id);
CREATE INDEX IF NOT EXISTS idx_consensus_votes_vote ON consensus_votes(vote);
CREATE INDEX IF NOT EXISTS idx_consensus_votes_created_at ON consensus_votes(created_at);

-- Agent reputation indexes
CREATE INDEX IF NOT EXISTS idx_agent_reputation_overall_score ON agent_reputation_scores(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_reputation_last_updated ON agent_reputation_scores(last_updated);
CREATE INDEX IF NOT EXISTS idx_agent_reputation_total_validations ON agent_reputation_scores(total_validations DESC);

-- Validation outcomes indexes
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_agent_id ON validation_outcomes(agent_id);
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_pattern_id ON validation_outcomes(pattern_id);
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_is_correct ON validation_outcomes(is_correct);
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_created_at ON validation_outcomes(created_at);

-- Network topology indexes
CREATE INDEX IF NOT EXISTS idx_network_nodes_node_type ON consensus_network_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_network_nodes_status ON consensus_network_nodes(current_status);
CREATE INDEX IF NOT EXISTS idx_network_nodes_last_heartbeat ON consensus_network_nodes(last_heartbeat);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type_scope ON consensus_performance_metrics(metric_type, metric_scope);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON consensus_performance_metrics(measurement_timestamp);

-- =====================================================
-- VIEWS FOR ANALYTICS AND REPORTING
-- =====================================================

-- View: consensus_validation_summary
-- Summary statistics for consensus validations
CREATE OR REPLACE VIEW consensus_validation_summary AS
SELECT 
    DATE(r.created_at) as validation_date,
    r.consensus_protocol,
    r.voting_mechanism,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN res.consensus_achieved THEN 1 END) as successful_consensus,
    COUNT(CASE WHEN r.status = 'failed' THEN 1 END) as failed_requests,
    COUNT(CASE WHEN r.status = 'timeout' THEN 1 END) as timeout_requests,
    AVG(res.execution_time_seconds) as avg_execution_time,
    AVG(res.confidence) as avg_confidence,
    AVG(res.total_votes) as avg_participation
FROM consensus_validation_requests r
LEFT JOIN consensus_validation_results res ON r.request_id = res.request_id
GROUP BY DATE(r.created_at), r.consensus_protocol, r.voting_mechanism
ORDER BY validation_date DESC;

-- View: agent_reputation_ranking
-- Ranking of agents by reputation score
CREATE OR REPLACE VIEW agent_reputation_ranking AS
SELECT 
    agent_id,
    overall_score,
    accuracy_score,
    reliability_score,
    participation_score,
    total_validations,
    correct_predictions,
    consensus_agreements,
    CASE 
        WHEN total_validations > 0 THEN (correct_predictions::NUMERIC / total_validations)
        ELSE 0
    END as accuracy_rate,
    CASE 
        WHEN total_validations > 0 THEN (consensus_agreements::NUMERIC / total_validations)
        ELSE 0
    END as consensus_agreement_rate,
    ROW_NUMBER() OVER (ORDER BY overall_score DESC, total_validations DESC) as ranking,
    last_updated
FROM agent_reputation_scores
WHERE total_validations >= 5 -- Only include agents with meaningful participation
ORDER BY overall_score DESC, total_validations DESC;

-- View: consensus_network_health
-- Overview of consensus network health
CREATE OR REPLACE VIEW consensus_network_health AS
SELECT 
    node_type,
    COUNT(*) as total_nodes,
    COUNT(CASE WHEN current_status = 'online' THEN 1 END) as online_nodes,
    COUNT(CASE WHEN current_status = 'offline' THEN 1 END) as offline_nodes,
    COUNT(CASE WHEN current_status = 'degraded' THEN 1 END) as degraded_nodes,
    COUNT(CASE WHEN last_heartbeat > NOW() - INTERVAL '5 minutes' THEN 1 END) as recent_heartbeats,
    AVG(EXTRACT(EPOCH FROM (NOW() - last_heartbeat))/60) as avg_minutes_since_heartbeat
FROM consensus_network_nodes
GROUP BY node_type
ORDER BY node_type;

-- View: pattern_consensus_trends
-- Trends in pattern consensus decisions over time
CREATE OR REPLACE VIEW pattern_consensus_trends AS
SELECT 
    DATE(decision_timestamp) as decision_date,
    decision,
    consensus_protocol,
    COUNT(*) as decision_count,
    AVG(confidence) as avg_confidence,
    AVG(array_length(participating_agents, 1)) as avg_participants,
    AVG(execution_time_seconds) as avg_execution_time
FROM pattern_consensus_history
WHERE decision_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(decision_timestamp), decision, consensus_protocol
ORDER BY decision_date DESC, decision_count DESC;

-- =====================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================

-- Trigger function to update agent reputation when validation outcomes change
CREATE OR REPLACE FUNCTION update_agent_reputation_on_outcome()
RETURNS TRIGGER AS $$
BEGIN
    -- Update reputation scores based on new validation outcome
    UPDATE agent_reputation_scores 
    SET 
        total_validations = total_validations + 1,
        correct_predictions = correct_predictions + CASE WHEN NEW.is_correct THEN 1 ELSE 0 END,
        last_updated = NOW()
    WHERE agent_id = NEW.agent_id;
    
    -- Insert agent if not exists
    INSERT INTO agent_reputation_scores (agent_id, total_validations, correct_predictions)
    VALUES (NEW.agent_id, 1, CASE WHEN NEW.is_correct THEN 1 ELSE 0 END)
    ON CONFLICT (agent_id) DO NOTHING;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for validation outcomes
DROP TRIGGER IF EXISTS trigger_update_reputation_on_outcome ON validation_outcomes;
CREATE TRIGGER trigger_update_reputation_on_outcome
    AFTER INSERT ON validation_outcomes
    FOR EACH ROW
    EXECUTE FUNCTION update_agent_reputation_on_outcome();

-- Trigger function to update consensus statistics
CREATE OR REPLACE FUNCTION update_consensus_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update protocol statistics when consensus results are inserted
    INSERT INTO consensus_protocol_statistics (
        protocol_type,
        coordinator_id,
        total_requests,
        successful_requests,
        failed_requests,
        timeout_requests,
        statistical_period,
        period_start,
        period_end
    )
    VALUES (
        NEW.consensus_protocol_used,
        'system', -- Default coordinator
        1,
        CASE WHEN NEW.consensus_achieved THEN 1 ELSE 0 END,
        CASE WHEN NOT NEW.consensus_achieved AND NEW.error_message IS NOT NULL THEN 1 ELSE 0 END,
        0, -- Will be updated by separate process for timeouts
        'daily',
        DATE_TRUNC('day', NEW.created_at),
        DATE_TRUNC('day', NEW.created_at) + INTERVAL '1 day'
    )
    ON CONFLICT (protocol_type, coordinator_id, statistical_period, period_start)
    DO UPDATE SET
        total_requests = consensus_protocol_statistics.total_requests + 1,
        successful_requests = consensus_protocol_statistics.successful_requests + 
            CASE WHEN NEW.consensus_achieved THEN 1 ELSE 0 END,
        failed_requests = consensus_protocol_statistics.failed_requests + 
            CASE WHEN NOT NEW.consensus_achieved AND NEW.error_message IS NOT NULL THEN 1 ELSE 0 END;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for consensus statistics
DROP TRIGGER IF EXISTS trigger_update_consensus_statistics ON consensus_validation_results;
CREATE TRIGGER trigger_update_consensus_statistics
    AFTER INSERT ON consensus_validation_results
    FOR EACH ROW
    EXECUTE FUNCTION update_consensus_statistics();

-- =====================================================
-- STORED PROCEDURES FOR COMPLEX OPERATIONS
-- =====================================================

-- Function to calculate comprehensive agent reputation scores
CREATE OR REPLACE FUNCTION calculate_agent_reputation(p_agent_id VARCHAR(255))
RETURNS JSONB AS $$
DECLARE
    v_reputation RECORD;
    v_recent_outcomes INTEGER;
    v_recent_correct INTEGER;
    v_peer_ratings NUMERIC;
    v_result JSONB;
BEGIN
    -- Get current reputation record
    SELECT * INTO v_reputation 
    FROM agent_reputation_scores 
    WHERE agent_id = p_agent_id;
    
    IF NOT FOUND THEN
        RETURN jsonb_build_object('error', 'Agent not found');
    END IF;
    
    -- Calculate recent performance (last 30 days)
    SELECT 
        COUNT(*),
        COUNT(CASE WHEN is_correct THEN 1 END)
    INTO v_recent_outcomes, v_recent_correct
    FROM validation_outcomes 
    WHERE agent_id = p_agent_id 
    AND created_at >= NOW() - INTERVAL '30 days';
    
    -- Calculate average peer ratings
    SELECT AVG(rating) INTO v_peer_ratings
    FROM peer_ratings 
    WHERE rated_agent_id = p_agent_id
    AND created_at >= NOW() - INTERVAL '90 days';
    
    -- Build result JSON
    v_result := jsonb_build_object(
        'agent_id', p_agent_id,
        'overall_score', v_reputation.overall_score,
        'component_scores', jsonb_build_object(
            'accuracy', v_reputation.accuracy_score,
            'reliability', v_reputation.reliability_score,
            'timeliness', v_reputation.timeliness_score,
            'participation', v_reputation.participation_score,
            'quality', v_reputation.quality_score,
            'peer', v_reputation.peer_score
        ),
        'statistics', jsonb_build_object(
            'total_validations', v_reputation.total_validations,
            'correct_predictions', v_reputation.correct_predictions,
            'consensus_agreements', v_reputation.consensus_agreements,
            'accuracy_rate', CASE WHEN v_reputation.total_validations > 0 
                          THEN v_reputation.correct_predictions::NUMERIC / v_reputation.total_validations 
                          ELSE 0 END,
            'recent_outcomes', v_recent_outcomes,
            'recent_accuracy', CASE WHEN v_recent_outcomes > 0 
                            THEN v_recent_correct::NUMERIC / v_recent_outcomes 
                            ELSE 0 END,
            'peer_rating', COALESCE(v_peer_ratings, 0.5)
        ),
        'last_updated', v_reputation.last_updated
    );
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Function to get consensus validation analytics
CREATE OR REPLACE FUNCTION get_consensus_analytics(p_days INTEGER DEFAULT 30)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
    v_total_requests INTEGER;
    v_successful_requests INTEGER;
    v_avg_confidence NUMERIC;
    v_protocol_breakdown JSONB;
BEGIN
    -- Get overall statistics
    SELECT 
        COUNT(*),
        COUNT(CASE WHEN res.consensus_achieved THEN 1 END),
        AVG(res.confidence)
    INTO v_total_requests, v_successful_requests, v_avg_confidence
    FROM consensus_validation_requests req
    LEFT JOIN consensus_validation_results res ON req.request_id = res.request_id
    WHERE req.created_at >= NOW() - (p_days || ' days')::INTERVAL;
    
    -- Get protocol breakdown
    SELECT jsonb_object_agg(consensus_protocol, protocol_stats)
    INTO v_protocol_breakdown
    FROM (
        SELECT 
            req.consensus_protocol,
            jsonb_build_object(
                'total_requests', COUNT(*),
                'success_rate', COUNT(CASE WHEN res.consensus_achieved THEN 1 END)::NUMERIC / COUNT(*),
                'avg_confidence', AVG(res.confidence),
                'avg_execution_time', AVG(res.execution_time_seconds)
            ) as protocol_stats
        FROM consensus_validation_requests req
        LEFT JOIN consensus_validation_results res ON req.request_id = res.request_id
        WHERE req.created_at >= NOW() - (p_days || ' days')::INTERVAL
        GROUP BY req.consensus_protocol
    ) protocol_data;
    
    -- Build result
    v_result := jsonb_build_object(
        'analysis_period_days', p_days,
        'total_requests', COALESCE(v_total_requests, 0),
        'successful_requests', COALESCE(v_successful_requests, 0),
        'success_rate', CASE WHEN v_total_requests > 0 
                       THEN v_successful_requests::NUMERIC / v_total_requests 
                       ELSE 0 END,
        'average_confidence', COALESCE(v_avg_confidence, 0),
        'protocol_breakdown', COALESCE(v_protocol_breakdown, '{}'::JSONB),
        'generated_at', NOW()
    );
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- INITIAL DATA AND CONFIGURATION
-- =====================================================

-- Insert default consensus network node for system coordinator
INSERT INTO consensus_network_nodes (
    node_id, 
    node_type, 
    capabilities, 
    protocols_supported, 
    current_status
) VALUES (
    'system_consensus_coordinator',
    'consensus_coordinator',
    ARRAY['consensus_coordination', 'bft_consensus', 'raft_consensus', 'voting_mechanisms'],
    ARRAY['bft', 'raft', 'voting'],
    'online'
) ON CONFLICT (node_id) DO NOTHING;

-- Create default performance metrics entries
INSERT INTO consensus_performance_metrics (
    metric_type,
    metric_scope,
    metric_value,
    measurement_unit,
    measurement_period
) VALUES 
    ('system_uptime', 'system', 100.0, 'percentage', 'daily'),
    ('consensus_success_rate', 'system', 0.0, 'percentage', 'daily'),
    ('average_consensus_time', 'system', 0.0, 'seconds', 'daily'),
    ('network_connectivity', 'system', 100.0, 'percentage', 'daily')
ON CONFLICT DO NOTHING;

-- =====================================================
-- COMMENTS AND DOCUMENTATION
-- =====================================================

COMMENT ON TABLE consensus_validation_requests IS 'Stores requests for consensus-based pattern validation with protocol and voting mechanism specifications';
COMMENT ON TABLE consensus_validation_results IS 'Results of consensus validation processes including decisions, confidence, and participation metrics';
COMMENT ON TABLE consensus_votes IS 'Individual votes cast by agents in consensus processes with statistical evidence and reasoning';
COMMENT ON TABLE agent_reputation_scores IS 'Comprehensive reputation tracking for consensus agents including accuracy, reliability, and peer ratings';
COMMENT ON TABLE validation_outcomes IS 'Individual validation outcomes used for reputation calculation and performance tracking';
COMMENT ON TABLE peer_ratings IS 'Peer-to-peer ratings between agents for reputation and trust management';
COMMENT ON TABLE consensus_network_nodes IS 'Network topology information for consensus coordinators and validation agents';
COMMENT ON TABLE consensus_performance_metrics IS 'Performance and analytics metrics for consensus operations and network health';
COMMENT ON TABLE pattern_consensus_history IS 'Historical record of consensus decisions for patterns with trends analysis';

-- Log schema creation
DO $$ 
BEGIN 
    RAISE NOTICE 'Consensus validation database schema (007_consensus_validation.sql) created successfully';
    RAISE NOTICE 'Tables created: consensus_validation_requests, consensus_validation_results, consensus_votes';
    RAISE NOTICE 'Reputation system: agent_reputation_scores, validation_outcomes, peer_ratings';
    RAISE NOTICE 'Network topology: consensus_network_nodes, consensus_network_connections';
    RAISE NOTICE 'Analytics: consensus_performance_metrics, consensus_protocol_statistics';
    RAISE NOTICE 'Views and functions for reporting and analysis created';
END $$;