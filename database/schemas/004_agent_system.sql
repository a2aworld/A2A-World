-- A2A World Platform - Agent System Schema
-- Tables for agent registration, task management, and system monitoring

SET search_path TO a2a_world, public;

-- Agent registry for all autonomous agents in the system
CREATE TABLE agents (
    id VARCHAR(255) PRIMARY KEY, -- Agent identifier (e.g., "pattern-discovery-001")
    agent_name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) CHECK (agent_type IN (
        'pattern_discovery', 'kml_parser', 'cultural_analysis', 'validation',
        'data_ingestion', 'monitoring', 'narrative_generation', 'correlation'
    )),
    agent_version VARCHAR(50),
    status VARCHAR(50) DEFAULT 'inactive' CHECK (status IN (
        'active', 'inactive', 'busy', 'error', 'maintenance', 'stopped'
    )),
    health_status VARCHAR(50) DEFAULT 'unknown' CHECK (health_status IN (
        'healthy', 'warning', 'critical', 'unknown', 'degraded'
    )),
    capabilities JSONB, -- Array of capabilities this agent provides
    resource_requirements JSONB, -- CPU, memory, storage requirements
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    heartbeat_interval_seconds INTEGER DEFAULT 30,
    start_time TIMESTAMP WITH TIME ZONE,
    last_task_completed TIMESTAMP WITH TIME ZONE,
    total_tasks_processed BIGINT DEFAULT 0,
    total_tasks_failed BIGINT DEFAULT 0,
    average_task_duration_ms INTEGER,
    current_task_id UUID,
    configuration JSONB,
    host_info JSONB, -- Host machine information
    process_id INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Task queue for coordinating work between agents
CREATE TABLE agent_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_type VARCHAR(100) CHECK (task_type IN (
        'parse_kml_file', 'discover_patterns', 'validate_pattern', 'analyze_correlation',
        'process_cultural_data', 'generate_narrative', 'spatial_analysis',
        'environmental_analysis', 'astronomical_analysis'
    )),
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10), -- 1 = highest priority
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN (
        'pending', 'assigned', 'in_progress', 'completed', 'failed', 'cancelled', 'retry'
    )),
    assigned_agent_id VARCHAR(255) REFERENCES agents(id) ON DELETE SET NULL,
    created_by VARCHAR(255), -- Agent or user that created the task
    task_parameters JSONB NOT NULL,
    input_data JSONB,
    output_data JSONB,
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage BETWEEN 0 AND 100),
    estimated_duration_ms INTEGER,
    actual_duration_ms INTEGER,
    max_retries INTEGER DEFAULT 3,
    retry_count INTEGER DEFAULT 0,
    last_retry_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    error_details JSONB,
    dependencies UUID[], -- Array of task IDs this task depends on
    scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    timeout_seconds INTEGER DEFAULT 3600, -- 1 hour default timeout
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent performance metrics and monitoring data
CREATE TABLE agent_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) REFERENCES agents(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6),
    metric_unit VARCHAR(50),
    metric_type VARCHAR(50) CHECK (metric_type IN (
        'counter', 'gauge', 'histogram', 'summary', 'timer'
    )),
    tags JSONB, -- Key-value pairs for metric dimensions
    timestamp_utc TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent communication logs for debugging and monitoring
CREATE TABLE agent_communications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sender_agent_id VARCHAR(255) REFERENCES agents(id) ON DELETE SET NULL,
    receiver_agent_id VARCHAR(255) REFERENCES agents(id) ON DELETE SET NULL,
    message_type VARCHAR(100) CHECK (message_type IN (
        'task_request', 'task_response', 'heartbeat', 'status_update',
        'error_report', 'data_sharing', 'coordination', 'shutdown_signal'
    )),
    message_payload JSONB,
    correlation_id UUID, -- For tracking request-response pairs
    priority INTEGER DEFAULT 5,
    delivery_status VARCHAR(50) DEFAULT 'pending' CHECK (delivery_status IN (
        'pending', 'delivered', 'failed', 'timeout', 'acknowledged'
    )),
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    delivered_at TIMESTAMP WITH TIME ZONE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent collaboration sessions for coordinated work
CREATE TABLE agent_collaborations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    collaboration_name VARCHAR(255),
    collaboration_type VARCHAR(100) CHECK (collaboration_type IN (
        'pattern_validation', 'data_processing_pipeline', 'correlation_analysis',
        'multi_stage_discovery', 'cross_validation', 'consensus_building'
    )),
    coordinator_agent_id VARCHAR(255) REFERENCES agents(id) ON DELETE SET NULL,
    participating_agents VARCHAR(255)[], -- Array of agent IDs
    status VARCHAR(50) DEFAULT 'initialized' CHECK (status IN (
        'initialized', 'in_progress', 'completed', 'failed', 'cancelled'
    )),
    collaboration_goal TEXT,
    shared_context JSONB,
    results JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    success_rate DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Resource locks to prevent conflicts in shared resource access
CREATE TABLE resource_locks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_type VARCHAR(100) CHECK (resource_type IN (
        'dataset', 'pattern', 'cultural_data', 'geospatial_feature',
        'analysis_region', 'computation_cluster'
    )),
    resource_id VARCHAR(255) NOT NULL,
    locked_by_agent VARCHAR(255) REFERENCES agents(id) ON DELETE CASCADE,
    lock_type VARCHAR(50) CHECK (lock_type IN (
        'read_lock', 'write_lock', 'exclusive_lock'
    )),
    lock_reason TEXT,
    acquired_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System health and status monitoring
CREATE TABLE system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_type VARCHAR(100) CHECK (component_type IN (
        'database', 'message_queue', 'file_system', 'external_api',
        'agent_pool', 'computation_resources', 'storage'
    )),
    component_name VARCHAR(255),
    health_status VARCHAR(50) CHECK (health_status IN (
        'healthy', 'warning', 'critical', 'down', 'maintenance'
    )),
    health_score DECIMAL(3,2) CHECK (health_score BETWEEN 0 AND 1),
    response_time_ms INTEGER,
    error_rate DECIMAL(5,4),
    throughput_per_second DECIMAL(10,2),
    resource_utilization JSONB, -- CPU, memory, disk usage
    last_check TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    check_interval_seconds INTEGER DEFAULT 60,
    alert_threshold DECIMAL(3,2) DEFAULT 0.8,
    escalation_rules JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent configuration templates and profiles
CREATE TABLE agent_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_name VARCHAR(255) UNIQUE NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    default_configuration JSONB,
    resource_limits JSONB,
    performance_targets JSONB,
    scaling_rules JSONB,
    monitoring_config JSONB,
    deployment_config JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    version VARCHAR(50),
    created_by VARCHAR(255),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for agent system tables
CREATE INDEX idx_agents_type ON agents(agent_type);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_health ON agents(health_status);
CREATE INDEX idx_agents_last_heartbeat ON agents(last_heartbeat);
CREATE INDEX idx_agents_start_time ON agents(start_time);

CREATE INDEX idx_agent_tasks_type ON agent_tasks(task_type);
CREATE INDEX idx_agent_tasks_status ON agent_tasks(status);
CREATE INDEX idx_agent_tasks_priority ON agent_tasks(priority);
CREATE INDEX idx_agent_tasks_assigned_agent ON agent_tasks(assigned_agent_id);
CREATE INDEX idx_agent_tasks_created_by ON agent_tasks(created_by);
CREATE INDEX idx_agent_tasks_scheduled ON agent_tasks(scheduled_for);
CREATE INDEX idx_agent_tasks_created_at ON agent_tasks(created_at);
CREATE INDEX idx_agent_tasks_status_priority ON agent_tasks(status, priority);

CREATE INDEX idx_agent_metrics_agent ON agent_metrics(agent_id);
CREATE INDEX idx_agent_metrics_name ON agent_metrics(metric_name);
CREATE INDEX idx_agent_metrics_timestamp ON agent_metrics(timestamp_utc);
CREATE INDEX idx_agent_metrics_type ON agent_metrics(metric_type);

CREATE INDEX idx_agent_communications_sender ON agent_communications(sender_agent_id);
CREATE INDEX idx_agent_communications_receiver ON agent_communications(receiver_agent_id);
CREATE INDEX idx_agent_communications_type ON agent_communications(message_type);
CREATE INDEX idx_agent_communications_correlation ON agent_communications(correlation_id);
CREATE INDEX idx_agent_communications_status ON agent_communications(delivery_status);
CREATE INDEX idx_agent_communications_sent ON agent_communications(sent_at);

CREATE INDEX idx_agent_collaborations_coordinator ON agent_collaborations(coordinator_agent_id);
CREATE INDEX idx_agent_collaborations_status ON agent_collaborations(status);
CREATE INDEX idx_agent_collaborations_type ON agent_collaborations(collaboration_type);
CREATE INDEX idx_agent_collaborations_started ON agent_collaborations(started_at);

CREATE INDEX idx_resource_locks_resource ON resource_locks(resource_type, resource_id);
CREATE INDEX idx_resource_locks_agent ON resource_locks(locked_by_agent);
CREATE INDEX idx_resource_locks_expires ON resource_locks(expires_at);
CREATE INDEX idx_resource_locks_acquired ON resource_locks(acquired_at);

CREATE INDEX idx_system_health_component ON system_health(component_type, component_name);
CREATE INDEX idx_system_health_status ON system_health(health_status);
CREATE INDEX idx_system_health_last_check ON system_health(last_check);
CREATE INDEX idx_system_health_score ON system_health(health_score);

CREATE INDEX idx_agent_profiles_type ON agent_profiles(agent_type);
CREATE INDEX idx_agent_profiles_active ON agent_profiles(is_active);
CREATE INDEX idx_agent_profiles_name ON agent_profiles(profile_name);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_agents_capabilities_gin ON agents USING gin(capabilities);
CREATE INDEX idx_agents_configuration_gin ON agents USING gin(configuration);
CREATE INDEX idx_agent_tasks_parameters_gin ON agent_tasks USING gin(task_parameters);
CREATE INDEX idx_agent_tasks_input_gin ON agent_tasks USING gin(input_data);
CREATE INDEX idx_agent_tasks_output_gin ON agent_tasks USING gin(output_data);
CREATE INDEX idx_agent_metrics_tags_gin ON agent_metrics USING gin(tags);
CREATE INDEX idx_agent_communications_payload_gin ON agent_communications USING gin(message_payload);
CREATE INDEX idx_agent_collaborations_context_gin ON agent_collaborations USING gin(shared_context);
CREATE INDEX idx_system_health_utilization_gin ON system_health USING gin(resource_utilization);
CREATE INDEX idx_agent_profiles_config_gin ON agent_profiles USING gin(default_configuration);

-- Create array indexes
CREATE INDEX idx_agent_tasks_dependencies_gin ON agent_tasks USING gin(dependencies);
CREATE INDEX idx_agent_collaborations_participants_gin ON agent_collaborations USING gin(participating_agents);

-- Apply updated_at triggers
CREATE TRIGGER update_agents_updated_at 
    BEFORE UPDATE ON agents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_tasks_updated_at 
    BEFORE UPDATE ON agent_tasks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_collaborations_updated_at 
    BEFORE UPDATE ON agent_collaborations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_profiles_updated_at 
    BEFORE UPDATE ON agent_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create constraint to prevent tasks from depending on themselves
CREATE OR REPLACE FUNCTION check_task_dependencies()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.id = ANY(NEW.dependencies) THEN
        RAISE EXCEPTION 'Task cannot depend on itself';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER check_agent_task_dependencies 
    BEFORE INSERT OR UPDATE ON agent_tasks
    FOR EACH ROW EXECUTE FUNCTION check_task_dependencies();

-- Create constraint to prevent resource locks from expiring in the past
ALTER TABLE resource_locks 
ADD CONSTRAINT resource_locks_valid_expiry 
CHECK (expires_at IS NULL OR expires_at > acquired_at);

-- Create unique constraint to prevent duplicate active resource locks
CREATE UNIQUE INDEX idx_resource_locks_unique_active 
ON resource_locks(resource_type, resource_id, locked_by_agent) 
WHERE expires_at IS NULL OR expires_at > NOW();

-- Create partial index for pending tasks
CREATE INDEX idx_agent_tasks_pending_priority 
ON agent_tasks(priority, created_at) 
WHERE status = 'pending';