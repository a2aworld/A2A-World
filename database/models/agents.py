"""
A2A World Platform - Agent System Models

SQLAlchemy models for agent registration, task management, and system monitoring.
"""

from sqlalchemy import (
    Boolean, Column, Integer, String, Text, DateTime, 
    Numeric, CheckConstraint, ForeignKey
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base


class Agent(Base):
    """Agent registry for all autonomous agents in the system."""
    
    __tablename__ = "agents"
    
    # Override id to use string instead of UUID for agents
    id = Column(String(255), primary_key=True)  # Agent identifier
    agent_name = Column(String(255), nullable=False)
    agent_type = Column(String(100), index=True)
    agent_version = Column(String(50))
    status = Column(String(50), default="inactive")
    health_status = Column(String(50), default="unknown")
    capabilities = Column(JSONB)
    resource_requirements = Column(JSONB)
    last_heartbeat = Column(DateTime(timezone=True))
    heartbeat_interval_seconds = Column(Integer, default=30)
    start_time = Column(DateTime(timezone=True))
    last_task_completed = Column(DateTime(timezone=True))
    total_tasks_processed = Column(Integer, default=0)
    total_tasks_failed = Column(Integer, default=0)
    average_task_duration_ms = Column(Integer)
    current_task_id = Column(UUID(as_uuid=True))
    configuration = Column(JSONB)
    host_info = Column(JSONB)
    process_id = Column(Integer)
    
    # Override base class fields since we don't want created_at/updated_at as UUID
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    
    # Relationships
    assigned_tasks = relationship("AgentTask", back_populates="assigned_agent")
    metrics = relationship("AgentMetric", back_populates="agent", cascade="all, delete-orphan")
    sent_communications = relationship("AgentCommunication", foreign_keys="AgentCommunication.sender_agent_id", back_populates="sender_agent")
    received_communications = relationship("AgentCommunication", foreign_keys="AgentCommunication.receiver_agent_id", back_populates="receiver_agent")
    coordinated_collaborations = relationship("AgentCollaboration", back_populates="coordinator_agent")
    resource_locks = relationship("ResourceLock", back_populates="locked_by_agent_obj", cascade="all, delete-orphan")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("agent_type IN ('pattern_discovery', 'kml_parser', 'cultural_analysis', 'validation', 'data_ingestion', 'monitoring', 'narrative_generation', 'correlation')", name="check_agent_type"),
        CheckConstraint("status IN ('active', 'inactive', 'busy', 'error', 'maintenance', 'stopped')", name="check_status"),
        CheckConstraint("health_status IN ('healthy', 'warning', 'critical', 'unknown', 'degraded')", name="check_health_status"),
    )
    
    def __repr__(self):
        return f"<Agent(id='{self.id}', agent_type='{self.agent_type}', status='{self.status}')>"


class AgentTask(Base):
    """Task queue for coordinating work between agents."""
    
    __tablename__ = "agent_tasks"
    
    task_type = Column(String(100))
    priority = Column(Integer, default=5)  # 1 = highest priority
    status = Column(String(50), default="pending")
    assigned_agent_id = Column(String(255), ForeignKey("agents.id", ondelete="SET NULL"))
    created_by = Column(String(255))  # Agent or user that created the task
    task_parameters = Column(JSONB, nullable=False)
    input_data = Column(JSONB)
    output_data = Column(JSONB)
    progress_percentage = Column(Integer, default=0)
    estimated_duration_ms = Column(Integer)
    actual_duration_ms = Column(Integer)
    max_retries = Column(Integer, default=3)
    retry_count = Column(Integer, default=0)
    last_retry_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    error_details = Column(JSONB)
    dependencies = Column(ARRAY(UUID(as_uuid=True)))  # Array of task IDs
    scheduled_for = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    timeout_seconds = Column(Integer, default=3600)  # 1 hour default
    metadata = Column(JSONB)
    
    # Relationships
    assigned_agent = relationship("Agent", back_populates="assigned_tasks")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("task_type IN ('parse_kml_file', 'discover_patterns', 'validate_pattern', 'analyze_correlation', 'process_cultural_data', 'generate_narrative', 'spatial_analysis', 'environmental_analysis', 'astronomical_analysis')", name="check_task_type"),
        CheckConstraint("priority BETWEEN 1 AND 10", name="check_priority"),
        CheckConstraint("status IN ('pending', 'assigned', 'in_progress', 'completed', 'failed', 'cancelled', 'retry')", name="check_status"),
        CheckConstraint("progress_percentage BETWEEN 0 AND 100", name="check_progress"),
    )
    
    def __repr__(self):
        return f"<AgentTask(id='{self.id}', task_type='{self.task_type}', status='{self.status}')>"


class AgentMetric(Base):
    """Agent performance metrics and monitoring data."""
    
    __tablename__ = "agent_metrics"
    
    agent_id = Column(String(255), ForeignKey("agents.id", ondelete="CASCADE"))
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Numeric(15, 6))
    metric_unit = Column(String(50))
    metric_type = Column(String(50))
    tags = Column(JSONB)  # Key-value pairs for metric dimensions
    timestamp_utc = Column(DateTime(timezone=True))
    
    # Relationships
    agent = relationship("Agent", back_populates="metrics")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("metric_type IN ('counter', 'gauge', 'histogram', 'summary', 'timer')", name="check_metric_type"),
    )
    
    def __repr__(self):
        return f"<AgentMetric(agent_id='{self.agent_id}', metric_name='{self.metric_name}')>"


class AgentCommunication(Base):
    """Agent communication logs for debugging and monitoring."""
    
    __tablename__ = "agent_communications"
    
    sender_agent_id = Column(String(255), ForeignKey("agents.id", ondelete="SET NULL"))
    receiver_agent_id = Column(String(255), ForeignKey("agents.id", ondelete="SET NULL"))
    message_type = Column(String(100))
    message_payload = Column(JSONB)
    correlation_id = Column(UUID(as_uuid=True))  # For tracking request-response pairs
    priority = Column(Integer, default=5)
    delivery_status = Column(String(50), default="pending")
    sent_at = Column(DateTime(timezone=True))
    delivered_at = Column(DateTime(timezone=True))
    acknowledged_at = Column(DateTime(timezone=True))
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    metadata = Column(JSONB)
    
    # Relationships
    sender_agent = relationship("Agent", foreign_keys=[sender_agent_id], back_populates="sent_communications")
    receiver_agent = relationship("Agent", foreign_keys=[receiver_agent_id], back_populates="received_communications")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("message_type IN ('task_request', 'task_response', 'heartbeat', 'status_update', 'error_report', 'data_sharing', 'coordination', 'shutdown_signal')", name="check_message_type"),
        CheckConstraint("delivery_status IN ('pending', 'delivered', 'failed', 'timeout', 'acknowledged')", name="check_delivery_status"),
    )
    
    def __repr__(self):
        return f"<AgentCommunication(message_type='{self.message_type}', delivery_status='{self.delivery_status}')>"


class AgentCollaboration(Base):
    """Agent collaboration sessions for coordinated work."""
    
    __tablename__ = "agent_collaborations"
    
    collaboration_name = Column(String(255))
    collaboration_type = Column(String(100))
    coordinator_agent_id = Column(String(255), ForeignKey("agents.id", ondelete="SET NULL"))
    participating_agents = Column(ARRAY(String(255)))  # Array of agent IDs
    status = Column(String(50), default="initialized")
    collaboration_goal = Column(Text)
    shared_context = Column(JSONB)
    results = Column(JSONB)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_ms = Column(Integer)
    success_rate = Column(Numeric(5, 4))
    metadata = Column(JSONB)
    
    # Relationships
    coordinator_agent = relationship("Agent", back_populates="coordinated_collaborations")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("collaboration_type IN ('pattern_validation', 'data_processing_pipeline', 'correlation_analysis', 'multi_stage_discovery', 'cross_validation', 'consensus_building')", name="check_collaboration_type"),
        CheckConstraint("status IN ('initialized', 'in_progress', 'completed', 'failed', 'cancelled')", name="check_status"),
    )
    
    def __repr__(self):
        return f"<AgentCollaboration(collaboration_name='{self.collaboration_name}', status='{self.status}')>"


class ResourceLock(Base):
    """Resource locks to prevent conflicts in shared resource access."""
    
    __tablename__ = "resource_locks"
    
    resource_type = Column(String(100))
    resource_id = Column(String(255), nullable=False)
    locked_by_agent = Column(String(255), ForeignKey("agents.id", ondelete="CASCADE"))
    lock_type = Column(String(50))
    lock_reason = Column(Text)
    acquired_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    metadata = Column(JSONB)
    
    # Relationships
    locked_by_agent_obj = relationship("Agent", back_populates="resource_locks")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("resource_type IN ('dataset', 'pattern', 'cultural_data', 'geospatial_feature', 'analysis_region', 'computation_cluster')", name="check_resource_type"),
        CheckConstraint("lock_type IN ('read_lock', 'write_lock', 'exclusive_lock')", name="check_lock_type"),
    )
    
    def __repr__(self):
        return f"<ResourceLock(resource_type='{self.resource_type}', resource_id='{self.resource_id}', lock_type='{self.lock_type}')>"


class SystemHealth(Base):
    """System health and status monitoring."""
    
    __tablename__ = "system_health"
    
    component_type = Column(String(100))
    component_name = Column(String(255))
    health_status = Column(String(50))
    health_score = Column(Numeric(3, 2))
    response_time_ms = Column(Integer)
    error_rate = Column(Numeric(5, 4))
    throughput_per_second = Column(Numeric(10, 2))
    resource_utilization = Column(JSONB)  # CPU, memory, disk usage
    last_check = Column(DateTime(timezone=True))
    check_interval_seconds = Column(Integer, default=60)
    alert_threshold = Column(Numeric(3, 2), default=0.8)
    escalation_rules = Column(JSONB)
    metadata = Column(JSONB)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("component_type IN ('database', 'message_queue', 'file_system', 'external_api', 'agent_pool', 'computation_resources', 'storage')", name="check_component_type"),
        CheckConstraint("health_status IN ('healthy', 'warning', 'critical', 'down', 'maintenance')", name="check_health_status"),
        CheckConstraint("health_score BETWEEN 0 AND 1", name="check_health_score"),
    )
    
    def __repr__(self):
        return f"<SystemHealth(component_type='{self.component_type}', health_status='{self.health_status}')>"


class AgentProfile(Base):
    """Agent configuration templates and profiles."""
    
    __tablename__ = "agent_profiles"
    
    profile_name = Column(String(255), unique=True, nullable=False)
    agent_type = Column(String(100), nullable=False)
    default_configuration = Column(JSONB)
    resource_limits = Column(JSONB)
    performance_targets = Column(JSONB)
    scaling_rules = Column(JSONB)
    monitoring_config = Column(JSONB)
    deployment_config = Column(JSONB)
    is_active = Column(Boolean, default=True)
    version = Column(String(50))
    created_by = Column(String(255))
    description = Column(Text)
    
    def __repr__(self):
        return f"<AgentProfile(profile_name='{self.profile_name}', agent_type='{self.agent_type}')>"