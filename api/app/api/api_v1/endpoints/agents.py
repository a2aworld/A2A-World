"""
A2A World Platform - Agent Management Endpoints

Comprehensive endpoints for managing and monitoring autonomous agents
with lifecycle control, task management, and performance monitoring.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

try:
    from sqlalchemy.orm import Session
    from database.models.agents import Agent, AgentTask, AgentMetric, AgentProfile
    from database.connection import get_database_session
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from agents.core.messaging import get_nats_client
    from agents.core.registry import get_consul_registry
    AGENT_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    AGENT_INFRASTRUCTURE_AVAILABLE = False

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class AgentStartRequest(BaseModel):
    """Agent startup configuration"""
    agent_type: str = Field(..., description="Type of agent to start")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    capabilities: Optional[List[str]] = Field(None, description="Agent capabilities")
    resource_limits: Optional[Dict[str, Any]] = Field(None, description="Resource limits")

class AgentStatusResponse(BaseModel):
    """Agent status response"""
    agent_id: str
    agent_type: str
    status: str
    health_status: str
    capabilities: List[str]
    start_time: Optional[str] = None
    last_heartbeat: Optional[str] = None
    processed_tasks: int = 0
    failed_tasks: int = 0
    current_tasks: int = 0
    metrics: Dict[str, Any] = {}

class AgentTaskRequest(BaseModel):
    """Task assignment request"""
    task_type: str = Field(..., description="Type of task to assign")
    parameters: Dict[str, Any] = Field(..., description="Task parameters")
    priority: int = Field(5, ge=1, le=10, description="Task priority (1=highest)")
    timeout_seconds: Optional[int] = Field(3600, description="Task timeout")
    dependencies: Optional[List[str]] = Field(None, description="Task dependencies")

class AgentMetricsResponse(BaseModel):
    """Agent metrics response"""
    agent_id: str
    metrics: Dict[str, Any]
    performance_stats: Dict[str, float]
    resource_usage: Dict[str, float]
    collected_at: str

# Helper functions
def get_db() -> Session:
    """Get database session dependency."""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    return get_database_session()

async def get_agent_registry():
    """Get agent registry (Consul) connection."""
    if not AGENT_INFRASTRUCTURE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent infrastructure not available")
    return await get_consul_registry()

# API Endpoints

@router.get("/", response_model=Dict[str, Any])
async def list_agents(
    status: Optional[str] = Query(None, description="Filter by agent status"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum agents to return"),
    offset: int = Query(0, ge=0, description="Number of agents to skip"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    List all registered agents with filtering and pagination.
    Provides comprehensive agent status and capability information.
    """
    try:
        query = db.query(Agent)
        
        # Apply filters
        if status:
            query = query.filter(Agent.status == status)
        
        if agent_type:
            query = query.filter(Agent.agent_type == agent_type)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        agents = query.offset(offset).limit(limit).all()
        
        # Format agent information
        agent_list = []
        status_counts = {"active": 0, "inactive": 0, "error": 0, "maintenance": 0}
        
        for agent in agents:
            # Update status counts
            if agent.status in status_counts:
                status_counts[agent.status] += 1
            elif agent.status in ["running", "busy"]:
                status_counts["active"] += 1
            elif agent.status in ["stopped", "offline"]:
                status_counts["inactive"] += 1
            elif agent.status == "error":
                status_counts["error"] += 1
            
            agent_info = {
                "agent_id": agent.id,
                "agent_name": agent.agent_name,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "health_status": agent.health_status,
                "capabilities": agent.capabilities or [],
                "start_time": agent.start_time.isoformat() if agent.start_time else None,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                "processed_tasks": agent.total_tasks_processed,
                "failed_tasks": agent.total_tasks_failed,
                "current_task_id": str(agent.current_task_id) if agent.current_task_id else None,
                "host_info": agent.host_info
            }
            agent_list.append(agent_info)
        
        return {
            "agents": agent_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
            "status_summary": status_counts,
            "supported_types": ["pattern_discovery", "kml_parser", "cultural_analysis", "validation", "data_ingestion", "monitoring"],
            "infrastructure_available": AGENT_INFRASTRUCTURE_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agents: {str(e)}")

@router.get("/{agent_id}", response_model=AgentStatusResponse)
async def get_agent_details(
    agent_id: str,
    include_metrics: bool = Query(False, description="Include recent metrics"),
    include_tasks: bool = Query(False, description="Include recent tasks"),
    db: Session = Depends(get_db)
) -> AgentStatusResponse:
    """
    Get detailed information about a specific agent.
    Includes status, capabilities, metrics, and task history.
    """
    try:
        # Get agent from database
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Build response
        response_data = {
            "agent_id": agent.id,
            "agent_type": agent.agent_type,
            "status": agent.status,
            "health_status": agent.health_status,
            "capabilities": agent.capabilities or [],
            "start_time": agent.start_time.isoformat() if agent.start_time else None,
            "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            "processed_tasks": agent.total_tasks_processed,
            "failed_tasks": agent.total_tasks_failed,
            "current_tasks": 1 if agent.current_task_id else 0,
            "metrics": {}
        }
        
        # Add recent metrics if requested
        if include_metrics:
            recent_date = datetime.utcnow() - timedelta(hours=1)
            recent_metrics = db.query(AgentMetric).filter(
                AgentMetric.agent_id == agent_id,
                AgentMetric.timestamp_utc >= recent_date
            ).all()
            
            metrics_data = {}
            for metric in recent_metrics:
                if metric.metric_name not in metrics_data:
                    metrics_data[metric.metric_name] = []
                metrics_data[metric.metric_name].append({
                    "value": float(metric.metric_value),
                    "timestamp": metric.timestamp_utc.isoformat(),
                    "unit": metric.metric_unit
                })
            
            response_data["metrics"] = metrics_data
        
        # Add recent tasks if requested
        if include_tasks:
            recent_tasks = db.query(AgentTask).filter(
                AgentTask.assigned_agent_id == agent_id
            ).order_by(AgentTask.created_at.desc()).limit(10).all()
            
            response_data["recent_tasks"] = [
                {
                    "task_id": str(task.id),
                    "task_type": task.task_type,
                    "status": task.status,
                    "priority": task.priority,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }
                for task in recent_tasks
            ]
        
        return AgentStatusResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent details for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent details: {str(e)}")

@router.post("/{agent_id}/start")
async def start_agent(
    agent_id: str,
    start_request: AgentStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Start a specific agent with configuration.
    Creates agent record and initiates startup process.
    """
    try:
        # Check if agent already exists
        existing_agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if existing_agent and existing_agent.status in ["active", "running", "busy"]:
            raise HTTPException(status_code=400, detail="Agent is already running")
        
        # Create or update agent record
        if existing_agent:
            existing_agent.status = "starting"
            existing_agent.configuration = start_request.configuration
            existing_agent.capabilities = start_request.capabilities or []
            existing_agent.updated_at = datetime.utcnow()
            agent = existing_agent
        else:
            agent = Agent(
                id=agent_id,
                agent_name=f"{start_request.agent_type}-{agent_id}",
                agent_type=start_request.agent_type,
                status="starting",
                health_status="unknown",
                capabilities=start_request.capabilities or [],
                configuration=start_request.configuration or {},
                resource_requirements=start_request.resource_limits,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(agent)
        
        db.commit()
        
        # In a real implementation, this would trigger actual agent startup
        # For now, we'll simulate the process
        background_tasks.add_task(_simulate_agent_startup, agent_id, db)
        
        return {
            "agent_id": agent_id,
            "action": "start",
            "status": "initiated",
            "message": f"Agent {agent_id} startup initiated",
            "agent_type": start_request.agent_type,
            "configuration_applied": start_request.configuration is not None,
            "expected_capabilities": start_request.capabilities or []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start agent: {str(e)}")

@router.post("/{agent_id}/stop")
async def stop_agent(
    agent_id: str,
    graceful: bool = Query(True, description="Perform graceful shutdown"),
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Stop a specific agent gracefully or forcefully.
    Handles task completion and cleanup.
    """
    try:
        # Get agent
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        if agent.status in ["stopped", "offline", "inactive"]:
            raise HTTPException(status_code=400, detail="Agent is already stopped")
        
        # Update agent status
        agent.status = "stopping" if graceful else "force_stopping"
        agent.updated_at = datetime.utcnow()
        db.commit()
        
        # In a real implementation, this would send shutdown signals
        background_tasks.add_task(_simulate_agent_shutdown, agent_id, graceful, db)
        
        return {
            "agent_id": agent_id,
            "action": "stop",
            "status": "initiated",
            "graceful": graceful,
            "message": f"Agent {agent_id} {'graceful ' if graceful else 'forced '}shutdown initiated",
            "current_tasks": 1 if agent.current_task_id else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop agent: {str(e)}")

@router.post("/{agent_id}/restart")
async def restart_agent(
    agent_id: str,
    start_request: Optional[AgentStartRequest] = None,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Restart an agent with optional new configuration.
    Performs stop followed by start sequence.
    """
    try:
        # Get agent
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Update configuration if provided
        if start_request:
            agent.configuration = start_request.configuration or agent.configuration
            agent.capabilities = start_request.capabilities or agent.capabilities
        
        agent.status = "restarting"
        agent.updated_at = datetime.utcnow()
        db.commit()
        
        # Simulate restart process
        background_tasks.add_task(_simulate_agent_restart, agent_id, db)
        
        return {
            "agent_id": agent_id,
            "action": "restart",
            "status": "initiated",
            "message": f"Agent {agent_id} restart initiated",
            "configuration_updated": start_request is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart agent: {str(e)}")

@router.get("/{agent_id}/tasks")
async def get_agent_tasks(
    agent_id: str,
    status: Optional[str] = Query(None, description="Filter by task status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get task queue and history for specific agent.
    Supports filtering by status and pagination.
    """
    try:
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        query = db.query(AgentTask).filter(AgentTask.assigned_agent_id == agent_id)
        
        if status:
            query = query.filter(AgentTask.status == status)
        
        total = query.count()
        tasks = query.order_by(AgentTask.created_at.desc()).offset(offset).limit(limit).all()
        
        task_list = []
        for task in tasks:
            task_info = {
                "task_id": str(task.id),
                "task_type": task.task_type,
                "status": task.status,
                "priority": task.priority,
                "progress_percentage": task.progress_percentage,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error_message": task.error_message,
                "retry_count": task.retry_count
            }
            task_list.append(task_info)
        
        # Get status distribution
        status_stats = db.query(AgentTask.status).filter(
            AgentTask.assigned_agent_id == agent_id
        ).all()
        status_distribution = {}
        for (status_val,) in status_stats:
            status_distribution[status_val] = status_distribution.get(status_val, 0) + 1
        
        return {
            "agent_id": agent_id,
            "tasks": task_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
            "status_distribution": status_distribution
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tasks for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent tasks: {str(e)}")

@router.post("/{agent_id}/tasks", response_model=Dict[str, Any])
async def assign_task_to_agent(
    agent_id: str,
    task_request: AgentTaskRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Assign a task to a specific agent.
    Creates task record and adds to agent's queue.
    """
    try:
        # Verify agent exists and is capable
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        if agent.status not in ["active", "running", "idle"]:
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not available for tasks")
        
        # Create task
        task = AgentTask(
            task_type=task_request.task_type,
            priority=task_request.priority,
            status="pending",
            assigned_agent_id=agent_id,
            created_by="api_user",
            task_parameters=task_request.parameters,
            timeout_seconds=task_request.timeout_seconds,
            dependencies=task_request.dependencies or [],
            created_at=datetime.utcnow()
        )
        
        db.add(task)
        db.commit()
        
        return {
            "task_id": str(task.id),
            "agent_id": agent_id,
            "task_type": task_request.task_type,
            "status": "assigned",
            "priority": task_request.priority,
            "estimated_start": "immediate",
            "timeout_seconds": task_request.timeout_seconds
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign task to agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign task: {str(e)}")

@router.get("/{agent_id}/metrics", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    agent_id: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of metrics to retrieve"),
    db: Session = Depends(get_db)
) -> AgentMetricsResponse:
    """
    Get performance metrics and statistics for specific agent.
    Includes resource usage, task performance, and health metrics.
    """
    try:
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get metrics from specified time range
        start_time = datetime.utcnow() - timedelta(hours=hours)
        metrics = db.query(AgentMetric).filter(
            AgentMetric.agent_id == agent_id,
            AgentMetric.timestamp_utc >= start_time
        ).all()
        
        # Organize metrics by name
        metrics_data = {}
        for metric in metrics:
            if metric.metric_name not in metrics_data:
                metrics_data[metric.metric_name] = []
            
            metrics_data[metric.metric_name].append({
                "value": float(metric.metric_value),
                "timestamp": metric.timestamp_utc.isoformat(),
                "unit": metric.metric_unit,
                "type": metric.metric_type
            })
        
        # Calculate performance statistics
        performance_stats = {
            "average_task_duration_ms": agent.average_task_duration_ms or 0,
            "task_success_rate": 0.0,
            "tasks_per_hour": 0.0
        }
        
        if agent.total_tasks_processed > 0:
            performance_stats["task_success_rate"] = (
                (agent.total_tasks_processed - agent.total_tasks_failed) /
                agent.total_tasks_processed
            )
        
        if agent.start_time:
            uptime_hours = (datetime.utcnow() - agent.start_time).total_seconds() / 3600
            if uptime_hours > 0:
                performance_stats["tasks_per_hour"] = agent.total_tasks_processed / uptime_hours
        
        # Calculate resource usage averages
        resource_usage = {}
        for metric_name in ["cpu_percent", "memory_percent", "threads", "open_files"]:
            if metric_name in metrics_data and metrics_data[metric_name]:
                values = [m["value"] for m in metrics_data[metric_name]]
                resource_usage[metric_name] = {
                    "current": values[-1] if values else 0,
                    "average": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values)
                }
        
        return AgentMetricsResponse(
            agent_id=agent_id,
            metrics=metrics_data,
            performance_stats=performance_stats,
            resource_usage=resource_usage,
            collected_at=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent metrics: {str(e)}")

@router.put("/{agent_id}/config")
async def update_agent_config(
    agent_id: str,
    configuration: Dict[str, Any],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Update agent configuration dynamically.
    Configuration changes are applied without restarting the agent.
    """
    try:
        # Get agent
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Store old configuration for rollback
        old_config = agent.configuration.copy() if agent.configuration else {}
        
        # Update configuration
        if agent.configuration:
            agent.configuration.update(configuration)
        else:
            agent.configuration = configuration
        
        agent.updated_at = datetime.utcnow()
        db.commit()
        
        return {
            "agent_id": agent_id,
            "status": "updated",
            "message": "Agent configuration updated successfully",
            "configuration": agent.configuration,
            "changes_applied": len(configuration),
            "requires_restart": False  # In real implementation, check if restart needed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update config for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update agent configuration: {str(e)}")

# Background task simulation functions
async def _simulate_agent_startup(agent_id: str, db: Session):
    """Simulate agent startup process."""
    import asyncio
    await asyncio.sleep(5)  # Simulate startup time
    
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent:
        agent.status = "active"
        agent.health_status = "healthy"
        agent.start_time = datetime.utcnow()
        agent.last_heartbeat = datetime.utcnow()
        db.commit()

async def _simulate_agent_shutdown(agent_id: str, graceful: bool, db: Session):
    """Simulate agent shutdown process."""
    import asyncio
    await asyncio.sleep(3 if graceful else 1)
    
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent:
        agent.status = "stopped"
        agent.health_status = "down"
        agent.current_task_id = None
        db.commit()

async def _simulate_agent_restart(agent_id: str, db: Session):
    """Simulate agent restart process."""
    import asyncio
    await asyncio.sleep(2)  # Stop
    await asyncio.sleep(5)  # Start
    
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent:
        agent.status = "active"
        agent.health_status = "healthy"
        agent.start_time = datetime.utcnow()
        agent.last_heartbeat = datetime.utcnow()
        db.commit()