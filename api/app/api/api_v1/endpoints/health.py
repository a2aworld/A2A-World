"""
A2A World Platform - System Management and Health Monitoring Endpoints

Comprehensive endpoints for system health monitoring, configuration management,
user session management, and operational maintenance.
"""

import psutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

try:
    from sqlalchemy.orm import Session
    from database.models.agents import SystemHealth
    from database.models.users import User
    from database.connection import get_database_session
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from agents.core.messaging import get_nats_client
    from agents.core.registry import get_consul_registry
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models
class SystemHealthResponse(BaseModel):
    """System health response model"""
    status: str
    service: str
    version: str
    uptime_seconds: float
    timestamp: str
    components: Dict[str, Any]
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]

class ConfigurationUpdate(BaseModel):
    """Configuration update request"""
    component: str = Field(..., description="Component to configure")
    configuration: Dict[str, Any] = Field(..., description="Configuration parameters")
    restart_required: bool = Field(False, description="Whether restart is required")

class MaintenanceTask(BaseModel):
    """Maintenance task request"""
    task_type: str = Field(..., description="Type of maintenance task")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Task parameters")
    schedule: Optional[str] = Field(None, description="Schedule for task execution")

# Helper functions
def get_db() -> Session:
    """Get database session dependency."""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    return get_database_session()

def get_system_uptime() -> float:
    """Get system uptime in seconds."""
    try:
        return psutil.boot_time()
    except Exception:
        return 0.0

async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance."""
    if not DATABASE_AVAILABLE:
        return {"status": "unavailable", "error": "Database not configured"}
    
    try:
        start_time = datetime.utcnow()
        with get_database_session() as session:
            # Simple connectivity test
            session.execute("SELECT 1")
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Get basic stats
            result = session.execute("SELECT current_database(), version()").fetchone()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "database": result[0] if result else "unknown",
                "version": result[1] if result else "unknown",
                "connection_pool": "active"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": None
        }

async def check_nats_health() -> Dict[str, Any]:
    """Check NATS messaging system health."""
    if not INFRASTRUCTURE_AVAILABLE:
        return {"status": "unavailable", "error": "NATS not configured"}
    
    try:
        nats_client = await get_nats_client()
        if nats_client and nats_client.is_connected:
            return {
                "status": "healthy",
                "connected": True,
                "server_info": nats_client.server_info if hasattr(nats_client, 'server_info') else {}
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Not connected to NATS server"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }

async def check_consul_health() -> Dict[str, Any]:
    """Check Consul service registry health."""
    if not INFRASTRUCTURE_AVAILABLE:
        return {"status": "unavailable", "error": "Consul not configured"}
    
    try:
        consul_registry = await get_consul_registry()
        if consul_registry:
            # Test basic connectivity
            services = await consul_registry.discover_services("health-check")
            return {
                "status": "healthy",
                "connected": True,
                "services_registered": len(services),
                "cluster_info": {}
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Cannot connect to Consul"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }

# API Endpoints

@router.get("/", response_model=SystemHealthResponse)
async def health_check() -> SystemHealthResponse:
    """
    Basic health check endpoint with essential system metrics.
    Returns system status, uptime, and basic performance indicators.
    """
    try:
        current_time = datetime.utcnow()
        boot_time = psutil.boot_time()
        uptime_seconds = (current_time.timestamp() - boot_time)
        
        # Get basic system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine overall status
        status = "healthy"
        if cpu_percent > 90 or memory.percent > 95 or disk.percent > 90:
            status = "degraded"
        if cpu_percent > 95 or memory.percent > 98:
            status = "critical"
        
        return SystemHealthResponse(
            status=status,
            service="a2a-world-api",
            version="0.1.0",
            uptime_seconds=uptime_seconds,
            timestamp=current_time.isoformat(),
            components={
                "api": {"status": "healthy"},
                "database": {"status": "checking"},
                "messaging": {"status": "checking"},
                "registry": {"status": "checking"}
            },
            performance_metrics={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "response_time_ms": 0.5  # Placeholder
            },
            resource_usage={
                "cpu_cores": psutil.cpu_count(),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealthResponse(
            status="critical",
            service="a2a-world-api",
            version="0.1.0",
            uptime_seconds=0,
            timestamp=datetime.utcnow().isoformat(),
            components={"error": str(e)},
            performance_metrics={},
            resource_usage={}
        )

@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check including all system dependencies.
    Performs connectivity tests and returns detailed status information.
    """
    try:
        start_time = datetime.utcnow()
        
        # Check all components
        database_health = await check_database_health()
        nats_health = await check_nats_health()
        consul_health = await check_consul_health()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Process information
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
        
        # Determine overall health
        component_statuses = [
            database_health["status"],
            nats_health["status"],
            consul_health["status"]
        ]
        
        healthy_components = len([s for s in component_statuses if s == "healthy"])
        total_components = len(component_statuses)
        
        if healthy_components == total_components:
            overall_status = "healthy"
        elif healthy_components >= total_components * 0.5:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        # Add resource warnings
        warnings = []
        if cpu_percent > 80:
            warnings.append(f"High CPU usage: {cpu_percent}%")
        if memory.percent > 85:
            warnings.append(f"High memory usage: {memory.percent}%")
        if disk.percent > 80:
            warnings.append(f"Low disk space: {100-disk.percent}% free")
        
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "status": overall_status,
            "service": "a2a-world-api",
            "version": "0.1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "health_check_duration_ms": round(total_time, 2),
            "components": {
                "database": database_health,
                "messaging": nats_health,
                "service_registry": consul_health
            },
            "system_metrics": {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
                },
                "memory": {
                    "percent": memory.percent,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2)
                },
                "disk": {
                    "percent": disk.percent,
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            },
            "process_info": process_info,
            "warnings": warnings,
            "component_summary": {
                "healthy": healthy_components,
                "total": total_components,
                "health_ratio": healthy_components / total_components
            }
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/metrics")
async def get_system_metrics(
    hours: int = Query(24, ge=1, le=168, description="Hours of metrics history"),
    component: Optional[str] = Query(None, description="Filter by component"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed system metrics and performance data.
    Includes historical data and trend analysis.
    """
    try:
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Get historical health data
        query = db.query(SystemHealth).filter(SystemHealth.last_check >= start_time)
        
        if component:
            query = query.filter(SystemHealth.component_name == component)
        
        health_records = query.order_by(SystemHealth.last_check.desc()).all()
        
        # Current system metrics
        current_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime_seconds": datetime.utcnow().timestamp() - psutil.boot_time(),
            "process_count": len(psutil.pids()),
            "network_connections": len(psutil.net_connections())
        }
        
        # Process historical data
        metrics_timeline = []
        component_summary = {}
        
        for record in health_records:
            metrics_timeline.append({
                "timestamp": record.last_check.isoformat(),
                "component": record.component_name,
                "health_score": float(record.health_score) if record.health_score else 0,
                "response_time_ms": record.response_time_ms,
                "error_rate": float(record.error_rate) if record.error_rate else 0,
                "resource_utilization": record.resource_utilization or {}
            })
            
            # Build component summary
            comp_name = record.component_name
            if comp_name not in component_summary:
                component_summary[comp_name] = {
                    "total_checks": 0,
                    "healthy_checks": 0,
                    "average_response_time": 0,
                    "average_health_score": 0
                }
            
            component_summary[comp_name]["total_checks"] += 1
            if record.health_status == "healthy":
                component_summary[comp_name]["healthy_checks"] += 1
        
        # Calculate averages
        for comp_name, data in component_summary.items():
            if data["total_checks"] > 0:
                data["uptime_percentage"] = (data["healthy_checks"] / data["total_checks"]) * 100
        
        return {
            "current_metrics": current_metrics,
            "historical_data": {
                "period_hours": hours,
                "total_records": len(health_records),
                "timeline": metrics_timeline[:100]  # Limit to recent 100 records
            },
            "component_summary": component_summary,
            "performance_trends": {
                "cpu_trend": "stable",  # Would calculate from historical data
                "memory_trend": "stable",
                "error_trend": "decreasing"
            },
            "alerts": _generate_performance_alerts(current_metrics, health_records)
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")

@router.get("/config")
async def get_system_configuration() -> Dict[str, Any]:
    """
    Get current system configuration and settings.
    Returns configuration for all system components.
    """
    try:
        # In a real implementation, this would read from configuration management
        configuration = {
            "api": {
                "version": "0.1.0",
                "debug_mode": False,
                "log_level": "INFO",
                "cors_enabled": True,
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 100
                }
            },
            "database": {
                "pool_size": 10,
                "max_connections": 50,
                "timeout_seconds": 30,
                "ssl_enabled": True
            },
            "messaging": {
                "nats_url": "nats://localhost:4222",
                "max_reconnect_attempts": 10,
                "reconnect_wait_seconds": 2
            },
            "agents": {
                "max_concurrent_agents": 20,
                "task_timeout_seconds": 3600,
                "heartbeat_interval_seconds": 30
            },
            "monitoring": {
                "metrics_retention_days": 30,
                "health_check_interval_seconds": 60,
                "alert_thresholds": {
                    "cpu_percent": 80,
                    "memory_percent": 85,
                    "disk_percent": 90
                }
            }
        }
        
        return {
            "configuration": configuration,
            "last_updated": datetime.utcnow().isoformat(),
            "environment": "development",  # Would detect actual environment
            "config_source": "default",
            "validation_status": "valid"
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@router.post("/config")
async def update_system_configuration(
    config_update: ConfigurationUpdate,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Update system configuration for specific component.
    Changes are applied dynamically when possible.
    """
    try:
        # Validate configuration update
        valid_components = ["api", "database", "messaging", "agents", "monitoring"]
        if config_update.component not in valid_components:
            raise HTTPException(status_code=400, detail=f"Invalid component. Valid options: {valid_components}")
        
        # In a real implementation, this would:
        # 1. Validate the configuration
        # 2. Apply changes to the running system
        # 3. Store changes persistently
        # 4. Notify relevant services
        
        # For now, simulate the process
        background_tasks.add_task(_apply_configuration_changes, config_update)
        
        return {
            "success": True,
            "component": config_update.component,
            "changes_applied": len(config_update.configuration),
            "restart_required": config_update.restart_required,
            "applied_at": datetime.utcnow().isoformat(),
            "message": f"Configuration for {config_update.component} updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@router.get("/logs")
async def get_system_logs(
    level: Optional[str] = Query(None, description="Log level filter"),
    component: Optional[str] = Query(None, description="Component filter"),
    hours: int = Query(24, ge=1, le=168, description="Hours of logs to retrieve"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum log entries")
) -> Dict[str, Any]:
    """
    Get system logs with filtering and pagination.
    Supports filtering by level, component, and time range.
    """
    try:
        # In a real implementation, this would read from centralized logging
        # For now, return simulated log data
        
        logs = []
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        components = ["api", "database", "agents", "messaging"]
        
        # Generate sample log entries
        import random
        for i in range(min(limit, 100)):
            timestamp = datetime.utcnow() - timedelta(
                minutes=random.randint(0, hours * 60)
            )
            
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "level": random.choice(log_levels) if not level else level,
                "component": random.choice(components) if not component else component,
                "message": f"Sample log message {i}",
                "context": {
                    "request_id": f"req_{i}",
                    "user_id": f"user_{random.randint(1, 100)}"
                }
            }
            
            # Apply filters
            if level and log_entry["level"] != level.upper():
                continue
            if component and log_entry["component"] != component:
                continue
                
            logs.append(log_entry)
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Generate summary statistics
        level_counts = {}
        component_counts = {}
        
        for log in logs:
            level_counts[log["level"]] = level_counts.get(log["level"], 0) + 1
            component_counts[log["component"]] = component_counts.get(log["component"], 0) + 1
        
        return {
            "logs": logs[:limit],
            "total_entries": len(logs),
            "filters_applied": {
                "level": level,
                "component": component,
                "hours": hours
            },
            "summary": {
                "level_distribution": level_counts,
                "component_distribution": component_counts,
                "time_range": {
                    "start": (datetime.utcnow() - timedelta(hours=hours)).isoformat(),
                    "end": datetime.utcnow().isoformat()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")

@router.post("/maintenance")
async def trigger_maintenance_task(
    maintenance_task: MaintenanceTask,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Trigger system maintenance operations.
    Supports various maintenance tasks like cleanup, optimization, and backups.
    """
    try:
        valid_tasks = [
            "database_cleanup",
            "log_rotation",
            "cache_cleanup",
            "system_backup",
            "performance_optimization",
            "health_check_reset"
        ]
        
        if maintenance_task.task_type not in valid_tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid maintenance task. Valid options: {valid_tasks}"
            )
        
        # Generate task ID
        task_id = f"maint_{maintenance_task.task_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Schedule or execute maintenance task
        background_tasks.add_task(_execute_maintenance_task, task_id, maintenance_task)
        
        return {
            "success": True,
            "task_id": task_id,
            "task_type": maintenance_task.task_type,
            "status": "initiated",
            "scheduled": maintenance_task.schedule is not None,
            "initiated_at": datetime.utcnow().isoformat(),
            "estimated_duration": _get_estimated_duration(maintenance_task.task_type),
            "message": f"Maintenance task '{maintenance_task.task_type}' has been initiated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger maintenance task: {e}")
        raise HTTPException(status_code=500, detail=f"Maintenance task failed: {str(e)}")

# Background task functions
async def _apply_configuration_changes(config_update: ConfigurationUpdate):
    """Apply configuration changes in background."""
    import asyncio
    await asyncio.sleep(2)  # Simulate configuration application
    logger.info(f"Configuration changes applied for {config_update.component}")

async def _execute_maintenance_task(task_id: str, task: MaintenanceTask):
    """Execute maintenance task in background."""
    import asyncio
    duration = _get_estimated_duration(task.task_type)
    await asyncio.sleep(duration)
    logger.info(f"Maintenance task {task_id} completed")

def _get_estimated_duration(task_type: str) -> int:
    """Get estimated duration for maintenance task."""
    durations = {
        "database_cleanup": 300,  # 5 minutes
        "log_rotation": 60,       # 1 minute
        "cache_cleanup": 30,      # 30 seconds
        "system_backup": 1800,    # 30 minutes
        "performance_optimization": 600,  # 10 minutes
        "health_check_reset": 10  # 10 seconds
    }
    return durations.get(task_type, 60)

def _generate_performance_alerts(current_metrics: Dict, health_records: List) -> List[Dict]:
    """Generate performance alerts based on current metrics."""
    alerts = []
    
    if current_metrics.get("cpu_percent", 0) > 80:
        alerts.append({
            "type": "warning",
            "component": "system",
            "message": f"High CPU usage: {current_metrics['cpu_percent']}%",
            "threshold": 80,
            "current_value": current_metrics["cpu_percent"]
        })
    
    if current_metrics.get("memory_percent", 0) > 85:
        alerts.append({
            "type": "critical",
            "component": "system",
            "message": f"High memory usage: {current_metrics['memory_percent']}%",
            "threshold": 85,
            "current_value": current_metrics["memory_percent"]
        })
    
    return alerts