"""
A2A World Platform - Task Queue System

Task queue and job processing system for coordinating work between agents.
Integrates with database for persistence and NATS for real-time distribution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import uuid
import json

from agents.core.messaging import AgentMessage, NATSClient
from agents.core.registry import ConsulRegistry, AgentServiceInfo


class TaskStatus(Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 5
    LOW = 8
    BACKGROUND = 10


class Task:
    """Represents a task in the queue system."""
    
    def __init__(
        self,
        task_id: Optional[str] = None,
        task_type: str = "",
        priority: int = TaskPriority.NORMAL.value,
        parameters: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
        created_by: str = "system",
        dependencies: Optional[List[str]] = None,
        timeout_seconds: int = 3600,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.task_type = task_type
        self.priority = priority
        self.status = TaskStatus.PENDING.value
        self.parameters = parameters or {}
        self.input_data = input_data or {}
        self.output_data: Dict[str, Any] = {}
        self.created_by = created_by
        self.assigned_agent_id: Optional[str] = None
        self.dependencies = dependencies or []
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_count = 0
        self.metadata = metadata or {}
        
        # Timestamps
        self.created_at = datetime.utcnow()
        self.scheduled_for = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.last_retry_at: Optional[datetime] = None
        
        # Progress and error tracking
        self.progress_percentage = 0
        self.estimated_duration_ms: Optional[int] = None
        self.actual_duration_ms: Optional[int] = None
        self.error_message: Optional[str] = None
        self.error_details: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "status": self.status,
            "parameters": self.parameters,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "created_by": self.created_by,
            "assigned_agent_id": self.assigned_agent_id,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "scheduled_for": self.scheduled_for.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "progress_percentage": self.progress_percentage,
            "estimated_duration_ms": self.estimated_duration_ms,
            "actual_duration_ms": self.actual_duration_ms,
            "error_message": self.error_message,
            "error_details": self.error_details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        task = cls(
            task_id=data.get("task_id"),
            task_type=data.get("task_type", ""),
            priority=data.get("priority", TaskPriority.NORMAL.value),
            parameters=data.get("parameters", {}),
            input_data=data.get("input_data", {}),
            created_by=data.get("created_by", "system"),
            dependencies=data.get("dependencies", []),
            timeout_seconds=data.get("timeout_seconds", 3600),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {})
        )
        
        # Set additional fields
        task.status = data.get("status", TaskStatus.PENDING.value)
        task.output_data = data.get("output_data", {})
        task.assigned_agent_id = data.get("assigned_agent_id")
        task.retry_count = data.get("retry_count", 0)
        task.progress_percentage = data.get("progress_percentage", 0)
        task.estimated_duration_ms = data.get("estimated_duration_ms")
        task.actual_duration_ms = data.get("actual_duration_ms")
        task.error_message = data.get("error_message")
        task.error_details = data.get("error_details", {})
        
        # Parse timestamps
        if data.get("created_at"):
            task.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("scheduled_for"):
            task.scheduled_for = datetime.fromisoformat(data["scheduled_for"])
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        if data.get("last_retry_at"):
            task.last_retry_at = datetime.fromisoformat(data["last_retry_at"])
        
        return task
    
    def is_ready_to_execute(self, completed_tasks: set) -> bool:
        """Check if task is ready for execution (dependencies satisfied)."""
        if self.status != TaskStatus.PENDING.value:
            return False
        
        if datetime.utcnow() < self.scheduled_for:
            return False
        
        # Check dependencies
        for dep_id in self.dependencies:
            if dep_id not in completed_tasks:
                return False
        
        return True
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED.value
    
    def is_expired(self) -> bool:
        """Check if task has exceeded its timeout."""
        if not self.started_at:
            return False
        
        timeout_delta = timedelta(seconds=self.timeout_seconds)
        return datetime.utcnow() > (self.started_at + timeout_delta)


class TaskQueue:
    """
    Distributed task queue system for agent coordination.
    Manages task distribution, execution tracking, and load balancing.
    """
    
    def __init__(
        self,
        nats_client: NATSClient,
        registry: ConsulRegistry,
        db_connection=None,  # Database connection for persistence
        queue_name: str = "a2a-tasks"
    ):
        self.nats = nats_client
        self.registry = registry
        self.db = db_connection
        self.queue_name = queue_name
        self.logger = logging.getLogger("task_queue")
        
        # In-memory task tracking
        self.pending_tasks: Dict[str, Task] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: set = set()
        self.failed_tasks: Dict[str, Task] = {}
        
        # Task handlers by type
        self.task_handlers: Dict[str, Callable] = {}
        
        # Queue management
        self.task_dispatcher_running = False
        self.task_monitor_running = False
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the task queue system."""
        try:
            # Subscribe to task-related NATS subjects
            await self.nats.subscribe(
                "agents.tasks.submit",
                self._handle_task_submission
            )
            
            await self.nats.subscribe(
                "agents.tasks.request",
                self._handle_task_request
            )
            
            await self.nats.subscribe(
                "agents.tasks.update",
                self._handle_task_update
            )
            
            await self.nats.subscribe(
                "agents.tasks.complete",
                self._handle_task_completion
            )
            
            # Start background workers
            self._dispatcher_task = asyncio.create_task(self._task_dispatcher())
            self._monitor_task = asyncio.create_task(self._task_monitor())
            
            self.task_dispatcher_running = True
            self.task_monitor_running = True
            
            # Load pending tasks from database
            await self._load_pending_tasks()
            
            self.logger.info("Task queue system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start task queue: {e}")
            raise
    
    async def stop(self):
        """Stop the task queue system."""
        self.task_dispatcher_running = False
        self.task_monitor_running = False
        
        # Cancel background tasks
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()
        
        # Wait for tasks to complete
        if self._dispatcher_task or self._monitor_task:
            await asyncio.gather(
                self._dispatcher_task, self._monitor_task,
                return_exceptions=True
            )
        
        self.logger.info("Task queue system stopped")
    
    async def submit_task(self, task: Task) -> bool:
        """Submit a new task to the queue."""
        try:
            # Store task in database if connection available
            if self.db:
                await self._persist_task(task)
            
            # Add to pending tasks
            self.pending_tasks[task.task_id] = task
            
            # Notify about new task
            message = AgentMessage.create(
                sender_id="task_queue",
                message_type="task_submitted",
                payload=task.to_dict()
            )
            
            await self.nats.publish("agents.tasks.notify", message)
            
            self.logger.info(f"Task {task.task_id} submitted: {task.task_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to a specific agent."""
        try:
            if task_id not in self.pending_tasks:
                self.logger.warning(f"Task {task_id} not found in pending tasks")
                return False
            
            task = self.pending_tasks.pop(task_id)
            task.assigned_agent_id = agent_id
            task.status = TaskStatus.ASSIGNED.value
            task.started_at = datetime.utcnow()
            
            # Move to active tasks
            self.active_tasks[task_id] = task
            
            # Update database
            if self.db:
                await self._update_task_status(task)
            
            # Notify agent
            message = AgentMessage.create(
                sender_id="task_queue",
                receiver_id=agent_id,
                message_type="task_assigned",
                payload=task.to_dict()
            )
            
            await self.nats.publish(f"agents.{agent_id}.tasks", message)
            
            self.logger.info(f"Task {task_id} assigned to agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to assign task {task_id} to {agent_id}: {e}")
            return False
    
    async def complete_task(self, task_id: str, result: Dict[str, Any], agent_id: str) -> bool:
        """Mark a task as completed with results."""
        try:
            if task_id not in self.active_tasks:
                self.logger.warning(f"Active task {task_id} not found")
                return False
            
            task = self.active_tasks.pop(task_id)
            task.status = TaskStatus.COMPLETED.value
            task.completed_at = datetime.utcnow()
            task.output_data = result
            task.progress_percentage = 100
            
            # Calculate duration
            if task.started_at:
                duration = task.completed_at - task.started_at
                task.actual_duration_ms = int(duration.total_seconds() * 1000)
            
            # Add to completed tasks
            self.completed_tasks.add(task_id)
            
            # Update database
            if self.db:
                await self._update_task_status(task)
            
            # Notify completion
            message = AgentMessage.create(
                sender_id=agent_id,
                message_type="task_completed",
                payload={
                    "task_id": task_id,
                    "result": result,
                    "duration_ms": task.actual_duration_ms
                }
            )
            
            await self.nats.publish("agents.tasks.completed", message)
            
            self.logger.info(f"Task {task_id} completed by agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    async def fail_task(self, task_id: str, error: str, agent_id: str, retry: bool = True) -> bool:
        """Mark a task as failed."""
        try:
            if task_id not in self.active_tasks:
                self.logger.warning(f"Active task {task_id} not found")
                return False
            
            task = self.active_tasks.pop(task_id)
            task.error_message = error
            task.last_retry_at = datetime.utcnow()
            
            # Determine if task should be retried
            if retry and task.can_retry():
                task.retry_count += 1
                task.status = TaskStatus.RETRY.value
                task.assigned_agent_id = None  # Reset assignment
                
                # Add back to pending tasks for retry
                self.pending_tasks[task_id] = task
                
                self.logger.info(f"Task {task_id} will be retried (attempt {task.retry_count}/{task.max_retries})")
            else:
                task.status = TaskStatus.FAILED.value
                self.failed_tasks[task_id] = task
                
                self.logger.warning(f"Task {task_id} failed permanently: {error}")
            
            # Update database
            if self.db:
                await self._update_task_status(task)
            
            # Notify failure
            message = AgentMessage.create(
                sender_id=agent_id,
                message_type="task_failed",
                payload={
                    "task_id": task_id,
                    "error": error,
                    "retry_count": task.retry_count,
                    "will_retry": task.status == TaskStatus.RETRY.value
                }
            )
            
            await self.nats.publish("agents.tasks.failed", message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to handle task failure {task_id}: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a task."""
        # Check all task collections
        for task_dict in [self.pending_tasks, self.active_tasks, self.failed_tasks]:
            if task_id in task_dict:
                return task_dict[task_id].to_dict()
        
        # Check completed tasks (minimal info)
        if task_id in self.completed_tasks:
            return {"task_id": task_id, "status": TaskStatus.COMPLETED.value}
        
        # Check database if available
        if self.db:
            return await self._get_task_from_db(task_id)
        
        return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tasks": len(self.pending_tasks) + len(self.active_tasks) + len(self.completed_tasks) + len(self.failed_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _task_dispatcher(self):
        """Background task for assigning tasks to available agents."""
        while self.task_dispatcher_running:
            try:
                if not self.pending_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Get tasks ready for execution (sorted by priority)
                ready_tasks = []
                for task in self.pending_tasks.values():
                    if task.is_ready_to_execute(self.completed_tasks):
                        ready_tasks.append(task)
                
                ready_tasks.sort(key=lambda t: (t.priority, t.created_at))
                
                if not ready_tasks:
                    await asyncio.sleep(5)
                    continue
                
                # Find available agents for each task
                for task in ready_tasks[:10]:  # Process up to 10 tasks at once
                    try:
                        # Find agents capable of handling this task type
                        capable_agents = await self.registry.find_agents_by_capability(task.task_type)
                        
                        if not capable_agents:
                            self.logger.debug(f"No agents available for task type: {task.task_type}")
                            continue
                        
                        # Filter for healthy agents
                        healthy_agents = [agent for agent in capable_agents if agent.status == "healthy"]
                        
                        if not healthy_agents:
                            continue
                        
                        # Select agent (simple round-robin for now)
                        selected_agent = healthy_agents[0]  # Could implement better selection logic
                        
                        # Assign task
                        success = await self.assign_task(task.task_id, selected_agent.agent_id)
                        
                        if success:
                            break  # Assign one task at a time to avoid overwhelming
                        
                    except Exception as e:
                        self.logger.error(f"Error assigning task {task.task_id}: {e}")
                
                await asyncio.sleep(1)  # Brief pause between dispatch cycles
                
            except Exception as e:
                self.logger.error(f"Task dispatcher error: {e}")
                await asyncio.sleep(5)
    
    async def _task_monitor(self):
        """Background task for monitoring task timeouts and health."""
        while self.task_monitor_running:
            try:
                current_time = datetime.utcnow()
                expired_tasks = []
                
                # Check for expired active tasks
                for task_id, task in self.active_tasks.items():
                    if task.is_expired():
                        expired_tasks.append(task_id)
                
                # Handle expired tasks
                for task_id in expired_tasks:
                    await self.fail_task(
                        task_id, 
                        f"Task timeout after {self.active_tasks[task_id].timeout_seconds} seconds",
                        self.active_tasks[task_id].assigned_agent_id or "system",
                        retry=True
                    )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Task monitor error: {e}")
                await asyncio.sleep(60)
    
    # NATS message handlers
    async def _handle_task_submission(self, message: AgentMessage):
        """Handle task submission via NATS."""
        try:
            task_data = message.payload
            task = Task.from_dict(task_data)
            await self.submit_task(task)
        except Exception as e:
            self.logger.error(f"Error handling task submission: {e}")
    
    async def _handle_task_request(self, message: AgentMessage):
        """Handle task request from agent."""
        try:
            agent_id = message.sender_id
            capabilities = message.payload.get("capabilities", [])
            
            # Find suitable task
            for task_id, task in self.pending_tasks.items():
                if task.task_type in capabilities and task.is_ready_to_execute(self.completed_tasks):
                    await self.assign_task(task_id, agent_id)
                    break
        except Exception as e:
            self.logger.error(f"Error handling task request: {e}")
    
    async def _handle_task_update(self, message: AgentMessage):
        """Handle task progress update from agent."""
        try:
            task_id = message.payload.get("task_id")
            progress = message.payload.get("progress", 0)
            
            if task_id in self.active_tasks:
                self.active_tasks[task_id].progress_percentage = progress
                
                if self.db:
                    await self._update_task_status(self.active_tasks[task_id])
                    
        except Exception as e:
            self.logger.error(f"Error handling task update: {e}")
    
    async def _handle_task_completion(self, message: AgentMessage):
        """Handle task completion from agent."""
        try:
            task_id = message.payload.get("task_id")
            result = message.payload.get("result", {})
            agent_id = message.sender_id
            
            if message.payload.get("success", True):
                await self.complete_task(task_id, result, agent_id)
            else:
                error = message.payload.get("error", "Unknown error")
                await self.fail_task(task_id, error, agent_id)
                
        except Exception as e:
            self.logger.error(f"Error handling task completion: {e}")
    
    # Database persistence methods (placeholder implementations)
    async def _persist_task(self, task: Task):
        """Persist task to database."""
        # TODO: Implement database persistence using the agent_tasks table
        pass
    
    async def _update_task_status(self, task: Task):
        """Update task status in database."""
        # TODO: Implement database update
        pass
    
    async def _load_pending_tasks(self):
        """Load pending tasks from database on startup."""
        # TODO: Implement database loading
        pass
    
    async def _get_task_from_db(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task from database."""
        # TODO: Implement database query
        return None


# Factory functions for task creation
def create_parse_task(file_path: str, file_type: str, priority: int = TaskPriority.NORMAL.value) -> Task:
    """Create a file parsing task."""
    return Task(
        task_type="parse_kml_file" if file_type.lower() == "kml" else "parse_file",
        priority=priority,
        parameters={
            "file_path": file_path,
            "file_type": file_type
        },
        metadata={"category": "parsing"}
    )


def create_discovery_task(dataset_id: str, algorithm: str = "hdbscan", priority: int = TaskPriority.NORMAL.value) -> Task:
    """Create a pattern discovery task."""
    return Task(
        task_type="discover_patterns",
        priority=priority,
        parameters={
            "dataset_id": dataset_id,
            "algorithm": algorithm
        },
        metadata={"category": "discovery"}
    )


def create_validation_task(pattern_id: str, priority: int = TaskPriority.HIGH.value) -> Task:
    """Create a pattern validation task."""
    return Task(
        task_type="validate_pattern",
        priority=priority,
        parameters={
            "pattern_id": pattern_id
        },
        metadata={"category": "validation"}
    )