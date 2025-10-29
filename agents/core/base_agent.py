"""
A2A World Platform - Base Agent Class

Abstract base class for all autonomous agents in the A2A World system.
Provides common functionality for NATS messaging, Consul service discovery,
task processing, and lifecycle management.
"""

import asyncio
import logging
import psutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import uuid
import json
import os
import signal

from agents.core.messaging import NATSClient, AgentMessaging, AgentMessage, get_nats_client
from agents.core.registry import ConsulRegistry, get_consul_registry
from agents.core.task_queue import TaskQueue, Task
from agents.core.config import AgentConfig, get_config_manager, load_agent_config
from agents.core.health_server import HealthCheckServer, create_health_server


class BaseAgent(ABC):
    """
    Abstract base class for all autonomous agents.
    
    Provides comprehensive functionality including:
    - NATS messaging integration with pub/sub patterns
    - Consul service registration and discovery
    - Task queue integration for distributed processing
    - Health monitoring and metrics collection
    - Configuration management with dynamic updates
    - Graceful shutdown handling with cleanup
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "base",
        config: Optional[AgentConfig] = None,
        config_file: Optional[str] = None
    ):
        # Load configuration
        if config:
            self.config = config
        else:
            self.config = load_agent_config(agent_type, config_file)
        
        # Set agent identification
        self.agent_id = agent_id or self.config.agent_id or f"{agent_type}-{uuid.uuid4().hex[:8]}"
        self.agent_type = agent_type
        self.config.agent_id = self.agent_id  # Update config with actual agent ID
        
        # Agent state
        self.status = "initializing"
        self.health_status = "unknown"
        self.last_heartbeat = None
        self.start_time = None
        self.processed_tasks = 0
        self.failed_tasks = 0
        self.current_tasks: Set[str] = set()
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self._setup_logging()
        
        # Core components (to be initialized)
        self.nats_client: Optional[NATSClient] = None
        self.messaging: Optional[AgentMessaging] = None
        self.registry: Optional[ConsulRegistry] = None
        self.task_queue: Optional[TaskQueue] = None
        self.health_server: Optional[HealthCheckServer] = None
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.subscription_ids: List[str] = []
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self._shutdown_handlers = []
        
        # Metrics and health data
        self.metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "tasks_per_minute": 0.0,
            "error_rate": 0.0,
            "uptime_seconds": 0
        }
        
        # Capabilities that this agent provides
        self.capabilities: List[str] = self._get_capabilities()
        
        self.logger.info(f"Created agent {self.agent_id} of type {self.agent_type}")
    
    async def initialize(self) -> bool:
        """
        Initialize the agent including connections and service registration.
        """
        try:
            self.logger.info(f"Initializing agent {self.agent_id}")
            
            # Initialize NATS connection
            self.nats_client = await get_nats_client(
                url=self.config.nats_url,
                name=self.agent_id
            )
            
            if not self.nats_client.is_connected:
                raise ConnectionError("Failed to connect to NATS server")
            
            # Initialize messaging
            self.messaging = AgentMessaging(self.nats_client, self.agent_id)
            
            # Initialize Consul registry
            self.registry = await get_consul_registry(
                host=self.config.consul_host,
                port=self.config.consul_port,
                token=self.config.consul_token
            )
            
            # Initialize task queue
            self.task_queue = TaskQueue(
                nats_client=self.nats_client,
                registry=self.registry
            )
            
            # Register with Consul
            health_check_url = None
            if self.config.health_check_port > 0:
                health_check_url = f"http://localhost:{self.config.health_check_port}{self.config.health_check_path}"
            
            success = await self.registry.register_agent(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                capabilities=self.capabilities,
                address="127.0.0.1",
                port=self.config.health_check_port,
                health_check_url=health_check_url,
                metadata={
                    "config": self.config.to_dict(),
                    "pid": os.getpid(),
                    "version": "1.0.0"
                }
            )
            
            if not success:
                raise RuntimeError("Failed to register agent with Consul")
            
            # Setup message subscriptions
            await self._setup_subscriptions()
            
            # Start health check server if port configured
            if self.config.health_check_port > 0:
                try:
                    self.health_server = await create_health_server(self)
                    self.logger.info(f"Health check server started on port {self.health_server.actual_port}")
                except Exception as e:
                    self.logger.warning(f"Failed to start health server: {e}")
            
            # Perform agent-specific initialization
            await self.agent_initialize()
            
            self.status = "initialized"
            self.health_status = "healthy"
            self.start_time = datetime.utcnow()
            
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.status = "error"
            self.health_status = "critical"
            return False
    
    async def start(self) -> None:
        """
        Start the agent and begin processing.
        """
        if not await self.initialize():
            return
        
        self.logger.info(f"Starting agent {self.agent_id}")
        self.status = "running"
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Start background tasks
        await self._start_background_tasks()
        
        try:
            # Main agent loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Error in agent main loop: {e}")
            self.status = "error"
            self.health_status = "critical"
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the agent.
        """
        if self.status == "stopped":
            return
        
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self.status = "shutting_down"
        
        try:
            # Run custom shutdown handlers
            for handler in self._shutdown_handlers:
                try:
                    await handler()
                except Exception as e:
                    self.logger.error(f"Error in shutdown handler: {e}")
            
            # Stop health check server
            if self.health_server:
                try:
                    await self.health_server.stop()
                    self.logger.info("Health check server stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping health server: {e}")
            
            # Perform agent-specific cleanup
            await self.agent_cleanup()
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Unsubscribe from NATS topics
            if self.nats_client:
                for sub_id in self.subscription_ids:
                    try:
                        await self.nats_client.unsubscribe(sub_id)
                    except Exception as e:
                        self.logger.error(f"Error unsubscribing {sub_id}: {e}")
            
            # Deregister from Consul
            if self.registry:
                await self.registry.deregister_agent(self.agent_id)
            
            # Close NATS connection
            if self.nats_client:
                await self.nats_client.disconnect()
            
            self.status = "stopped"
            self.health_status = "down"
            self.logger.info(f"Agent {self.agent_id} shut down complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _main_loop(self) -> None:
        """
        Main agent processing loop.
        """
        while not self.shutdown_event.is_set():
            try:
                # Process agent-specific work
                await self.process()
                
                # Small delay between processing cycles
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                await asyncio.sleep(5)
    
    async def _start_background_tasks(self) -> None:
        """
        Start background maintenance tasks.
        """
        # Heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.background_tasks.append(heartbeat_task)
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_loop())
        self.background_tasks.append(metrics_task)
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitor_loop())
        self.background_tasks.append(health_task)
        
        # Task queue if enabled
        if self.task_queue:
            await self.task_queue.start()
    
    async def _heartbeat_loop(self) -> None:
        """
        Send periodic heartbeat signals.
        """
        while not self.shutdown_event.is_set():
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _send_heartbeat(self) -> None:
        """
        Send heartbeat message via NATS and update Consul health.
        """
        self.last_heartbeat = datetime.utcnow()
        
        # Calculate uptime
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (self.last_heartbeat - self.start_time).total_seconds()
            self.metrics["uptime_seconds"] = uptime_seconds
        
        heartbeat_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "health_status": self.health_status,
            "timestamp": self.last_heartbeat.isoformat(),
            "processed_tasks": self.processed_tasks,
            "failed_tasks": self.failed_tasks,
            "current_tasks": len(self.current_tasks),
            "metrics": self.metrics,
            "uptime_seconds": uptime_seconds
        }
        
        # Send heartbeat via NATS
        if self.messaging:
            await self.messaging.send_heartbeat(self.status, heartbeat_data)
        
        # Update Consul health check
        if self.registry:
            await self.registry.update_agent_status(
                self.agent_id,
                "healthy" if self.health_status == "healthy" else "unhealthy",
                heartbeat_data
            )
        
        self.logger.debug(f"Heartbeat sent: {self.status}")
    
    async def _metrics_loop(self) -> None:
        """
        Collect and update system metrics.
        """
        while not self.shutdown_event.is_set():
            try:
                await self._collect_metrics()
                await asyncio.sleep(60)  # Collect metrics every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_metrics(self) -> None:
        """
        Collect system and agent metrics.
        """
        try:
            # System metrics
            process = psutil.Process()
            self.metrics.update({
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_rss_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
            })
            
            # Calculate task processing rate
            if self.start_time:
                uptime_minutes = (datetime.utcnow() - self.start_time).total_seconds() / 60
                if uptime_minutes > 0:
                    self.metrics["tasks_per_minute"] = self.processed_tasks / uptime_minutes
            
            # Calculate error rate
            total_tasks = self.processed_tasks + self.failed_tasks
            if total_tasks > 0:
                self.metrics["error_rate"] = self.failed_tasks / total_tasks
            
            # Agent-specific metrics
            custom_metrics = await self.collect_metrics()
            if custom_metrics:
                self.metrics.update(custom_metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    async def _health_monitor_loop(self) -> None:
        """
        Monitor agent health and update status.
        """
        while not self.shutdown_event.is_set():
            try:
                await self._check_health()
                await asyncio.sleep(30)  # Health check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_health(self) -> None:
        """
        Perform health checks and update status.
        """
        try:
            health_issues = []
            
            # Check resource usage
            if self.metrics.get("cpu_percent", 0) > 90:
                health_issues.append("High CPU usage")
            
            if self.metrics.get("memory_percent", 0) > 95:
                health_issues.append("High memory usage")
            
            # Check error rate
            if self.metrics.get("error_rate", 0) > 0.1:  # 10% error rate
                health_issues.append("High error rate")
            
            # Check NATS connection
            if not self.nats_client or not self.nats_client.is_connected:
                health_issues.append("NATS connection lost")
            
            # Agent-specific health checks
            custom_health = await self.check_health()
            if custom_health:
                health_issues.extend(custom_health)
            
            # Update health status
            if not health_issues:
                self.health_status = "healthy"
            elif len(health_issues) == 1 and "High" in health_issues[0]:
                self.health_status = "degraded"
            else:
                self.health_status = "critical"
            
            if health_issues:
                self.logger.warning(f"Health issues detected: {', '.join(health_issues)}")
        
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            self.health_status = "critical"
    
    async def _setup_subscriptions(self) -> None:
        """
        Setup NATS message subscriptions.
        """
        if not self.messaging:
            return
        
        # Subscribe to task assignments
        task_sub_id = await self.messaging.subscribe_to_tasks(self._handle_task_message)
        self.subscription_ids.append(task_sub_id)
        
        # Subscribe to broadcasts
        broadcast_sub_id = await self.messaging.subscribe_to_broadcasts(self._handle_broadcast_message)
        self.subscription_ids.append(broadcast_sub_id)
        
        # Agent-specific subscriptions
        await self.setup_subscriptions()
    
    async def _handle_task_message(self, message: AgentMessage) -> None:
        """
        Handle incoming task assignment messages.
        """
        try:
            if message.message_type == "task_assigned":
                task_data = message.payload
                task = Task.from_dict(task_data)
                
                # Process the task
                await self.handle_task(task)
                
        except Exception as e:
            self.logger.error(f"Error handling task message: {e}")
    
    async def _handle_broadcast_message(self, message: AgentMessage) -> None:
        """
        Handle broadcast messages.
        """
        try:
            if message.message_type == "shutdown_signal":
                self.logger.info("Received shutdown signal")
                self.shutdown_event.set()
            elif message.message_type == "config_update":
                await self._handle_config_update(message.payload)
            else:
                await self.handle_broadcast(message)
                
        except Exception as e:
            self.logger.error(f"Error handling broadcast message: {e}")
    
    async def _handle_config_update(self, config_data: Dict[str, Any]) -> None:
        """
        Handle configuration updates.
        """
        try:
            # Update configuration
            config_manager = get_config_manager(self.registry)
            old_config = self.config
            
            # Create new config instance
            config_class = type(self.config)
            new_config = config_class(**config_data)
            self.config = new_config
            
            # Apply configuration changes
            await self.apply_config_update(old_config, new_config)
            
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
    
    def _setup_logging(self) -> None:
        """
        Setup logging configuration.
        """
        # Set log level
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create formatter
        formatter = logging.Formatter(self.config.log_format)
        
        # Create console handler if not already exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_signal_handlers(self) -> None:
        """
        Setup signal handlers for graceful shutdown.
        """
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}")
            asyncio.create_task(self._trigger_shutdown())
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def _trigger_shutdown(self) -> None:
        """
        Trigger graceful shutdown.
        """
        self.shutdown_event.set()
    
    def add_shutdown_handler(self, handler) -> None:
        """
        Add a custom shutdown handler.
        """
        self._shutdown_handlers.append(handler)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status information.
        """
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "health_status": self.health_status,
            "capabilities": self.capabilities,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": uptime_seconds,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "processed_tasks": self.processed_tasks,
            "failed_tasks": self.failed_tasks,
            "current_tasks": len(self.current_tasks),
            "metrics": self.metrics,
            "config": self.config.to_dict()
        }
    
    # Abstract methods for subclasses to implement
    
    @abstractmethod
    async def process(self) -> None:
        """
        Process a single iteration of agent work.
        Must be implemented by subclasses.
        """
        pass
    
    async def agent_initialize(self) -> None:
        """
        Agent-specific initialization logic.
        Override in subclasses if needed.
        """
        pass
    
    async def agent_cleanup(self) -> None:
        """
        Agent-specific cleanup logic.
        Override in subclasses if needed.
        """
        pass
    
    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect agent-specific metrics.
        Override in subclasses to provide custom metrics.
        """
        return None
    
    async def check_health(self) -> Optional[List[str]]:
        """
        Perform agent-specific health checks.
        Override in subclasses to provide custom health checks.
        Return list of health issues (empty list = healthy).
        """
        return None
    
    async def setup_subscriptions(self) -> None:
        """
        Setup agent-specific message subscriptions.
        Override in subclasses if needed.
        """
        pass
    
    async def handle_task(self, task: Task) -> None:
        """
        Handle assigned task processing.
        Override in subclasses to implement task handling.
        """
        self.logger.info(f"Received task {task.task_id}: {task.task_type}")
        
        # Default implementation - mark as completed
        if self.task_queue:
            await self.task_queue.complete_task(
                task.task_id,
                {"message": "Task completed by base implementation"},
                self.agent_id
            )
    
    async def handle_broadcast(self, message: AgentMessage) -> None:
        """
        Handle broadcast messages.
        Override in subclasses to handle custom broadcast messages.
        """
        pass
    
    async def apply_config_update(self, old_config: AgentConfig, new_config: AgentConfig) -> None:
        """
        Apply configuration updates.
        Override in subclasses to handle config changes.
        """
        pass
    
    def _get_capabilities(self) -> List[str]:
        """
        Get list of capabilities this agent provides.
        Override in subclasses to define specific capabilities.
        """
        return [self.agent_type, "base_agent", "health_check"]