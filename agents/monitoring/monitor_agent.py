"""
A2A World Platform - Monitor Agent

Agent responsible for system health monitoring, performance tracking,
and alerting across the multi-agent system infrastructure.
"""

import asyncio
import logging
import psutil
import aiohttp
import json
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import uuid

from agents.core.base_agent import BaseAgent
from agents.core.config import MonitorAgentConfig
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from agents.core.registry import AgentServiceInfo


class SystemAlert:
    """Represents a system alert."""
    
    def __init__(
        self,
        alert_id: str,
        severity: str,
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.alert_id = alert_id
        self.severity = severity  # critical, warning, info
        self.component = component
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        self.acknowledged = False
        self.resolved = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "component": self.component,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


class HealthCheck:
    """Represents a health check for a system component."""
    
    def __init__(
        self,
        name: str,
        check_type: str,
        endpoint: Optional[str] = None,
        timeout: int = 10
    ):
        self.name = name
        self.check_type = check_type  # http, tcp, agent, system
        self.endpoint = endpoint
        self.timeout = timeout
        self.last_check = None
        self.status = "unknown"
        self.response_time = 0.0
        self.error_message = None
        self.consecutive_failures = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "check_type": self.check_type,
            "endpoint": self.endpoint,
            "status": self.status,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "response_time": self.response_time,
            "error_message": self.error_message,
            "consecutive_failures": self.consecutive_failures
        }


class MonitorAgent(BaseAgent):
    """
    Agent that monitors system health and performance.
    
    Responsibilities:
    - Monitor agent health and status
    - Track system resource usage
    - Monitor infrastructure components (NATS, Consul, DB)
    - Generate and manage alerts
    - Collect and aggregate metrics
    - Provide health dashboards and reports
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[MonitorAgentConfig] = None,
        config_file: Optional[str] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="monitoring",
            config=config or MonitorAgentConfig(),
            config_file=config_file
        )
        
        # Monitoring state
        self.agents_status: Dict[str, Dict[str, Any]] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.infrastructure_health: Dict[str, HealthCheck] = {}
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Metrics history (last 24 hours worth of data points)
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))
        
        # Alert cooldowns to prevent spam
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Health checks configuration
        self.health_checks = self._initialize_health_checks()
        
        # Monitoring statistics
        self.alerts_generated = 0
        self.health_checks_performed = 0
        self.agents_monitored = 0
        
        self.logger.info(f"MonitorAgent {self.agent_id} initialized with {len(self.health_checks)} health checks")
    
    async def process(self) -> None:
        """
        Main monitoring loop - perform all monitoring activities.
        """
        try:
            # Perform system monitoring activities
            await self._monitor_system_resources()
            await self._monitor_agents()
            await self._monitor_infrastructure()
            await self._process_alerts()
            
            # Update metrics history
            await self._update_metrics_history()
            
        except Exception as e:
            self.logger.error(f"Error in monitoring process: {e}")
    
    async def agent_initialize(self) -> None:
        """
        Monitor agent specific initialization.
        """
        try:
            # Initialize monitoring components
            await self._setup_monitoring()
            
            # Start background monitoring tasks
            await self._start_monitoring_tasks()
            
            self.logger.info("MonitorAgent initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MonitorAgent: {e}")
            raise
    
    async def setup_subscriptions(self) -> None:
        """
        Setup monitoring-specific message subscriptions.
        """
        if not self.messaging:
            return
        
        # Subscribe to heartbeats from all agents
        heartbeat_sub_id = await self.nats_client.subscribe(
            "agents.heartbeat",
            self._handle_heartbeat,
            queue_group="monitor-heartbeats"
        )
        self.subscription_ids.append(heartbeat_sub_id)
        
        # Subscribe to alert notifications
        alert_sub_id = await self.nats_client.subscribe(
            "agents.monitoring.alert",
            self._handle_alert_notification,
            queue_group="monitor-alerts"
        )
        self.subscription_ids.append(alert_sub_id)
        
        # Subscribe to monitoring queries
        query_sub_id = await self.nats_client.subscribe(
            "agents.monitoring.query",
            self._handle_monitoring_query,
            queue_group="monitor-queries"
        )
        self.subscription_ids.append(query_sub_id)
    
    async def handle_task(self, task: Task) -> None:
        """
        Handle monitoring task processing.
        """
        self.logger.info(f"Processing monitoring task {task.task_id}: {task.task_type}")
        
        try:
            task_id = task.task_id
            self.current_tasks.add(task_id)
            
            result = None
            
            if task.task_type == "health_check":
                result = await self._health_check_task(task)
            elif task.task_type == "system_report":
                result = await self._system_report_task(task)
            elif task.task_type == "alert_management":
                result = await self._alert_management_task(task)
            elif task.task_type == "metrics_collection":
                result = await self._metrics_collection_task(task)
            else:
                raise ValueError(f"Unknown monitoring task type: {task.task_type}")
            
            # Report success
            if self.task_queue:
                await self.task_queue.complete_task(task_id, result, self.agent_id)
            
            self.processed_tasks += 1
            self.logger.info(f"Completed monitoring task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing monitoring task {task.task_id}: {e}")
            
            if self.task_queue:
                await self.task_queue.fail_task(task.task_id, str(e), self.agent_id)
            
            self.failed_tasks += 1
        
        finally:
            self.current_tasks.discard(task.task_id)
    
    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect monitoring-specific metrics.
        """
        return {
            "alerts_generated": self.alerts_generated,
            "active_alerts": len(self.active_alerts),
            "agents_monitored": len(self.agents_status),
            "health_checks_performed": self.health_checks_performed,
            "infrastructure_components": len(self.infrastructure_health),
            "healthy_components": len([hc for hc in self.infrastructure_health.values() if hc.status == "healthy"]),
            "system_cpu_percent": self.system_metrics.get("cpu_percent", 0),
            "system_memory_percent": self.system_metrics.get("memory_percent", 0),
            "system_disk_percent": self.system_metrics.get("disk_percent", 0)
        }
    
    async def check_health(self) -> Optional[List[str]]:
        """
        Perform monitor agent specific health checks.
        """
        issues = []
        
        # Check if we're receiving heartbeats from agents
        active_agents = len([a for a in self.agents_status.values() 
                           if a.get("status") == "running"])
        
        if active_agents == 0:
            issues.append("No active agents detected")
        
        # Check system resource usage
        if self.system_metrics.get("cpu_percent", 0) > 95:
            issues.append("Critical CPU usage")
        
        if self.system_metrics.get("memory_percent", 0) > 95:
            issues.append("Critical memory usage")
        
        # Check infrastructure health
        unhealthy_components = [name for name, hc in self.infrastructure_health.items() 
                              if hc.status == "unhealthy"]
        
        if unhealthy_components:
            issues.append(f"Unhealthy infrastructure: {', '.join(unhealthy_components)}")
        
        return issues
    
    def _get_capabilities(self) -> List[str]:
        """
        Get monitoring agent capabilities.
        """
        return [
            "monitoring",
            "health_check",
            "system_monitoring",
            "alert_management",
            "metrics_collection",
            "performance_monitoring",
            "infrastructure_monitoring",
            "agent_monitoring"
        ]
    
    def _initialize_health_checks(self) -> Dict[str, HealthCheck]:
        """
        Initialize health checks for system components.
        """
        checks = {}
        
        # NATS health check
        checks["nats"] = HealthCheck(
            name="NATS Message Broker",
            check_type="http",
            endpoint="http://localhost:8222/",
            timeout=self.config.health_check_timeout
        )
        
        # Consul health check
        checks["consul"] = HealthCheck(
            name="Consul Service Discovery",
            check_type="http", 
            endpoint="http://localhost:8500/v1/status/leader",
            timeout=self.config.health_check_timeout
        )
        
        # Database health check (if available)
        if hasattr(self.config, 'database_url'):
            checks["database"] = HealthCheck(
                name="PostgreSQL Database",
                check_type="tcp",
                endpoint="localhost:5432",
                timeout=self.config.health_check_timeout
            )
        
        # Redis health check
        checks["redis"] = HealthCheck(
            name="Redis Cache",
            check_type="tcp",
            endpoint="localhost:6379",
            timeout=self.config.health_check_timeout
        )
        
        return checks
    
    async def _setup_monitoring(self) -> None:
        """
        Setup monitoring infrastructure.
        """
        # Initialize system metrics
        await self._collect_system_metrics()
        
        # Perform initial health checks
        await self._perform_health_checks()
        
        self.logger.info("Monitoring infrastructure setup complete")
    
    async def _start_monitoring_tasks(self) -> None:
        """
        Start background monitoring tasks.
        """
        # System monitoring task
        system_task = asyncio.create_task(self._system_monitoring_loop())
        self.background_tasks.append(system_task)
        
        # Agent monitoring task
        agent_task = asyncio.create_task(self._agent_monitoring_loop())
        self.background_tasks.append(agent_task)
        
        # Infrastructure monitoring task
        infra_task = asyncio.create_task(self._infrastructure_monitoring_loop())
        self.background_tasks.append(infra_task)
        
        # Alert processing task
        alert_task = asyncio.create_task(self._alert_processing_loop())
        self.background_tasks.append(alert_task)
    
    async def _system_monitoring_loop(self) -> None:
        """
        Background task for system resource monitoring.
        """
        while not self.shutdown_event.is_set():
            try:
                await self._collect_system_metrics()
                await self._check_system_thresholds()
                await asyncio.sleep(self.config.system_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _agent_monitoring_loop(self) -> None:
        """
        Background task for agent health monitoring.
        """
        while not self.shutdown_event.is_set():
            try:
                await self._check_agent_health()
                await asyncio.sleep(self.config.agent_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Agent monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _infrastructure_monitoring_loop(self) -> None:
        """
        Background task for infrastructure monitoring.
        """
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Check infrastructure every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Infrastructure monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_processing_loop(self) -> None:
        """
        Background task for alert processing and cleanup.
        """
        while not self.shutdown_event.is_set():
            try:
                await self._process_alert_queue()
                await self._cleanup_resolved_alerts()
                await asyncio.sleep(30)  # Process alerts every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert processing loop error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_system_resources(self) -> None:
        """
        Monitor system resource usage.
        """
        await self._collect_system_metrics()
        await self._check_system_thresholds()
    
    async def _monitor_agents(self) -> None:
        """
        Monitor agent health and status.
        """
        if self.registry:
            try:
                # Get all registered agents
                agents = await self.registry.discover_agents()
                self.agents_monitored = len(agents)
                
                for agent in agents:
                    agent_id = agent.agent_id
                    
                    # Update agent status
                    self.agents_status[agent_id] = {
                        "agent_id": agent_id,
                        "agent_type": agent.agent_type,
                        "status": agent.status,
                        "last_seen": datetime.utcnow().isoformat(),
                        "capabilities": agent.capabilities,
                        "address": agent.address,
                        "metadata": agent.metadata
                    }
                
            except Exception as e:
                self.logger.error(f"Error monitoring agents: {e}")
    
    async def _monitor_infrastructure(self) -> None:
        """
        Monitor infrastructure components.
        """
        await self._perform_health_checks()
    
    async def _collect_system_metrics(self) -> None:
        """
        Collect system-wide metrics.
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            self.system_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk_percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "process_count": process_count,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _perform_health_checks(self) -> None:
        """
        Perform health checks on infrastructure components.
        """
        for name, health_check in self.infrastructure_health.items():
            try:
                await self._execute_health_check(health_check)
                self.health_checks_performed += 1
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                health_check.status = "unhealthy"
                health_check.error_message = str(e)
                health_check.consecutive_failures += 1
    
    async def _execute_health_check(self, health_check: HealthCheck) -> None:
        """
        Execute a single health check.
        """
        start_time = time.time()
        
        try:
            if health_check.check_type == "http":
                await self._http_health_check(health_check)
            elif health_check.check_type == "tcp":
                await self._tcp_health_check(health_check)
            else:
                raise ValueError(f"Unknown health check type: {health_check.check_type}")
            
            # Success
            health_check.status = "healthy"
            health_check.error_message = None
            health_check.consecutive_failures = 0
            
        except Exception as e:
            health_check.status = "unhealthy"
            health_check.error_message = str(e)
            health_check.consecutive_failures += 1
            
            # Generate alert if consecutive failures exceed threshold
            if health_check.consecutive_failures >= 3:
                await self._generate_alert(
                    severity="critical",
                    component=health_check.name,
                    message=f"Health check failing: {health_check.error_message}",
                    details=health_check.to_dict()
                )
        
        finally:
            health_check.response_time = time.time() - start_time
            health_check.last_check = datetime.utcnow()
    
    async def _http_health_check(self, health_check: HealthCheck) -> None:
        """
        Perform HTTP health check.
        """
        timeout = aiohttp.ClientTimeout(total=health_check.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(health_check.endpoint) as response:
                if response.status >= 400:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
    
    async def _tcp_health_check(self, health_check: HealthCheck) -> None:
        """
        Perform TCP health check.
        """
        if not health_check.endpoint:
            raise ValueError("TCP endpoint required")
        
        host, port = health_check.endpoint.split(':')
        port = int(port)
        
        # Simple TCP connection test
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=health_check.timeout
            )
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            raise Exception(f"TCP connection failed: {e}")
    
    async def _check_system_thresholds(self) -> None:
        """
        Check system metrics against thresholds and generate alerts.
        """
        # CPU threshold
        cpu_percent = self.system_metrics.get("cpu_percent", 0)
        if cpu_percent > self.config.cpu_threshold:
            await self._generate_alert(
                severity="warning" if cpu_percent < 95 else "critical",
                component="System CPU",
                message=f"High CPU usage: {cpu_percent:.1f}%",
                details={"cpu_percent": cpu_percent, "threshold": self.config.cpu_threshold}
            )
        
        # Memory threshold
        memory_percent = self.system_metrics.get("memory_percent", 0)
        if memory_percent > self.config.memory_threshold:
            await self._generate_alert(
                severity="warning" if memory_percent < 95 else "critical",
                component="System Memory",
                message=f"High memory usage: {memory_percent:.1f}%",
                details={"memory_percent": memory_percent, "threshold": self.config.memory_threshold}
            )
        
        # Disk threshold
        disk_percent = self.system_metrics.get("disk_percent", 0)
        if disk_percent > self.config.disk_threshold:
            await self._generate_alert(
                severity="warning" if disk_percent < 95 else "critical",
                component="System Disk",
                message=f"High disk usage: {disk_percent:.1f}%",
                details={"disk_percent": disk_percent, "threshold": self.config.disk_threshold}
            )
    
    async def _check_agent_health(self) -> None:
        """
        Check agent health and detect stale agents.
        """
        current_time = datetime.utcnow()
        stale_threshold = timedelta(minutes=5)  # Agents are stale after 5 minutes
        
        for agent_id, agent_data in list(self.agents_status.items()):
            try:
                last_seen = datetime.fromisoformat(agent_data.get("last_seen", ""))
                
                if current_time - last_seen > stale_threshold:
                    # Agent appears to be offline or unresponsive
                    await self._generate_alert(
                        severity="warning",
                        component=f"Agent {agent_id}",
                        message=f"Agent appears offline (last seen: {last_seen})",
                        details={
                            "agent_id": agent_id,
                            "agent_type": agent_data.get("agent_type"),
                            "last_seen": last_seen.isoformat()
                        }
                    )
                    
                    # Mark as offline
                    agent_data["status"] = "offline"
            
            except (ValueError, TypeError):
                # Invalid timestamp, remove from tracking
                self.logger.warning(f"Invalid timestamp for agent {agent_id}")
                del self.agents_status[agent_id]
    
    async def _generate_alert(
        self, 
        severity: str, 
        component: str, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Generate and process a system alert.
        """
        # Check alert cooldown
        cooldown_key = f"{component}:{message}"
        if cooldown_key in self.alert_cooldowns:
            last_alert = self.alert_cooldowns[cooldown_key]
            if datetime.utcnow() - last_alert < timedelta(seconds=self.config.alert_cooldown):
                return  # Skip duplicate alert within cooldown period
        
        # Create alert
        alert = SystemAlert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            component=component,
            message=message,
            details=details
        )
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert.to_dict())
        self.alert_cooldowns[cooldown_key] = alert.timestamp
        self.alerts_generated += 1
        
        # Log alert
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "critical": logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self.logger.log(log_level, f"ALERT [{severity.upper()}] {component}: {message}")
        
        # Publish alert if enabled
        if self.config.enable_alerts and self.messaging:
            alert_message = AgentMessage.create(
                sender_id=self.agent_id,
                message_type="system_alert",
                payload=alert.to_dict()
            )
            
            await self.nats_client.publish("agents.monitoring.alerts", alert_message)
    
    async def _process_alerts(self) -> None:
        """
        Process and manage active alerts.
        """
        await self._process_alert_queue()
        await self._cleanup_resolved_alerts()
    
    async def _process_alert_queue(self) -> None:
        """
        Process queued alerts and notifications.
        """
        # This could integrate with external alerting systems
        # For now, we just log critical alerts
        critical_alerts = [alert for alert in self.active_alerts.values() 
                          if alert.severity == "critical" and not alert.acknowledged]
        
        if critical_alerts:
            self.logger.critical(f"Active critical alerts: {len(critical_alerts)}")
    
    async def _cleanup_resolved_alerts(self) -> None:
        """
        Clean up resolved or old alerts.
        """
        current_time = datetime.utcnow()
        alert_ttl = timedelta(hours=24)  # Keep alerts for 24 hours
        
        alerts_to_remove = []
        
        for alert_id, alert in self.active_alerts.items():
            # Auto-resolve old alerts
            if current_time - alert.timestamp > alert_ttl:
                alert.resolved = True
                alerts_to_remove.append(alert_id)
        
        # Remove resolved alerts
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
    
    async def _update_metrics_history(self) -> None:
        """
        Update metrics history for trending analysis.
        """
        timestamp = datetime.utcnow()
        
        # Store key metrics
        self.metrics_history["cpu_percent"].append({
            "timestamp": timestamp.isoformat(),
            "value": self.system_metrics.get("cpu_percent", 0)
        })
        
        self.metrics_history["memory_percent"].append({
            "timestamp": timestamp.isoformat(), 
            "value": self.system_metrics.get("memory_percent", 0)
        })
        
        self.metrics_history["active_agents"].append({
            "timestamp": timestamp.isoformat(),
            "value": len([a for a in self.agents_status.values() if a.get("status") == "running"])
        })
        
        self.metrics_history["active_alerts"].append({
            "timestamp": timestamp.isoformat(),
            "value": len(self.active_alerts)
        })
    
    # Message handlers
    
    async def _handle_heartbeat(self, message: AgentMessage) -> None:
        """
        Handle heartbeat messages from agents.
        """
        try:
            payload = message.payload
            agent_id = payload.get("agent_id")
            
            if agent_id:
                # Update agent status
                self.agents_status[agent_id] = {
                    "agent_id": agent_id,
                    "agent_type": payload.get("agent_type", "unknown"),
                    "status": payload.get("status", "unknown"),
                    "health_status": payload.get("health_status", "unknown"),
                    "last_seen": datetime.utcnow().isoformat(),
                    "processed_tasks": payload.get("processed_tasks", 0),
                    "metrics": payload.get("metrics", {}),
                    "uptime_seconds": payload.get("uptime_seconds", 0)
                }
                
                self.logger.debug(f"Received heartbeat from agent {agent_id}")
        
        except Exception as e:
            self.logger.error(f"Error handling heartbeat: {e}")
    
    async def _handle_alert_notification(self, message: AgentMessage) -> None:
        """
        Handle alert notifications from other agents.
        """
        try:
            alert_data = message.payload
            
            # Create alert from external notification
            await self._generate_alert(
                severity=alert_data.get("severity", "info"),
                component=alert_data.get("component", f"Agent {message.sender_id}"),
                message=alert_data.get("message", "External alert"),
                details=alert_data.get("details", {})
            )
            
        except Exception as e:
            self.logger.error(f"Error handling alert notification: {e}")
    
    async def _handle_monitoring_query(self, message: AgentMessage) -> None:
        """
        Handle monitoring data queries.
        """
        try:
            query_type = message.payload.get("query_type")
            
            if query_type == "system_status":
                response_data = await self._get_system_status()
            elif query_type == "agent_status":
                response_data = dict(self.agents_status)
            elif query_type == "alerts":
                response_data = [alert.to_dict() for alert in self.active_alerts.values()]
            elif query_type == "metrics":
                response_data = self.system_metrics
            else:
                response_data = {"error": f"Unknown query type: {query_type}"}
            
            # Send response
            response = AgentMessage.create(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="monitoring_response",
                payload=response_data,
                correlation_id=message.correlation_id
            )
            
            if message.reply_to:
                await self.nats_client.publish(message.reply_to, response)
        
        except Exception as e:
            self.logger.error(f"Error handling monitoring query: {e}")
    
    # Task handlers
    
    async def _health_check_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle health check task.
        """
        component = task.parameters.get("component")
        
        if component and component in self.infrastructure_health:
            health_check = self.infrastructure_health[component]
            await self._execute_health_check(health_check)
            return health_check.to_dict()
        else:
            # Perform all health checks
            await self._perform_health_checks()
            return {
                component: hc.to_dict() 
                for component, hc in self.infrastructure_health.items()
            }
    
    async def _system_report_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle system report generation task.
        """
        return await self._get_system_status()
    
    async def _alert_management_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle alert management task.
        """
        action = task.parameters.get("action", "list")
        alert_id = task.parameters.get("alert_id")
        
        if action == "list":
            return [alert.to_dict() for alert in self.active_alerts.values()]
        elif action == "acknowledge" and alert_id:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                return {"status": "acknowledged", "alert_id": alert_id}
            else:
                return {"error": "Alert not found", "alert_id": alert_id}
        elif action == "resolve" and alert_id:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                del self.active_alerts[alert_id]
                return {"status": "resolved", "alert_id": alert_id}
            else:
                return {"error": "Alert not found", "alert_id": alert_id}
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _metrics_collection_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle metrics collection task.
        """
        await self._collect_system_metrics()
        
        metric_type = task.parameters.get("metric_type", "all")
        
        if metric_type == "system":
            return self.system_metrics
        elif metric_type == "agents":
            return dict(self.agents_status)
        elif metric_type == "alerts":
            return [alert.to_dict() for alert in self.active_alerts.values()]
        else:
            return {
                "system": self.system_metrics,
                "agents": dict(self.agents_status),
                "alerts": [alert.to_dict() for alert in self.active_alerts.values()],
                "infrastructure": {name: hc.to_dict() for name, hc in self.infrastructure_health.items()}
            }
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status report.
        """
        # Calculate summary statistics
        healthy_agents = len([a for a in self.agents_status.values() if a.get("status") == "running"])
        total_agents = len(self.agents_status)
        
        healthy_infrastructure = len([hc for hc in self.infrastructure_health.values() if hc.status == "healthy"])
        total_infrastructure = len(self.infrastructure_health)
        
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == "critical"])
        
        # Overall system health
        if critical_alerts > 0:
            overall_health = "critical"
        elif len(self.active_alerts) > 0 or healthy_infrastructure < total_infrastructure:
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": overall_health,
            "system_metrics": self.system_metrics,
            "agents": {
                "total": total_agents,
                "healthy": healthy_agents,
                "unhealthy": total_agents - healthy_agents,
                "details": dict(self.agents_status)
            },
            "infrastructure": {
                "total": total_infrastructure,
                "healthy": healthy_infrastructure,
                "unhealthy": total_infrastructure - healthy_infrastructure,
                "details": {name: hc.to_dict() for name, hc in self.infrastructure_health.items()}
            },
            "alerts": {
                "total": len(self.active_alerts),
                "critical": critical_alerts,
                "warning": len([a for a in self.active_alerts.values() if a.severity == "warning"]),
                "info": len([a for a in self.active_alerts.values() if a.severity == "info"]),
                "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()]
            },
            "monitoring_stats": {
                "alerts_generated": self.alerts_generated,
                "health_checks_performed": self.health_checks_performed,
                "agents_monitored": self.agents_monitored,
                "uptime_seconds": self.metrics.get("uptime_seconds", 0)
            }
        }


# Main entry point for running the agent
async def main():
    """
    Main entry point for running the MonitorAgent.
    """
    import signal
    import sys
    
    # Create and configure agent
    agent = MonitorAgent()
    
    # Setup graceful shutdown
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down...")
        asyncio.create_task(agent.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the agent
        await agent.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Agent failed: {e}")
        sys.exit(1)
    
    print("MonitorAgent stopped")


if __name__ == "__main__":
    asyncio.run(main())