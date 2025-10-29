"""
A2A World Platform - Consul Service Registry

Consul integration for agent service discovery, health checks, and configuration management.
Provides agent registration, capability matching, and dynamic scaling support.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import uuid

import consul
import consul.aio


class AgentServiceInfo:
    """Information about a registered agent service."""
    
    def __init__(
        self,
        service_id: str,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        address: str,
        port: int,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.service_id = service_id
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.address = address
        self.port = port
        self.status = status
        self.metadata = metadata or {}
        self.registered_at = datetime.utcnow()
        self.last_health_check = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "service_id": self.service_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "address": self.address,
            "port": self.port,
            "status": self.status,
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentServiceInfo':
        """Create from dictionary."""
        service = cls(
            service_id=data["service_id"],
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            capabilities=data["capabilities"],
            address=data["address"],
            port=data["port"],
            status=data.get("status", "active"),
            metadata=data.get("metadata", {})
        )
        
        if data.get("registered_at"):
            service.registered_at = datetime.fromisoformat(data["registered_at"])
        if data.get("last_health_check"):
            service.last_health_check = datetime.fromisoformat(data["last_health_check"])
        
        return service


class ConsulRegistry:
    """
    Consul-based service registry for agent management.
    Handles registration, discovery, health checks, and configuration.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8500,
        token: Optional[str] = None,
        datacenter: str = "dc1"
    ):
        self.host = host
        self.port = port
        self.token = token
        self.datacenter = datacenter
        self.consul = consul.Consul(host=host, port=port, token=token, dc=datacenter)
        self.aio_consul = consul.aio.Consul(host=host, port=port, token=token, dc=datacenter)
        self.logger = logging.getLogger("consul.registry")
        self.registered_services: Dict[str, AgentServiceInfo] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        address: str = "127.0.0.1",
        port: int = 0,
        health_check_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register an agent service with Consul."""
        try:
            service_id = f"a2a-agent-{agent_id}"
            
            # Prepare service registration
            service_config = {
                "Name": f"a2a-agent-{agent_type}",
                "ID": service_id,
                "Tags": [
                    f"agent_type:{agent_type}",
                    f"agent_id:{agent_id}",
                    "a2a-platform",
                    "agent-service"
                ] + [f"capability:{cap}" for cap in capabilities],
                "Address": address,
                "Port": port,
                "Meta": {
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "capabilities": json.dumps(capabilities),
                    "registered_at": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            }
            
            # Add health check if URL provided
            if health_check_url:
                service_config["Check"] = {
                    "HTTP": health_check_url,
                    "Interval": "30s",
                    "Timeout": "10s",
                    "DeregisterCriticalServiceAfter": "90s"
                }
            else:
                # Use TTL health check if no HTTP endpoint
                service_config["Check"] = {
                    "TTL": "60s",
                    "DeregisterCriticalServiceAfter": "120s"
                }
            
            # Register service
            success = await self.aio_consul.agent.service.register(**service_config)
            
            if success:
                # Store service info
                service_info = AgentServiceInfo(
                    service_id=service_id,
                    agent_id=agent_id,
                    agent_type=agent_type,
                    capabilities=capabilities,
                    address=address,
                    port=port,
                    metadata=metadata
                )
                self.registered_services[service_id] = service_info
                
                # Start health check maintenance if using TTL
                if not health_check_url:
                    await self._start_health_check_maintenance(service_id)
                
                self.logger.info(f"Registered agent {agent_id} as service {service_id}")
                return True
            else:
                self.logger.error(f"Failed to register agent {agent_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error registering agent {agent_id}: {e}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent service from Consul."""
        try:
            service_id = f"a2a-agent-{agent_id}"
            
            # Stop health check maintenance
            if service_id in self.health_check_tasks:
                self.health_check_tasks[service_id].cancel()
                del self.health_check_tasks[service_id]
            
            # Deregister service
            success = await self.aio_consul.agent.service.deregister(service_id)
            
            if success:
                if service_id in self.registered_services:
                    del self.registered_services[service_id]
                self.logger.info(f"Deregistered agent {agent_id}")
                return True
            else:
                self.logger.error(f"Failed to deregister agent {agent_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deregistering agent {agent_id}: {e}")
            return False
    
    async def discover_agents(
        self,
        agent_type: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        status_filter: str = "passing"
    ) -> List[AgentServiceInfo]:
        """Discover agents by type, capabilities, or status."""
        try:
            # Get all agent services
            services = await self.aio_consul.health.service(
                "a2a-agent" if not agent_type else f"a2a-agent-{agent_type}",
                passing=True if status_filter == "passing" else None
            )
            
            discovered_agents = []
            
            for service_data in services[1]:  # services[1] contains the actual service data
                service = service_data.get("Service", {})
                checks = service_data.get("Checks", [])
                
                # Extract service information
                service_id = service.get("ID", "")
                meta = service.get("Meta", {})
                tags = service.get("Tags", [])
                
                if not service_id.startswith("a2a-agent-"):
                    continue
                
                # Parse capabilities from metadata or tags
                service_capabilities = []
                if "capabilities" in meta:
                    try:
                        service_capabilities = json.loads(meta["capabilities"])
                    except json.JSONDecodeError:
                        pass
                
                # Extract capabilities from tags
                for tag in tags:
                    if tag.startswith("capability:"):
                        cap = tag.split(":", 1)[1]
                        if cap not in service_capabilities:
                            service_capabilities.append(cap)
                
                # Filter by capabilities if specified
                if capabilities:
                    if not any(cap in service_capabilities for cap in capabilities):
                        continue
                
                # Determine health status
                health_status = "healthy"
                for check in checks:
                    if check.get("Status") != "passing":
                        health_status = "unhealthy"
                        break
                
                agent_info = AgentServiceInfo(
                    service_id=service_id,
                    agent_id=meta.get("agent_id", service_id.replace("a2a-agent-", "")),
                    agent_type=meta.get("agent_type", "unknown"),
                    capabilities=service_capabilities,
                    address=service.get("Address", "localhost"),
                    port=service.get("Port", 0),
                    status=health_status,
                    metadata=meta
                )
                
                discovered_agents.append(agent_info)
            
            self.logger.info(f"Discovered {len(discovered_agents)} agents")
            return discovered_agents
            
        except Exception as e:
            self.logger.error(f"Error discovering agents: {e}")
            return []
    
    async def find_agents_by_capability(self, capability: str) -> List[AgentServiceInfo]:
        """Find agents that have a specific capability."""
        return await self.discover_agents(capabilities=[capability])
    
    async def get_agent_by_id(self, agent_id: str) -> Optional[AgentServiceInfo]:
        """Get specific agent by ID."""
        service_id = f"a2a-agent-{agent_id}"
        
        try:
            services = await self.aio_consul.health.service(
                service_id,
                passing=False  # Get all statuses
            )
            
            for service_data in services[1]:
                service = service_data.get("Service", {})
                if service.get("ID") == service_id:
                    meta = service.get("Meta", {})
                    
                    capabilities = []
                    if "capabilities" in meta:
                        try:
                            capabilities = json.loads(meta["capabilities"])
                        except json.JSONDecodeError:
                            pass
                    
                    return AgentServiceInfo(
                        service_id=service_id,
                        agent_id=agent_id,
                        agent_type=meta.get("agent_type", "unknown"),
                        capabilities=capabilities,
                        address=service.get("Address", "localhost"),
                        port=service.get("Port", 0),
                        metadata=meta
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting agent {agent_id}: {e}")
            return None
    
    async def update_agent_status(self, agent_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update agent status and metadata."""
        try:
            service_id = f"a2a-agent-{agent_id}"
            
            # Update TTL health check
            if status == "healthy":
                await self.aio_consul.agent.check.ttl_pass(f"service:{service_id}")
            else:
                await self.aio_consul.agent.check.ttl_fail(
                    f"service:{service_id}",
                    f"Agent status: {status}"
                )
            
            # Update metadata if provided
            if metadata and service_id in self.registered_services:
                self.registered_services[service_id].metadata.update(metadata)
                self.registered_services[service_id].status = status
                self.registered_services[service_id].last_health_check = datetime.utcnow()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating agent status {agent_id}: {e}")
            return False
    
    async def get_configuration(self, key: str, default: Any = None) -> Any:
        """Get configuration value from Consul KV store."""
        try:
            index, data = await self.aio_consul.kv.get(f"a2a-world/config/{key}")
            if data:
                try:
                    return json.loads(data["Value"].decode())
                except json.JSONDecodeError:
                    return data["Value"].decode()
            return default
            
        except Exception as e:
            self.logger.error(f"Error getting configuration {key}: {e}")
            return default
    
    async def set_configuration(self, key: str, value: Any) -> bool:
        """Set configuration value in Consul KV store."""
        try:
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            
            success = await self.aio_consul.kv.put(f"a2a-world/config/{key}", value_str)
            return success
            
        except Exception as e:
            self.logger.error(f"Error setting configuration {key}: {e}")
            return False
    
    async def watch_configuration(self, key: str, callback):
        """Watch for configuration changes."""
        # This would be implemented with Consul's blocking queries
        # For now, we'll provide a basic polling implementation
        last_value = await self.get_configuration(key)
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                current_value = await self.get_configuration(key)
                
                if current_value != last_value:
                    await callback(key, current_value, last_value)
                    last_value = current_value
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error watching configuration {key}: {e}")
                await asyncio.sleep(10)
    
    async def _start_health_check_maintenance(self, service_id: str):
        """Start periodic TTL health check maintenance."""
        async def maintain_health():
            while True:
                try:
                    await asyncio.sleep(30)  # Send TTL pass every 30 seconds
                    await self.aio_consul.agent.check.ttl_pass(f"service:{service_id}")
                    
                    if service_id in self.registered_services:
                        self.registered_services[service_id].last_health_check = datetime.utcnow()
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Health check maintenance error for {service_id}: {e}")
                    await asyncio.sleep(5)
        
        task = asyncio.create_task(maintain_health())
        self.health_check_tasks[service_id] = task
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status and metrics."""
        try:
            # Get all agent services
            agents = await self.discover_agents()
            
            # Calculate statistics
            agent_types = {}
            capabilities = set()
            healthy_count = 0
            
            for agent in agents:
                agent_types[agent.agent_type] = agent_types.get(agent.agent_type, 0) + 1
                capabilities.update(agent.capabilities)
                if agent.status == "healthy":
                    healthy_count += 1
            
            return {
                "total_agents": len(agents),
                "healthy_agents": healthy_count,
                "unhealthy_agents": len(agents) - healthy_count,
                "agent_types": agent_types,
                "available_capabilities": list(capabilities),
                "cluster_health": "healthy" if healthy_count == len(agents) else "degraded",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cluster status: {e}")
            return {
                "total_agents": 0,
                "healthy_agents": 0,
                "unhealthy_agents": 0,
                "agent_types": {},
                "available_capabilities": [],
                "cluster_health": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup registry resources."""
        # Cancel all health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.health_check_tasks:
            await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
        
        self.health_check_tasks.clear()
        self.registered_services.clear()


# Global registry instance
_global_registry: Optional[ConsulRegistry] = None


async def get_consul_registry(
    host: str = "localhost",
    port: int = 8500,
    token: Optional[str] = None
) -> ConsulRegistry:
    """Get or create global Consul registry instance."""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ConsulRegistry(host=host, port=port, token=token)
    
    return _global_registry


async def cleanup_registry():
    """Cleanup global registry instance."""
    global _global_registry
    
    if _global_registry:
        await _global_registry.cleanup()
        _global_registry = None