"""
A2A World Platform - NATS Messaging System

NATS integration for agent communication, event streaming, and coordination.
Provides pub/sub patterns, message serialization, and topic organization.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg


@dataclass
class AgentMessage:
    """Standardized message format for agent communication."""
    
    id: str
    sender_id: str
    receiver_id: Optional[str]
    message_type: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    priority: int = 5
    reply_to: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(
        cls,
        sender_id: str,
        message_type: str,
        payload: Dict[str, Any],
        receiver_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: int = 5
    ) -> 'AgentMessage':
        """Create a new agent message with auto-generated ID and timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id,
            priority=priority,
            metadata={}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        return cls(**data)


class NATSClient:
    """
    NATS client wrapper for agent communication.
    Handles connections, message serialization, and topic organization.
    """
    
    def __init__(self, url: str = "nats://localhost:4222", name: str = "agent"):
        self.url = url
        self.name = name
        self.nc: Optional[NATS] = None
        self.js = None  # JetStream context
        self.logger = logging.getLogger(f"nats.{name}")
        self.subscriptions: Dict[str, Any] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Connect to NATS server with error handling."""
        try:
            self.nc = await nats.connect(
                servers=self.url,
                name=self.name,
                error_cb=self._error_callback,
                disconnected_cb=self._disconnected_callback,
                reconnected_cb=self._reconnected_callback,
                closed_cb=self._closed_callback,
                max_reconnect_attempts=10,
                reconnect_time_wait=2
            )
            
            # Initialize JetStream context
            self.js = self.nc.jetstream()
            self.is_connected = True
            
            self.logger.info(f"Connected to NATS server: {self.url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to NATS: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from NATS server."""
        if self.nc and not self.nc.is_closed:
            await self.nc.close()
            self.is_connected = False
            self.logger.info("Disconnected from NATS server")
    
    async def publish(
        self,
        subject: str,
        message: Union[AgentMessage, Dict[str, Any], str],
        reply_to: Optional[str] = None
    ) -> None:
        """Publish a message to a NATS subject."""
        if not self.is_connected or not self.nc:
            raise RuntimeError("Not connected to NATS server")
        
        try:
            # Serialize message
            if isinstance(message, AgentMessage):
                payload = json.dumps(message.to_dict()).encode()
            elif isinstance(message, dict):
                payload = json.dumps(message).encode()
            else:
                payload = str(message).encode()
            
            await self.nc.publish(subject, payload, reply=reply_to)
            self.logger.debug(f"Published message to {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish to {subject}: {e}")
            raise
    
    async def subscribe(
        self,
        subject: str,
        handler: Callable[[AgentMessage], None],
        queue_group: Optional[str] = None
    ) -> str:
        """Subscribe to a NATS subject with message handler."""
        if not self.is_connected or not self.nc:
            raise RuntimeError("Not connected to NATS server")
        
        try:
            async def message_callback(msg: Msg):
                try:
                    # Deserialize message
                    data = json.loads(msg.data.decode())
                    agent_message = AgentMessage.from_dict(data)
                    
                    # Call handler
                    await handler(agent_message)
                    
                except Exception as e:
                    self.logger.error(f"Error processing message from {subject}: {e}")
            
            # Subscribe with optional queue group for load balancing
            if queue_group:
                sub = await self.nc.subscribe(subject, cb=message_callback, queue=queue_group)
            else:
                sub = await self.nc.subscribe(subject, cb=message_callback)
            
            subscription_id = str(uuid.uuid4())
            self.subscriptions[subscription_id] = sub
            
            self.logger.info(f"Subscribed to {subject} with queue group: {queue_group}")
            return subscription_id
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {subject}: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a NATS subject."""
        if subscription_id in self.subscriptions:
            sub = self.subscriptions[subscription_id]
            await sub.unsubscribe()
            del self.subscriptions[subscription_id]
            self.logger.info(f"Unsubscribed from subscription {subscription_id}")
    
    async def request(
        self,
        subject: str,
        message: Union[AgentMessage, Dict[str, Any]],
        timeout: float = 5.0
    ) -> AgentMessage:
        """Send a request and wait for response."""
        if not self.is_connected or not self.nc:
            raise RuntimeError("Not connected to NATS server")
        
        try:
            # Serialize request
            if isinstance(message, AgentMessage):
                payload = json.dumps(message.to_dict()).encode()
            else:
                payload = json.dumps(message).encode()
            
            # Send request
            response = await self.nc.request(subject, payload, timeout=timeout)
            
            # Deserialize response
            response_data = json.loads(response.data.decode())
            return AgentMessage.from_dict(response_data)
            
        except Exception as e:
            self.logger.error(f"Request to {subject} failed: {e}")
            raise
    
    async def _error_callback(self, error):
        """Handle NATS connection errors."""
        self.logger.error(f"NATS error: {error}")
    
    async def _disconnected_callback(self):
        """Handle NATS disconnection."""
        self.logger.warning("Disconnected from NATS server")
        self.is_connected = False
    
    async def _reconnected_callback(self):
        """Handle NATS reconnection."""
        self.logger.info("Reconnected to NATS server")
        self.is_connected = True
    
    async def _closed_callback(self):
        """Handle NATS connection closure."""
        self.logger.info("NATS connection closed")
        self.is_connected = False


class AgentMessaging:
    """
    High-level messaging interface for agents.
    Provides topic organization and message routing patterns.
    """
    
    def __init__(self, nats_client: NATSClient, agent_id: str):
        self.nats = nats_client
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"messaging.{agent_id}")
        
        # Topic patterns for different message types
        self.topics = {
            "heartbeat": "agents.heartbeat",
            "task_request": f"agents.{agent_id}.tasks",
            "task_response": "agents.tasks.responses",
            "discovery": "agents.discovery",
            "validation": "agents.validation",
            "coordination": "agents.coordination",
            "broadcast": "agents.broadcast",
            "monitoring": "agents.monitoring"
        }
    
    async def send_heartbeat(self, status: str, metrics: Dict[str, Any]) -> None:
        """Send periodic heartbeat message."""
        message = AgentMessage.create(
            sender_id=self.agent_id,
            message_type="heartbeat",
            payload={
                "status": status,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await self.nats.publish(self.topics["heartbeat"], message)
    
    async def request_task(self, task_type: str, parameters: Dict[str, Any]) -> Optional[AgentMessage]:
        """Request a task from the task queue."""
        message = AgentMessage.create(
            sender_id=self.agent_id,
            message_type="task_request",
            payload={
                "task_type": task_type,
                "parameters": parameters,
                "requester": self.agent_id
            }
        )
        
        try:
            response = await self.nats.request("agents.tasks.queue", message, timeout=10.0)
            return response
        except Exception as e:
            self.logger.error(f"Task request failed: {e}")
            return None
    
    async def submit_task_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Submit task completion result."""
        message = AgentMessage.create(
            sender_id=self.agent_id,
            message_type="task_result",
            payload={
                "task_id": task_id,
                "result": result,
                "completed_by": self.agent_id
            }
        )
        
        await self.nats.publish(self.topics["task_response"], message)
    
    async def publish_discovery(self, pattern_data: Dict[str, Any]) -> None:
        """Publish pattern discovery results."""
        message = AgentMessage.create(
            sender_id=self.agent_id,
            message_type="pattern_discovered",
            payload=pattern_data
        )

        await self.nats.publish(self.topics["discovery"], message)

    async def publish_validation(self, validation_data: Dict[str, Any]) -> None:
        """Publish validation completion results."""
        message = AgentMessage.create(
            sender_id=self.agent_id,
            message_type="validation_completed",
            payload=validation_data
        )

        await self.nats.publish(self.topics["validation"], message)
    
    async def request_validation(self, pattern_id: str, pattern_data: Dict[str, Any]) -> Optional[AgentMessage]:
        """Request pattern validation from validation agents."""
        message = AgentMessage.create(
            sender_id=self.agent_id,
            message_type="validation_request",
            payload={
                "pattern_id": pattern_id,
                "pattern_data": pattern_data
            }
        )
        
        try:
            response = await self.nats.request(self.topics["validation"], message, timeout=30.0)
            return response
        except Exception as e:
            self.logger.error(f"Validation request failed: {e}")
            return None
    
    async def broadcast_status(self, status_data: Dict[str, Any]) -> None:
        """Broadcast status update to all agents."""
        message = AgentMessage.create(
            sender_id=self.agent_id,
            message_type="status_broadcast",
            payload=status_data
        )
        
        await self.nats.publish(self.topics["broadcast"], message)
    
    async def subscribe_to_discoveries(self, handler: Callable[[AgentMessage], None]) -> str:
        """Subscribe to pattern discovery notifications."""
        return await self.nats.subscribe(
            self.topics["discovery"], 
            handler, 
            queue_group=f"{self.agent_id}-discoveries"
        )
    
    async def subscribe_to_tasks(self, handler: Callable[[AgentMessage], None]) -> str:
        """Subscribe to task assignments."""
        return await self.nats.subscribe(
            self.topics["task_request"], 
            handler, 
            queue_group=f"{self.agent_id}-tasks"
        )
    
    async def subscribe_to_broadcasts(self, handler: Callable[[AgentMessage], None]) -> str:
        """Subscribe to system broadcasts."""
        return await self.nats.subscribe(
            self.topics["broadcast"], 
            handler
        )


# Global NATS client instance (can be shared across agents)
_global_nats_client: Optional[NATSClient] = None


async def get_nats_client(url: str = "nats://localhost:4222", name: str = "agent") -> NATSClient:
    """Get or create global NATS client instance."""
    global _global_nats_client
    
    if _global_nats_client is None or not _global_nats_client.is_connected:
        _global_nats_client = NATSClient(url, name)
        await _global_nats_client.connect()
    
    return _global_nats_client


async def cleanup_nats():
    """Cleanup global NATS client."""
    global _global_nats_client
    
    if _global_nats_client:
        await _global_nats_client.disconnect()
        _global_nats_client = None