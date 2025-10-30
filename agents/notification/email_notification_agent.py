"""
A2A World Platform - Email Notification Agent

Autonomous agent for handling email notifications based on system events.
Subscribes to NATS topics and sends email notifications to subscribed users.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from agents.core.base_agent import BaseAgent
from agents.core.messaging import AgentMessage
from agents.core.task_queue import Task
from api.app.core.email_service import send_notification_email, EmailMessage


class EmailNotificationAgent(BaseAgent):
    """
    Email notification agent that monitors system events and sends email notifications.

    Subscribes to:
    - agents.validation (validation results)
    - agents.discovery (pattern discoveries)
    - agents.consensus (consensus updates)
    - system.maintenance (maintenance notifications)
    - agents.broadcast (system broadcasts)

    Sends notifications to subscribed email addresses based on event types.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(
            agent_id=agent_id or "email-notification-agent",
            agent_type="email_notification"
        )

        # Notification subscriptions cache (in production, this would be in database)
        self.email_subscriptions: Dict[str, List[str]] = {
            "validation": ["researchers@a2aworld.ai", "validators@a2aworld.ai"],
            "pattern_discovery": ["researchers@a2aworld.ai", "analysts@a2aworld.ai"],
            "consensus": ["participants@a2aworld.ai", "coordinators@a2aworld.ai"],
            "maintenance": ["admin@a2aworld.ai", "ops@a2aworld.ai"],
            "system": ["admin@a2aworld.ai"]
        }

        # Notification processing stats
        self.notifications_sent = 0
        self.notifications_failed = 0
        self.pending_notifications: List[Dict[str, Any]] = []

    async def agent_initialize(self) -> None:
        """Agent-specific initialization."""
        self.logger.info("Initializing email notification agent")

        # Test email service connectivity
        try:
            from api.app.core.email_service import get_email_service
            email_service = get_email_service()
            test_result = await email_service.test_connection()
            if test_result:
                self.logger.info("Email service connection test successful")
            else:
                self.logger.warning("Email service connection test failed")
        except Exception as e:
            self.logger.error(f"Failed to test email service: {e}")

    async def setup_subscriptions(self) -> None:
        """Setup NATS message subscriptions for different event types."""
        if not self.messaging:
            return

        # Subscribe to validation results
        validation_sub = await self.messaging.nats.subscribe(
            "agents.validation",
            self._handle_validation_message,
            queue_group="email-notifications"
        )
        self.subscription_ids.append(validation_sub)

        # Subscribe to pattern discoveries
        discovery_sub = await self.messaging.nats.subscribe(
            "agents.discovery",
            self._handle_discovery_message,
            queue_group="email-notifications"
        )
        self.subscription_ids.append(discovery_sub)

        # Subscribe to consensus updates
        consensus_sub = await self.messaging.nats.subscribe(
            "agents.consensus",
            self._handle_consensus_message,
            queue_group="email-notifications"
        )
        self.subscription_ids.append(consensus_sub)

        # Subscribe to system maintenance notifications
        maintenance_sub = await self.messaging.nats.subscribe(
            "system.maintenance",
            self._handle_maintenance_message,
            queue_group="email-notifications"
        )
        self.subscription_ids.append(maintenance_sub)

        # Subscribe to system broadcasts
        broadcast_sub = await self.messaging.nats.subscribe(
            "agents.broadcast",
            self._handle_system_broadcast,
            queue_group="email-notifications"
        )
        self.subscription_ids.append(broadcast_sub)

        self.logger.info("Email notification subscriptions established")

    async def process(self) -> None:
        """Main processing loop - handle pending notifications."""
        if self.pending_notifications:
            # Process pending notifications in batches
            batch_size = min(5, len(self.pending_notifications))
            batch = self.pending_notifications[:batch_size]

            for notification in batch:
                try:
                    await self._send_notification(notification)
                    self.notifications_sent += 1
                except Exception as e:
                    self.logger.error(f"Failed to send notification: {e}")
                    self.notifications_failed += 1

            # Remove processed notifications
            self.pending_notifications = self.pending_notifications[batch_size:]

        # Small delay to prevent busy looping
        await asyncio.sleep(1)

    async def _handle_validation_message(self, message: AgentMessage) -> None:
        """Handle validation result messages."""
        try:
            payload = message.payload

            # Prepare notification data
            notification_data = {
                "dataset_id": payload.get("dataset_id", "unknown"),
                "validation_type": payload.get("validation_type", "general"),
                "validation_status": payload.get("status", "completed"),
                "completed_at": payload.get("timestamp", datetime.utcnow().isoformat()),
                "processing_time": payload.get("processing_time", 0),
                "issues": payload.get("issues", []),
                "recommendations": payload.get("recommendations", []),
                "metrics": payload.get("metrics", []),
                "dashboard_url": f"https://a2aworld.ai/dashboard/validation/{payload.get('dataset_id', '')}"
            }

            # Queue notification
            await self._queue_notification("validation", notification_data)

        except Exception as e:
            self.logger.error(f"Error handling validation message: {e}")

    async def _handle_discovery_message(self, message: AgentMessage) -> None:
        """Handle pattern discovery messages."""
        try:
            payload = message.payload

            # Prepare notification data
            notification_data = {
                "pattern_title": payload.get("title", "New Pattern Discovered"),
                "pattern_description": payload.get("description", ""),
                "pattern_id": payload.get("pattern_id", ""),
                "discovered_at": payload.get("timestamp", datetime.utcnow().isoformat()),
                "data_source": payload.get("data_source", "unknown"),
                "pattern_type": payload.get("pattern_type", "unknown"),
                "confidence": payload.get("confidence", 0),
                "confidence_level": self._get_confidence_level(payload.get("confidence", 0)),
                "insights": payload.get("insights", []),
                "related_patterns": payload.get("related_patterns", []),
                "analysis_url": f"https://a2aworld.ai/dashboard/patterns/{payload.get('pattern_id', '')}",
                "dashboard_url": "https://a2aworld.ai/dashboard"
            }

            # Queue notification
            await self._queue_notification("pattern_discovery", notification_data)

        except Exception as e:
            self.logger.error(f"Error handling discovery message: {e}")

    async def _handle_consensus_message(self, message: AgentMessage) -> None:
        """Handle consensus update messages."""
        try:
            payload = message.payload

            # Prepare notification data
            notification_data = {
                "consensus_title": payload.get("title", "Consensus Update"),
                "consensus_description": payload.get("description", ""),
                "consensus_status": payload.get("status", "in_progress"),
                "total_participants": payload.get("total_participants", 0),
                "votes_cast": payload.get("votes_cast", 0),
                "consensus_percentage": payload.get("consensus_percentage", 0),
                "time_remaining": payload.get("time_remaining", "unknown"),
                "participants": payload.get("participants", []),
                "timeline": payload.get("timeline", []),
                "decision_reached": payload.get("decision_reached", False),
                "final_decision": payload.get("final_decision", ""),
                "decision_confidence": payload.get("decision_confidence", 0),
                "next_steps": payload.get("next_steps", []),
                "consensus_url": f"https://a2aworld.ai/dashboard/consensus/{payload.get('consensus_id', '')}",
                "dashboard_url": "https://a2aworld.ai/dashboard"
            }

            # Queue notification
            await self._queue_notification("consensus", notification_data)

        except Exception as e:
            self.logger.error(f"Error handling consensus message: {e}")

    async def _handle_maintenance_message(self, message: AgentMessage) -> None:
        """Handle system maintenance messages."""
        try:
            payload = message.payload

            # Prepare notification data
            notification_data = {
                "maintenance_type": payload.get("type", "scheduled"),
                "maintenance_message": payload.get("message", "System maintenance scheduled"),
                "maintenance_start": payload.get("start_time", "unknown"),
                "maintenance_duration": payload.get("duration", "unknown"),
                "maintenance_end": payload.get("end_time", "unknown"),
                "impact_level": payload.get("impact_level", "medium"),
                "affected_services": payload.get("affected_services", []),
                "maintenance_reason": payload.get("reason", ""),
                "what_to_expect": payload.get("what_to_expect", []),
                "alternative_access": payload.get("alternative_access", ""),
                "status_page_url": "https://status.a2aworld.ai",
                "dashboard_url": "https://a2aworld.ai/dashboard"
            }

            # Queue notification
            await self._queue_notification("maintenance", notification_data)

        except Exception as e:
            self.logger.error(f"Error handling maintenance message: {e}")

    async def _handle_system_broadcast(self, message: AgentMessage) -> None:
        """Handle system broadcast messages."""
        try:
            payload = message.payload

            # Only send notifications for important system events
            if payload.get("priority", "low") in ["high", "critical"]:
                notification_data = {
                    "broadcast_type": payload.get("type", "system_broadcast"),
                    "message": payload.get("message", ""),
                    "priority": payload.get("priority", "low"),
                    "timestamp": payload.get("timestamp", datetime.utcnow().isoformat()),
                    "action_required": payload.get("action_required", False),
                    "dashboard_url": "https://a2aworld.ai/dashboard"
                }

                await self._queue_notification("system", notification_data)

        except Exception as e:
            self.logger.error(f"Error handling system broadcast: {e}")

    async def _queue_notification(self, notification_type: str, data: Dict[str, Any]) -> None:
        """Queue a notification for processing."""
        notification = {
            "type": notification_type,
            "data": data,
            "queued_at": datetime.utcnow().isoformat(),
            "recipients": self.email_subscriptions.get(notification_type, [])
        }

        self.pending_notifications.append(notification)
        self.logger.info(f"Queued {notification_type} notification for {len(notification['recipients'])} recipients")

    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """Send a queued notification."""
        notification_type = notification["type"]
        data = notification["data"]
        recipients = notification["recipients"]

        if not recipients:
            self.logger.warning(f"No recipients configured for {notification_type} notifications")
            return

        # Send notification email
        success = await send_notification_email(recipients, notification_type, data)

        if success:
            self.logger.info(f"Successfully sent {notification_type} notification to {len(recipients)} recipients")
        else:
            self.logger.error(f"Failed to send {notification_type} notification")
            raise Exception(f"Email sending failed for {notification_type}")

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence percentage to descriptive level."""
        if confidence >= 90:
            return "Very High"
        elif confidence >= 75:
            return "High"
        elif confidence >= 60:
            return "Medium"
        elif confidence >= 40:
            return "Low"
        else:
            return "Very Low"

    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect agent-specific metrics."""
        return {
            "notifications_sent": self.notifications_sent,
            "notifications_failed": self.notifications_failed,
            "pending_notifications": len(self.pending_notifications),
            "subscription_counts": {k: len(v) for k, v in self.email_subscriptions.items()}
        }

    async def check_health(self) -> Optional[List[str]]:
        """Perform agent-specific health checks."""
        issues = []

        # Check if we have email subscriptions configured
        total_subscriptions = sum(len(recipients) for recipients in self.email_subscriptions.values())
        if total_subscriptions == 0:
            issues.append("No email subscriptions configured")

        # Check pending notification queue size
        if len(self.pending_notifications) > 100:
            issues.append("Large pending notification queue")

        # Check failure rate
        total_processed = self.notifications_sent + self.notifications_failed
        if total_processed > 0:
            failure_rate = self.notifications_failed / total_processed
            if failure_rate > 0.1:  # 10% failure rate
                issues.append("High notification failure rate")

        return issues if issues else None

    def _get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return [
            "email_notification",
            "event_monitoring",
            "nats_subscriber",
            "notification_service"
        ]