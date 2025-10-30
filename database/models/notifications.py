"""
A2A World Platform - Notification Database Models

Database models for email subscriptions and notification history.
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, ForeignKey, JSON, func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from .base import Base


class EmailSubscription(Base):
    """
    Email subscription model for managing user notification preferences.

    Tracks which users want to receive notifications for different event types.
    """

    __tablename__ = "email_subscriptions"

    # Foreign key to users table (if exists)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)

    # Subscription details
    email = Column(String(255), nullable=False, index=True)
    subscriber_name = Column(String(255))

    # Notification preferences
    validation_notifications = Column(Boolean, default=True)
    pattern_discovery_notifications = Column(Boolean, default=True)
    consensus_notifications = Column(Boolean, default=False)
    maintenance_notifications = Column(Boolean, default=True)
    system_notifications = Column(Boolean, default=False)

    # Subscription settings
    notification_frequency = Column(String(50), default='immediate')  # immediate, daily, weekly
    is_active = Column(Boolean, default=True)

    # Subscription metadata
    subscribed_at = Column(DateTime(timezone=True), server_default=func.now())
    unsubscribed_at = Column(DateTime(timezone=True), nullable=True)
    subscription_source = Column(String(100), default='platform')  # platform, api, admin

    # Additional preferences
    preferred_language = Column(String(10), default='en')
    timezone = Column(String(50), default='UTC')

    # JSON field for custom preferences
    custom_preferences = Column(JSON, default=dict)

    def __repr__(self):
        return f"<EmailSubscription(email='{self.email}', active={self.is_active})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        data = super().to_dict()
        # Add notification types as a list
        data['notification_types'] = []
        if self.validation_notifications:
            data['notification_types'].append('validation')
        if self.pattern_discovery_notifications:
            data['notification_types'].append('pattern_discovery')
        if self.consensus_notifications:
            data['notification_types'].append('consensus')
        if self.maintenance_notifications:
            data['notification_types'].append('maintenance')
        if self.system_notifications:
            data['notification_types'].append('system')
        return data

    def is_subscribed_to(self, notification_type: str) -> bool:
        """Check if user is subscribed to a specific notification type."""
        type_map = {
            'validation': self.validation_notifications,
            'pattern_discovery': self.pattern_discovery_notifications,
            'consensus': self.consensus_notifications,
            'maintenance': self.maintenance_notifications,
            'system': self.system_notifications
        }
        return type_map.get(notification_type, False)


class NotificationHistory(Base):
    """
    Notification history model for tracking sent emails.

    Records all email notifications sent, their status, and delivery information.
    """

    __tablename__ = "notification_history"

    # Notification details
    notification_type = Column(String(100), nullable=False, index=True)  # validation, pattern_discovery, etc.
    subject = Column(String(500), nullable=False)
    recipient_email = Column(String(255), nullable=False, index=True)

    # Content and metadata
    template_used = Column(String(100))
    notification_data = Column(JSON)  # The data used to render the template

    # Delivery status
    status = Column(String(50), default='pending')  # pending, sent, delivered, failed, bounced
    sent_at = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=True)

    # Error information
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # SMTP delivery details
    smtp_message_id = Column(String(255))
    smtp_response = Column(Text)

    # Related entities (optional foreign keys)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey('email_subscriptions.id'), nullable=True)
    related_entity_id = Column(String(255))  # Could be dataset_id, pattern_id, etc.
    related_entity_type = Column(String(100))  # dataset, pattern, consensus, etc.

    # Agent information
    sent_by_agent = Column(String(255))  # Agent ID that sent the notification

    # Relationships
    subscription = relationship("EmailSubscription", backref="notification_history")

    def __repr__(self):
        return f"<NotificationHistory(type='{self.notification_type}', email='{self.recipient_email}', status='{self.status}')>"

    def mark_sent(self, smtp_message_id: str = None):
        """Mark notification as sent."""
        self.status = 'sent'
        self.sent_at = func.now()
        if smtp_message_id:
            self.smtp_message_id = smtp_message_id

    def mark_delivered(self):
        """Mark notification as delivered."""
        self.status = 'delivered'
        self.delivered_at = func.now()

    def mark_failed(self, error_message: str):
        """Mark notification as failed."""
        self.status = 'failed'
        self.failed_at = func.now()
        self.error_message = error_message
        self.retry_count += 1

    def can_retry(self) -> bool:
        """Check if notification can be retried."""
        return self.retry_count < self.max_retries and self.status in ['pending', 'failed']

    def to_dict(self):
        """Convert to dictionary for API responses."""
        data = super().to_dict()
        # Convert timestamps to ISO format
        if self.sent_at:
            data['sent_at'] = self.sent_at.isoformat()
        if self.delivered_at:
            data['delivered_at'] = self.delivered_at.isoformat()
        if self.failed_at:
            data['failed_at'] = self.failed_at.isoformat()
        return data


class NotificationTemplate(Base):
    """
    Email template model for storing reusable notification templates.

    Allows dynamic template management and customization.
    """

    __tablename__ = "notification_templates"

    # Template identification
    template_name = Column(String(100), nullable=False, unique=True, index=True)
    template_type = Column(String(100), nullable=False)  # validation, pattern_discovery, etc.
    subject_template = Column(String(500), nullable=False)

    # Template content
    html_content = Column(Text, nullable=False)
    text_content = Column(Text)  # Plain text fallback

    # Template metadata
    version = Column(String(20), default='1.0')
    is_active = Column(Boolean, default=True)
    created_by = Column(String(255))

    # Template variables (JSON schema)
    required_variables = Column(JSON, default=list)  # List of required template variables
    optional_variables = Column(JSON, default=list)  # List of optional template variables

    # Usage statistics
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<NotificationTemplate(name='{self.template_name}', type='{self.template_type}', active={self.is_active})>"

    def increment_usage(self):
        """Increment usage count and update last used timestamp."""
        self.usage_count += 1
        self.last_used = func.now()

    def validate_variables(self, variables: dict) -> list:
        """Validate that required template variables are provided."""
        missing = []
        for var in self.required_variables:
            if var not in variables:
                missing.append(var)
        return missing