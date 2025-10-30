"""
A2A World Platform - Notification API Endpoints

FastAPI endpoints for email notification management, subscriptions, and sending.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from api.app.core.database import get_db
from api.app.core.email_service import send_notification_email, get_email_service
from database.models.notifications import (
    EmailSubscription, NotificationHistory, NotificationTemplate
)


router = APIRouter()


# Pydantic models for API requests/responses

class EmailSubscriptionCreate(BaseModel):
    """Request model for creating email subscriptions."""

    email: EmailStr
    subscriber_name: Optional[str] = None
    validation_notifications: bool = True
    pattern_discovery_notifications: bool = True
    consensus_notifications: bool = False
    maintenance_notifications: bool = True
    system_notifications: bool = False
    notification_frequency: str = Field(default="immediate", regex="^(immediate|daily|weekly)$")
    preferred_language: str = "en"
    timezone: str = "UTC"


class EmailSubscriptionUpdate(BaseModel):
    """Request model for updating email subscriptions."""

    subscriber_name: Optional[str] = None
    validation_notifications: Optional[bool] = None
    pattern_discovery_notifications: Optional[bool] = None
    consensus_notifications: Optional[bool] = None
    maintenance_notifications: Optional[bool] = None
    system_notifications: Optional[bool] = None
    notification_frequency: Optional[str] = Field(None, regex="^(immediate|daily|weekly)$")
    preferred_language: Optional[str] = None
    timezone: Optional[str] = None
    is_active: Optional[bool] = None


class EmailSubscriptionResponse(BaseModel):
    """Response model for email subscriptions."""

    id: UUID
    email: EmailStr
    subscriber_name: Optional[str]
    notification_types: List[str]
    notification_frequency: str
    is_active: bool
    subscribed_at: datetime
    preferred_language: str
    timezone: str

    class Config:
        from_attributes = True


class SendNotificationRequest(BaseModel):
    """Request model for sending manual notifications."""

    recipients: List[EmailStr]
    notification_type: str = Field(..., regex="^(validation|pattern_discovery|consensus|maintenance|system)$")
    subject: str
    template_data: Dict[str, Any]
    cc: Optional[List[EmailStr]] = None
    bcc: Optional[List[EmailStr]] = None


class NotificationHistoryResponse(BaseModel):
    """Response model for notification history."""

    id: UUID
    notification_type: str
    subject: str
    recipient_email: EmailStr
    status: str
    sent_at: Optional[datetime]
    delivered_at: Optional[datetime]
    failed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int

    class Config:
        from_attributes = True


class EmailTestRequest(BaseModel):
    """Request model for testing email configuration."""

    test_email: EmailStr


# API Endpoints

@router.post("/subscriptions", response_model=EmailSubscriptionResponse)
async def create_email_subscription(
    subscription: EmailSubscriptionCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new email subscription.

    Allows users to subscribe to different types of email notifications.
    """
    # Check if email already exists
    existing = db.query(EmailSubscription).filter(
        EmailSubscription.email == subscription.email,
        EmailSubscription.is_active == True
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email address is already subscribed"
        )

    # Create new subscription
    db_subscription = EmailSubscription(
        email=subscription.email,
        subscriber_name=subscription.subscriber_name,
        validation_notifications=subscription.validation_notifications,
        pattern_discovery_notifications=subscription.pattern_discovery_notifications,
        consensus_notifications=subscription.consensus_notifications,
        maintenance_notifications=subscription.maintenance_notifications,
        system_notifications=subscription.system_notifications,
        notification_frequency=subscription.notification_frequency,
        preferred_language=subscription.preferred_language,
        timezone=subscription.timezone
    )

    db.add(db_subscription)
    db.commit()
    db.refresh(db_subscription)

    return db_subscription


@router.get("/subscriptions", response_model=List[EmailSubscriptionResponse])
async def list_email_subscriptions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    List email subscriptions with pagination.

    Returns all active email subscriptions by default.
    """
    query = db.query(EmailSubscription)

    if active_only:
        query = query.filter(EmailSubscription.is_active == True)

    subscriptions = query.offset(skip).limit(limit).all()
    return subscriptions


@router.get("/subscriptions/{subscription_id}", response_model=EmailSubscriptionResponse)
async def get_email_subscription(
    subscription_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific email subscription by ID."""
    subscription = db.query(EmailSubscription).filter(
        EmailSubscription.id == subscription_id
    ).first()

    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    return subscription


@router.put("/subscriptions/{subscription_id}", response_model=EmailSubscriptionResponse)
async def update_email_subscription(
    subscription_id: UUID,
    updates: EmailSubscriptionUpdate,
    db: Session = Depends(get_db)
):
    """Update an email subscription."""
    subscription = db.query(EmailSubscription).filter(
        EmailSubscription.id == subscription_id
    ).first()

    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    # Update fields
    update_data = updates.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(subscription, field, value)

    subscription.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(subscription)

    return subscription


@router.delete("/subscriptions/{subscription_id}")
async def delete_email_subscription(
    subscription_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete (unsubscribe) an email subscription."""
    subscription = db.query(EmailSubscription).filter(
        EmailSubscription.id == subscription_id
    ).first()

    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    # Soft delete by marking as inactive
    subscription.is_active = False
    subscription.unsubscribed_at = datetime.utcnow()

    db.commit()

    return {"message": "Subscription deactivated successfully"}


@router.post("/send")
async def send_manual_notification(
    request: SendNotificationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Send a manual notification email.

    This endpoint allows administrators to send custom notifications.
    """
    # Send email in background
    background_tasks.add_task(
        _send_notification_background,
        request.recipients,
        request.notification_type,
        request.template_data,
        request.cc,
        request.bcc,
        db
    )

    return {"message": "Notification queued for sending"}


@router.post("/test")
async def test_email_configuration(
    request: EmailTestRequest
):
    """
    Test email configuration by sending a test email.

    This endpoint tests the SMTP configuration by sending a test email.
    """
    email_service = get_email_service()

    test_data = {
        "title": "Email Configuration Test",
        "message": "This is a test email to verify your SMTP configuration is working correctly.",
        "timestamp": datetime.utcnow().isoformat()
    }

    from api.app.core.email_service import EmailMessage
    message = EmailMessage(
        to=[request.test_email],
        subject="A2A World - Email Test",
        template_name="validation_notification",
        template_data=test_data
    )
    success = await email_service.send_email(message)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to send test email. Check SMTP configuration."
        )

    return {"message": "Test email sent successfully"}


@router.get("/history", response_model=List[NotificationHistoryResponse])
async def get_notification_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    notification_type: Optional[str] = None,
    status: Optional[str] = None,
    recipient_email: Optional[EmailStr] = None,
    db: Session = Depends(get_db)
):
    """
    Get notification history with filtering and pagination.

    Allows filtering by type, status, and recipient email.
    """
    query = db.query(NotificationHistory)

    if notification_type:
        query = query.filter(NotificationHistory.notification_type == notification_type)

    if status:
        query = query.filter(NotificationHistory.status == status)

    if recipient_email:
        query = query.filter(NotificationHistory.recipient_email == recipient_email)

    # Order by most recent first
    query = query.order_by(NotificationHistory.created_at.desc())

    history = query.offset(skip).limit(limit).all()
    return history


@router.get("/subscriptions/search")
async def search_subscriptions_by_email(
    email: EmailStr,
    db: Session = Depends(get_db)
):
    """
    Search for subscriptions by email address.

    Returns subscription details for a given email.
    """
    subscriptions = db.query(EmailSubscription).filter(
        EmailSubscription.email == email,
        EmailSubscription.is_active == True
    ).all()

    if not subscriptions:
        return {"subscriptions": [], "message": "No active subscriptions found"}

    return {
        "subscriptions": [
            {
                "id": str(sub.id),
                "notification_types": sub.to_dict()["notification_types"],
                "frequency": sub.notification_frequency,
                "subscribed_at": sub.subscribed_at.isoformat()
            }
            for sub in subscriptions
        ]
    }


# Background task functions

async def _send_notification_background(
    recipients: List[str],
    notification_type: str,
    template_data: Dict[str, Any],
    cc: Optional[List[str]],
    bcc: Optional[List[str]],
    db: Session
):
    """Background task to send notification emails."""
    try:
        # Send the notification
        success = await send_notification_email(recipients, notification_type, template_data)

        # Log to database
        for recipient in recipients:
            history_entry = NotificationHistory(
                notification_type=notification_type,
                subject=f"A2A World - {notification_type.replace('_', ' ').title()}",
                recipient_email=recipient,
                notification_data=template_data,
                status="sent" if success else "failed",
                sent_by_agent="api_endpoint"
            )

            if success:
                history_entry.sent_at = datetime.utcnow()
            else:
                history_entry.error_message = "Email sending failed"

            db.add(history_entry)

        db.commit()

    except Exception as e:
        # Log error but don't crash
        print(f"Error sending notification: {e}")

