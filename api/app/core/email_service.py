"""
A2A World Platform - Email Service

SMTP-based email service for @a2aworld.ai domain.
Handles email sending, template rendering, and configuration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Environment, FileSystemLoader, select_autoescape
import yaml

from ..core.config import settings


@dataclass
class EmailConfig:
    """Email configuration settings."""

    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    from_email: str
    from_name: str
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30


@dataclass
class EmailMessage:
    """Email message structure."""

    to: List[str]
    subject: str
    template_name: str
    template_data: Dict[str, Any]
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    reply_to: Optional[str] = None


class EmailService:
    """
    Async email service using SMTP for @a2aworld.ai domain.
    Handles template rendering and email sending.
    """

    def __init__(self, config: EmailConfig):
        self.config = config
        self.logger = logging.getLogger("email_service")
        self.smtp_client: Optional[aiosmtplib.SMTP] = None

        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent.parent / "templates" / "emails"
        template_dir.mkdir(parents=True, exist_ok=True)

        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Email templates mapping
        self.templates = {
            "validation_notification": "validation_notification.html",
            "pattern_discovery_alert": "pattern_discovery_alert.html",
            "consensus_update": "consensus_update.html",
            "system_maintenance": "system_maintenance.html",
            "subscription_confirmation": "subscription_confirmation.html"
        }

    async def connect(self) -> bool:
        """Establish SMTP connection."""
        try:
            self.smtp_client = aiosmtplib.SMTP(
                hostname=self.config.smtp_host,
                port=self.config.smtp_port,
                use_tls=self.config.use_tls,
                timeout=self.config.timeout
            )

            await self.smtp_client.connect()
            await self.smtp_client.login(
                self.config.smtp_username,
                self.config.smtp_password
            )

            self.logger.info(f"Connected to SMTP server: {self.config.smtp_host}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to SMTP server: {e}")
            return False

    async def disconnect(self) -> None:
        """Close SMTP connection."""
        if self.smtp_client:
            try:
                await self.smtp_client.quit()
                self.logger.info("Disconnected from SMTP server")
            except Exception as e:
                self.logger.error(f"Error disconnecting from SMTP: {e}")

    async def send_email(self, message: EmailMessage) -> bool:
        """
        Send an email using template rendering.

        Args:
            message: EmailMessage object with template data

        Returns:
            bool: Success status
        """
        if not self.smtp_client:
            if not await self.connect():
                return False

        try:
            # Render template
            html_content = self._render_template(message.template_name, message.template_data)
            text_content = self._strip_html(html_content)  # Fallback text version

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = message.subject
            msg['From'] = f"{self.config.from_name} <{self.config.from_email}>"
            msg['To'] = ', '.join(message.to)

            if message.cc:
                msg['Cc'] = ', '.join(message.cc)

            if message.reply_to:
                msg['Reply-To'] = message.reply_to

            # Attach text and HTML versions
            text_part = MIMEText(text_content, 'plain')
            html_part = MIMEText(html_content, 'html')

            msg.attach(text_part)
            msg.attach(html_part)

            # Prepare recipients
            recipients = message.to.copy()
            if message.cc:
                recipients.extend(message.cc)
            if message.bcc:
                recipients.extend(message.bcc)

            # Send email
            await self.smtp_client.send_message(msg, to_addrs=recipients)

            self.logger.info(f"Email sent successfully to {len(recipients)} recipients")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False

    async def send_bulk_email(self, messages: List[EmailMessage]) -> Dict[str, bool]:
        """
        Send multiple emails efficiently.

        Args:
            messages: List of EmailMessage objects

        Returns:
            Dict mapping template names to success status
        """
        results = {}

        for message in messages:
            success = await self.send_email(message)
            results[message.template_name] = success

            # Small delay to avoid overwhelming SMTP server
            await asyncio.sleep(0.1)

        return results

    def _render_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """Render email template with data."""
        try:
            template_file = self.templates.get(template_name)
            if not template_file:
                raise ValueError(f"Template '{template_name}' not found")

            template = self.template_env.get_template(template_file)
            return template.render(**data)

        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            # Return basic HTML fallback
            return f"""
            <html>
            <body>
                <h2>A2A World Platform Notification</h2>
                <p>{data.get('message', 'Notification content')}</p>
                <p>Timestamp: {datetime.utcnow().isoformat()}</p>
            </body>
            </html>
            """

    def _strip_html(self, html_content: str) -> str:
        """Strip HTML tags for plain text version."""
        import re
        # Simple HTML tag removal
        clean = re.compile('<.*?>')
        return re.sub(clean, '', html_content)

    async def test_connection(self) -> bool:
        """Test SMTP connection and authentication."""
        try:
            if not await self.connect():
                return False

            # Send test email to verify configuration
            test_message = EmailMessage(
                to=[self.config.smtp_username],  # Send to ourselves
                subject="A2A World Email Service Test",
                template_name="validation_notification",
                template_data={
                    "title": "Email Service Test",
                    "message": "This is a test email to verify SMTP configuration.",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            success = await self.send_email(test_message)
            await self.disconnect()
            return success

        except Exception as e:
            self.logger.error(f"SMTP test failed: {e}")
            return False


# Global email service instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get or create global email service instance."""
    global _email_service

    if _email_service is None:
        # Load configuration from settings
        config = EmailConfig(
            smtp_host=getattr(settings, 'SMTP_HOST', 'smtp.gmail.com'),
            smtp_port=getattr(settings, 'SMTP_PORT', 587),
            smtp_username=getattr(settings, 'SMTP_USERNAME', ''),
            smtp_password=getattr(settings, 'SMTP_PASSWORD', ''),
            from_email=getattr(settings, 'FROM_EMAIL', 'noreply@a2aworld.ai'),
            from_name=getattr(settings, 'FROM_NAME', 'A2A World Platform'),
            use_tls=getattr(settings, 'SMTP_USE_TLS', True),
            use_ssl=getattr(settings, 'SMTP_USE_SSL', False),
            timeout=getattr(settings, 'SMTP_TIMEOUT', 30)
        )

        _email_service = EmailService(config)

    return _email_service


async def send_notification_email(
    recipients: List[str],
    notification_type: str,
    data: Dict[str, Any]
) -> bool:
    """
    Convenience function to send notification emails.

    Args:
        recipients: List of email addresses
        notification_type: Type of notification (validation, pattern_discovery, etc.)
        data: Template data

    Returns:
        bool: Success status
    """
    service = get_email_service()

    # Map notification types to templates
    template_mapping = {
        "validation": "validation_notification",
        "pattern_discovery": "pattern_discovery_alert",
        "consensus": "consensus_update",
        "maintenance": "system_maintenance"
    }

    template_name = template_mapping.get(notification_type, "validation_notification")

    message = EmailMessage(
        to=recipients,
        subject=f"A2A World - {notification_type.replace('_', ' ').title()}",
        template_name=template_name,
        template_data=data
    )

    return await service.send_email(message)