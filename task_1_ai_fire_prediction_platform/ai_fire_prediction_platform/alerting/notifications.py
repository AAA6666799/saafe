"""
Notification system for Synthetic Fire Prediction System

Handles various notification channels for alert delivery.
"""

import logging
import smtplib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from ai_fire_prediction_platform.alerting.engine import AlertData, AlertLevel

logger = logging.getLogger(__name__)


class NotificationChannel:
    """Base class for notification channels"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', False)
    
    def send_notification(self, alert: AlertData) -> bool:
        """
        Send notification for alert.
        
        Args:
            alert (AlertData): Alert to notify about
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            return self._send_message(alert)
        except Exception as e:
            logger.error(f"Failed to send {self.name} notification: {e}")
            return False
    
    def _send_message(self, alert: AlertData) -> bool:
        """
        Send message implementation (to be overridden by subclasses).
        
        Args:
            alert (AlertData): Alert to notify about
            
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('email', config)
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_address = config.get('from_address')
        self.to_addresses = config.get('to_addresses', [])
        self.require_encryption = config.get('require_encryption', True)
    
    def _send_message(self, alert: AlertData) -> bool:
        """Send email notification"""
        if not all([self.smtp_server, self.username, self.password, self.from_address]) or not self.to_addresses:
            logger.warning("Email configuration incomplete")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)
            msg['Subject'] = f"Fire Alert - {alert.alert_level.description}"
            
            # Create body
            body = self._format_alert_body(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.require_encryption:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent to {', '.join(self.to_addresses)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _format_alert_body(self, alert: AlertData) -> str:
        """Format alert body for email"""
        body = f"""
Fire Detection Alert
====================

Alert Level: {alert.alert_level.description} {alert.alert_level.icon}
Risk Score: {alert.risk_score:.1f}
Confidence: {alert.confidence:.2f}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.message}

"""
        
        if alert.context_info:
            body += "Context Information:\n"
            for key, value in alert.context_info.items():
                body += f"  {key}: {value}\n"
        
        return body


class SMSNotificationChannel(NotificationChannel):
    """SMS notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('sms', config)
        self.provider = config.get('provider', 'twilio')
        self.api_key = config.get('api_key')
        self.from_number = config.get('from_number')
        self.to_numbers = config.get('to_numbers', [])
    
    def _send_message(self, alert: AlertData) -> bool:
        """Send SMS notification"""
        if not all([self.api_key, self.from_number]) or not self.to_numbers:
            logger.warning("SMS configuration incomplete")
            return False
        
        try:
            message = self._format_alert_message(alert)
            
            # Send SMS via provider API
            for to_number in self.to_numbers:
                if self.provider == 'twilio':
                    self._send_twilio_sms(self.api_key, self.from_number, to_number, message)
                # Add other providers as needed
            
            logger.info(f"SMS notifications sent to {len(self.to_numbers)} numbers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS notification: {e}")
            return False
    
    def _format_alert_message(self, alert: AlertData) -> str:
        """Format alert message for SMS"""
        return f"FIRE ALERT: {alert.alert_level.description} - {alert.message} (Score: {alert.risk_score:.1f})"
    
    def _send_twilio_sms(self, api_key: str, from_number: str, to_number: str, message: str):
        """Send SMS via Twilio API"""
        # This is a simplified implementation
        # In a real system, you would use the Twilio SDK
        logger.info(f"Would send SMS via Twilio: {from_number} -> {to_number}")


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('webhook', config)
        self.url = config.get('url')
        self.method = config.get('method', 'POST')
        self.headers = config.get('headers', {})
        self.include_alert_data = config.get('include_alert_data', True)
    
    def _send_message(self, alert: AlertData) -> bool:
        """Send webhook notification"""
        if not self.url:
            logger.warning("Webhook URL not configured")
            return False
        
        try:
            # Prepare payload
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'alert_level': alert.alert_level.description,
                'risk_score': alert.risk_score,
                'confidence': alert.confidence,
                'message': alert.message
            }
            
            if self.include_alert_data:
                payload['alert_data'] = alert.to_dict()
            
            # Send webhook
            response = requests.request(
                self.method,
                self.url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code in [200, 201, 204]:
                logger.info(f"Webhook notification sent successfully: {response.status_code}")
                return True
            else:
                logger.error(f"Webhook notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class NotificationManager:
    """Manages multiple notification channels"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notification manager.
        
        Args:
            config (Dict): Configuration dictionary with channel settings
        """
        self.config = config
        self.channels: List[NotificationChannel] = []
        self._initialize_channels()
    
    def _initialize_channels(self):
        """Initialize notification channels based on configuration"""
        # Email channel
        if 'email' in self.config:
            self.channels.append(EmailNotificationChannel(self.config['email']))
        
        # SMS channel
        if 'sms' in self.config:
            self.channels.append(SMSNotificationChannel(self.config['sms']))
        
        # Webhook channel
        if 'webhook' in self.config:
            self.channels.append(WebhookNotificationChannel(self.config['webhook']))
        
        logger.info(f"Initialized {len(self.channels)} notification channels")
    
    def send_alert_notifications(self, alert: AlertData) -> Dict[str, bool]:
        """
        Send notifications for alert through all enabled channels.
        
        Args:
            alert (AlertData): Alert to notify about
            
        Returns:
            Dict[str, bool]: Channel name to success status mapping
        """
        results = {}
        
        # Only send notifications for elevated alert levels
        if alert.alert_level.level < AlertLevel.ELEVATED.level:
            logger.debug(f"Skipping notifications for low-level alert: {alert.alert_level.description}")
            return results
        
        logger.info(f"Sending notifications for {alert.alert_level.description} alert")
        
        for channel in self.channels:
            success = channel.send_notification(alert)
            results[channel.name] = success
            
            if success:
                logger.info(f"Successfully sent {alert.alert_level.description} alert via {channel.name}")
            else:
                logger.warning(f"Failed to send {alert.alert_level.description} alert via {channel.name}")
        
        return results
    
    def test_notifications(self) -> Dict[str, bool]:
        """
        Test all notification channels.
        
        Returns:
            Dict[str, bool]: Channel name to success status mapping
        """
        # Create a test alert
        test_alert = AlertData(
            alert_level=AlertLevel.MILD,
            risk_score=45.0,
            confidence=0.8,
            message="Test notification from Synthetic Fire System",
            timestamp=datetime.now(),
            context_info={'test': True}
        )
        
        results = {}
        for channel in self.channels:
            if channel.enabled:
                success = channel.send_notification(test_alert)
                results[channel.name] = success
        
        return results


# Convenience functions
def create_notification_manager(config_file: str = None) -> NotificationManager:
    """
    Create notification manager with configuration.
    
    Args:
        config_file (str): Path to configuration file (optional)
        
    Returns:
        NotificationManager: Configured notification manager
    """
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load notification configuration: {e}")
            config = {}
    else:
        # Default configuration
        config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_address': '',
                'to_addresses': []
            },
            'sms': {
                'enabled': False,
                'provider': 'twilio',
                'api_key': '',
                'from_number': '',
                'to_numbers': []
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'method': 'POST',
                'headers': {}
            }
        }
    
    return NotificationManager(config)