"""
Alerting System for FLIR+SCD41 Fire Detection System Performance Monitoring.

This module implements automated alerting for performance degradation,
system health issues, and other critical events.
"""

import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class Alert:
    """Represents a system alert."""
    
    def __init__(self, alert_type: str, severity: str, message: str, 
                 timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """
        Initialize alert.
        
        Args:
            alert_type: Type of alert (e.g., 'performance_degradation', 'system_health')
            severity: Severity level ('info', 'warning', 'critical')
            message: Alert message
            timestamp: Alert timestamp
            metadata: Additional alert metadata
        """
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.alert_id = f"{alert_type}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class AlertingSystem:
    """Manages alert generation, filtering, and notification."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize alerting system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.alert_history = []
        self.alert_suppression = {}  # For preventing alert spam
        self.notification_channels = self.config.get('notification_channels', {
            'email': {'enabled': False},
            'sms': {'enabled': False},
            'webhook': {'enabled': False}
        })
        
        # Load alert history if file exists
        self.alert_history_file = self.config.get('alert_history_file', 'alert_history.json')
        self._load_alert_history()
        
        logger.info("Alerting System initialized")
    
    def _load_alert_history(self):
        """Load alert history from file."""
        if os.path.exists(self.alert_history_file):
            try:
                with open(self.alert_history_file, 'r') as f:
                    data = json.load(f)
                    for alert_dict in data.get('alerts', []):
                        alert = Alert(
                            alert_type=alert_dict['alert_type'],
                            severity=alert_dict['severity'],
                            message=alert_dict['message'],
                            timestamp=datetime.fromisoformat(alert_dict['timestamp']),
                            metadata=alert_dict.get('metadata', {})
                        )
                        alert.alert_id = alert_dict.get('alert_id', alert.alert_id)
                        self.alert_history.append(alert)
                logger.info(f"Loaded {len(self.alert_history)} historical alerts")
            except Exception as e:
                logger.warning(f"Failed to load alert history: {e}")
    
    def _save_alert_history(self):
        """Save alert history to file."""
        try:
            data = {
                'alerts': [alert.to_dict() for alert in self.alert_history],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.alert_history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save alert history: {e}")
    
    def should_suppress_alert(self, alert: Alert) -> bool:
        """
        Check if alert should be suppressed to prevent spam.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert should be suppressed, False otherwise
        """
        suppression_key = f"{alert.alert_type}_{alert.severity}"
        suppression_config = self.config.get('alert_suppression', {})
        suppression_duration = suppression_config.get(suppression_key, 300)  # 5 minutes default
        
        # Check if we have a recent alert of the same type
        for recent_alert in reversed(self.alert_history[-10:]):  # Check last 10 alerts
            if (recent_alert.alert_type == alert.alert_type and 
                recent_alert.severity == alert.severity and
                (alert.timestamp - recent_alert.timestamp).total_seconds() < suppression_duration):
                return True
        
        return False
    
    def generate_alert(self, alert_type: str, severity: str, message: str, 
                      metadata: Dict[str, Any] = None) -> Optional[Alert]:
        """
        Generate and process an alert.
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            Alert object if generated, None if suppressed
        """
        alert = Alert(alert_type, severity, message, metadata=metadata)
        
        # Check if alert should be suppressed
        if self.should_suppress_alert(alert):
            logger.info(f"Suppressing alert: {alert.alert_type} - {alert.message}")
            return None
        
        # Add to history
        self.alert_history.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Save history
        self._save_alert_history()
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.info(f"Generated alert: {alert.severity.upper()} - {alert.message}")
        return alert
    
    def _send_notifications(self, alert: Alert):
        """
        Send notifications for alert.
        
        Args:
            alert: Alert to notify about
        """
        # Send email notification
        if self.notification_channels.get('email', {}).get('enabled', False):
            self._send_email_notification(alert)
        
        # Send SMS notification
        if self.notification_channels.get('sms', {}).get('enabled', False):
            self._send_sms_notification(alert)
        
        # Send webhook notification
        if self.notification_channels.get('webhook', {}).get('enabled', False):
            self._send_webhook_notification(alert)
    
    def _send_email_notification(self, alert: Alert):
        """
        Send email notification for alert.
        
        Args:
            alert: Alert to notify about
        """
        try:
            email_config = self.notification_channels.get('email', {})
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            username = email_config.get('username')
            password = email_config.get('password')
            sender = email_config.get('sender')
            recipients = email_config.get('recipients', [])
            
            if not all([smtp_server, username, password, sender, recipients]):
                logger.warning("Email configuration incomplete")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] Fire Detection System Alert"
            
            body = f"""
Fire Detection System Alert

Type: {alert.alert_type}
Severity: {alert.severity}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Message: {alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}

This is an automated alert from the FLIR+SCD41 Fire Detection System.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_sms_notification(self, alert: Alert):
        """
        Send SMS notification for alert.
        
        Args:
            alert: Alert to notify about
        """
        try:
            sms_config = self.notification_channels.get('sms', {})
            provider = sms_config.get('provider', 'twilio')
            api_key = sms_config.get('api_key')
            from_number = sms_config.get('from_number')
            to_numbers = sms_config.get('to_numbers', [])
            
            if not all([api_key, from_number, to_numbers]):
                logger.warning("SMS configuration incomplete")
                return
            
            # Send SMS via provider API
            for to_number in to_numbers:
                if provider == 'twilio':
                    self._send_twilio_sms(api_key, from_number, to_number, alert)
                # Add other providers as needed
            
            logger.info(f"SMS notifications sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send SMS notification: {e}")
    
    def _send_twilio_sms(self, api_key: str, from_number: str, to_number: str, alert: Alert):
        """
        Send SMS via Twilio API.
        
        Args:
            api_key: Twilio API key
            from_number: From phone number
            to_number: To phone number
            alert: Alert to send
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Import twilio library
        # 2. Create Twilio client
        # 3. Send SMS message
        pass
    
    def _send_webhook_notification(self, alert: Alert):
        """
        Send webhook notification for alert.
        
        Args:
            alert: Alert to notify about
        """
        try:
            webhook_config = self.notification_channels.get('webhook', {})
            url = webhook_config.get('url')
            method = webhook_config.get('method', 'POST')
            headers = webhook_config.get('headers', {})
            payload_template = webhook_config.get('payload_template', {})
            
            if not url:
                logger.warning("Webhook URL not configured")
                return
            
            # Prepare payload
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            # Merge with template if provided
            if payload_template:
                payload.update(payload_template)
            
            # Send webhook
            if method.upper() == 'POST':
                response = requests.post(url, json=payload, headers=headers)
            elif method.upper() == 'GET':
                response = requests.get(url, params=payload, headers=headers)
            else:
                logger.warning(f"Unsupported webhook method: {method}")
                return
            
            if response.status_code >= 400:
                logger.error(f"Webhook request failed with status {response.status_code}")
            else:
                logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """
        Get recent alerts within specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alerts_by_type(self, alert_type: str) -> List[Alert]:
        """
        Get alerts by type.
        
        Args:
            alert_type: Alert type to filter by
            
        Returns:
            List of alerts of specified type
        """
        return [alert for alert in self.alert_history if alert.alert_type == alert_type]
    
    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """
        Get alerts by severity.
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of alerts of specified severity
        """
        return [alert for alert in self.alert_history if alert.severity == severity]
    
    def clear_alert_history(self):
        """Clear alert history."""
        self.alert_history.clear()
        self._save_alert_history()
        logger.info("Alert history cleared")

# Convenience functions
def create_alerting_system(config: Dict[str, Any] = None) -> AlertingSystem:
    """
    Create an alerting system instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AlertingSystem instance
    """
    return AlertingSystem(config)

def generate_performance_alert(alerting_system: AlertingSystem, 
                             performance_issue: str, 
                             current_value: float,
                             threshold: float,
                             metadata: Dict[str, Any] = None) -> Optional[Alert]:
    """
    Generate a performance-related alert.
    
    Args:
        alerting_system: Alerting system instance
        performance_issue: Description of performance issue
        current_value: Current metric value
        threshold: Threshold that was exceeded
        metadata: Additional metadata
        
    Returns:
        Alert object if generated, None if suppressed
    """
    severity = 'critical' if abs(current_value - threshold) > (threshold * 0.1) else 'warning'
    
    message = f"Performance issue detected: {performance_issue}. Current: {current_value:.4f}, Threshold: {threshold:.4f}"
    
    alert_metadata = {
        'current_value': current_value,
        'threshold': threshold,
        'difference': current_value - threshold
    }
    
    if metadata:
        alert_metadata.update(metadata)
    
    return alerting_system.generate_alert(
        alert_type='performance_degradation',
        severity=severity,
        message=message,
        metadata=alert_metadata
    )

__all__ = ['Alert', 'AlertingSystem', 'create_alerting_system', 'generate_performance_alert']