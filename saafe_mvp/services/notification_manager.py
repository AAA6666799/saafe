"""
Unified notification manager that coordinates SMS, email, and push notifications
Provides a single interface for sending alerts across all notification channels
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .sms_service import SMSService, SMSConfig, AlertType as SMSAlertType
from .email_service import EmailService, EmailConfig, AlertType as EmailAlertType
from .push_notification_service import PushNotificationService, PushConfig, AlertType as PushAlertType
from ..utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, safe_execute
from ..utils.fallback_manager import FallbackManager


class AlertLevel(Enum):
    """Unified alert levels"""
    NORMAL = "normal"
    MILD = "mild"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    TEST = "test"


@dataclass
class NotificationConfig:
    """Unified notification configuration"""
    sms_config: Optional[SMSConfig] = None
    email_config: Optional[EmailConfig] = None
    push_config: Optional[PushConfig] = None
    
    # Notification preferences
    sms_enabled: bool = True
    email_enabled: bool = True
    push_enabled: bool = True
    
    # Contact information
    phone_numbers: List[str] = None
    email_addresses: List[str] = None
    
    # Alert level preferences
    sms_min_level: AlertLevel = AlertLevel.ELEVATED
    email_min_level: AlertLevel = AlertLevel.MILD
    push_min_level: AlertLevel = AlertLevel.MILD

    def __post_init__(self):
        if self.phone_numbers is None:
            self.phone_numbers = []
        if self.email_addresses is None:
            self.email_addresses = []


@dataclass
class NotificationResult:
    """Result of notification delivery across all channels"""
    alert_level: AlertLevel
    timestamp: datetime
    sms_results: List[Any] = None
    email_results: List[Any] = None
    push_results: List[Any] = None
    total_sent: int = 0
    total_failed: int = 0

    def __post_init__(self):
        if self.sms_results is None:
            self.sms_results = []
        if self.email_results is None:
            self.email_results = []
        if self.push_results is None:
            self.push_results = []
        
        # Calculate totals
        all_results = self.sms_results + self.email_results + self.push_results
        self.total_sent = sum(1 for r in all_results if r.success)
        self.total_failed = sum(1 for r in all_results if not r.success)


class NotificationManager:
    """Unified notification manager for all alert channels"""
    
    def __init__(self, config: NotificationConfig, error_handler: ErrorHandler = None, 
                 fallback_manager: FallbackManager = None):
        """Initialize notification manager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize error handling and fallback systems
        from ..utils.error_handler import get_error_handler
        from ..utils.fallback_manager import get_fallback_manager
        self.error_handler = error_handler or get_error_handler()
        self.fallback_manager = fallback_manager or get_fallback_manager()
        
        # Initialize services with error handling
        self.sms_service = None
        self.email_service = None
        self.push_service = None
        
        if config.sms_config and config.sms_enabled:
            try:
                self.sms_service = SMSService(config.sms_config)
            except Exception as e:
                self.error_handler.handle_error(
                    e, ErrorCategory.NOTIFICATION_ERROR, "NotificationManager",
                    context={'service': 'sms_init'}
                )
            
        if config.email_config and config.email_enabled:
            try:
                self.email_service = EmailService(config.email_config)
            except Exception as e:
                self.error_handler.handle_error(
                    e, ErrorCategory.NOTIFICATION_ERROR, "NotificationManager",
                    context={'service': 'email_init'}
                )
            
        if config.push_config and config.push_enabled:
            try:
                self.push_service = PushNotificationService(config.push_config)
            except Exception as e:
                self.error_handler.handle_error(
                    e, ErrorCategory.NOTIFICATION_ERROR, "NotificationManager",
                    context={'service': 'push_init'}
                )
        
        self.logger.info("Notification manager initialized with error handling")
    
    def _convert_alert_level(self, alert_level: AlertLevel, service_type: str) -> Any:
        """Convert unified alert level to service-specific alert type"""
        mapping = {
            AlertLevel.NORMAL: "normal",
            AlertLevel.MILD: "mild", 
            AlertLevel.ELEVATED: "elevated",
            AlertLevel.CRITICAL: "critical",
            AlertLevel.TEST: "test"
        }
        
        level_str = mapping.get(alert_level, "normal")
        
        if service_type == "sms":
            return SMSAlertType(level_str)
        elif service_type == "email":
            return EmailAlertType(level_str)
        elif service_type == "push":
            return PushAlertType(level_str)
        
        return level_str
    
    def _should_send_notification(self, alert_level: AlertLevel, service_type: str) -> bool:
        """Check if notification should be sent based on minimum level settings"""
        level_hierarchy = {
            AlertLevel.NORMAL: 0,
            AlertLevel.MILD: 1,
            AlertLevel.ELEVATED: 2,
            AlertLevel.CRITICAL: 3,
            AlertLevel.TEST: 0  # Always send test notifications
        }
        
        if alert_level == AlertLevel.TEST:
            return True
        
        current_level = level_hierarchy.get(alert_level, 0)
        
        if service_type == "sms":
            min_level = level_hierarchy.get(self.config.sms_min_level, 2)
        elif service_type == "email":
            min_level = level_hierarchy.get(self.config.email_min_level, 1)
        elif service_type == "push":
            min_level = level_hierarchy.get(self.config.push_min_level, 1)
        else:
            min_level = 0
        
        return current_level >= min_level
    
    def send_alert(self, alert_level: AlertLevel, **kwargs) -> NotificationResult:
        """
        Send alert notification across all enabled channels
        
        Args:
            alert_level: Level of alert to send
            **kwargs: Additional data for notifications (risk_score, location, etc.)
            
        Returns:
            NotificationResult with delivery status across all channels
        """
        self.logger.info(f"Sending {alert_level.value} alert notification")
        
        result = NotificationResult(
            alert_level=alert_level,
            timestamp=datetime.now()
        )
        
        # Send SMS notifications with error handling
        if (self.sms_service and 
            self.sms_service.is_available() and 
            self.config.phone_numbers and
            self._should_send_notification(alert_level, "sms")):
            
            sms_results = safe_execute(
                lambda: self._send_sms_notifications(alert_level, **kwargs),
                ErrorCategory.NOTIFICATION_ERROR, "NotificationManager.send_sms",
                default_return=[],
                context={'alert_level': alert_level.value, 'phone_count': len(self.config.phone_numbers)}
            )
            result.sms_results = sms_results
            if sms_results:
                self.logger.info(f"SMS notifications sent: {len(sms_results)}")
        
        # Send email notifications with error handling
        if (self.email_service and 
            self.email_service.is_available() and 
            self.config.email_addresses and
            self._should_send_notification(alert_level, "email")):
            
            email_results = safe_execute(
                lambda: self._send_email_notifications(alert_level, **kwargs),
                ErrorCategory.NOTIFICATION_ERROR, "NotificationManager.send_email",
                default_return=[],
                context={'alert_level': alert_level.value, 'email_count': len(self.config.email_addresses)}
            )
            result.email_results = email_results
            if email_results:
                self.logger.info(f"Email notifications sent: {len(email_results)}")
        
        # Send push notifications with error handling
        if (self.push_service and 
            self.push_service.is_available() and
            self._should_send_notification(alert_level, "push")):
            
            push_results = safe_execute(
                lambda: self._send_push_notifications(alert_level, **kwargs),
                ErrorCategory.NOTIFICATION_ERROR, "NotificationManager.send_push",
                default_return=[],
                context={'alert_level': alert_level.value}
            )
            result.push_results = push_results
            if push_results:
                self.logger.info(f"Push notifications sent: {len(push_results)}")
        
        # Handle offline mode if all notifications failed
        if result.total_sent == 0 and alert_level in [AlertLevel.CRITICAL, AlertLevel.ELEVATED]:
            self.fallback_manager.offline_mode.queue_notification({
                'alert_level': alert_level.value,
                'timestamp': result.timestamp.isoformat(),
                'kwargs': kwargs
            })
        
        # Recalculate totals after all notifications
        all_results = result.sms_results + result.email_results + result.push_results
        result.total_sent = sum(1 for r in all_results if r.success)
        result.total_failed = sum(1 for r in all_results if not r.success)
        
        self.logger.info(f"Alert notification complete: {result.total_sent} sent, {result.total_failed} failed")
        return result
    
    def send_fire_alert(self, risk_score: float = None, location: str = None, message: str = None, **kwargs) -> Dict[str, Any]:
        """
        Send fire alert notification across all enabled channels
        
        Args:
            risk_score: Fire risk score (0-100)
            location: Location of detected fire
            message: Custom message for the alert
            **kwargs: Additional alert parameters
            
        Returns:
            Dict with notification results in format expected by tests
        """
        # Determine alert level based on risk score
        if risk_score is not None:
            if risk_score >= 80:
                alert_level = AlertLevel.CRITICAL
            elif risk_score >= 60:
                alert_level = AlertLevel.ELEVATED
            elif risk_score >= 30:
                alert_level = AlertLevel.MILD
            else:
                alert_level = AlertLevel.NORMAL
        else:
            alert_level = AlertLevel.CRITICAL  # Default for fire alerts
        
        # Create fire-specific message if not provided
        if message is None:
            message = f"🔥 FIRE ALERT: Risk level {risk_score:.1f}%" if risk_score else "🔥 FIRE DETECTED"
        if location:
            message += f" at {location}"
        
        # Send alert using the main send_alert method
        # Remove message from kwargs if it exists to avoid duplication
        kwargs.pop('message', None)
        result = self.send_alert(alert_level, message=message, **kwargs)
        
        # Convert to format expected by tests
        return {
            "sms": [{"success": r.success, "message_id": r.message_id} for r in result.sms_results],
            "email": [{"success": r.success, "message_id": r.message_id} for r in result.email_results],
            "push": {"success": len(result.push_results) > 0 and all(r.success for r in result.push_results), 
                    "message_id": result.push_results[0].message_id if result.push_results else None}
        }

    def send_test_notifications(self) -> NotificationResult:
        """
        Send test notifications across all enabled channels
        
        Returns:
            NotificationResult with test delivery status
        """
        return self.send_alert(AlertLevel.TEST, message="Test notification from Saafe")
    
    def add_phone_number(self, phone_number: str) -> bool:
        """Add phone number to notification list"""
        if self.sms_service:
            is_valid, formatted = self.sms_service.validate_phone_number(phone_number)
            if is_valid and formatted not in self.config.phone_numbers:
                self.config.phone_numbers.append(formatted)
                self.logger.info(f"Added phone number: {formatted}")
                return True
        return False
    
    def remove_phone_number(self, phone_number: str) -> bool:
        """Remove phone number from notification list"""
        if phone_number in self.config.phone_numbers:
            self.config.phone_numbers.remove(phone_number)
            self.logger.info(f"Removed phone number: {phone_number}")
            return True
        return False
    
    def add_email_address(self, email_address: str) -> bool:
        """Add email address to notification list"""
        if self.email_service:
            is_valid, cleaned = self.email_service.validate_email_address(email_address)
            if is_valid and cleaned not in self.config.email_addresses:
                self.config.email_addresses.append(cleaned)
                self.logger.info(f"Added email address: {cleaned}")
                return True
        return False
    
    def remove_email_address(self, email_address: str) -> bool:
        """Remove email address from notification list"""
        if email_address in self.config.email_addresses:
            self.config.email_addresses.remove(email_address)
            self.logger.info(f"Removed email address: {email_address}")
            return True
        return False
    
    def add_push_subscription(self, subscription_data: Dict) -> bool:
        """Add push notification subscription"""
        if self.push_service:
            return self.push_service.add_subscription(subscription_data)
        return False
    
    def remove_push_subscription(self, endpoint: str) -> bool:
        """Remove push notification subscription"""
        if self.push_service:
            return self.push_service.remove_subscription(endpoint)
        return False
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all notification services"""
        status = {}
        
        if self.sms_service:
            status['sms'] = self.sms_service.get_status()
        else:
            status['sms'] = {'available': False, 'enabled': False}
        
        if self.email_service:
            status['email'] = self.email_service.get_status()
        else:
            status['email'] = {'available': False, 'enabled': False}
        
        if self.push_service:
            status['push'] = self.push_service.get_status()
        else:
            status['push'] = {'available': False, 'enabled': False}
        
        return status
    
    def get_contact_info(self) -> Dict[str, List[str]]:
        """Get current contact information"""
        return {
            'phone_numbers': self.config.phone_numbers.copy(),
            'email_addresses': self.config.email_addresses.copy(),
            'push_subscriptions': len(self.push_service.subscriptions) if self.push_service else 0
        }
    
    def update_notification_preferences(self, 
                                      sms_enabled: Optional[bool] = None,
                                      email_enabled: Optional[bool] = None,
                                      push_enabled: Optional[bool] = None,
                                      sms_min_level: Optional[AlertLevel] = None,
                                      email_min_level: Optional[AlertLevel] = None,
                                      push_min_level: Optional[AlertLevel] = None):
        """Update notification preferences"""
        if sms_enabled is not None:
            self.config.sms_enabled = sms_enabled
        if email_enabled is not None:
            self.config.email_enabled = email_enabled
        if push_enabled is not None:
            self.config.push_enabled = push_enabled
        if sms_min_level is not None:
            self.config.sms_min_level = sms_min_level
        if email_min_level is not None:
            self.config.email_min_level = email_min_level
        if push_min_level is not None:
            self.config.push_min_level = push_min_level
        
        self.logger.info("Notification preferences updated")
    
    def _send_sms_notifications(self, alert_level: AlertLevel, **kwargs):
        """Send SMS notifications with error handling."""
        sms_alert_type = self._convert_alert_level(alert_level, "sms")
        return self.sms_service.send_alert_sms(
            self.config.phone_numbers, 
            sms_alert_type, 
            **kwargs
        )
    
    def _send_email_notifications(self, alert_level: AlertLevel, **kwargs):
        """Send email notifications with error handling."""
        email_alert_type = self._convert_alert_level(alert_level, "email")
        return self.email_service.send_alert_email(
            self.config.email_addresses, 
            email_alert_type, 
            **kwargs
        )
    
    def _send_push_notifications(self, alert_level: AlertLevel, **kwargs):
        """Send push notifications with error handling."""
        push_alert_type = self._convert_alert_level(alert_level, "push")
        return self.push_service.send_alert_push(push_alert_type, **kwargs)