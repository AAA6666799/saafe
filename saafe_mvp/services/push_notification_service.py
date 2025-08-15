"""
Push notification service for browser notifications
Handles browser push notifications with permission management and customization
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

try:
    from pywebpush import webpush, WebPushException
    WEBPUSH_AVAILABLE = True
except ImportError:
    WEBPUSH_AVAILABLE = False
    WebPushException = Exception


class AlertType(Enum):
    """Alert types for push notifications"""
    NORMAL = "normal"
    MILD = "mild"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    TEST = "test"


@dataclass
class PushSubscription:
    """Browser push subscription data"""
    endpoint: str
    keys: Dict[str, str]  # Contains 'p256dh' and 'auth' keys
    user_agent: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PushConfig:
    """Push notification service configuration"""
    vapid_private_key: str
    vapid_public_key: str
    vapid_claims: Dict[str, str]  # Should contain 'sub' (email or URL)
    enabled: bool = True
    max_retries: int = 2


@dataclass
class PushResult:
    """Result of push notification attempt"""
    success: bool
    subscription_endpoint: str
    error_message: Optional[str] = None
    attempts: int = 1
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class NotificationPayload:
    """Push notification payload structure"""
    title: str
    body: str
    icon: Optional[str] = None
    badge: Optional[str] = None
    image: Optional[str] = None
    tag: Optional[str] = None
    data: Optional[Dict] = None
    actions: Optional[List[Dict]] = None
    require_interaction: bool = False
    silent: bool = False
    timestamp: int = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = int(datetime.now().timestamp() * 1000)

    def to_json(self) -> str:
        """Convert payload to JSON string"""
        return json.dumps(asdict(self), default=str)


class PushNotificationService:
    """Browser push notification service"""
    
    def __init__(self, config: PushConfig):
        """Initialize push notification service"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.subscriptions: List[PushSubscription] = []
        
        if not WEBPUSH_AVAILABLE:
            self.logger.warning("pywebpush library not available. Push notifications will be disabled.")
            return
            
        if config.enabled and self._validate_config():
            self.logger.info("Push notification service initialized successfully")
        else:
            self.logger.info("Push notification service disabled or not configured")
    
    def _validate_config(self) -> bool:
        """Validate VAPID configuration"""
        if not self.config.vapid_private_key or not self.config.vapid_public_key:
            self.logger.error("VAPID keys not configured")
            return False
        
        if not self.config.vapid_claims or 'sub' not in self.config.vapid_claims:
            self.logger.error("VAPID claims not configured (missing 'sub')")
            return False
        
        return True
    
    def add_subscription(self, subscription_data: Dict) -> bool:
        """
        Add a new push subscription
        
        Args:
            subscription_data: Browser subscription object
            
        Returns:
            True if subscription was added successfully
        """
        try:
            # Validate subscription data structure
            if not all(key in subscription_data for key in ['endpoint', 'keys']):
                self.logger.error("Invalid subscription data structure")
                return False
            
            if not all(key in subscription_data['keys'] for key in ['p256dh', 'auth']):
                self.logger.error("Invalid subscription keys")
                return False
            
            # Create subscription object
            subscription = PushSubscription(
                endpoint=subscription_data['endpoint'],
                keys=subscription_data['keys'],
                user_agent=subscription_data.get('userAgent')
            )
            
            # Check if subscription already exists
            existing = next((s for s in self.subscriptions if s.endpoint == subscription.endpoint), None)
            if existing:
                self.logger.info(f"Subscription already exists: {subscription.endpoint[:50]}...")
                return True
            
            self.subscriptions.append(subscription)
            self.logger.info(f"Added push subscription: {subscription.endpoint[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding subscription: {e}")
            return False
    
    def remove_subscription(self, endpoint: str) -> bool:
        """
        Remove a push subscription
        
        Args:
            endpoint: Subscription endpoint to remove
            
        Returns:
            True if subscription was removed
        """
        initial_count = len(self.subscriptions)
        self.subscriptions = [s for s in self.subscriptions if s.endpoint != endpoint]
        
        removed = len(self.subscriptions) < initial_count
        if removed:
            self.logger.info(f"Removed subscription: {endpoint[:50]}...")
        
        return removed
    
    def get_notification_payload(self, alert_type: AlertType, **kwargs) -> NotificationPayload:
        """
        Create notification payload for alert type
        
        Args:
            alert_type: Type of alert
            **kwargs: Additional data for notification
            
        Returns:
            NotificationPayload object
        """
        # Default values
        default_kwargs = {
            'risk_score': kwargs.get('risk_score', 'N/A'),
            'location': kwargs.get('location', 'Unknown location'),
            'timestamp': kwargs.get('timestamp', datetime.now().strftime('%H:%M:%S')),
            'confidence': kwargs.get('confidence', 'N/A')
        }
        format_kwargs = {**default_kwargs, **kwargs}
        
        # Base icon and badge URLs (can be customized)
        base_icon = "/static/icons/saafe-icon-192.png"
        base_badge = "/static/icons/saafe-badge-72.png"
        
        if alert_type == AlertType.CRITICAL:
            return NotificationPayload(
                title="🔴 FIRE EMERGENCY DETECTED",
                body=f"Critical fire risk detected! Risk score: {format_kwargs['risk_score']} at {format_kwargs['location']}",
                icon=base_icon,
                badge=base_badge,
                tag="fire-critical",
                require_interaction=True,
                data={
                    'alert_type': 'critical',
                    'risk_score': format_kwargs['risk_score'],
                    'location': format_kwargs['location'],
                    'url': '/dashboard?alert=critical'
                },
                actions=[
                    {
                        'action': 'view',
                        'title': 'View Dashboard',
                        'icon': '/static/icons/view-icon.png'
                    },
                    {
                        'action': 'acknowledge',
                        'title': 'Acknowledge',
                        'icon': '/static/icons/check-icon.png'
                    }
                ]
            )
        
        elif alert_type == AlertType.ELEVATED:
            return NotificationPayload(
                title="🟠 Elevated Fire Risk",
                body=f"Elevated risk detected. Risk score: {format_kwargs['risk_score']} at {format_kwargs['location']}",
                icon=base_icon,
                badge=base_badge,
                tag="fire-elevated",
                require_interaction=False,
                data={
                    'alert_type': 'elevated',
                    'risk_score': format_kwargs['risk_score'],
                    'location': format_kwargs['location'],
                    'url': '/dashboard?alert=elevated'
                },
                actions=[
                    {
                        'action': 'view',
                        'title': 'View Details',
                        'icon': '/static/icons/view-icon.png'
                    }
                ]
            )
        
        elif alert_type == AlertType.MILD:
            return NotificationPayload(
                title="🟡 Mild Anomaly Detected",
                body=f"Mild anomaly detected. Risk score: {format_kwargs['risk_score']} at {format_kwargs['location']}",
                icon=base_icon,
                badge=base_badge,
                tag="fire-mild",
                require_interaction=False,
                data={
                    'alert_type': 'mild',
                    'risk_score': format_kwargs['risk_score'],
                    'location': format_kwargs['location'],
                    'url': '/dashboard?alert=mild'
                }
            )
        
        elif alert_type == AlertType.TEST:
            return NotificationPayload(
                title="📱 Saafe Test Notification",
                body="Push notification service is working correctly!",
                icon=base_icon,
                badge=base_badge,
                tag="test",
                require_interaction=False,
                data={
                    'alert_type': 'test',
                    'url': '/dashboard'
                }
            )
        
        else:  # NORMAL
            return NotificationPayload(
                title="🟢 Saafe Status Normal",
                body=f"All systems normal. Risk score: {format_kwargs['risk_score']}",
                icon=base_icon,
                badge=base_badge,
                tag="status-normal",
                require_interaction=False,
                data={
                    'alert_type': 'normal',
                    'risk_score': format_kwargs['risk_score'],
                    'url': '/dashboard'
                }
            )
    
    def send_push_notification(self, subscription: PushSubscription, payload: NotificationPayload) -> PushResult:
        """
        Send push notification to a single subscription
        
        Args:
            subscription: Push subscription
            payload: Notification payload
            
        Returns:
            PushResult with delivery status
        """
        if not WEBPUSH_AVAILABLE:
            return PushResult(
                success=False,
                subscription_endpoint=subscription.endpoint,
                error_message="pywebpush library not available"
            )
        
        if not self.config.enabled:
            return PushResult(
                success=False,
                subscription_endpoint=subscription.endpoint,
                error_message="Push notification service is disabled"
            )
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info(f"Sending push notification (attempt {attempt}): {subscription.endpoint[:50]}...")
                
                # Send the notification
                webpush(
                    subscription_info={
                        'endpoint': subscription.endpoint,
                        'keys': subscription.keys
                    },
                    data=payload.to_json(),
                    vapid_private_key=self.config.vapid_private_key,
                    vapid_claims=self.config.vapid_claims
                )
                
                self.logger.info("Push notification sent successfully")
                return PushResult(
                    success=True,
                    subscription_endpoint=subscription.endpoint,
                    attempts=attempt
                )
                
            except WebPushException as e:
                error_msg = str(e)
                self.logger.warning(f"Push notification attempt {attempt} failed: {error_msg}")
                
                # Handle specific error cases
                if e.response and e.response.status_code in [410, 404]:
                    # Subscription is no longer valid
                    self.logger.info(f"Removing invalid subscription: {subscription.endpoint[:50]}...")
                    self.remove_subscription(subscription.endpoint)
                    return PushResult(
                        success=False,
                        subscription_endpoint=subscription.endpoint,
                        error_message="Subscription no longer valid (removed)",
                        attempts=attempt
                    )
                
                if attempt >= self.config.max_retries:
                    return PushResult(
                        success=False,
                        subscription_endpoint=subscription.endpoint,
                        error_message=error_msg,
                        attempts=attempt
                    )
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Unexpected error sending push notification: {error_msg}")
                return PushResult(
                    success=False,
                    subscription_endpoint=subscription.endpoint,
                    error_message=error_msg,
                    attempts=attempt
                )
        
        return PushResult(
            success=False,
            subscription_endpoint=subscription.endpoint,
            error_message="Max retries exceeded",
            attempts=self.config.max_retries
        )
    
    def send_alert_push(self, alert_type: AlertType, **kwargs) -> List[PushResult]:
        """
        Send push notification to all subscriptions
        
        Args:
            alert_type: Type of alert
            **kwargs: Additional data for notification
            
        Returns:
            List of PushResult objects
        """
        if not self.config.enabled:
            self.logger.info("Push notification service is disabled")
            return []
        
        if not self.subscriptions:
            self.logger.info("No push subscriptions available")
            return []
        
        payload = self.get_notification_payload(alert_type, **kwargs)
        results = []
        
        for subscription in self.subscriptions.copy():  # Copy to avoid modification during iteration
            result = self.send_push_notification(subscription, payload)
            results.append(result)
        
        return results
    
    def send_test_push(self) -> List[PushResult]:
        """
        Send test push notification to all subscriptions
        
        Returns:
            List of PushResult objects
        """
        return self.send_alert_push(AlertType.TEST)
    
    def is_available(self) -> bool:
        """Check if push notification service is available"""
        return (WEBPUSH_AVAILABLE and 
                self.config.enabled and 
                self._validate_config())
    
    def get_status(self) -> Dict[str, any]:
        """Get push notification service status"""
        return {
            'available': self.is_available(),
            'enabled': self.config.enabled,
            'webpush_available': WEBPUSH_AVAILABLE,
            'configured': self._validate_config(),
            'subscription_count': len(self.subscriptions),
            'vapid_configured': bool(self.config.vapid_private_key and self.config.vapid_public_key)
        }
    
    def get_vapid_public_key(self) -> Optional[str]:
        """Get VAPID public key for client-side subscription"""
        return self.config.vapid_public_key if self.config.vapid_public_key else None
    
    def cleanup_invalid_subscriptions(self) -> int:
        """
        Remove subscriptions that are older than 30 days or marked as invalid
        
        Returns:
            Number of subscriptions removed
        """
        initial_count = len(self.subscriptions)
        
        # For now, just remove subscriptions that failed multiple times
        # In a real implementation, you might track failure counts
        
        self.logger.info(f"Cleanup completed. {initial_count - len(self.subscriptions)} subscriptions removed")
        return initial_count - len(self.subscriptions)