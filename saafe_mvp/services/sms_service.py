"""
SMS notification service using Twilio API
Handles SMS delivery with retry logic and message templating
"""

import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    Client = None
    TwilioException = Exception


class AlertType(Enum):
    """Alert types for SMS templates"""
    NORMAL = "normal"
    MILD = "mild"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    TEST = "test"


@dataclass
class SMSConfig:
    """SMS service configuration"""
    account_sid: str
    auth_token: str
    from_number: str
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0  # Initial delay in seconds


@dataclass
class SMSResult:
    """Result of SMS sending attempt"""
    success: bool
    message_sid: Optional[str] = None
    error_message: Optional[str] = None
    phone_number: str = ""
    attempts: int = 1
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SMSService:
    """SMS notification service with Twilio integration"""
    
    def __init__(self, config: SMSConfig):
        """Initialize SMS service with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        
        if not TWILIO_AVAILABLE:
            self.logger.warning("Twilio library not available. SMS service will be disabled.")
            return
            
        if config.enabled and config.account_sid and config.auth_token:
            try:
                self.client = Client(config.account_sid, config.auth_token)
                self.logger.info("SMS service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Twilio client: {e}")
                self.client = None
        else:
            self.logger.info("SMS service disabled or not configured")
    
    def validate_phone_number(self, phone_number: str) -> Tuple[bool, str]:
        """
        Validate and format phone number
        
        Args:
            phone_number: Raw phone number string
            
        Returns:
            Tuple of (is_valid, formatted_number)
        """
        if not phone_number:
            return False, ""
        
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone_number)
        
        # Check if it's a valid length
        if len(digits_only) == 10:
            # US number without country code
            formatted = f"+1{digits_only}"
        elif len(digits_only) == 11 and digits_only.startswith('1'):
            # US number with country code
            formatted = f"+{digits_only}"
        elif len(digits_only) >= 10 and len(digits_only) <= 15:
            # International number
            formatted = f"+{digits_only}"
        else:
            return False, ""
        
        # Basic validation pattern for international numbers
        pattern = r'^\+[1-9]\d{1,14}$'
        if re.match(pattern, formatted):
            return True, formatted
        
        return False, ""
    
    def get_message_template(self, alert_type: AlertType, **kwargs) -> str:
        """
        Get SMS message template for alert type
        
        Args:
            alert_type: Type of alert
            **kwargs: Template variables
            
        Returns:
            Formatted SMS message
        """
        templates = {
            AlertType.NORMAL: "🟢 Saafe: All systems normal. Risk score: {risk_score}",
            AlertType.MILD: "🟡 Saafe: Mild anomaly detected. Risk score: {risk_score}. {details}",
            AlertType.ELEVATED: "🟠 Saafe: Elevated risk detected. Risk score: {risk_score}. {details}",
            AlertType.CRITICAL: "🔴 SAAFE ALERT: FIRE EMERGENCY DETECTED! Risk score: {risk_score}. Location: {location}. Time: {timestamp}",
            AlertType.TEST: "📱 Saafe: Test notification. SMS service is working correctly."
        }
        
        template = templates.get(alert_type, "Saafe notification: {message}")
        
        # Set default values
        default_kwargs = {
            'risk_score': kwargs.get('risk_score', 'N/A'),
            'details': kwargs.get('details', 'No additional details'),
            'location': kwargs.get('location', 'Unknown'),
            'timestamp': kwargs.get('timestamp', datetime.now().strftime('%H:%M:%S')),
            'message': kwargs.get('message', 'System notification')
        }
        
        # Merge with provided kwargs
        format_kwargs = {**default_kwargs, **kwargs}
        
        try:
            return template.format(**format_kwargs)
        except KeyError as e:
            self.logger.warning(f"Missing template variable {e}, using fallback")
            return f"Saafe notification: {format_kwargs.get('message', 'System alert')}"
    
    def send_sms_with_retry(self, phone_number: str, message: str) -> SMSResult:
        """
        Send SMS with exponential backoff retry logic
        
        Args:
            phone_number: Validated phone number
            message: SMS message content
            
        Returns:
            SMSResult with delivery status
        """
        if not self.client:
            return SMSResult(
                success=False,
                error_message="SMS service not available",
                phone_number=phone_number
            )
        
        last_error = None
        delay = self.config.retry_delay
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info(f"Sending SMS to {phone_number} (attempt {attempt})")
                
                message_obj = self.client.messages.create(
                    body=message,
                    from_=self.config.from_number,
                    to=phone_number
                )
                
                self.logger.info(f"SMS sent successfully. SID: {message_obj.sid}")
                return SMSResult(
                    success=True,
                    message_sid=message_obj.sid,
                    phone_number=phone_number,
                    attempts=attempt
                )
                
            except TwilioException as e:
                last_error = str(e)
                self.logger.warning(f"SMS attempt {attempt} failed: {e}")
                
                # Don't retry for certain error types
                if "invalid" in str(e).lower() or "unsubscribed" in str(e).lower():
                    break
                    
                if attempt < self.config.max_retries:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Unexpected error sending SMS: {e}")
                break
        
        return SMSResult(
            success=False,
            error_message=last_error,
            phone_number=phone_number,
            attempts=self.config.max_retries
        )
    
    def send_alert_sms(self, phone_numbers: List[str], alert_type: AlertType, **kwargs) -> List[SMSResult]:
        """
        Send alert SMS to multiple phone numbers
        
        Args:
            phone_numbers: List of phone numbers
            alert_type: Type of alert
            **kwargs: Template variables
            
        Returns:
            List of SMSResult objects
        """
        if not self.config.enabled:
            self.logger.info("SMS service is disabled")
            return []
        
        results = []
        message = self.get_message_template(alert_type, **kwargs)
        
        for phone_number in phone_numbers:
            # Validate phone number
            is_valid, formatted_number = self.validate_phone_number(phone_number)
            
            if not is_valid:
                self.logger.warning(f"Invalid phone number: {phone_number}")
                results.append(SMSResult(
                    success=False,
                    error_message="Invalid phone number format",
                    phone_number=phone_number
                ))
                continue
            
            # Send SMS
            result = self.send_sms_with_retry(formatted_number, message)
            results.append(result)
        
        return results
    
    def send_test_sms(self, phone_number: str) -> SMSResult:
        """
        Send test SMS to verify service configuration
        
        Args:
            phone_number: Phone number to test
            
        Returns:
            SMSResult with test status
        """
        is_valid, formatted_number = self.validate_phone_number(phone_number)
        
        if not is_valid:
            return SMSResult(
                success=False,
                error_message="Invalid phone number format",
                phone_number=phone_number
            )
        
        message = self.get_message_template(AlertType.TEST)
        return self.send_sms_with_retry(formatted_number, message)
    
    def is_available(self) -> bool:
        """Check if SMS service is available and configured"""
        return self.client is not None and self.config.enabled
    
    def get_status(self) -> Dict[str, any]:
        """Get SMS service status information"""
        return {
            'available': self.is_available(),
            'enabled': self.config.enabled,
            'twilio_available': TWILIO_AVAILABLE,
            'configured': bool(self.config.account_sid and self.config.auth_token),
            'from_number': self.config.from_number if self.config.from_number else 'Not configured'
        }