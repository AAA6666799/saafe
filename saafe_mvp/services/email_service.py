"""
Email notification service with SMTP support
Handles email delivery with HTML templates and fallback mechanisms
"""

import smtplib
import ssl
import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from enum import Enum


class AlertType(Enum):
    """Alert types for email templates"""
    NORMAL = "normal"
    MILD = "mild"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    TEST = "test"


@dataclass
class EmailConfig:
    """Email service configuration"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_email: str
    from_name: str = "Saafe Fire Detection"
    use_tls: bool = True
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0  # Initial delay in seconds


@dataclass
class EmailResult:
    """Result of email sending attempt"""
    success: bool
    email_address: str
    error_message: Optional[str] = None
    attempts: int = 1
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EmailService:
    """Email notification service with SMTP support"""
    
    def __init__(self, config: EmailConfig):
        """Initialize email service with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if config.enabled:
            self.logger.info("Email service initialized")
        else:
            self.logger.info("Email service disabled")
    
    def validate_email_address(self, email: str) -> Tuple[bool, str]:
        """
        Validate email address format
        
        Args:
            email: Email address to validate
            
        Returns:
            Tuple of (is_valid, cleaned_email)
        """
        if not email:
            return False, ""
        
        # Clean the email
        cleaned_email = email.strip().lower()
        
        # Basic email validation pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(pattern, cleaned_email):
            return True, cleaned_email
        
        return False, ""
    
    def get_html_template(self, alert_type: AlertType, **kwargs) -> str:
        """
        Get HTML email template for alert type
        
        Args:
            alert_type: Type of alert
            **kwargs: Template variables
            
        Returns:
            HTML email content
        """
        # Common CSS styles
        styles = """
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { padding: 20px; text-align: center; color: white; }
            .content { padding: 30px; }
            .footer { padding: 20px; background-color: #f8f9fa; text-align: center; font-size: 12px; color: #666; }
            .alert-info { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }
            .timestamp { color: #666; font-size: 14px; }
            .risk-score { font-size: 24px; font-weight: bold; margin: 10px 0; }
            .details { margin: 15px 0; }
        </style>
        """
        
        # Template variables with defaults
        default_kwargs = {
            'risk_score': kwargs.get('risk_score', 'N/A'),
            'details': kwargs.get('details', 'No additional details available'),
            'location': kwargs.get('location', 'Unknown location'),
            'timestamp': kwargs.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'confidence': kwargs.get('confidence', 'N/A'),
            'processing_time': kwargs.get('processing_time', 'N/A')
        }
        format_kwargs = {**default_kwargs, **kwargs}
        
        if alert_type == AlertType.CRITICAL:
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>🔴 FIRE EMERGENCY DETECTED</title>
                {styles}
            </head>
            <body>
                <div class="container">
                    <div class="header" style="background-color: #dc3545;">
                        <h1>🔥 FIRE EMERGENCY DETECTED</h1>
                    </div>
                    <div class="content">
                        <h2 style="color: #dc3545;">IMMEDIATE ACTION REQUIRED</h2>
                        <div class="alert-info" style="border-left: 4px solid #dc3545;">
                            <div class="risk-score" style="color: #dc3545;">Risk Score: {format_kwargs['risk_score']}</div>
                            <div class="timestamp">Detected at: {format_kwargs['timestamp']}</div>
                            <div class="details">Location: {format_kwargs['location']}</div>
                        </div>
                        <h3>Alert Details:</h3>
                        <p>{format_kwargs['details']}</p>
                        <div class="alert-info">
                            <strong>System Information:</strong><br>
                            Model Confidence: {format_kwargs['confidence']}<br>
                            Processing Time: {format_kwargs['processing_time']}ms
                        </div>
                        <p style="color: #dc3545; font-weight: bold;">
                            Please verify the situation immediately and take appropriate action.
                        </p>
                    </div>
                    <div class="footer">
                        <p>This is an automated alert from Saafe Fire Detection System</p>
                        <p>If this is a false alarm, please check your system configuration</p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        elif alert_type == AlertType.ELEVATED:
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>🟠 Elevated Fire Risk Detected</title>
                {styles}
            </head>
            <body>
                <div class="container">
                    <div class="header" style="background-color: #fd7e14;">
                        <h1>🟠 Elevated Fire Risk</h1>
                    </div>
                    <div class="content">
                        <h2 style="color: #fd7e14;">Elevated Risk Detected</h2>
                        <div class="alert-info" style="border-left: 4px solid #fd7e14;">
                            <div class="risk-score" style="color: #fd7e14;">Risk Score: {format_kwargs['risk_score']}</div>
                            <div class="timestamp">Detected at: {format_kwargs['timestamp']}</div>
                            <div class="details">Location: {format_kwargs['location']}</div>
                        </div>
                        <h3>Alert Details:</h3>
                        <p>{format_kwargs['details']}</p>
                        <div class="alert-info">
                            <strong>System Information:</strong><br>
                            Model Confidence: {format_kwargs['confidence']}<br>
                            Processing Time: {format_kwargs['processing_time']}ms
                        </div>
                        <p>Please monitor the situation closely.</p>
                    </div>
                    <div class="footer">
                        <p>This is an automated alert from Saafe Fire Detection System</p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        elif alert_type == AlertType.MILD:
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>🟡 Mild Anomaly Detected</title>
                {styles}
            </head>
            <body>
                <div class="container">
                    <div class="header" style="background-color: #ffc107;">
                        <h1 style="color: #000;">🟡 Mild Anomaly</h1>
                    </div>
                    <div class="content">
                        <h2 style="color: #ffc107;">Mild Anomaly Detected</h2>
                        <div class="alert-info" style="border-left: 4px solid #ffc107;">
                            <div class="risk-score" style="color: #ffc107;">Risk Score: {format_kwargs['risk_score']}</div>
                            <div class="timestamp">Detected at: {format_kwargs['timestamp']}</div>
                            <div class="details">Location: {format_kwargs['location']}</div>
                        </div>
                        <h3>Alert Details:</h3>
                        <p>{format_kwargs['details']}</p>
                        <div class="alert-info">
                            <strong>System Information:</strong><br>
                            Model Confidence: {format_kwargs['confidence']}<br>
                            Processing Time: {format_kwargs['processing_time']}ms
                        </div>
                    </div>
                    <div class="footer">
                        <p>This is an automated alert from Saafe Fire Detection System</p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        elif alert_type == AlertType.TEST:
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>📧 Email Service Test</title>
                {styles}
            </head>
            <body>
                <div class="container">
                    <div class="header" style="background-color: #17a2b8;">
                        <h1>📧 Email Service Test</h1>
                    </div>
                    <div class="content">
                        <h2 style="color: #17a2b8;">Email Service Working Correctly</h2>
                        <div class="alert-info" style="border-left: 4px solid #17a2b8;">
                            <div class="timestamp">Test sent at: {format_kwargs['timestamp']}</div>
                        </div>
                        <p>This is a test email to verify that your Saafe email notification service is configured correctly.</p>
                        <p>If you received this email, your email notifications are working properly.</p>
                    </div>
                    <div class="footer">
                        <p>This is a test message from Saafe Fire Detection System</p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        else:  # NORMAL
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>🟢 System Normal</title>
                {styles}
            </head>
            <body>
                <div class="container">
                    <div class="header" style="background-color: #28a745;">
                        <h1>🟢 All Systems Normal</h1>
                    </div>
                    <div class="content">
                        <h2 style="color: #28a745;">System Status: Normal</h2>
                        <div class="alert-info" style="border-left: 4px solid #28a745;">
                            <div class="risk-score" style="color: #28a745;">Risk Score: {format_kwargs['risk_score']}</div>
                            <div class="timestamp">Status at: {format_kwargs['timestamp']}</div>
                            <div class="details">Location: {format_kwargs['location']}</div>
                        </div>
                        <p>All fire detection systems are operating normally.</p>
                    </div>
                    <div class="footer">
                        <p>This is an automated status update from Saafe Fire Detection System</p>
                    </div>
                </div>
            </body>
            </html>
            """
    
    def get_subject_line(self, alert_type: AlertType, **kwargs) -> str:
        """Get email subject line for alert type"""
        subjects = {
            AlertType.CRITICAL: "🔴 FIRE EMERGENCY DETECTED - Immediate Action Required",
            AlertType.ELEVATED: "🟠 Elevated Fire Risk Detected - Monitor Situation",
            AlertType.MILD: "🟡 Mild Anomaly Detected - Saafe Alert",
            AlertType.NORMAL: "🟢 Saafe Status: All Systems Normal",
            AlertType.TEST: "📧 Saafe Email Service Test"
        }
        
        subject = subjects.get(alert_type, "Saafe Fire Detection Alert")
        
        # Add location if provided
        if kwargs.get('location'):
            subject += f" - {kwargs['location']}"
        
        return subject
    
    def send_email_with_retry(self, to_email: str, subject: str, html_content: str) -> EmailResult:
        """
        Send email with retry logic
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email content
            
        Returns:
            EmailResult with delivery status
        """
        if not self.config.enabled:
            return EmailResult(
                success=False,
                email_address=to_email,
                error_message="Email service is disabled"
            )
        
        last_error = None
        delay = self.config.retry_delay
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info(f"Sending email to {to_email} (attempt {attempt})")
                
                # Create message
                msg = MIMEMultipart('alternative')
                msg['Subject'] = subject
                msg['From'] = f"{self.config.from_name} <{self.config.from_email}>"
                msg['To'] = to_email
                
                # Add HTML content
                html_part = MIMEText(html_content, 'html')
                msg.attach(html_part)
                
                # Create SMTP connection
                if self.config.use_tls:
                    context = ssl.create_default_context()
                    server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                    server.starttls(context=context)
                else:
                    server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                
                # Login and send
                if self.config.username and self.config.password:
                    server.login(self.config.username, self.config.password)
                
                server.send_message(msg)
                server.quit()
                
                self.logger.info(f"Email sent successfully to {to_email}")
                return EmailResult(
                    success=True,
                    email_address=to_email,
                    attempts=attempt
                )
                
            except smtplib.SMTPException as e:
                last_error = str(e)
                self.logger.warning(f"SMTP error on attempt {attempt}: {e}")
                
                # Don't retry for authentication errors
                if "authentication" in str(e).lower() or "login" in str(e).lower():
                    break
                    
                if attempt < self.config.max_retries:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Unexpected error sending email: {e}")
                break
        
        return EmailResult(
            success=False,
            email_address=to_email,
            error_message=last_error,
            attempts=self.config.max_retries
        )
    
    def send_alert_email(self, email_addresses: List[str], alert_type: AlertType, **kwargs) -> List[EmailResult]:
        """
        Send alert email to multiple recipients
        
        Args:
            email_addresses: List of email addresses
            alert_type: Type of alert
            **kwargs: Template variables
            
        Returns:
            List of EmailResult objects
        """
        if not self.config.enabled:
            self.logger.info("Email service is disabled")
            return []
        
        results = []
        subject = self.get_subject_line(alert_type, **kwargs)
        html_content = self.get_html_template(alert_type, **kwargs)
        
        for email_address in email_addresses:
            # Validate email address
            is_valid, cleaned_email = self.validate_email_address(email_address)
            
            if not is_valid:
                self.logger.warning(f"Invalid email address: {email_address}")
                results.append(EmailResult(
                    success=False,
                    email_address=email_address,
                    error_message="Invalid email address format"
                ))
                continue
            
            # Send email
            result = self.send_email_with_retry(cleaned_email, subject, html_content)
            results.append(result)
        
        return results
    
    def send_test_email(self, email_address: str) -> EmailResult:
        """
        Send test email to verify service configuration
        
        Args:
            email_address: Email address to test
            
        Returns:
            EmailResult with test status
        """
        is_valid, cleaned_email = self.validate_email_address(email_address)
        
        if not is_valid:
            return EmailResult(
                success=False,
                email_address=email_address,
                error_message="Invalid email address format"
            )
        
        subject = self.get_subject_line(AlertType.TEST)
        html_content = self.get_html_template(AlertType.TEST)
        
        return self.send_email_with_retry(cleaned_email, subject, html_content)
    
    def is_available(self) -> bool:
        """Check if email service is available and configured"""
        return (self.config.enabled and 
                bool(self.config.smtp_server) and 
                bool(self.config.from_email))
    
    def get_status(self) -> Dict[str, any]:
        """Get email service status information"""
        return {
            'available': self.is_available(),
            'enabled': self.config.enabled,
            'configured': bool(self.config.smtp_server and self.config.from_email),
            'smtp_server': self.config.smtp_server if self.config.smtp_server else 'Not configured',
            'from_email': self.config.from_email if self.config.from_email else 'Not configured',
            'use_tls': self.config.use_tls
        }