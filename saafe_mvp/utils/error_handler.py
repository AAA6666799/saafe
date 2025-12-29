"""
Comprehensive error handling system for Saafe MVP.

This module provides centralized error handling, user-friendly error messages,
logging system, and graceful degradation mechanisms.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import json


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    MODEL_ERROR = "model_error"
    DATA_ERROR = "data_error"
    NOTIFICATION_ERROR = "notification_error"
    UI_ERROR = "ui_error"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"


# Alias for backward compatibility
ErrorType = ErrorCategory


@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    recovery_suggestions: List[str]
    timestamp: datetime
    component: str
    traceback_info: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    resolved: bool = False


class SafeguardError(Exception):
    """Base exception class for Saafe MVP errors."""
    
    def __init__(self, message: str, error_type: ErrorCategory = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type or ErrorCategory.SYSTEM_ERROR
        self.context = context or {}


class ModelError(SafeguardError):
    """Error related to AI model operations."""
    
    def __init__(self, message: str, model_id: str = None, **kwargs):
        super().__init__(message, ErrorCategory.MODEL_ERROR, kwargs)
        self.model_id = model_id


class DataError(SafeguardError):
    """Error related to data processing or validation."""
    
    def __init__(self, message: str, data_source: str = None, **kwargs):
        super().__init__(message, ErrorCategory.DATA_ERROR, kwargs)
        self.data_source = data_source


class NotificationError(SafeguardError):
    """Error related to notification services."""
    
    def __init__(self, message: str, service: str = None, **kwargs):
        super().__init__(message, ErrorCategory.NOTIFICATION_ERROR, kwargs)
        self.service = service


class UIError(SafeguardError):
    """Error related to user interface operations."""
    
    def __init__(self, message: str, component: str = None, **kwargs):
        super().__init__(message, ErrorCategory.UI_ERROR, kwargs)
        self.component = component


class ErrorHandler:
    """
    Centralized error handling system with logging and recovery suggestions.
    """
    
    def __init__(self, log_file: Optional[Path] = None, log_level: int = logging.INFO):
        """
        Initialize error handler.
        
        Args:
            log_file (Path): Path to log file (uses default if None)
            log_level (int): Logging level
        """
        self.log_file = log_file or Path("logs/safeguard_errors.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.error_count_by_category: Dict[ErrorCategory, int] = {}
        self.error_handlers: Dict[ErrorCategory, Callable] = {}
        
        # Recovery mechanisms
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        
        # Initialize default error handlers
        self._setup_default_handlers()
        
        self.logger.info("ErrorHandler initialized")
    
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Setup comprehensive logging system."""
        logger = logging.getLogger('safeguard_error_handler')
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_default_handlers(self):
        """Setup default error handlers for each category."""
        self.error_handlers = {
            ErrorCategory.MODEL_ERROR: self._handle_model_error,
            ErrorCategory.DATA_ERROR: self._handle_data_error,
            ErrorCategory.NOTIFICATION_ERROR: self._handle_notification_error,
            ErrorCategory.UI_ERROR: self._handle_ui_error,
            ErrorCategory.SYSTEM_ERROR: self._handle_system_error,
            ErrorCategory.CONFIGURATION_ERROR: self._handle_configuration_error,
            ErrorCategory.NETWORK_ERROR: self._handle_network_error,
            ErrorCategory.VALIDATION_ERROR: self._handle_validation_error,
        }
    
    def handle_error(self, 
                    exception: Exception,
                    category: ErrorCategory,
                    component: str,
                    context: Optional[Dict[str, Any]] = None,
                    severity: Optional[ErrorSeverity] = None) -> ErrorInfo:
        """
        Handle an error with comprehensive logging and recovery suggestions.
        
        Args:
            exception (Exception): The exception that occurred
            category (ErrorCategory): Category of the error
            component (str): Component where error occurred
            context (Dict): Additional context information
            severity (ErrorSeverity): Error severity (auto-detected if None)
            
        Returns:
            ErrorInfo: Comprehensive error information
        """
        # Auto-detect severity if not provided
        if severity is None:
            severity = self._detect_severity(exception, category)
        
        # Generate error ID
        error_id = f"{category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Get user-friendly message and recovery suggestions
        user_message, recovery_suggestions = self._get_user_friendly_info(exception, category)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(exception),
            user_message=user_message,
            recovery_suggestions=recovery_suggestions,
            timestamp=datetime.now(),
            component=component,
            traceback_info=traceback.format_exc(),
            context=context or {}
        )
        
        # Log the error
        self._log_error(error_info)
        
        # Track error
        self.error_history.append(error_info)
        self.error_count_by_category[category] = self.error_count_by_category.get(category, 0) + 1
        
        # Execute category-specific handler
        if category in self.error_handlers:
            try:
                self.error_handlers[category](error_info)
            except Exception as handler_error:
                self.logger.error(f"Error handler failed: {handler_error}")
        
        # Attempt recovery if critical
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._attempt_recovery(error_info)
        
        return error_info
    
    def _detect_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Auto-detect error severity based on exception type and category."""
        # Critical errors
        if isinstance(exception, (SystemExit, KeyboardInterrupt, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(exception, (RuntimeError, OSError, ImportError)):
            return ErrorSeverity.HIGH
        
        # Category-specific severity
        if category == ErrorCategory.MODEL_ERROR:
            if "CUDA" in str(exception) or "GPU" in str(exception):
                return ErrorSeverity.MEDIUM  # Can fallback to CPU
            return ErrorSeverity.HIGH
        
        if category == ErrorCategory.SYSTEM_ERROR:
            return ErrorSeverity.HIGH
        
        if category == ErrorCategory.NOTIFICATION_ERROR:
            return ErrorSeverity.MEDIUM  # System can continue without notifications
        
        if category == ErrorCategory.DATA_ERROR:
            return ErrorSeverity.MEDIUM
        
        # Default to medium
        return ErrorSeverity.MEDIUM
    
    def _get_user_friendly_info(self, exception: Exception, category: ErrorCategory) -> tuple:
        """Get user-friendly error message and recovery suggestions."""
        error_messages = {
            ErrorCategory.MODEL_ERROR: {
                "message": "AI model encountered an issue",
                "suggestions": [
                    "The system will attempt to use a backup model",
                    "Check if model files are present and not corrupted",
                    "Restart the application if the issue persists",
                    "Contact support if the problem continues"
                ]
            },
            ErrorCategory.DATA_ERROR: {
                "message": "Data processing encountered an issue",
                "suggestions": [
                    "The system will use cached data patterns",
                    "Check sensor data for unusual values",
                    "Restart the data generation if needed",
                    "Verify system time and date settings"
                ]
            },
            ErrorCategory.NOTIFICATION_ERROR: {
                "message": "Notification service is temporarily unavailable",
                "suggestions": [
                    "Check your internet connection",
                    "Verify notification settings and credentials",
                    "The system will continue monitoring without notifications",
                    "Try testing notifications in settings"
                ]
            },
            ErrorCategory.UI_ERROR: {
                "message": "User interface encountered a display issue",
                "suggestions": [
                    "Refresh the page or restart the application",
                    "Check your browser compatibility",
                    "Clear browser cache if using web interface",
                    "Try reducing the update frequency in settings"
                ]
            },
            ErrorCategory.SYSTEM_ERROR: {
                "message": "System encountered an unexpected error",
                "suggestions": [
                    "Restart the application",
                    "Check available memory and disk space",
                    "Verify system permissions",
                    "Contact technical support if the issue persists"
                ]
            },
            ErrorCategory.CONFIGURATION_ERROR: {
                "message": "Configuration issue detected",
                "suggestions": [
                    "Check configuration file format",
                    "Verify all required settings are present",
                    "Reset to default configuration if needed",
                    "Refer to documentation for correct format"
                ]
            },
            ErrorCategory.NETWORK_ERROR: {
                "message": "Network connection issue",
                "suggestions": [
                    "Check your internet connection",
                    "Verify firewall settings",
                    "The system will operate in offline mode",
                    "Retry the operation when connection is restored"
                ]
            },
            ErrorCategory.VALIDATION_ERROR: {
                "message": "Data validation failed",
                "suggestions": [
                    "Check input data format",
                    "Verify data is within expected ranges",
                    "The system will use default values",
                    "Correct the input and try again"
                ]
            }
        }
        
        # Get category-specific info
        info = error_messages.get(category, {
            "message": "An unexpected error occurred",
            "suggestions": [
                "Restart the application",
                "Check system logs for more details",
                "Contact support if the issue persists"
            ]
        })
        
        # Add specific error details if helpful
        error_str = str(exception).lower()
        if "cuda" in error_str or "gpu" in error_str:
            info["suggestions"].insert(0, "GPU issue detected - system will use CPU processing")
        elif "memory" in error_str:
            info["suggestions"].insert(0, "Memory issue detected - try closing other applications")
        elif "permission" in error_str:
            info["suggestions"].insert(0, "Permission issue - check file and folder permissions")
        
        return info["message"], info["suggestions"]
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_message = (
            f"[{error_info.error_id}] {error_info.category.value.upper()} in {error_info.component}: "
            f"{error_info.message}"
        )
        
        if error_info.context:
            log_message += f" | Context: {error_info.context}"
        
        # Log based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            if error_info.traceback_info:
                self.logger.critical(f"Traceback:\n{error_info.traceback_info}")
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            if error_info.traceback_info:
                self.logger.error(f"Traceback:\n{error_info.traceback_info}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _attempt_recovery(self, error_info: ErrorInfo):
        """Attempt automatic recovery for critical errors."""
        if error_info.category in self.recovery_strategies:
            for recovery_func in self.recovery_strategies[error_info.category]:
                try:
                    success = recovery_func(error_info)
                    if success:
                        error_info.resolved = True
                        self.logger.info(f"Recovery successful for error {error_info.error_id}")
                        break
                except Exception as recovery_error:
                    self.logger.error(f"Recovery attempt failed: {recovery_error}")
    
    # Default error handlers
    def _handle_model_error(self, error_info: ErrorInfo):
        """Handle model-related errors."""
        self.logger.warning(f"Model error in {error_info.component}: {error_info.message}")
        # Model manager should handle fallback automatically
    
    def _handle_data_error(self, error_info: ErrorInfo):
        """Handle data-related errors."""
        self.logger.warning(f"Data error in {error_info.component}: {error_info.message}")
        # Data generators should have fallback patterns
    
    def _handle_notification_error(self, error_info: ErrorInfo):
        """Handle notification-related errors."""
        self.logger.warning(f"Notification error in {error_info.component}: {error_info.message}")
        # System should continue without notifications
    
    def _handle_ui_error(self, error_info: ErrorInfo):
        """Handle UI-related errors."""
        self.logger.warning(f"UI error in {error_info.component}: {error_info.message}")
        # UI should gracefully degrade
    
    def _handle_system_error(self, error_info: ErrorInfo):
        """Handle system-related errors."""
        self.logger.error(f"System error in {error_info.component}: {error_info.message}")
        # May require application restart
    
    def _handle_configuration_error(self, error_info: ErrorInfo):
        """Handle configuration-related errors."""
        self.logger.error(f"Configuration error in {error_info.component}: {error_info.message}")
        # Should use default configuration
    
    def _handle_network_error(self, error_info: ErrorInfo):
        """Handle network-related errors."""
        self.logger.warning(f"Network error in {error_info.component}: {error_info.message}")
        # System should work offline
    
    def _handle_validation_error(self, error_info: ErrorInfo):
        """Handle validation-related errors."""
        self.logger.warning(f"Validation error in {error_info.component}: {error_info.message}")
        # Should use default/safe values
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        return {
            'total_errors': len(self.error_history),
            'errors_by_category': dict(self.error_count_by_category),
            'recent_errors': [
                {
                    'id': error.error_id,
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'component': error.component,
                    'message': error.user_message,
                    'timestamp': error.timestamp.isoformat(),
                    'resolved': error.resolved
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def get_recovery_suggestions(self, error_id: str) -> List[str]:
        """Get recovery suggestions for a specific error."""
        for error in self.error_history:
            if error.error_id == error_id:
                return error.recovery_suggestions
        return []
    
    def mark_error_resolved(self, error_id: str) -> bool:
        """Mark an error as resolved."""
        for error in self.error_history:
            if error.error_id == error_id:
                error.resolved = True
                self.logger.info(f"Error {error_id} marked as resolved")
                return True
        return False
    
    def clear_error_history(self):
        """Clear error history (keep recent critical errors)."""
        critical_errors = [
            error for error in self.error_history 
            if error.severity == ErrorSeverity.CRITICAL and not error.resolved
        ]
        self.error_history = critical_errors
        self.error_count_by_category.clear()
        self.logger.info("Error history cleared")
    
    def export_error_log(self, file_path: Path) -> bool:
        """Export error log to file."""
        try:
            error_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_errors': len(self.error_history),
                'errors_by_category': {k.value: v for k, v in self.error_count_by_category.items()},
                'errors': [
                    {
                        'id': error.error_id,
                        'category': error.category.value,
                        'severity': error.severity.value,
                        'message': error.message,
                        'user_message': error.user_message,
                        'component': error.component,
                        'timestamp': error.timestamp.isoformat(),
                        'context': error.context,
                        'resolved': error.resolved,
                        'recovery_suggestions': error.recovery_suggestions
                    }
                    for error in self.error_history
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(error_data, f, indent=2)
            
            self.logger.info(f"Error log exported to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export error log: {e}")
            return False


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(exception: Exception,
                category: ErrorCategory,
                component: str,
                context: Optional[Dict[str, Any]] = None,
                severity: Optional[ErrorSeverity] = None) -> ErrorInfo:
    """Convenience function to handle errors using global handler."""
    return get_error_handler().handle_error(exception, category, component, context, severity)


def safe_execute(func: Callable, 
                category: ErrorCategory,
                component: str,
                default_return: Any = None,
                context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func (Callable): Function to execute
        category (ErrorCategory): Error category if function fails
        component (str): Component name
        default_return (Any): Default return value on error
        context (Dict): Additional context
        
    Returns:
        Any: Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        handle_error(e, category, component, context)
        return default_return


# Decorator for automatic error handling
def error_handler(category: ErrorCategory, component: str = None):
    """
    Decorator for automatic error handling.
    
    Args:
        category (ErrorCategory): Error category
        component (str): Component name (uses function name if None)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            comp_name = component or func.__name__
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(e, category, comp_name)
                return None
        return wrapper
    return decorator