"""
Utilities module for Saafe MVP
Contains helper functions, configuration management, and common utilities.
"""

from .error_handler import (
    ErrorHandler, ErrorCategory, ErrorType, ErrorSeverity, ErrorInfo,
    SafeguardError, ModelError, DataError, NotificationError, UIError
)
from .fallback_manager import FallbackManager, FallbackStrategy, FallbackResult

__all__ = [
    'ErrorHandler',
    'ErrorCategory', 
    'ErrorType',
    'ErrorSeverity',
    'ErrorInfo',
    'SafeguardError',
    'ModelError',
    'DataError', 
    'NotificationError',
    'UIError',
    'FallbackManager',
    'FallbackStrategy',
    'FallbackResult'
]