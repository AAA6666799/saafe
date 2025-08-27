"""
Logging utility for the synthetic fire prediction system.

This module provides a unified logging interface for the entire system.
"""

import os
import logging
import logging.handlers
import sys
from typing import Optional, Dict, Any
import json
from datetime import datetime


class SystemLogger:
    """
    System logger class that provides a unified logging interface.
    
    This class sets up logging with console and file handlers, and provides
    methods for logging messages at different levels.
    """
    
    def __init__(self, 
                name: str, 
                log_level: str = 'INFO',
                log_dir: str = 'logs',
                log_to_console: bool = True,
                log_to_file: bool = True,
                log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        """
        Initialize the system logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_format: Log message format
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_dir = log_dir
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.log_format = log_format
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(log_format)
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Add rotating file handler for long-running processes
            rotating_log_file = os.path.join(log_dir, f"{name}.log")
            rotating_handler = logging.handlers.RotatingFileHandler(
                rotating_log_file, maxBytes=10*1024*1024, backupCount=5
            )
            rotating_handler.setFormatter(formatter)
            self.logger.addHandler(rotating_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: Message to log
            **kwargs: Additional key-value pairs to include in the log
        """
        if kwargs:
            message = f"{message} {json.dumps(kwargs)}"
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log
            **kwargs: Additional key-value pairs to include in the log
        """
        if kwargs:
            message = f"{message} {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
            **kwargs: Additional key-value pairs to include in the log
        """
        if kwargs:
            message = f"{message} {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
            exc_info: Whether to include exception information
            **kwargs: Additional key-value pairs to include in the log
        """
        if kwargs:
            message = f"{message} {json.dumps(kwargs)}"
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: Message to log
            exc_info: Whether to include exception information
            **kwargs: Additional key-value pairs to include in the log
        """
        if kwargs:
            message = f"{message} {json.dumps(kwargs)}"
        self.logger.critical(message, exc_info=exc_info)
    
    def exception(self, message: str, **kwargs) -> None:
        """
        Log an exception message.
        
        Args:
            message: Message to log
            **kwargs: Additional key-value pairs to include in the log
        """
        if kwargs:
            message = f"{message} {json.dumps(kwargs)}"
        self.logger.exception(message)


# Dictionary to store logger instances
_loggers: Dict[str, SystemLogger] = {}


def get_logger(name: str, 
              log_level: Optional[str] = None,
              log_dir: Optional[str] = None,
              log_to_console: bool = True,
              log_to_file: bool = True) -> SystemLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        log_level: Optional logging level (uses system default if not provided)
        log_dir: Optional log directory (uses system default if not provided)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        
    Returns:
        SystemLogger instance
    """
    global _loggers
    
    if name in _loggers:
        return _loggers[name]
    
    # Import here to avoid circular imports
    from src.config.base import get_config
    
    try:
        config = get_config()
        system_config = config.get('system', {})
        
        if log_level is None:
            log_level = system_config.get('log_level', 'INFO')
        
        if log_dir is None:
            log_dir = system_config.get('log_dir', 'logs')
    
    except Exception:
        # If config is not available, use defaults
        if log_level is None:
            log_level = 'INFO'
        
        if log_dir is None:
            log_dir = 'logs'
    
    logger = SystemLogger(
        name=name,
        log_level=log_level,
        log_dir=log_dir,
        log_to_console=log_to_console,
        log_to_file=log_to_file
    )
    
    _loggers[name] = logger
    return logger