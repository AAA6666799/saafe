"""
Core Utility Functions

Common utility functions used across the system.
"""

import time
import logging
import functools
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import numpy as np


def get_timestamp() -> float:
    """Get current timestamp as float"""
    return time.time()


def get_datetime_string() -> str:
    """Get current datetime as formatted string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_array_shape(array: np.ndarray, expected_shape: tuple, 
                        name: str = "array") -> None:
    """Validate numpy array shape"""
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match expected {expected_shape}")


def validate_probability(value: float, name: str = "probability") -> None:
    """Validate probability value is between 0 and 1"""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate_positive(value: float, name: str = "value") -> None:
    """Validate value is positive"""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
        
        return result
    return wrapper


def retry_decorator(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function execution on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    
            raise last_exception
        return wrapper
    return decorator


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_array(array: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Normalize array to specified range"""
    if array.size == 0:
        return array
    
    array_min = np.min(array)
    array_max = np.max(array)
    
    if array_max == array_min:
        return np.full_like(array, (min_val + max_val) / 2)
    
    normalized = (array - array_min) / (array_max - array_min)
    return normalized * (max_val - min_val) + min_val


def calculate_z_score(value: float, mean: float, std: float) -> float:
    """Calculate z-score for anomaly detection"""
    if std == 0:
        return 0.0
    return (value - mean) / std


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate moving average of data"""
    if len(data) < window_size:
        return data
    
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("ai_fire_prediction_platform")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger