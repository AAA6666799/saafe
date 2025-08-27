"""
Custom Exception Classes

Defines system-specific exceptions for better error handling and debugging.
"""


class SyntheticFireSystemError(Exception):
    """Base exception for all system errors"""
    pass


class DataGenerationError(SyntheticFireSystemError):
    """Raised when synthetic data generation fails"""
    pass


class FeatureExtractionError(SyntheticFireSystemError):
    """Raised when feature extraction fails"""
    pass


class ModelError(SyntheticFireSystemError):
    """Raised when model operations fail"""
    pass


class AgentError(SyntheticFireSystemError):
    """Raised when agent operations fail"""
    pass


class HardwareError(SyntheticFireSystemError):
    """Raised when hardware interface operations fail"""
    pass


class ConfigurationError(SyntheticFireSystemError):
    """Raised when configuration is invalid"""
    pass


class ValidationError(SyntheticFireSystemError):
    """Raised when data validation fails"""
    pass


class SystemComponentError(SyntheticFireSystemError):
    """Raised when system component operations fail"""
    pass