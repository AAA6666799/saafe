"""
Feature Fusion Package.
"""

# Import the main fusion system
from .feature_fusion_system import FeatureFusionSystem

# Import fusion components
from .early import *
from .late import *
from .hybrid import *
from .selection import *

# Import enhanced fusion modules
from .cross_sensor_correlation_analyzer import CrossSensorCorrelationAnalyzer
from .cross_sensor_fusion_extractor import CrossSensorFusionExtractor

__all__ = [
    'FeatureFusionSystem',
    'CrossSensorCorrelationAnalyzer',
    'CrossSensorFusionExtractor'
]