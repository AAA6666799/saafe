"""
Environmental feature extractors package for the synthetic fire prediction system.

This package provides implementations of feature extractors for environmental data.
"""

# Import feature extractors
from .temperature_pattern_extractor import TemperaturePatternExtractor
from .humidity_correlation_analyzer import HumidityCorrelationAnalyzer
from .pressure_change_extractor import PressureChangeExtractor
from .environmental_trend_analyzer import EnvironmentalTrendAnalyzer
from .environmental_anomaly_detector import EnvironmentalAnomalyDetector

__all__ = [
    'TemperaturePatternExtractor',
    'HumidityCorrelationAnalyzer',
    'PressureChangeExtractor',
    'EnvironmentalTrendAnalyzer',
    'EnvironmentalAnomalyDetector'
]