"""
Environmental data generation package for synthetic fire prediction system.

This package provides components for generating synthetic environmental data
including temperature, humidity, pressure, and VOC patterns.
"""

from .environmental_data_generator import EnvironmentalDataGenerator
from .voc_pattern_generator import VOCPatternGenerator
from .correlation_engine import CorrelationEngine
from .environmental_variation_model import EnvironmentalVariationModel

__all__ = [
    'EnvironmentalDataGenerator',
    'VOCPatternGenerator',
    'CorrelationEngine',
    'EnvironmentalVariationModel'
]