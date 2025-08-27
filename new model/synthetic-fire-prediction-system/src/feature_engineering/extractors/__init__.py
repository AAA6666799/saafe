"""
Feature extractors package for the synthetic fire prediction system.

This package provides implementations of feature extractors for different data types,
including thermal, gas, and environmental data.
"""

# Import feature extractors
try:
    from .thermal.thermal_feature_extractor import BasicThermalFeatureExtractor
except ImportError:
    pass

__all__ = [
    'BasicThermalFeatureExtractor'
]