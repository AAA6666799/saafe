"""
Feature fusion package for the synthetic fire prediction system.

This package provides components for fusing features from different sources
(thermal, gas, environmental, temporal) to create more powerful predictive features.
"""

from .feature_fusion_system import FeatureFusionSystem, FusionPipeline

__all__ = ['FeatureFusionSystem', 'FusionPipeline']