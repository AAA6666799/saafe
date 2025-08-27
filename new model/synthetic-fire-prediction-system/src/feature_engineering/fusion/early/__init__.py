"""
Early fusion techniques for the synthetic fire prediction system.

This package provides implementations of early fusion techniques that combine
features at the data level or early in the feature extraction process.
"""

from .data_level_fusion import DataLevelFusion
from .feature_concatenation import FeatureConcatenation
from .feature_averaging import FeatureAveraging
from .weighted_feature_combination import WeightedFeatureCombination

__all__ = [
    'DataLevelFusion',
    'FeatureConcatenation',
    'FeatureAveraging',
    'WeightedFeatureCombination'
]