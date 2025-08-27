"""
Machine Learning components for the fire prediction system.

This package contains various machine learning components for fire prediction:
- Base interfaces for fire models
- Model implementations (classification, identification, progression, confidence)
- Training pipeline for model training and evaluation
"""

from .base import (
    FireModel,
    FireClassificationModel,
    FireIdentificationModel,
    FireProgressionModel,
    ConfidenceEstimationModel
)

from .models import (
    BinaryFireClassifier,
    MultiClassFireClassifier,
    EnsembleFireClassifier,
    DeepLearningFireClassifier
)

from .training.pipeline import ModelTrainingPipeline

__all__ = [
    # Base interfaces
    'FireModel',
    'FireClassificationModel',
    'FireIdentificationModel',
    'FireProgressionModel',
    'ConfidenceEstimationModel',
    
    # Classification models
    'BinaryFireClassifier',
    'MultiClassFireClassifier',
    'EnsembleFireClassifier',
    'DeepLearningFireClassifier',
    
    # Training pipeline
    'ModelTrainingPipeline'
]