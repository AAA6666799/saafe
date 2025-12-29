"""
Decision-making agents for the synthetic fire prediction system.

This package contains agents responsible for making decisions about fire detection,
classification, alert generation, and response recommendations.
"""

from .fire_detection import FireDetectionAgent
from .fire_classification import FireClassificationAgent
from .alert_generation import AlertGenerationAgent, Alert
from .response_recommendation import ResponseRecommendationAgent, ResponseAction, ResponsePlan

__all__ = [
    'FireDetectionAgent',
    'FireClassificationAgent',
    'AlertGenerationAgent',
    'Alert',
    'ResponseRecommendationAgent',
    'ResponseAction',
    'ResponsePlan'
]