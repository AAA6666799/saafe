"""
Saafe MVP - AI-Powered Fire Detection System

A comprehensive fire detection system using advanced AI models,
real-time sensor data processing, and multi-channel alerting.
"""

__version__ = "1.0.0"
__author__ = "Saafe Team"
__email__ = "team@saafe.ai"

from .core.alert_engine import AlertEngine
from .core.fire_detection_pipeline import FireDetectionPipeline
from .core.data_stream import DataStreamManager
from .models.model_manager import ModelManager
from .services.notification_manager import NotificationManager

__all__ = [
    "AlertEngine",
    "FireDetectionPipeline", 
    "DataStreamManager",
    "ModelManager",
    "NotificationManager"
]