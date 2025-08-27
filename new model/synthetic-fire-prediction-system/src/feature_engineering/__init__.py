"""
Feature Engineering package for the synthetic fire prediction system.

This package provides functionality for extracting features from synthetic datasets,
managing feature extraction workflows, and storing and versioning extracted features.
"""

from .framework import FeatureExtractionFramework
from .orchestrator import FeatureExtractionOrchestrator
from .job_manager import FeatureExtractionJobManager
from .storage import FeatureStorageSystem
from .versioning_fix import FeatureVersioningSystemFixed as FeatureVersioningSystem
from .aws_integration import AWSFeatureExtractionIntegration

__all__ = [
    'FeatureExtractionFramework',
    'FeatureExtractionOrchestrator',
    'FeatureExtractionJobManager',
    'FeatureStorageSystem',
    'FeatureVersioningSystem',
    'AWSFeatureExtractionIntegration'
]