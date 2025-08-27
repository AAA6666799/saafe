"""
Synthetic Fire Prediction System.

A comprehensive AI-powered fire detection and prediction system that combines
synthetic data generation, advanced machine learning, multi-agent coordination,
and hardware abstraction for robust fire safety applications.
"""

# Import main integrated system
from .integrated_system import IntegratedFireDetectionSystem, create_integrated_fire_system
from .system import create_system

# Import ML components
from .ml import (
    # Base interfaces
    FireModel,
    FireClassificationModel,
    FireIdentificationModel,
    FireProgressionModel,
    ConfidenceEstimationModel,
    
    # Classification models
    BinaryFireClassifier,
    MultiClassFireClassifier,
    EnsembleFireClassifier,
    DeepLearningFireClassifier,
    
    # Training pipeline
    ModelTrainingPipeline
)

# Import major subsystems (with optional imports)
try:
    from .hardware import SensorManager, create_sensor_manager
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

try:
    from .agents.coordination import MultiAgentFireDetectionSystem, create_multi_agent_fire_system
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

try:
    from .ml.ensemble import ModelEnsembleManager, create_fire_prediction_ensemble
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Saafe AI Team"

__all__ = [
    # Main integrated system
    'IntegratedFireDetectionSystem',
    'create_integrated_fire_system', 
    'create_system',
    
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

# Add optional components if available
if HARDWARE_AVAILABLE:
    __all__.extend(['SensorManager', 'create_sensor_manager'])

if AGENTS_AVAILABLE:
    __all__.extend(['MultiAgentFireDetectionSystem', 'create_multi_agent_fire_system'])

if ENSEMBLE_AVAILABLE:
    __all__.extend(['ModelEnsembleManager', 'create_fire_prediction_ensemble'])