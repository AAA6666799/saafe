"""
Ensemble components for the fire prediction system.
"""

from .model_ensemble_manager import ModelEnsembleManager

def create_fire_prediction_ensemble(config):
    """Create a fire prediction ensemble."""
    return ModelEnsembleManager(config)

__all__ = ['ModelEnsembleManager', 'create_fire_prediction_ensemble']