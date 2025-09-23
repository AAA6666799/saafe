"""
Ensemble models for synthetic fire prediction system
"""

import numpy as np
from typing import Tuple, Dict, Any, List
import os
import pickle

from synthetic_fire_system.core.interfaces import Model, PredictionResult
from synthetic_fire_system.models.baseline import RandomForestModel, LogisticRegressionModel
from synthetic_fire_system.models.temporal import LSTMTemporalModel


class EnsembleModel(Model):
    """Ensemble of multiple models for fire prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
        # Initialize models based on configuration
        if config.get('use_baseline_models', True):
            self.models['random_forest'] = RandomForestModel(config)
            self.models['logistic_regression'] = LogisticRegressionModel(config)
            self.weights['random_forest'] = 0.4
            self.weights['logistic_regression'] = 0.3
        
        if config.get('use_temporal_models', True):
            self.models['lstm'] = LSTMTemporalModel(config)
            self.weights['lstm'] = 0.3
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make ensemble prediction and return (probability, confidence)"""
        if not self.is_trained or not self.models:
            # Return neutral prediction if not trained or no models
            return 0.5, 0.0
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for model_name, model in self.models.items():
            try:
                prob, conf = model.predict(features)
                predictions[model_name] = prob
                confidences[model_name] = conf
            except Exception:
                # If a model fails, use neutral prediction
                predictions[model_name] = 0.5
                confidences[model_name] = 0.0
        
        # Weighted average of predictions
        weighted_sum = 0.0
        weight_sum = 0.0
        total_confidence = 0.0
        
        for model_name in predictions:
            weight = self.weights.get(model_name, 1.0 / len(predictions))
            weighted_sum += predictions[model_name] * weight
            weight_sum += weight
            total_confidence += confidences[model_name] * weight
        
        if weight_sum > 0:
            ensemble_probability = weighted_sum / weight_sum
            ensemble_confidence = total_confidence / weight_sum
        else:
            ensemble_probability = 0.5
            ensemble_confidence = 0.0
        
        return ensemble_probability, ensemble_confidence
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train all models in the ensemble"""
        for model_name, model in self.models.items():
            try:
                model.train(features, labels)
            except Exception as e:
                print(f"Warning: Failed to train {model_name} model: {e}")
        
        self.is_trained = any(model.is_trained for model in self.models.values())
    
    def save(self, path: str) -> None:
        """Save ensemble model to disk"""
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory:  # Only create directory if path has a directory component
            os.makedirs(directory, exist_ok=True)
        
        # Save ensemble configuration
        ensemble_data = {
            'weights': self.weights,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = f"{path}_{model_name}"
            try:
                model.save(model_path)
            except Exception as e:
                print(f"Warning: Failed to save {model_name} model: {e}")
    
    def load(self, path: str) -> None:
        """Load ensemble model from disk"""
        # Load ensemble configuration
        if os.path.exists(path):
            with open(path, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.weights = ensemble_data.get('weights', self.weights)
            self.is_trained = ensemble_data.get('is_trained', False)
        
        # Load individual models
        for model_name, model in self.models.items():
            model_path = f"{path}_{model_name}"
            try:
                model.load(model_path)
            except Exception as e:
                print(f"Warning: Failed to load {model_name} model: {e}")
        
        # Update ensemble trained status
        self.is_trained = any(model.is_trained for model in self.models.values())
    
    def get_model_votes(self, features: np.ndarray) -> Dict[str, float]:
        """Get individual model votes for explainability"""
        votes = {}
        
        if not self.is_trained:
            return votes
        
        for model_name, model in self.models.items():
            try:
                prob, _ = model.predict(features)
                votes[model_name] = prob
            except Exception:
                votes[model_name] = 0.5
        
        return votes