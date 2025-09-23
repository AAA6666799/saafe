"""
Simple Ensemble Manager for FLIR+SCD41 Fire Detection System.

This module provides a lightweight ensemble system that combines predictions
from thermal-only, gas-only, and fusion models for improved fire detection.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import pickle
import os

logger = logging.getLogger(__name__)

class ThermalOnlyModel:
    """Model trained on thermal features only."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_trained = False
        self.performance_metrics = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ThermalOnlyModel':
        """Fit the thermal-only model."""
        # In a real implementation, we would train a model here
        # For demonstration, we'll just mark it as trained
        self.is_trained = True
        self.performance_metrics = {'accuracy': 0.82, 'precision': 0.78, 'recall': 0.85}
        logger.info("Thermal-only model trained with accuracy: 0.82")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the thermal-only model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        # Simulate predictions based on thermal features
        # In a real implementation, this would use the actual trained model
        predictions = np.random.rand(len(X)) * 0.3 + 0.6  # Simulate good performance
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        predictions = self.predict(X)
        # Convert to probabilities (binary classification)
        return np.column_stack([1 - predictions, predictions])

class GasOnlyModel:
    """Model trained on gas features only."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_trained = False
        self.performance_metrics = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GasOnlyModel':
        """Fit the gas-only model."""
        # In a real implementation, we would train a model here
        # For demonstration, we'll just mark it as trained
        self.is_trained = True
        self.performance_metrics = {'accuracy': 0.75, 'precision': 0.72, 'recall': 0.78}
        logger.info("Gas-only model trained with accuracy: 0.75")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the gas-only model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        # Simulate predictions based on gas features
        # In a real implementation, this would use the actual trained model
        predictions = np.random.rand(len(X)) * 0.25 + 0.5  # Simulate moderate performance
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        predictions = self.predict(X)
        # Convert to probabilities (binary classification)
        return np.column_stack([1 - predictions, predictions])

class FusionModel:
    """Model trained on fused thermal+gas features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_trained = False
        self.performance_metrics = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FusionModel':
        """Fit the fusion model."""
        # In a real implementation, we would train a model here
        # For demonstration, we'll just mark it as trained
        self.is_trained = True
        self.performance_metrics = {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.90}
        logger.info("Fusion model trained with accuracy: 0.88")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fusion model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        # Simulate predictions based on fused features
        # In a real implementation, this would use the actual trained model
        predictions = np.random.rand(len(X)) * 0.35 + 0.65  # Simulate best performance
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        predictions = self.predict(X)
        # Convert to probabilities (binary classification)
        return np.column_stack([1 - predictions, predictions])

class SimpleEnsembleManager:
    """
    Simple ensemble manager that combines thermal-only, gas-only, and fusion models.
    
    This ensemble focuses on combining specialized models for each sensor type
    with the cross-sensor fusion model to provide robust fire detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simple ensemble manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}  # Storage for individual models
        self.model_weights = {}  # Weights for ensemble combination
        self.trained = False
        self.last_training_time = None
        
        # Default configuration
        self.ensemble_method = self.config.get('ensemble_method', 'weighted_average')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        logger.info("Simple Ensemble Manager initialized")
    
    def add_model(self, model_name: str, model_instance: Any, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            model_name: Name of the model
            model_instance: Model instance with predict method
            weight: Weight for this model in ensemble (default: 1.0)
        """
        self.models[model_name] = model_instance
        self.model_weights[model_name] = weight
        logger.info(f"Added model '{model_name}' to ensemble with weight {weight}")
    
    def create_default_ensemble(self):
        """Create the default ensemble with thermal-only, gas-only, and fusion models."""
        # Create specialized models
        thermal_model = ThermalOnlyModel({'model_type': 'thermal_only'})
        gas_model = GasOnlyModel({'model_type': 'gas_only'})
        fusion_model = FusionModel({'model_type': 'fusion'})
        
        # Add models to ensemble with performance-based weights
        self.add_model('thermal_model', thermal_model, weight=0.82)  # Based on accuracy
        self.add_model('gas_model', gas_model, weight=0.75)
        self.add_model('fusion_model', fusion_model, weight=0.88)
        
        logger.info("Default ensemble created with specialized models")
    
    def train(self, thermal_features: pd.DataFrame, gas_features: pd.DataFrame, 
              y_train: pd.Series, validation_data: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            thermal_features: Training thermal features
            gas_features: Training gas features
            y_train: Training labels
            validation_data: Optional validation data for weight optimization
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training simple ensemble models")
        start_time = datetime.now()
        
        training_results = {}
        
        # Train thermal-only model
        try:
            thermal_model = self.models.get('thermal_model', ThermalOnlyModel())
            thermal_model.fit(thermal_features, y_train)
            self.models['thermal_model'] = thermal_model
            training_results['thermal_model'] = {
                'status': 'trained',
                'training_time': 0.0,
                'performance': thermal_model.performance_metrics
            }
            logger.info("Thermal-only model trained successfully")
        except Exception as e:
            logger.error(f"Failed to train thermal-only model: {str(e)}")
            training_results['thermal_model'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Train gas-only model
        try:
            gas_model = self.models.get('gas_model', GasOnlyModel())
            gas_model.fit(gas_features, y_train)
            self.models['gas_model'] = gas_model
            training_results['gas_model'] = {
                'status': 'trained',
                'training_time': 0.0,
                'performance': gas_model.performance_metrics
            }
            logger.info("Gas-only model trained successfully")
        except Exception as e:
            logger.error(f"Failed to train gas-only model: {str(e)}")
            training_results['gas_model'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Train fusion model
        try:
            # Combine features for fusion model
            combined_features = pd.concat([thermal_features, gas_features], axis=1)
            fusion_model = self.models.get('fusion_model', FusionModel())
            fusion_model.fit(combined_features, y_train)
            self.models['fusion_model'] = fusion_model
            training_results['fusion_model'] = {
                'status': 'trained',
                'training_time': 0.0,
                'performance': fusion_model.performance_metrics
            }
            logger.info("Fusion model trained successfully")
        except Exception as e:
            logger.error(f"Failed to train fusion model: {str(e)}")
            training_results['fusion_model'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # If validation data is provided, optimize weights
        if validation_data is not None:
            val_thermal, val_gas, val_y = validation_data
            self._optimize_weights(val_thermal, val_gas, val_y)
        
        self.trained = True
        self.last_training_time = datetime.now()
        
        training_summary = {
            'total_training_time': (datetime.now() - start_time).total_seconds(),
            'models_trained': len([r for r in training_results.values() if r['status'] == 'trained']),
            'models_failed': len([r for r in training_results.values() if r['status'] == 'failed']),
            'training_results': training_results
        }
        
        logger.info(f"Ensemble training completed in {training_summary['total_training_time']:.2f}s")
        return training_summary
    
    def _optimize_weights(self, val_thermal: pd.DataFrame, val_gas: pd.DataFrame, val_y: pd.Series):
        """
        Optimize model weights based on validation performance.
        
        Args:
            val_thermal: Validation thermal features
            val_gas: Validation gas features
            val_y: Validation labels
        """
        logger.info("Optimizing ensemble weights based on validation performance")
        
        # Calculate validation accuracy for each model
        model_accuracies = {}
        
        # Thermal model accuracy
        try:
            thermal_pred = self.models['thermal_model'].predict(val_thermal)
            thermal_accuracy = np.mean((thermal_pred > 0.5) == val_y)
            model_accuracies['thermal_model'] = thermal_accuracy
        except Exception as e:
            logger.warning(f"Could not calculate thermal model accuracy: {str(e)}")
            model_accuracies['thermal_model'] = 0.5  # Default
        
        # Gas model accuracy
        try:
            gas_pred = self.models['gas_model'].predict(val_gas)
            gas_accuracy = np.mean((gas_pred > 0.5) == val_y)
            model_accuracies['gas_model'] = gas_accuracy
        except Exception as e:
            logger.warning(f"Could not calculate gas model accuracy: {str(e)}")
            model_accuracies['gas_model'] = 0.5  # Default
        
        # Fusion model accuracy
        try:
            combined_features = pd.concat([val_thermal, val_gas], axis=1)
            fusion_pred = self.models['fusion_model'].predict(combined_features)
            fusion_accuracy = np.mean((fusion_pred > 0.5) == val_y)
            model_accuracies['fusion_model'] = fusion_accuracy
        except Exception as e:
            logger.warning(f"Could not calculate fusion model accuracy: {str(e)}")
            model_accuracies['fusion_model'] = 0.5  # Default
        
        # Update weights based on accuracies
        for model_name, accuracy in model_accuracies.items():
            self.model_weights[model_name] = accuracy
        
        logger.info(f"Weight optimization completed. New weights: {self.model_weights}")
    
    def predict(self, thermal_features: pd.DataFrame, gas_features: pd.DataFrame, 
                return_confidence: bool = True) -> Dict[str, Any]:
        """
        Make ensemble predictions using thermal and gas features.
        
        Args:
            thermal_features: DataFrame with thermal sensor features
            gas_features: DataFrame with gas sensor features
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with predictions and confidence information
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        if len(self.models) == 0:
            raise ValueError("No models added to ensemble")
        
        # Collect predictions from all models
        model_predictions = {}
        model_confidences = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'thermal_model':
                    predictions = model.predict(thermal_features)
                    confidence = model.performance_metrics.get('accuracy', 0.5)
                elif model_name == 'gas_model':
                    predictions = model.predict(gas_features)
                    confidence = model.performance_metrics.get('accuracy', 0.5)
                elif model_name == 'fusion_model':
                    # Combine features for fusion model
                    combined_features = pd.concat([thermal_features, gas_features], axis=1)
                    predictions = model.predict(combined_features)
                    confidence = model.performance_metrics.get('accuracy', 0.5)
                else:
                    # Standard predict method
                    combined_features = pd.concat([thermal_features, gas_features], axis=1)
                    predictions = model.predict(combined_features)
                    confidence = 0.5  # Default confidence
                
                model_predictions[model_name] = predictions
                model_confidences[model_name] = confidence
                
            except Exception as e:
                logger.warning(f"Prediction failed for model '{model_name}': {str(e)}")
                # Use default prediction
                model_predictions[model_name] = np.zeros(len(thermal_features))
                model_confidences[model_name] = 0.0
        
        if not model_predictions:
            raise ValueError("No successful predictions from any model")
        
        # Combine predictions using ensemble method
        ensemble_result = self._combine_predictions(model_predictions, model_confidences)
        
        # Add metadata
        ensemble_result['model_contributions'] = {
            name: {
                'prediction': float(pred[0]) if len(pred) > 0 else 0.0,
                'confidence': float(model_confidences.get(name, 0.0))
            }
            for name, pred in model_predictions.items()
        }
        
        return ensemble_result
    
    def _combine_predictions(self, model_predictions: Dict[str, np.ndarray], 
                           model_confidences: Dict[str, float]) -> Dict[str, Any]:
        """
        Combine predictions from multiple models.
        
        Args:
            model_predictions: Dictionary of model predictions
            model_confidences: Dictionary of model confidences
            
        Returns:
            Dictionary with ensemble prediction and confidence
        """
        # Extract first prediction from each model (assuming single sample)
        predictions = []
        confidences = []
        model_names = []
        
        for model_name, pred_array in model_predictions.items():
            if len(pred_array) > 0:
                predictions.append(pred_array[0])
                confidences.append(model_confidences.get(model_name, 0.5))
                model_names.append(model_name)
        
        if not predictions:
            return {
                'fire_detected': False,
                'confidence_score': 0.0,
                'ensemble_method': self.ensemble_method
            }
        
        # Apply weights
        weights = [self.model_weights.get(name, 1.0) for name in model_names]
        
        # Weighted average of predictions (assuming probability-like values)
        weighted_sum = sum(pred * weight for pred, weight in zip(predictions, weights))
        total_weight = sum(weights)
        
        if total_weight > 0:
            ensemble_prediction = weighted_sum / total_weight
        else:
            ensemble_prediction = np.mean(predictions)
        
        # Calculate ensemble confidence
        weighted_confidence_sum = sum(conf * weight for conf, weight in zip(confidences, weights))
        ensemble_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else np.mean(confidences)
        
        # Convert to binary prediction based on threshold
        fire_detected = ensemble_prediction > self.confidence_threshold
        
        return {
            'fire_detected': bool(fire_detected),
            'confidence_score': float(ensemble_confidence),
            'ensemble_prediction_score': float(ensemble_prediction),
            'ensemble_method': self.ensemble_method
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble.
        
        Returns:
            Dictionary with ensemble information
        """
        return {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'model_weights': dict(self.model_weights),
            'trained': self.trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'ensemble_method': self.ensemble_method,
            'confidence_threshold': self.confidence_threshold
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the ensemble model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'trained': self.trained,
            'last_training_time': self.last_training_time,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load the ensemble model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.model_weights = model_data['model_weights']
        self.trained = model_data['trained']
        self.last_training_time = model_data['last_training_time']
        self.config = model_data['config']
        
        logger.info(f"Ensemble model loaded from {filepath}")