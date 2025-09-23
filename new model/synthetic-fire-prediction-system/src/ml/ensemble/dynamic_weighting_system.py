"""
Dynamic Weighting System for FLIR+SCD41 Fire Detection Ensemble.

This module implements an adaptive ensemble system that dynamically adjusts
model weights based on environmental conditions, confidence levels, and recent performance.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
import json

from .simple_ensemble_manager import SimpleEnsembleManager, ThermalOnlyModel, GasOnlyModel, FusionModel

logger = logging.getLogger(__name__)


class DynamicWeightingSystem(SimpleEnsembleManager):
    """
    Dynamic ensemble system that adapts weights based on environmental conditions,
    model confidence, and recent performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dynamic weighting system.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Dynamic weighting configuration
        self.weighting_strategy = self.config.get('weighting_strategy', 'performance_adaptive')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        self.performance_window = self.config.get('performance_window', 100)
        self.environmental_adaptation = self.config.get('environmental_adaptation', True)
        
        # Performance tracking
        self.model_performance_history = {}
        self.prediction_confidence_history = {}
        self.environmental_conditions_history = deque(maxlen=self.performance_window)
        self.weight_history = deque(maxlen=self.performance_window)
        
        # Environmental condition thresholds
        self.environmental_thresholds = {
            'high_temperature': 50.0,  # °C
            'high_co2': 1000.0,        # ppm
            'low_light': 10.0,         # Not used but kept for future expansion
            'high_humidity': 80.0      # %
        }
        
        logger.info("Dynamic Weighting System initialized")
        logger.info(f"Weighting strategy: {self.weighting_strategy}")
    
    def create_default_ensemble(self):
        """Create the default ensemble with thermal-only, gas-only, and fusion models."""
        # Create specialized models
        thermal_model = ThermalOnlyModel({'model_type': 'thermal_only'})
        gas_model = GasOnlyModel({'model_type': 'gas_only'})
        fusion_model = FusionModel({'model_type': 'fusion'})
        
        # Add models to ensemble with initial equal weights
        self.add_model('thermal_model', thermal_model, weight=1.0)
        self.add_model('gas_model', gas_model, weight=1.0)
        self.add_model('fusion_model', fusion_model, weight=1.0)
        
        logger.info("Default dynamic ensemble created with specialized models")
    
    def train(self, thermal_features: pd.DataFrame, gas_features: pd.DataFrame, 
              y_train: pd.Series, validation_data: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble with dynamic weighting initialization.
        
        Args:
            thermal_features: Training thermal features
            gas_features: Training gas features
            y_train: Training labels
            validation_data: Optional validation data for weight optimization
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training dynamic ensemble models")
        start_time = datetime.now()
        
        # Train using parent method
        training_results = super().train(thermal_features, gas_features, y_train, validation_data)
        
        # Initialize performance tracking
        for model_name in self.models.keys():
            self.model_performance_history[model_name] = deque(maxlen=self.performance_window)
            self.prediction_confidence_history[model_name] = deque(maxlen=self.performance_window)
        
        # If validation data provided, initialize weights based on performance
        if validation_data is not None:
            val_thermal, val_gas, val_y = validation_data
            self._initialize_weights_from_performance(val_thermal, val_gas, val_y)
        
        self.trained = True
        self.last_training_time = datetime.now()
        
        logger.info(f"Dynamic ensemble training completed in {(datetime.now() - start_time).total_seconds():.2f}s")
        return training_results
    
    def _initialize_weights_from_performance(self, val_thermal: pd.DataFrame, val_gas: pd.DataFrame, val_y: pd.Series):
        """
        Initialize weights based on validation performance.
        
        Args:
            val_thermal: Validation thermal features
            val_gas: Validation gas features
            val_y: Validation labels
        """
        logger.info("Initializing weights from validation performance")
        
        model_accuracies = {}
        
        # Calculate validation accuracy for each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'thermal_model':
                    predictions = model.predict(val_thermal)
                elif model_name == 'gas_model':
                    predictions = model.predict(val_gas)
                elif model_name == 'fusion_model':
                    combined_features = pd.concat([val_thermal, val_gas], axis=1)
                    predictions = model.predict(combined_features)
                else:
                    combined_features = pd.concat([val_thermal, val_gas], axis=1)
                    predictions = model.predict(combined_features)
                
                accuracy = np.mean((predictions > 0.5) == val_y)
                model_accuracies[model_name] = accuracy
                
            except Exception as e:
                logger.warning(f"Could not calculate accuracy for model '{model_name}': {str(e)}")
                model_accuracies[model_name] = 0.5  # Default
        
        # Update weights based on accuracies
        for model_name, accuracy in model_accuracies.items():
            self.model_weights[model_name] = accuracy
        
        logger.info(f"Weight initialization completed. New weights: {self.model_weights}")
    
    def predict(self, thermal_features: pd.DataFrame, gas_features: pd.DataFrame, 
                return_confidence: bool = True,
                environmental_conditions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Make ensemble predictions with dynamic weight adjustment.
        
        Args:
            thermal_features: DataFrame with thermal sensor features
            gas_features: DataFrame with gas sensor features
            return_confidence: Whether to return confidence scores
            environmental_conditions: Current environmental conditions for adaptation
            
        Returns:
            Dictionary with predictions and confidence information
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        if len(self.models) == 0:
            raise ValueError("No models added to ensemble")
        
        # Adapt weights based on environmental conditions if provided
        if environmental_conditions and self.environmental_adaptation:
            self._adapt_weights_for_environment(environmental_conditions)
        
        # Collect predictions from all models
        model_predictions = {}
        model_confidences = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'thermal_model':
                    predictions = model.predict(thermal_features)
                    confidence = self._calculate_model_confidence(model_name, thermal_features, predictions)
                elif model_name == 'gas_model':
                    predictions = model.predict(gas_features)
                    confidence = self._calculate_model_confidence(model_name, gas_features, predictions)
                elif model_name == 'fusion_model':
                    combined_features = pd.concat([thermal_features, gas_features], axis=1)
                    predictions = model.predict(combined_features)
                    confidence = self._calculate_model_confidence(model_name, combined_features, predictions)
                else:
                    combined_features = pd.concat([thermal_features, gas_features], axis=1)
                    predictions = model.predict(combined_features)
                    confidence = self._calculate_model_confidence(model_name, combined_features, predictions)
                
                model_predictions[model_name] = predictions
                model_confidences[model_name] = confidence
                
            except Exception as e:
                logger.warning(f"Prediction failed for model '{model_name}': {str(e)}")
                # Use default prediction
                model_predictions[model_name] = np.zeros(len(thermal_features))
                model_confidences[model_name] = 0.0
        
        if not model_predictions:
            raise ValueError("No successful predictions from any model")
        
        # Update performance tracking
        self._update_performance_tracking(model_predictions, model_confidences, environmental_conditions)
        
        # Adapt weights based on recent performance
        self._adapt_weights_from_performance()
        
        # Combine predictions using dynamic ensemble method
        ensemble_result = self._combine_predictions_dynamically(model_predictions, model_confidences)
        
        # Add metadata
        ensemble_result['model_contributions'] = {
            name: {
                'prediction': float(pred[0]) if len(pred) > 0 else 0.0,
                'confidence': float(model_confidences.get(name, 0.0)),
                'weight': float(self.model_weights.get(name, 1.0))
            }
            for name, pred in model_predictions.items()
        }
        
        # Store current weights for history
        self.weight_history.append(self.model_weights.copy())
        
        return ensemble_result
    
    def _adapt_weights_for_environment(self, environmental_conditions: Dict[str, float]):
        """
        Adapt model weights based on current environmental conditions.
        
        Args:
            environmental_conditions: Dictionary with current environmental conditions
        """
        # Get current weights
        current_weights = self.model_weights.copy()
        
        # Check for high temperature conditions
        if environmental_conditions.get('temperature', 0) > self.environmental_thresholds['high_temperature']:
            # Increase weight for thermal model, decrease for gas model
            current_weights['thermal_model'] = min(1.5, current_weights.get('thermal_model', 1.0) * 1.2)
            current_weights['gas_model'] = max(0.5, current_weights.get('gas_model', 1.0) * 0.8)
        
        # Check for high CO2 conditions
        if environmental_conditions.get('co2', 0) > self.environmental_thresholds['high_co2']:
            # Increase weight for gas model, decrease for thermal model
            current_weights['gas_model'] = min(1.5, current_weights.get('gas_model', 1.0) * 1.2)
            current_weights['thermal_model'] = max(0.5, current_weights.get('thermal_model', 1.0) * 0.8)
        
        # Check for high humidity conditions
        if environmental_conditions.get('humidity', 0) > self.environmental_thresholds['high_humidity']:
            # Increase weight for fusion model as it can better handle complex conditions
            current_weights['fusion_model'] = min(1.5, current_weights.get('fusion_model', 1.0) * 1.1)
        
        # Normalize weights to sum to number of models (maintain relative scale)
        total_weight = sum(current_weights.values())
        if total_weight > 0:
            normalization_factor = len(current_weights) / total_weight
            for model_name in current_weights:
                current_weights[model_name] *= normalization_factor
        
        # Apply adaptation rate to prevent drastic changes
        for model_name in self.model_weights:
            self.model_weights[model_name] = (
                (1 - self.adaptation_rate) * self.model_weights.get(model_name, 1.0) +
                self.adaptation_rate * current_weights.get(model_name, 1.0)
            )
        
        logger.debug(f"Environmental adaptation applied. New weights: {self.model_weights}")
    
    def _calculate_model_confidence(self, model_name: str, features: pd.DataFrame, predictions: np.ndarray) -> float:
        """
        Calculate confidence for a model's predictions.
        
        Args:
            model_name: Name of the model
            features: Input features
            predictions: Model predictions
            
        Returns:
            Confidence score (0-1)
        """
        # Simple confidence based on prediction certainty (distance from 0.5)
        certainty = np.mean(np.abs(predictions - 0.5) * 2)  # Scale to 0-1
        
        # Factor in model performance history
        if model_name in self.model_performance_history:
            recent_performance = np.mean(list(self.model_performance_history[model_name])) if self.model_performance_history[model_name] else 0.5
        else:
            recent_performance = 0.5  # Default
        
        # Combine certainty and performance history
        confidence = 0.7 * certainty + 0.3 * recent_performance
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _update_performance_tracking(self, model_predictions: Dict[str, np.ndarray], 
                                   model_confidences: Dict[str, float],
                                   environmental_conditions: Optional[Dict[str, float]] = None):
        """
        Update performance tracking based on recent predictions.
        
        Args:
            model_predictions: Dictionary of model predictions
            model_confidences: Dictionary of model confidences
            environmental_conditions: Current environmental conditions
        """
        # Store confidence scores
        for model_name, confidence in model_confidences.items():
            if model_name not in self.prediction_confidence_history:
                self.prediction_confidence_history[model_name] = deque(maxlen=self.performance_window)
            self.prediction_confidence_history[model_name].append(confidence)
        
        # Store environmental conditions
        if environmental_conditions:
            self.environmental_conditions_history.append(environmental_conditions)
    
    def _adapt_weights_from_performance(self):
        """
        Adapt model weights based on recent performance and confidence.
        """
        if not self.prediction_confidence_history:
            return
        
        # Calculate average confidence for each model
        model_confidence_scores = {}
        for model_name, confidences in self.prediction_confidence_history.items():
            if confidences:
                model_confidence_scores[model_name] = float(np.mean(list(confidences)))
            else:
                model_confidence_scores[model_name] = 0.5  # Default
        
        if not model_confidence_scores:
            return
        
        # Apply confidence-based weighting with adaptation rate
        for model_name in self.model_weights:
            current_confidence = model_confidence_scores.get(model_name, 0.5)
            # Map confidence (0-1) to weight adjustment (0.5-1.5)
            confidence_weight = 0.5 + current_confidence  # Range 0.5-1.5
            
            # Apply adaptation with rate
            self.model_weights[model_name] = (
                (1 - self.adaptation_rate) * self.model_weights.get(model_name, 1.0) +
                self.adaptation_rate * confidence_weight
            )
        
        # Normalize weights
        self._normalize_weights()
        
        logger.debug(f"Performance-based adaptation applied. New weights: {self.model_weights}")
    
    def _normalize_weights(self):
        """Normalize model weights to maintain relative scale."""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            normalization_factor = len(self.model_weights) / total_weight
            for model_name in self.model_weights:
                self.model_weights[model_name] *= normalization_factor
    
    def _combine_predictions_dynamically(self, model_predictions: Dict[str, np.ndarray], 
                                       model_confidences: Dict[str, float]) -> Dict[str, Any]:
        """
        Combine predictions using dynamic weights and confidence-based voting.
        
        Args:
            model_predictions: Dictionary of model predictions
            model_confidences: Dictionary of model confidences
            
        Returns:
            Dictionary with ensemble prediction and confidence
        """
        # Extract first prediction from each model (assuming single sample)
        predictions = []
        confidences = []
        weights = []
        model_names = []
        
        for model_name, pred_array in model_predictions.items():
            if len(pred_array) > 0:
                predictions.append(pred_array[0])
                confidences.append(model_confidences.get(model_name, 0.5))
                weights.append(self.model_weights.get(model_name, 1.0))
                model_names.append(model_name)
        
        if not predictions:
            return {
                'fire_detected': False,
                'confidence_score': 0.0,
                'ensemble_method': 'dynamic_weighted_average'
            }
        
        # Apply confidence-weighted dynamic voting
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pred, conf, weight, name in zip(predictions, confidences, weights, model_names):
            # Combine model weight and confidence for final weight
            final_weight = weight * (0.7 + 0.3 * conf)  # Weight more influenced by model performance than confidence
            weighted_sum += pred * final_weight
            total_weight += final_weight
        
        if total_weight > 0:
            ensemble_prediction = weighted_sum / total_weight
        else:
            ensemble_prediction = np.mean(predictions)
        
        # Calculate ensemble confidence as weighted average of model confidences
        weighted_confidence_sum = 0.0
        confidence_total_weight = 0.0
        
        for conf, weight in zip(confidences, weights):
            weighted_confidence_sum += conf * weight
            confidence_total_weight += weight
        
        ensemble_confidence = (
            weighted_confidence_sum / confidence_total_weight if confidence_total_weight > 0 
            else np.mean(confidences)
        )
        
        # Convert to binary prediction based on threshold
        fire_detected = ensemble_prediction > self.confidence_threshold
        
        return {
            'fire_detected': bool(fire_detected),
            'confidence_score': float(ensemble_confidence),
            'ensemble_prediction_score': float(ensemble_prediction),
            'ensemble_method': 'dynamic_weighted_average'
        }
    
    def get_weight_history(self) -> List[Dict[str, float]]:
        """
        Get history of model weights over time.
        
        Returns:
            List of weight dictionaries over time
        """
        return list(self.weight_history)
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of model performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        summary = {}
        
        for model_name in self.models:
            confidences = self.prediction_confidence_history.get(model_name, [])
            if confidences:
                summary[model_name] = {
                    'avg_confidence': float(np.mean(list(confidences))),
                    'min_confidence': float(np.min(list(confidences))),
                    'max_confidence': float(np.max(list(confidences))),
                    'samples': len(confidences)
                }
            else:
                summary[model_name] = {
                    'avg_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0,
                    'samples': 0
                }
        
        return summary
    
    def reset_weights(self):
        """Reset model weights to equal values."""
        for model_name in self.model_weights:
            self.model_weights[model_name] = 1.0
        logger.info("Model weights reset to equal values")
    
    def export_weighting_configuration(self, filepath: str) -> None:
        """
        Export current weighting configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        config = {
            'model_weights': self.model_weights,
            'weighting_strategy': self.weighting_strategy,
            'confidence_threshold': self.confidence_threshold,
            'adaptation_rate': self.adaptation_rate,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Weighting configuration exported to {filepath}")
    
    def import_weighting_configuration(self, filepath: str) -> None:
        """
        Import weighting configuration from file.
        
        Args:
            filepath: Path to load configuration from
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.model_weights = config.get('model_weights', self.model_weights)
            self.weighting_strategy = config.get('weighting_strategy', self.weighting_strategy)
            self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
            self.adaptation_rate = config.get('adaptation_rate', self.adaptation_rate)
            
            logger.info(f"Weighting configuration imported from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to import weighting configuration: {str(e)}")
            raise


class EnvironmentalConditionAdapter:
    """
    Adapter for environmental condition-based weight adaptation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the environmental condition adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.condition_weights = self.config.get('condition_weights', {
            'temperature': 0.3,
            'co2': 0.3,
            'humidity': 0.2,
            'pressure': 0.1,
            'light': 0.1
        })
        
        logger.info("Environmental Condition Adapter initialized")
    
    def adapt_weights_for_conditions(self, base_weights: Dict[str, float], 
                                   conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Adapt weights based on environmental conditions.
        
        Args:
            base_weights: Base model weights
            conditions: Environmental conditions
            
        Returns:
            Adapted weights
        """
        adapted_weights = base_weights.copy()
        
        # Temperature adaptation
        temp = conditions.get('temperature', 20.0)
        if temp > 50:  # High temperature
            adapted_weights['thermal_model'] *= 1.3
            adapted_weights['gas_model'] *= 0.8
        elif temp < 10:  # Low temperature
            adapted_weights['thermal_model'] *= 0.9
            adapted_weights['gas_model'] *= 1.1
        
        # CO2 adaptation
        co2 = conditions.get('co2', 400.0)
        if co2 > 1000:  # High CO2
            adapted_weights['gas_model'] *= 1.3
            adapted_weights['thermal_model'] *= 0.8
        elif co2 < 300:  # Low CO2
            adapted_weights['gas_model'] *= 0.9
        
        # Humidity adaptation
        humidity = conditions.get('humidity', 50.0)
        if humidity > 80:  # High humidity
            adapted_weights['fusion_model'] *= 1.2
        elif humidity < 20:  # Low humidity
            adapted_weights['thermal_model'] *= 1.1
        
        return adapted_weights


class ConfidenceBasedVotingSystem:
    """
    System for confidence-based voting in ensemble predictions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the confidence-based voting system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.voting_method = self.config.get('voting_method', 'weighted')
        
        logger.info("Confidence-Based Voting System initialized")
    
    def vote(self, predictions: List[float], confidences: List[float], 
             weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Perform confidence-based voting.
        
        Args:
            predictions: List of model predictions
            confidences: List of model confidences
            weights: Optional list of model weights
            
        Returns:
            Dictionary with voting results
        """
        if not predictions or not confidences or len(predictions) != len(confidences):
            raise ValueError("Invalid predictions or confidences")
        
        if weights and len(weights) != len(predictions):
            raise ValueError("Weights length must match predictions length")
        
        # Default weights if not provided
        if weights is None:
            weights = [1.0] * len(predictions)
        
        if self.voting_method == 'weighted':
            return self._weighted_voting(predictions, confidences, weights)
        elif self.voting_method == 'threshold':
            return self._threshold_voting(predictions, confidences)
        else:
            return self._simple_average_voting(predictions, confidences, weights)
    
    def _weighted_voting(self, predictions: List[float], confidences: List[float], 
                        weights: List[float]) -> Dict[str, Any]:
        """
        Perform weighted voting based on confidence and weights.
        
        Args:
            predictions: List of model predictions
            confidences: List of model confidences
            weights: List of model weights
            
        Returns:
            Dictionary with voting results
        """
        # Calculate final weights combining model weights and confidences
        final_weights = [
            weight * (0.5 + 0.5 * confidence)  # Weight more influenced by model weight than confidence
            for weight, confidence in zip(weights, confidences)
        ]
        
        # Weighted average of predictions
        weighted_sum = sum(pred * weight for pred, weight in zip(predictions, final_weights))
        total_weight = sum(final_weights)
        
        if total_weight > 0:
            ensemble_prediction = weighted_sum / total_weight
        else:
            ensemble_prediction = np.mean(predictions)
        
        # Calculate ensemble confidence
        confidence_sum = sum(conf * weight for conf, weight in zip(confidences, final_weights))
        ensemble_confidence = confidence_sum / total_weight if total_weight > 0 else np.mean(confidences)
        
        # Convert to binary prediction
        fire_detected = ensemble_prediction > self.confidence_threshold
        
        return {
            'fire_detected': bool(fire_detected),
            'confidence_score': float(ensemble_confidence),
            'ensemble_prediction_score': float(ensemble_prediction),
            'voting_method': 'weighted'
        }
    
    def _threshold_voting(self, predictions: List[float], confidences: List[float]) -> Dict[str, Any]:
        """
        Perform threshold-based voting.
        
        Args:
            predictions: List of model predictions
            confidences: List of model confidences
            
        Returns:
            Dictionary with voting results
        """
        # Only consider predictions with high confidence
        high_confidence_predictions = [
            pred for pred, conf in zip(predictions, confidences) if conf >= self.confidence_threshold
        ]
        
        if high_confidence_predictions:
            ensemble_prediction = np.mean(high_confidence_predictions)
            # Confidence is average of high-confidence predictions
            high_confidence_scores = [
                conf for conf in confidences if conf >= self.confidence_threshold
            ]
            ensemble_confidence = np.mean(high_confidence_scores)
        else:
            # If no high-confidence predictions, use all predictions but lower confidence
            ensemble_prediction = np.mean(predictions)
            ensemble_confidence = np.mean(confidences) * 0.5  # Penalize low confidence
        
        fire_detected = ensemble_prediction > self.confidence_threshold
        
        return {
            'fire_detected': bool(fire_detected),
            'confidence_score': float(ensemble_confidence),
            'ensemble_prediction_score': float(ensemble_prediction),
            'voting_method': 'threshold'
        }
    
    def _simple_average_voting(self, predictions: List[float], confidences: List[float], 
                              weights: List[float]) -> Dict[str, Any]:
        """
        Perform simple average voting.
        
        Args:
            predictions: List of model predictions
            confidences: List of model confidences
            weights: List of model weights
            
        Returns:
            Dictionary with voting results
        """
        ensemble_prediction = np.mean(predictions)
        ensemble_confidence = np.mean(confidences)
        fire_detected = ensemble_prediction > self.confidence_threshold
        
        return {
            'fire_detected': bool(fire_detected),
            'confidence_score': float(ensemble_confidence),
            'ensemble_prediction_score': float(ensemble_prediction),
            'voting_method': 'simple_average'
        }