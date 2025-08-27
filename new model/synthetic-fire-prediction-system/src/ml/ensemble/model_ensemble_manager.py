"""
Model Ensemble Manager for the fire prediction system.

This module provides a comprehensive ensemble system that combines multiple models
(baseline, temporal, and specialized) with advanced confidence scoring and uncertainty
quantification for robust fire prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
from datetime import datetime
from collections import defaultdict
import pickle
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

from ..base import FireModel
from .classification import BinaryFireClassifier, EnsembleFireClassifier
from .confidence import ConfidenceScorer, UncertaintyEstimator, EnsembleVarianceAnalyzer

# Import temporal models if available
try:
    from .temporal import LSTMFireClassifier, GRUFireClassifier, LSTM_AVAILABLE, GRU_AVAILABLE
    TEMPORAL_AVAILABLE = True
except ImportError:
    LSTMFireClassifier = None
    GRUFireClassifier = None
    LSTM_AVAILABLE = False
    GRU_AVAILABLE = False
    TEMPORAL_AVAILABLE = False

# Import XGBoost if available
try:
    from .classification.xgboost_classifier import XGBoostFireClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBoostFireClassifier = None
    XGBOOST_AVAILABLE = False


class ModelEnsembleManager:
    """
    Comprehensive model ensemble manager for fire prediction.
    
    This class manages multiple models (baseline, temporal, specialized) and combines
    their predictions using advanced ensemble techniques with confidence scoring and
    uncertainty quantification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model ensemble manager.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ensemble configuration
        self.ensemble_strategy = config.get('ensemble_strategy', 'weighted_voting')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.3)
        self.enable_calibration = config.get('enable_calibration', True)
        self.cross_validation_folds = config.get('cross_validation_folds', 5)
        
        # Model configurations
        self.model_configs = config.get('model_configs', {})
        self.enabled_models = config.get('enabled_models', [])
        
        # Storage for models and components
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.confidence_scorer = None
        self.uncertainty_estimator = None
        self.variance_analyzer = None
        self.calibration_models = {}
        
        # Training and prediction history
        self.training_history = {
            'individual_models': {},
            'ensemble_metrics': {},
            'confidence_scores': [],
            'uncertainty_scores': []
        }
        
        # Metadata
        self.metadata = {
            'ensemble_type': 'multi_model_ensemble',
            'strategy': self.ensemble_strategy,
            'models_available': {
                'baseline': True,
                'xgboost': XGBOOST_AVAILABLE,
                'temporal_lstm': LSTM_AVAILABLE,
                'temporal_gru': GRU_AVAILABLE
            },
            'training_completed': False,
            'last_training_time': None,
            'total_models': 0,
            'best_individual_model': None,
            'ensemble_performance': {}
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ensemble components."""
        # Initialize confidence scorer
        confidence_config = self.config.get('confidence_config', {
            'methods': ['probability_based', 'variance_based', 'entropy_based'],
            'calibration_method': 'isotonic'
        })
        self.confidence_scorer = ConfidenceScorer(confidence_config)
        
        # Initialize uncertainty estimator
        uncertainty_config = self.config.get('uncertainty_config', {
            'epistemic_estimation': True,
            'aleatoric_estimation': True,
            'bootstrap_samples': 100
        })
        self.uncertainty_estimator = UncertaintyEstimator(uncertainty_config)
        
        # Initialize variance analyzer
        variance_config = self.config.get('variance_config', {
            'analysis_methods': ['prediction_variance', 'feature_importance_variance'],
            'diversity_measures': ['disagreement', 'double_fault']
        })
        self.variance_analyzer = EnsembleVarianceAnalyzer(variance_config)
        
        self.logger.info("Ensemble components initialized")
    
    def add_model(self, model_name: str, model_config: Dict[str, Any], model_type: str = 'classification'):
        """
        Add a model to the ensemble.
        
        Args:
            model_name: Unique name for the model
            model_config: Configuration for the model
            model_type: Type of model ('classification', 'temporal', etc.)
        """
        try:
            if model_type == 'random_forest':
                model_config['algorithm'] = 'random_forest'
                model = BinaryFireClassifier(model_config)
            
            elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                model = XGBoostFireClassifier(model_config)
            
            elif model_type == 'lstm' and LSTM_AVAILABLE:
                model = LSTMFireClassifier(model_config)
            
            elif model_type == 'gru' and GRU_AVAILABLE:
                model = GRUFireClassifier(model_config)
            
            elif model_type == 'ensemble':
                model = EnsembleFireClassifier(model_config)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.models[model_name] = {
                'model': model,
                'type': model_type,
                'config': model_config,
                'trained': False,
                'performance': None
            }
            
            self.logger.info(f"Added {model_type} model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add model {model_name}: {str(e)}")
            raise
    
    def create_default_ensemble(self):
        """Create a default ensemble with available models."""
        default_models = []
        
        # Add Random Forest (always available)
        rf_config = {
            'algorithm': 'random_forest',
            'n_estimators': 200,
            'max_depth': 15,
            'class_weight': 'balanced',
            'random_state': 42
        }
        self.add_model('random_forest', rf_config, 'random_forest')
        default_models.append('random_forest')
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_config = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'class_weight': 'balanced',
                'random_state': 42
            }
            self.add_model('xgboost', xgb_config, 'xgboost')
            default_models.append('xgboost')
        
        # Add LSTM if available
        if LSTM_AVAILABLE:
            lstm_config = {
                'sequence_length': 30,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True,
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 50,
                'early_stopping_patience': 10
            }
            self.add_model('lstm', lstm_config, 'lstm')
            default_models.append('lstm')
        
        # Add GRU if available
        if GRU_AVAILABLE:
            gru_config = {
                'sequence_length': 30,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True,
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 50,
                'early_stopping_patience': 10
            }
            self.add_model('gru', gru_config, 'gru')
            default_models.append('gru')
        
        # Add ensemble classifier
        ensemble_config = {
            'ensemble_method': 'stacking',
            'base_estimators': ['random_forest', 'gradient_boosting', 'logistic_regression'],
            'voting_type': 'soft',
            'cv': 5
        }
        self.add_model('ensemble_classifier', ensemble_config, 'ensemble')
        default_models.append('ensemble_classifier')
        
        self.enabled_models = default_models
        self.metadata['total_models'] = len(default_models)
        
        self.logger.info(f"Created default ensemble with {len(default_models)} models: {default_models}")
    
    def train_all_models(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Optional validation data
            
        Returns:
            Dictionary containing training results for all models
        """
        self.logger.info("Training all models in the ensemble")
        start_time = time.time()
        
        # Create default ensemble if no models are added
        if not self.models:
            self.create_default_ensemble()
        
        # Training results
        training_results = {}
        
        # Train each model
        for model_name, model_info in self.models.items():
            if model_name not in self.enabled_models:
                continue
                
            self.logger.info(f"Training model: {model_name}")
            model = model_info['model']
            
            try:
                # Train the model
                model_start_time = time.time()
                metrics = model.train(X_train, y_train, validation_data)
                training_time = time.time() - model_start_time
                
                # Update model info
                model_info['trained'] = True
                model_info['performance'] = metrics
                
                # Store training results
                training_results[model_name] = {
                    'metrics': metrics,
                    'training_time': training_time,
                    'status': 'success'
                }
                
                # Store in history
                self.training_history['individual_models'][model_name] = metrics
                
                self.logger.info(f"✓ {model_name} trained successfully in {training_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"✗ Failed to train {model_name}: {str(e)}")
                training_results[model_name] = {
                    'metrics': {},
                    'training_time': 0,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calculate ensemble weights based on individual model performance
        self._calculate_ensemble_weights(validation_data)
        
        # Train confidence and uncertainty estimators
        if validation_data is not None:
            self._train_confidence_estimators(validation_data[0], validation_data[1])
        
        # Update metadata
        total_training_time = time.time() - start_time
        self.metadata['training_completed'] = True
        self.metadata['last_training_time'] = datetime.now().isoformat()
        
        # Find best individual model
        best_model = self._find_best_individual_model()
        self.metadata['best_individual_model'] = best_model
        
        training_summary = {
            'total_training_time': total_training_time,
            'models_trained': len([r for r in training_results.values() if r['status'] == 'success']),
            'models_failed': len([r for r in training_results.values() if r['status'] == 'failed']),
            'best_individual_model': best_model,
            'ensemble_weights': dict(self.model_weights),
            'individual_results': training_results
        }
        
        self.logger.info(f"Ensemble training completed in {total_training_time:.2f}s")
        self.logger.info(f"Successfully trained: {training_summary['models_trained']}/{len(self.enabled_models)} models")
        
        return training_summary
    
    def _calculate_ensemble_weights(self, validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None):
        """Calculate ensemble weights based on model performance."""
        if validation_data is None:
            # Use equal weights if no validation data
            num_models = len([m for name, m in self.models.items() if m['trained'] and name in self.enabled_models])
            weight = 1.0 / num_models if num_models > 0 else 0.0
            
            for model_name in self.enabled_models:
                if self.models[model_name]['trained']:
                    self.model_weights[model_name] = weight
            return
        
        X_val, y_val = validation_data
        model_scores = {}
        
        # Calculate validation scores for each trained model
        for model_name in self.enabled_models:
            if not self.models[model_name]['trained']:
                continue
                
            try:
                model = self.models[model_name]['model']
                predictions = model.predict(X_val)
                accuracy = accuracy_score(y_val, predictions)
                model_scores[model_name] = accuracy
                
                self.logger.info(f"{model_name} validation accuracy: {accuracy:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Could not calculate validation score for {model_name}: {e}")
                model_scores[model_name] = 0.0
        
        # Calculate weights based on performance (softmax-like normalization)
        if model_scores:
            # Use exponential weighting to emphasize better models
            exp_scores = {name: np.exp(score * 5) for name, score in model_scores.items()}  # Scale factor 5
            total_exp_score = sum(exp_scores.values())
            
            if total_exp_score > 0:
                self.model_weights = {name: exp_score / total_exp_score 
                                    for name, exp_score in exp_scores.items()}
            else:
                # Fallback to equal weights
                num_models = len(model_scores)
                self.model_weights = {name: 1.0 / num_models for name in model_scores.keys()}
        
        self.logger.info(f"Ensemble weights calculated: {self.model_weights}")
    
    def _train_confidence_estimators(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Train confidence and uncertainty estimators."""
        try:
            # Collect predictions from all trained models
            model_predictions = {}
            model_probabilities = {}
            
            for model_name in self.enabled_models:
                if not self.models[model_name]['trained']:
                    continue
                    
                model = self.models[model_name]['model']
                predictions = model.predict(X_val)
                probabilities = model.predict_proba(X_val)
                
                model_predictions[model_name] = predictions
                model_probabilities[model_name] = probabilities
            
            # Train confidence scorer
            self.confidence_scorer.fit(model_probabilities, y_val)
            
            # Train uncertainty estimator
            self.uncertainty_estimator.fit(model_predictions, model_probabilities, y_val)
            
            # Train variance analyzer
            self.variance_analyzer.fit(model_predictions, model_probabilities, y_val)
            
            self.logger.info("Confidence and uncertainty estimators trained")
            
        except Exception as e:
            self.logger.warning(f"Failed to train confidence estimators: {e}")
    
    def _find_best_individual_model(self) -> Optional[str]:
        """Find the best performing individual model."""
        best_model = None
        best_score = 0.0
        
        for model_name, model_info in self.models.items():
            if not model_info['trained'] or model_name not in self.enabled_models:
                continue
                
            performance = model_info.get('performance', {})
            
            # Try to get validation accuracy, fallback to other metrics
            score = (performance.get('validation', {}).get('accuracy') or 
                    performance.get('best_val_accuracy') or
                    performance.get('accuracy', 0.0))
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        self.logger.info(f"Best individual model: {best_model} (score: {best_score:.4f})")
        return best_model
    
    def predict(self, X: pd.DataFrame, return_confidence: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Make ensemble predictions.
        
        Args:
            X: Features to predict
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions, and optionally confidence/uncertainty information
        """
        if not any(model_info['trained'] for model_info in self.models.values()):
            raise ValueError("No trained models available for prediction")
        
        # Collect predictions from all trained models
        model_predictions = {}
        model_probabilities = {}
        
        for model_name in self.enabled_models:
            if not self.models[model_name]['trained']:
                continue
                
            try:
                model = self.models[model_name]['model']
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)
                
                model_predictions[model_name] = predictions
                model_probabilities[model_name] = probabilities
                
            except Exception as e:
                self.logger.warning(f"Prediction failed for {model_name}: {e}")
        
        if not model_predictions:
            raise ValueError("No successful predictions from any model")
        
        # Combine predictions based on ensemble strategy
        ensemble_predictions = self._combine_predictions(model_predictions, model_probabilities)
        
        if not return_confidence:
            return ensemble_predictions
        
        # Calculate confidence and uncertainty scores
        confidence_info = self._calculate_confidence_scores(model_predictions, model_probabilities, ensemble_predictions)
        
        return ensemble_predictions, confidence_info
    
    def _combine_predictions(self, model_predictions: Dict[str, np.ndarray], 
                           model_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions from multiple models based on the ensemble strategy."""
        
        if self.ensemble_strategy == 'weighted_voting':
            # Weighted voting using model weights
            combined_probs = None
            total_weight = 0.0
            
            for model_name, probs in model_probabilities.items():
                weight = self.model_weights.get(model_name, 1.0)
                
                if combined_probs is None:
                    combined_probs = weight * probs
                else:
                    combined_probs += weight * probs
                
                total_weight += weight
            
            if total_weight > 0:
                combined_probs /= total_weight
            
            # Convert probabilities to predictions
            return np.argmax(combined_probs, axis=1)
        
        elif self.ensemble_strategy == 'majority_voting':
            # Simple majority voting
            all_predictions = np.column_stack(list(model_predictions.values()))
            
            # Find majority class for each sample
            ensemble_preds = []
            for i in range(all_predictions.shape[0]):
                sample_preds = all_predictions[i, :]
                unique, counts = np.unique(sample_preds, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                ensemble_preds.append(majority_class)
            
            return np.array(ensemble_preds)
        
        elif self.ensemble_strategy == 'average_probabilities':
            # Average probabilities then take argmax
            combined_probs = np.mean(list(model_probabilities.values()), axis=0)
            return np.argmax(combined_probs, axis=1)
        
        else:
            # Default to weighted voting
            return self._combine_predictions(model_predictions, model_probabilities)
    
    def _calculate_confidence_scores(self, model_predictions: Dict[str, np.ndarray],
                                   model_probabilities: Dict[str, np.ndarray],
                                   ensemble_predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive confidence and uncertainty scores."""
        confidence_info = {}
        
        try:
            # Basic confidence from probability scores
            if model_probabilities:
                avg_probs = np.mean(list(model_probabilities.values()), axis=0)
                confidence_info['probability_confidence'] = np.max(avg_probs, axis=1)
                
                # Entropy-based uncertainty
                epsilon = 1e-10  # Prevent log(0)
                entropy = -np.sum(avg_probs * np.log(avg_probs + epsilon), axis=1)
                confidence_info['entropy_uncertainty'] = entropy
            
            # Model agreement (how much models agree)
            if len(model_predictions) > 1:
                all_preds = np.column_stack(list(model_predictions.values()))
                
                # Calculate agreement rate for each sample
                agreement_rates = []
                for i in range(all_preds.shape[0]):
                    sample_preds = all_preds[i, :]
                    unique, counts = np.unique(sample_preds, return_counts=True)
                    max_agreement = np.max(counts) / len(sample_preds)
                    agreement_rates.append(max_agreement)
                
                confidence_info['model_agreement'] = np.array(agreement_rates)
            
            # Use trained confidence estimators if available
            if self.confidence_scorer is not None:
                try:
                    confidence_scores = self.confidence_scorer.predict_confidence(model_probabilities)
                    confidence_info['calibrated_confidence'] = confidence_scores
                except Exception as e:
                    self.logger.debug(f"Confidence scorer prediction failed: {e}")
            
            if self.uncertainty_estimator is not None:
                try:
                    uncertainty_scores = self.uncertainty_estimator.estimate_uncertainty(model_predictions, model_probabilities)
                    confidence_info['estimated_uncertainty'] = uncertainty_scores
                except Exception as e:
                    self.logger.debug(f"Uncertainty estimator prediction failed: {e}")
            
            # Combined confidence score (weighted average of available metrics)
            confidence_components = []
            if 'probability_confidence' in confidence_info:
                confidence_components.append(confidence_info['probability_confidence'])
            if 'model_agreement' in confidence_info:
                confidence_components.append(confidence_info['model_agreement'])
            
            if confidence_components:
                confidence_info['combined_confidence'] = np.mean(confidence_components, axis=0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence scores: {e}")
            # Fallback: return default confidence
            n_samples = len(ensemble_predictions)
            confidence_info['combined_confidence'] = np.full(n_samples, 0.5)
        
        return confidence_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the ensemble."""
        info = {
            'ensemble_strategy': self.ensemble_strategy,
            'total_models': len(self.models),
            'enabled_models': self.enabled_models,
            'trained_models': [name for name, model_info in self.models.items() 
                             if model_info['trained']],
            'model_weights': dict(self.model_weights),
            'best_individual_model': self.metadata.get('best_individual_model'),
            'confidence_threshold': self.confidence_threshold,
            'uncertainty_threshold': self.uncertainty_threshold,
            'availability': self.metadata['models_available']
        }
        
        # Add individual model information
        info['individual_models'] = {}
        for model_name, model_info in self.models.items():
            info['individual_models'][model_name] = {
                'type': model_info['type'],
                'trained': model_info['trained'],
                'performance': model_info.get('performance', {})
            }
        
        return info
    
    def save(self, filepath: str) -> None:
        """Save the ensemble system."""
        save_data = {
            'config': self.config,
            'models': {},
            'model_weights': self.model_weights,
            'enabled_models': self.enabled_models,
            'training_history': self.training_history,
            'metadata': self.metadata
        }
        
        # Save individual models
        model_dir = os.path.join(os.path.dirname(filepath), 'ensemble_models')
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model_info in self.models.items():
            if model_info['trained']:
                model_path = os.path.join(model_dir, f"{model_name}.pkl")
                model_info['model'].save(model_path)
                save_data['models'][model_name] = {
                    'type': model_info['type'],
                    'config': model_info['config'],
                    'trained': model_info['trained'],
                    'performance': model_info['performance'],
                    'path': model_path
                }
        
        # Save ensemble metadata
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved ensemble system to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the ensemble system."""
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        self.config = save_data['config']
        self.model_weights = save_data['model_weights']
        self.enabled_models = save_data['enabled_models']
        self.training_history = save_data['training_history']
        self.metadata = save_data['metadata']
        
        # Load individual models
        self.models = {}
        for model_name, model_info in save_data['models'].items():
            if model_info['trained']:
                # Recreate model and load it
                if model_info['type'] == 'random_forest':
                    model = BinaryFireClassifier(model_info['config'])
                elif model_info['type'] == 'xgboost' and XGBOOST_AVAILABLE:
                    model = XGBoostFireClassifier(model_info['config'])
                elif model_info['type'] == 'lstm' and LSTM_AVAILABLE:
                    model = LSTMFireClassifier(model_info['config'])
                elif model_info['type'] == 'gru' and GRU_AVAILABLE:
                    model = GRUFireClassifier(model_info['config'])
                else:
                    continue
                
                model.load(model_info['path'])
                
                self.models[model_name] = {
                    'model': model,
                    'type': model_info['type'],
                    'config': model_info['config'],
                    'trained': model_info['trained'],
                    'performance': model_info['performance']
                }
        
        self.logger.info(f"Loaded ensemble system from {filepath}")
        self.logger.info(f"Loaded {len(self.models)} models")


# Convenience function for creating a complete ensemble
def create_fire_prediction_ensemble(config: Optional[Dict[str, Any]] = None) -> ModelEnsembleManager:
    """
    Create a comprehensive fire prediction ensemble with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured ModelEnsembleManager
    """
    default_config = {
        'ensemble_strategy': 'weighted_voting',
        'confidence_threshold': 0.7,
        'uncertainty_threshold': 0.3,
        'enable_calibration': True,
        'cross_validation_folds': 5,
        'confidence_config': {
            'methods': ['probability_based', 'variance_based', 'entropy_based'],
            'calibration_method': 'isotonic'
        },
        'uncertainty_config': {
            'epistemic_estimation': True,
            'aleatoric_estimation': True,
            'bootstrap_samples': 50  # Reduced for performance
        },
        'variance_config': {
            'analysis_methods': ['prediction_variance', 'feature_importance_variance'],
            'diversity_measures': ['disagreement', 'double_fault']
        }
    }
    
    if config:
        default_config.update(config)
    
    return ModelEnsembleManager(default_config)