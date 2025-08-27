"""
Model Training Pipeline for baseline ML models.

This module provides a comprehensive training pipeline for baseline machine learning
models including Random Forest, XGBoost, and other traditional ML algorithms.
"""

import os
import json
import logging
import time
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from ..base import FireModel
from .binary_classifier import BinaryFireClassifier


class ModelTrainingPipeline:
    """
    Comprehensive training pipeline for baseline ML models.
    
    This class provides functionality for training, evaluating, and managing
    multiple baseline ML models with hyperparameter tuning, cross-validation,
    and comprehensive evaluation metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration dictionary containing pipeline parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pipeline configuration
        self.test_size = config.get('test_size', 0.2)
        self.validation_size = config.get('validation_size', 0.1)
        self.random_state = config.get('random_state', 42)
        self.cv_folds = config.get('cv_folds', 5)
        self.scoring = config.get('scoring', 'accuracy')
        self.n_jobs = config.get('n_jobs', -1)
        
        # Model configurations
        self.model_configs = config.get('models', {})
        self.hyperparameter_tuning = config.get('hyperparameter_tuning', False)
        self.tuning_method = config.get('tuning_method', 'randomized')  # 'grid' or 'randomized'
        self.tuning_iterations = config.get('tuning_iterations', 50)
        
        # Output configuration
        self.output_dir = config.get('output_dir', './models')
        self.save_models = config.get('save_models', True)
        
        # Initialize storage
        self.trained_models = {}
        self.training_results = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging for the training pipeline."""
        log_level = self.config.get('log_level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
    
    def prepare_data(self, 
                    X: pd.DataFrame, 
                    y: pd.Series,
                    feature_scaling: bool = True,
                    encode_labels: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting and preprocessing.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_scaling: Whether to apply feature scaling
            encode_labels: Whether to encode labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Preparing data for training")
        
        # Encode labels if required
        if encode_labels and y.dtype == 'object':
            self.label_encoders['target'] = LabelEncoder()
            y = pd.Series(self.label_encoders['target'].fit_transform(y), 
                         index=y.index, name=y.name)
        
        # Split data into train, validation, and test sets
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs validation
        val_size = self.validation_size / (1 - self.test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=self.random_state, stratify=y_temp
        )
        
        # Apply feature scaling if required
        if feature_scaling:
            self.scalers['features'] = StandardScaler()
            X_train = pd.DataFrame(
                self.scalers['features'].fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_val = pd.DataFrame(
                self.scalers['features'].transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            X_test = pd.DataFrame(
                self.scalers['features'].transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        self.logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_default_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default configurations for baseline models.
        
        Returns:
            Dictionary of default model configurations
        """
        return {
            'random_forest': {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'early_stopping_rounds': 50,
                'eval_metric': 'logloss'
            }
        }
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get hyperparameter grids for tuning.
        
        Returns:
            Dictionary of hyperparameter grids for each model
        """
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        }
    
    def train_single_model(self, 
                          model_name: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series,
                          config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels  
            X_val: Validation features
            y_val: Validation labels
            config: Optional model configuration
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        self.logger.info(f"Training model: {model_name}")
        
        # Get model configuration
        if config is None:
            default_configs = self.get_default_model_configs()
            config = default_configs.get(model_name, {})
        
        # Initialize model based on type
        if model_name == 'random_forest':
            model = BinaryFireClassifier(config)
        elif model_name == 'xgboost':
            try:
                from .xgboost_classifier import XGBoostFireClassifier
                model = XGBoostFireClassifier(config)
            except ImportError:
                self.logger.warning("XGBoost not available, falling back to Random Forest")
                config['algorithm'] = 'random_forest'
                model = BinaryFireClassifier(config)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Train the model
        start_time = time.time()
        metrics = model.train(X_train, y_train, validation_data=(X_val, y_val))
        training_time = time.time() - start_time
        
        metrics['total_training_time'] = training_time
        
        self.logger.info(f"Model {model_name} trained in {training_time:.2f} seconds")
        
        return model, metrics
    
    def perform_hyperparameter_tuning(self,
                                    model_name: str,
                                    X_train: pd.DataFrame,
                                    y_train: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a model.
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary containing best parameters and scores
        """
        self.logger.info(f"Performing hyperparameter tuning for {model_name}")
        
        # Get base model and parameter grid
        default_configs = self.get_default_model_configs()
        base_config = default_configs.get(model_name, {})
        param_grids = self.get_hyperparameter_grids()
        param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            self.logger.warning(f"No parameter grid found for {model_name}")
            return {'best_params': base_config, 'best_score': 0.0}
        
        # Create base model
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(random_state=self.random_state)
        elif model_name == 'xgboost':
            try:
                import xgboost as xgb
                base_model = xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            except ImportError:
                self.logger.warning("XGBoost not available for tuning")
                return {'best_params': base_config, 'best_score': 0.0}
        else:
            self.logger.warning(f"Hyperparameter tuning not implemented for {model_name}")
            return {'best_params': base_config, 'best_score': 0.0}
        
        # Perform tuning
        start_time = time.time()
        
        if self.tuning_method == 'grid':
            search = GridSearchCV(
                base_model, param_grid, cv=self.cv_folds,
                scoring=self.scoring, n_jobs=self.n_jobs,
                verbose=1
            )
        else:  # randomized
            search = RandomizedSearchCV(
                base_model, param_grid, cv=self.cv_folds,
                scoring=self.scoring, n_jobs=self.n_jobs,
                n_iter=self.tuning_iterations, random_state=self.random_state,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        self.logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        self.logger.info(f"Best score: {search.best_score_:.4f}")
        self.logger.info(f"Best params: {search.best_params_}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'tuning_time': tuning_time
        }
    
    def train_all_models(self,
                        X: pd.DataFrame,
                        y: pd.Series,
                        models_to_train: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train all specified models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            models_to_train: List of model names to train, if None train all
            
        Returns:
            Dictionary containing training results
        """
        self.logger.info("Starting comprehensive model training pipeline")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        
        # Determine which models to train
        if models_to_train is None:
            models_to_train = list(self.get_default_model_configs().keys())
        
        # Initialize results dictionary
        results = {
            'data_info': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'features': len(X.columns),
                'classes': sorted(y.unique().tolist())
            },
            'models': {},
            'summary': {}
        }
        
        # Train each model
        for model_name in models_to_train:
            self.logger.info(f"Training {model_name}...")
            
            try:
                model_config = self.model_configs.get(model_name, {})
                
                # Perform hyperparameter tuning if enabled
                if self.hyperparameter_tuning:
                    tuning_results = self.perform_hyperparameter_tuning(
                        model_name, X_train, y_train
                    )
                    model_config.update(tuning_results['best_params'])
                    results['models'][model_name] = {
                        'tuning_results': tuning_results
                    }
                
                # Train the model
                model, metrics = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, model_config
                )
                
                # Evaluate on test set
                test_predictions = model.predict(X_test)
                test_probabilities = model.predict_proba(X_test)
                test_metrics = model._calculate_metrics(y_test, test_predictions, test_probabilities)
                
                # Store results
                self.trained_models[model_name] = model
                self.training_results[model_name] = metrics
                
                results['models'][model_name].update({
                    'training_metrics': metrics,
                    'test_metrics': test_metrics,
                    'config': model_config
                })
                
                # Save model if requested
                if self.save_models:
                    model_path = os.path.join(self.output_dir, f"{model_name}_model.pkl")
                    model.save(model_path)
                
                self.logger.info(f"✅ {model_name} training completed successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train {model_name}: {str(e)}")
                results['models'][model_name] = {
                    'error': str(e),
                    'training_metrics': {},
                    'test_metrics': {}
                }
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        # Save training results
        results_path = os.path.join(self.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save scalers and encoders
        if self.scalers:
            scalers_path = os.path.join(self.output_dir, 'scalers.pkl')
            joblib.dump(self.scalers, scalers_path)
        
        if self.label_encoders:
            encoders_path = os.path.join(self.output_dir, 'label_encoders.pkl')
            joblib.dump(self.label_encoders, encoders_path)
        
        self.logger.info("Model training pipeline completed")
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of training results.
        
        Args:
            results: Training results dictionary
            
        Returns:
            Summary dictionary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': len([m for m in results['models'] if 'error' not in results['models'][m]]),
            'models_failed': len([m for m in results['models'] if 'error' in results['models'][m]]),
            'best_model': None,
            'best_score': 0.0,
            'model_comparison': {}
        }
        
        # Find best model based on validation accuracy
        best_score = 0.0
        best_model = None
        
        for model_name, model_results in results['models'].items():
            if 'error' not in model_results and 'test_metrics' in model_results:
                test_accuracy = model_results['test_metrics'].get('accuracy', 0.0)
                summary['model_comparison'][model_name] = {
                    'test_accuracy': test_accuracy,
                    'validation_accuracy': model_results.get('training_metrics', {}).get('validation', {}).get('accuracy', 0.0)
                }
                
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    best_model = model_name
        
        summary['best_model'] = best_model
        summary['best_score'] = best_score
        
        return summary
    
    def load_trained_models(self, models_dir: str) -> Dict[str, Any]:
        """
        Load previously trained models.
        
        Args:
            models_dir: Directory containing saved models
            
        Returns:
            Dictionary of loaded models
        """
        self.logger.info(f"Loading models from {models_dir}")
        
        loaded_models = {}
        
        # Load scalers and encoders if they exist
        scalers_path = os.path.join(models_dir, 'scalers.pkl')
        if os.path.exists(scalers_path):
            self.scalers = joblib.load(scalers_path)
        
        encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            self.label_encoders = joblib.load(encoders_path)
        
        # Load models
        for model_file in os.listdir(models_dir):
            if model_file.endswith('_model.pkl'):
                model_name = model_file.replace('_model.pkl', '')
                model_path = os.path.join(models_dir, model_file)
                
                try:
                    if model_name == 'random_forest':
                        model = BinaryFireClassifier({})
                    elif model_name == 'xgboost':
                        from .xgboost_classifier import XGBoostFireClassifier
                        model = XGBoostFireClassifier({})
                    else:
                        continue
                    
                    model.load(model_path)
                    loaded_models[model_name] = model
                    self.logger.info(f"Loaded {model_name} model")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {model_name}: {str(e)}")
        
        return loaded_models