"""
Classification models for fire prediction.
"""

import numpy as np
import pandas as pd
import os
import pickle
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..base import FireClassificationModel

class BinaryFireClassifier(FireClassificationModel):
    """
    Binary fire classification model using sklearn algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the binary fire classifier."""
        super().__init__(config)
        self.algorithm = config.get('algorithm', 'random_forest')
        self.model = None
        self.trained = False
        
        if self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 10),
                random_state=config.get('random_state', 42),
                class_weight=config.get('class_weight', 'balanced')
            )
        elif self.algorithm == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=config.get('random_state', 42),
                class_weight=config.get('class_weight', 'balanced'),
                max_iter=config.get('max_iter', 1000)
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, float]:
        """Train the classifier."""
        self.model.fit(X_train, y_train)
        self.trained = True
        
        # Calculate training metrics
        predictions = self.model.predict(X_train)
        probabilities = self.model.predict_proba(X_train)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_train, predictions),
            'precision': precision_score(y_train, predictions, average='weighted'),
            'recall': recall_score(y_train, predictions, average='weighted'),
            'f1_score': f1_score(y_train, predictions, average='weighted')
        }
        
        # Add AUC if probabilities are available
        if probabilities is not None:
            try:
                metrics['auc'] = roc_auc_score(y_train, probabilities)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Return uniform probabilities if not supported
            n_samples = len(X)
            return np.column_stack([np.ones(n_samples) * 0.5, np.ones(n_samples) * 0.5])
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted')
        }
        
        # Add AUC if probabilities are available
        if probabilities is not None:
            try:
                metrics['auc'] = roc_auc_score(y_test, probabilities)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'algorithm': self.algorithm,
            'trained': self.trained,
            'config': self.config
        }
        metadata_path = path + '.metadata'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, path: str) -> None:
        """Load a trained model from disk."""
        # Load model
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        metadata_path = path + '.metadata'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.algorithm = metadata.get('algorithm', self.algorithm)
            self.trained = metadata.get('trained', True)
            self.config = metadata.get('config', self.config)
        else:
            self.trained = True
    
    def save_metadata(self, path: str) -> None:
        """Save model metadata to disk."""
        metadata = {
            'algorithm': self.algorithm,
            'trained': self.trained,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Get confidence scores for predictions."""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            return np.max(proba, axis=1)
        else:
            return np.ones(len(X)) * 0.5


class EnsembleFireClassifier(FireClassificationModel):
    """
    Ensemble fire classifier combining multiple models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ensemble classifier.""" 
        super().__init__(config)
        self.models = []
        self.weights = config.get('weights', None)
        self.voting = config.get('voting', 'hard')
        self.trained = False
    
    def add_model(self, model: FireClassificationModel, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append({'model': model, 'weight': weight})
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, float]:
        """Train all models in the ensemble."""
        results = []
        for model_info in self.models:
            result = model_info['model'].train(X_train, y_train, validation_data)
            results.append(result)
        
        self.trained = True
        
        # Calculate ensemble predictions
        predictions = self.predict(X_train)
        probabilities = self.predict_proba(X_train)[:, 1] if self.predict_proba(X_train).shape[1] > 1 else None
        
        metrics = {
            'accuracy': accuracy_score(y_train, predictions),
            'precision': precision_score(y_train, predictions, average='weighted'),
            'recall': recall_score(y_train, predictions, average='weighted'),
            'f1_score': f1_score(y_train, predictions, average='weighted')
        }
        
        # Add AUC if probabilities are available
        if probabilities is not None:
            try:
                metrics['auc'] = roc_auc_score(y_train, probabilities)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        all_predictions = []
        all_probabilities = []
        
        for model_info in self.models:
            pred = model_info['model'].predict(X)
            prob = model_info['model'].predict_proba(X)[:, 1] if model_info['model'].predict_proba(X).shape[1] > 1 else np.ones(len(X)) * 0.5
            all_predictions.append(pred * model_info['weight'])
            all_probabilities.append(prob * model_info['weight'])
        
        if self.voting == 'hard':
            # Majority voting
            ensemble_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
        else:
            # Soft voting (weighted average of probabilities)
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)
        
        return ensemble_predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for ensemble."""
        if not self.trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        all_probabilities = []
        
        for model_info in self.models:
            prob = model_info['model'].predict_proba(X)
            weighted_prob = prob * model_info['weight']
            all_probabilities.append(weighted_prob)
        
        # Average probabilities
        ensemble_probabilities = np.mean(all_probabilities, axis=0)
        return ensemble_probabilities
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the ensemble on test data."""
        if not self.trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1] if self.predict_proba(X_test).shape[1] > 1 else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted')
        }
        
        # Add AUC if probabilities are available
        if probabilities is not None:
            try:
                metrics['auc'] = roc_auc_score(y_test, probabilities)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save the ensemble to disk."""
        if not self.trained:
            raise ValueError("Ensemble must be trained before saving")
        
        # Save ensemble configuration
        ensemble_data = {
            'weights': self.weights,
            'voting': self.voting,
            'trained': self.trained
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + '_ensemble.pkl', 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        # Save individual models
        for i, model_info in enumerate(self.models):
            model_path = f"{path}_model_{i}.pkl"
            model_info['model'].save(model_path)
    
    def load(self, path: str) -> None:
        """Load an ensemble from disk."""
        # Load ensemble configuration
        with open(path + '_ensemble.pkl', 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.weights = ensemble_data.get('weights', self.weights)
        self.voting = ensemble_data.get('voting', self.voting)
        self.trained = ensemble_data.get('trained', True)
        
        # Load individual models (this would need to be adapted based on how models are stored)
        # For now, we'll assume models are loaded separately
    
    def save_metadata(self, path: str) -> None:
        """Save ensemble metadata to disk."""
        metadata = {
            'weights': self.weights,
            'voting': self.voting,
            'trained': self.trained,
            'model_count': len(self.models)
        }
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble confidence scores."""
        if not self.trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        all_confidences = []
        
        for model_info in self.models:
            confidence = model_info['model'].get_confidence(X)
            all_confidences.append(confidence * model_info['weight'])
        
        return np.mean(all_confidences, axis=0)

__all__ = ['BinaryFireClassifier', 'EnsembleFireClassifier']