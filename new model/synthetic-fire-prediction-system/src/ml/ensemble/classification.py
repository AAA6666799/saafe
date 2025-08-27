"""
Classification models for fire prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    
    def train(self, data: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """Train the classifier."""
        self.model.fit(data, labels)
        
        # Calculate training metrics
        predictions = self.model.predict(data)
        probabilities = self.model.predict_proba(data)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1_score': f1_score(labels, predictions, average='weighted')
        }
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'metrics': metrics
        }
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        predictions = self.model.predict(data)
        probabilities = self.model.predict_proba(data)[:, 1] if hasattr(self.model, 'predict_proba') else np.ones_like(predictions) * 0.5
        return predictions, probabilities
    
    def get_confidence(self, data: pd.DataFrame) -> np.ndarray:
        """Get confidence scores for predictions."""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(data)
            return np.max(proba, axis=1)
        else:
            return np.ones(len(data)) * 0.5

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
    
    def add_model(self, model: FireClassificationModel, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append({'model': model, 'weight': weight})
    
    def train(self, data: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        results = []
        for model_info in self.models:
            result = model_info['model'].train(data, labels)
            results.append(result)
        
        # Calculate ensemble predictions
        predictions, probabilities = self.predict(data)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1_score': f1_score(labels, predictions, average='weighted')
        }
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'metrics': metrics,
            'individual_results': results
        }
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions."""
        all_predictions = []
        all_probabilities = []
        
        for model_info in self.models:
            pred, prob = model_info['model'].predict(data)
            all_predictions.append(pred * model_info['weight'])
            all_probabilities.append(prob * model_info['weight'])
        
        if self.voting == 'hard':
            # Majority voting
            ensemble_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
        else:
            # Soft voting (weighted average of probabilities)
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)
        
        return ensemble_predictions, ensemble_probabilities
    
    def get_confidence(self, data: pd.DataFrame) -> np.ndarray:
        """Get ensemble confidence scores."""
        all_confidences = []
        
        for model_info in self.models:
            confidence = model_info['model'].get_confidence(data)
            all_confidences.append(confidence * model_info['weight'])
        
        return np.mean(all_confidences, axis=0)

__all__ = ['BinaryFireClassifier', 'EnsembleFireClassifier']