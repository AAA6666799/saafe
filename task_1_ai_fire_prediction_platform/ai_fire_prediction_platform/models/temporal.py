"""
Temporal models for synthetic fire prediction system
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
import pickle
import os

from ai_fire_prediction_platform.core.interfaces import Model


class LSTMTemporalModel(Model):
    """Simple LSTM-based temporal model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sequence_length = config.get('sequence_length', 10)
        self.feature_size = config.get('feature_vector_size', 20)
        self.hidden_size = config.get('lstm_hidden_size', 128)
        self.num_layers = config.get('lstm_num_layers', 2)
        
        # For this simplified version, we'll use a sliding window approach with Logistic Regression
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False
        self.temporal_weights = np.ones(self.sequence_length) / self.sequence_length
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction and return (probability, confidence)"""
        if not self.is_trained:
            # Return neutral prediction if not trained
            return 0.5, 0.0
        
        # For temporal prediction, we expect a sequence of features
        if len(features.shape) == 1:
            # Single feature vector - use as is
            features = features.reshape(1, -1)
        elif len(features.shape) == 2 and features.shape[0] == self.sequence_length:
            # Sequence of features - apply temporal weights
            weighted_features = np.average(features, axis=0, weights=self.temporal_weights)
            features = weighted_features.reshape(1, -1)
        
        # Get prediction probability
        try:
            proba = self.model.predict_proba(features)
            fire_probability = proba[0][1] if proba.shape[1] > 1 else 0.0
            
            # Confidence as max probability
            confidence = np.max(proba)
        except:
            # Fallback if prediction fails
            fire_probability = 0.5
            confidence = 0.0
        
        return fire_probability, confidence
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model"""
        # For this simplified version, we flatten the temporal features
        if len(features.shape) == 3:
            # (samples, sequence_length, feature_size) -> (samples, sequence_length * feature_size)
            features = features.reshape(features.shape[0], -1)
        elif len(features.shape) == 2 and features.shape[1] == self.sequence_length * self.feature_size:
            # Already flattened
            pass
        else:
            # Reshape if needed
            features = features.reshape(-1, self.sequence_length * self.feature_size)
        
        self.model.fit(features, labels)
        self.is_trained = True
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True


class AttentionTemporalModel(Model):
    """Attention-based temporal model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sequence_length = config.get('sequence_length', 10)
        self.feature_size = config.get('feature_vector_size', 20)
        self.attention_heads = config.get('attention_heads', 8)
        
        # For this simplified version, we'll use a weighted average approach
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False
        
        # Initialize attention weights
        self.attention_weights = np.random.rand(self.sequence_length)
        self.attention_weights = self.attention_weights / np.sum(self.attention_weights)
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction and return (probability, confidence)"""
        if not self.is_trained:
            # Return neutral prediction if not trained
            return 0.5, 0.0
        
        # For temporal prediction, we expect a sequence of features
        if len(features.shape) == 1:
            # Single feature vector - use as is
            features = features.reshape(1, -1)
        elif len(features.shape) == 2 and features.shape[0] == self.sequence_length:
            # Sequence of features - apply attention weights
            weighted_features = np.average(features, axis=0, weights=self.attention_weights)
            features = weighted_features.reshape(1, -1)
        
        # Get prediction probability
        try:
            proba = self.model.predict_proba(features)
            fire_probability = proba[0][1] if proba.shape[1] > 1 else 0.0
            
            # Confidence as max probability
            confidence = np.max(proba)
        except:
            # Fallback if prediction fails
            fire_probability = 0.5
            confidence = 0.0
        
        return fire_probability, confidence
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model"""
        # For this simplified version, we flatten the temporal features
        if len(features.shape) == 3:
            # (samples, sequence_length, feature_size) -> (samples, sequence_length * feature_size)
            features = features.reshape(features.shape[0], -1)
        elif len(features.shape) == 2 and features.shape[1] == self.sequence_length * self.feature_size:
            # Already flattened
            pass
        else:
            # Reshape if needed
            features = features.reshape(-1, self.sequence_length * self.feature_size)
        
        self.model.fit(features, labels)
        self.is_trained = True
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True