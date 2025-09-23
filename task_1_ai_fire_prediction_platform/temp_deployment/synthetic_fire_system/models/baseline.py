"""
Baseline models for synthetic fire prediction system
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import os

from synthetic_fire_system.core.interfaces import Model


class RandomForestModel(Model):
    """Random Forest baseline model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.get('rf_n_estimators', 100),
            max_depth=config.get('rf_max_depth', 10),
            random_state=42
        )
        self.is_trained = False
        self.feature_size = config.get('feature_vector_size', 20)
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction and return (probability, confidence)"""
        if not self.is_trained:
            # Return neutral prediction if not trained
            return 0.5, 0.0
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get prediction probability
        proba = self.model.predict_proba(features)
        fire_probability = proba[0][1] if proba.shape[1] > 1 else 0.0
        
        # Confidence as max probability
        confidence = np.max(proba)
        
        return fire_probability, confidence
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model"""
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
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


class LogisticRegressionModel(Model):
    """Logistic Regression baseline model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = LogisticRegression(
            random_state=42,
            max_iter=config.get('lr_max_iter', 1000)
        )
        self.is_trained = False
        self.feature_size = config.get('feature_vector_size', 20)
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction and return (probability, confidence)"""
        if not self.is_trained:
            # Return neutral prediction if not trained
            return 0.5, 0.0
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get prediction probability
        proba = self.model.predict_proba(features)
        fire_probability = proba[0][1] if proba.shape[1] > 1 else 0.0
        
        # Confidence as max probability
        confidence = np.max(proba)
        
        return fire_probability, confidence
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model"""
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
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


class SVMModel(Model):
    """Support Vector Machine baseline model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = SVC(
            probability=True,
            random_state=42
        )
        self.is_trained = False
        self.feature_size = config.get('feature_vector_size', 20)
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction and return (probability, confidence)"""
        if not self.is_trained:
            # Return neutral prediction if not trained
            return 0.5, 0.0
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get prediction probability
        proba = self.model.predict_proba(features)
        fire_probability = proba[0][1] if proba.shape[1] > 1 else 0.0
        
        # Confidence as max probability
        confidence = np.max(proba)
        
        return fire_probability, confidence
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model"""
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
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