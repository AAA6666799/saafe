"""
Base interfaces for fire prediction models.

This module defines the base interfaces for various fire prediction models:
- FireModel: Base interface for all fire models
- FireClassificationModel: For fire detection and classification
- FireIdentificationModel: For identifying specific fire types
- FireProgressionModel: For predicting fire growth and spread
- ConfidenceEstimationModel: For estimating prediction confidence
"""

import os
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List


class FireModel(ABC):
    """
    Base interface for all fire prediction models.
    
    This abstract class defines the common functionality and interface
    that all fire prediction models must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fire model.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Optional tuple of (X_val, y_val) for validation
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        pass
    
    @abstractmethod
    def save_metadata(self, path: str) -> None:
        """
        Save model metadata to disk.
        
        Args:
            path: Path to save the metadata
        """
        pass


class FireClassificationModel(FireModel):
    """
    Base interface for fire classification models.
    
    This abstract class defines the interface for models that classify
    whether a fire is present or categorize fires into different types.
    """
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of class probabilities
        """
        pass


class FireIdentificationModel(FireModel):
    """
    Base interface for fire identification models.
    
    This abstract class defines the interface for models that identify
    specific characteristics of fires, such as fuel type, ignition source, etc.
    """
    
    @abstractmethod
    def identify_characteristics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify fire characteristics.
        
        Args:
            X: Features to analyze
            
        Returns:
            Dictionary of identified characteristics
        """
        pass
    
    @abstractmethod
    def get_confidence_scores(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get confidence scores for identified characteristics.
        
        Args:
            X: Features to analyze
            
        Returns:
            Dictionary mapping characteristics to confidence scores
        """
        pass


class FireProgressionModel(FireModel):
    """
    Base interface for fire progression models.
    
    This abstract class defines the interface for models that predict
    how fires will grow and spread over time.
    """
    
    @abstractmethod
    def predict_progression(self, 
                          X: pd.DataFrame, 
                          time_steps: int,
                          time_interval: str = 'minutes') -> Dict[str, np.ndarray]:
        """
        Predict fire progression over time.
        
        Args:
            X: Current fire state features
            time_steps: Number of time steps to predict
            time_interval: Time interval between steps ('seconds', 'minutes', 'hours')
            
        Returns:
            Dictionary mapping prediction types to arrays of predictions over time
        """
        pass
    
    @abstractmethod
    def predict_time_to_threshold(self, 
                                X: pd.DataFrame,
                                threshold_type: str,
                                threshold_value: float) -> Dict[str, float]:
        """
        Predict time until a fire reaches a specified threshold.
        
        Args:
            X: Current fire state features
            threshold_type: Type of threshold ('size', 'temperature', 'intensity', etc.)
            threshold_value: Value of the threshold
            
        Returns:
            Dictionary with predicted time and confidence interval
        """
        pass


class ConfidenceEstimationModel(FireModel):
    """
    Base interface for confidence estimation models.
    
    This abstract class defines the interface for models that estimate
    the confidence or uncertainty in fire predictions.
    """
    
    @abstractmethod
    def estimate_uncertainty(self, X: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """
        Estimate uncertainty in predictions.
        
        Args:
            X: Features used for prediction
            predictions: Predictions made by another model
            
        Returns:
            Array of uncertainty estimates
        """
        pass
    
    @abstractmethod
    def calibrate_probabilities(self, 
                              X: pd.DataFrame, 
                              probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate probability estimates.
        
        Args:
            X: Features used for prediction
            probabilities: Uncalibrated probability estimates
            
        Returns:
            Array of calibrated probabilities
        """
        pass
    
    @abstractmethod
    def get_confidence_intervals(self, 
                               X: pd.DataFrame, 
                               predictions: np.ndarray,
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals for predictions.
        
        Args:
            X: Features used for prediction
            predictions: Point predictions
            confidence_level: Confidence level (e.g., 0.95 for 95% confidence)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        pass