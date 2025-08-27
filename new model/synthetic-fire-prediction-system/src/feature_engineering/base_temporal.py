"""
Base interfaces for temporal feature extraction components.

This module defines the core interfaces and abstract classes for temporal feature extraction
components in the synthetic fire prediction system.
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from .base import FeatureExtractor


class TemporalFeatureExtractor(FeatureExtractor):
    """
    Abstract base class for temporal feature extractors.
    
    This class extends the base FeatureExtractor with temporal-specific methods
    for extracting features from time-series data.
    """
    
    @abstractmethod
    def extract_sequence_patterns(self,
                                time_series: pd.DataFrame,
                                column_name: str,
                                window_size: int) -> Dict[str, Any]:
        """
        Extract sequence patterns from time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            window_size: Size of the window for pattern extraction
            
        Returns:
            Dictionary containing extracted sequence patterns
        """
        pass
    
    @abstractmethod
    def analyze_trend(self,
                     time_series: pd.DataFrame,
                     column_name: str,
                     window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze trends in time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            window_size: Optional window size for trend analysis
            
        Returns:
            Dictionary containing trend analysis results
        """
        pass
    
    @abstractmethod
    def detect_seasonality(self,
                          time_series: pd.DataFrame,
                          column_name: str,
                          period: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect seasonality in time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            period: Optional period length for seasonality detection
            
        Returns:
            Dictionary containing seasonality detection results
        """
        pass
    
    @abstractmethod
    def detect_change_points(self,
                            time_series: pd.DataFrame,
                            column_name: str,
                            threshold: float) -> List[Dict[str, Any]]:
        """
        Detect change points in time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            threshold: Threshold for change point detection
            
        Returns:
            List of dictionaries containing change point information
        """
        pass
    
    @abstractmethod
    def detect_anomalies(self,
                        time_series: pd.DataFrame,
                        column_name: str,
                        window_size: int,
                        threshold: float) -> List[Dict[str, Any]]:
        """
        Detect anomalies in time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            window_size: Size of the window for anomaly detection
            threshold: Threshold for anomaly detection
            
        Returns:
            List of dictionaries containing anomaly information
        """
        pass