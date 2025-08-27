"""
Base interfaces for feature extraction components.

This module defines the core interfaces and abstract classes for all feature extraction
components in the synthetic fire prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime


class FeatureExtractor(ABC):
    """
    Base abstract class for all feature extractors in the system.
    
    This class defines the common interface that all feature extractors must implement,
    regardless of the type of data they process (thermal, gas, environmental).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature extractor with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for the extractor
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract features from the input data.
        
        Args:
            data: Input data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        pass
    
    @abstractmethod
    def to_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert extracted features to a pandas DataFrame.
        
        Args:
            features: Extracted features from the extract_features method
            
        Returns:
            DataFrame containing the features in a structured format
        """
        pass
    
    @abstractmethod
    def save(self, features: Dict[str, Any], filepath: str) -> None:
        """
        Save extracted features to a file.
        
        Args:
            features: Extracted features from the extract_features method
            filepath: Path to save the features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        pass


class ThermalFeatureExtractor(FeatureExtractor):
    """
    Abstract base class for thermal feature extractors.
    
    This class extends the base FeatureExtractor with thermal-specific methods.
    """
    
    @abstractmethod
    def extract_temperature_statistics(self, 
                                      thermal_frame: np.ndarray,
                                      regions: Optional[List[Tuple[int, int, int, int]]] = None) -> Dict[str, float]:
        """
        Extract temperature statistics from a thermal frame.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            regions: Optional list of regions (x, y, width, height) to analyze
            
        Returns:
            Dictionary containing temperature statistics (min, max, mean, etc.)
        """
        pass
    
    @abstractmethod
    def detect_hotspots(self, 
                       thermal_frame: np.ndarray,
                       threshold_temp: float) -> List[Dict[str, Any]]:
        """
        Detect hotspots in a thermal frame.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            threshold_temp: Temperature threshold for hotspot detection
            
        Returns:
            List of dictionaries containing hotspot information
        """
        pass


class GasFeatureExtractor(FeatureExtractor):
    """
    Abstract base class for gas feature extractors.
    
    This class extends the base FeatureExtractor with gas-specific methods.
    """
    
    @abstractmethod
    def calculate_concentration_slope(self,
                                     gas_readings: pd.DataFrame,
                                     gas_type: str,
                                     window_sizes: List[int]) -> Dict[str, float]:
        """
        Calculate concentration slope over different time windows.
        
        Args:
            gas_readings: DataFrame containing gas concentration readings
            gas_type: Type of gas to analyze
            window_sizes: List of window sizes (in samples) to calculate slopes for
            
        Returns:
            Dictionary mapping window size to calculated slope
        """
        pass
    
    @abstractmethod
    def detect_concentration_peaks(self,
                                  gas_readings: pd.DataFrame,
                                  gas_type: str,
                                  threshold: float) -> List[Dict[str, Any]]:
        """
        Detect peaks in gas concentration.
        
        Args:
            gas_readings: DataFrame containing gas concentration readings
            gas_type: Type of gas to analyze
            threshold: Threshold for peak detection
            
        Returns:
            List of dictionaries containing peak information
        """
        pass


class EnvironmentalFeatureExtractor(FeatureExtractor):
    """
    Abstract base class for environmental feature extractors.
    
    This class extends the base FeatureExtractor with environmental-specific methods.
    """
    
    @abstractmethod
    def calculate_environmental_statistics(self,
                                         env_readings: pd.DataFrame,
                                         parameters: List[str],
                                         window_size: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for environmental parameters.
        
        Args:
            env_readings: DataFrame containing environmental readings
            parameters: List of environmental parameters to analyze
            window_size: Window size (in samples) for statistics calculation
            
        Returns:
            Nested dictionary mapping parameter to statistics
        """
        pass
    
    @abstractmethod
    def calculate_dew_point(self,
                           temperature: float,
                           humidity: float) -> float:
        """
        Calculate dew point from temperature and humidity.
        
        Args:
            temperature: Temperature in degrees Celsius
            humidity: Relative humidity as a percentage
            
        Returns:
            Dew point in degrees Celsius
        """
        pass


class FeatureFusion(ABC):
    """
    Abstract base class for feature fusion components.
    
    This class defines the interface for combining features from multiple extractors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature fusion component.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        pass
    
    @abstractmethod
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        pass