"""
Temporal Signature Pattern Recognition for FLIR Thermal Data.

Identifies characteristic temperature rise patterns that indicate fire development.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TemporalSignatureAnalyzer:
    """Analyzes temporal patterns in thermal data to identify fire signatures."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the temporal signature analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.window_size = self.config.get('window_size', 5)  # Number of readings to analyze
        self.rise_threshold = self.config.get('rise_threshold', 2.0)  # °C per reading
        
        logger.info("Initialized Temporal Signature Analyzer")
        logger.info(f"Analysis window size: {self.window_size}")
    
    def analyze_temporal_patterns(self, thermal_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze temporal patterns in thermal data.
        
        Args:
            thermal_history: List of recent thermal feature dictionaries
            
        Returns:
            Dictionary with temporal pattern features
        """
        if len(thermal_history) < 2:
            return {}
        
        try:
            # Extract temperature trends
            temporal_features = {}
            
            # Temperature rise analysis
            temp_rise_features = self._analyze_temperature_rise(thermal_history)
            temporal_features.update(temp_rise_features)
            
            # Pattern consistency
            consistency_features = self._analyze_pattern_consistency(thermal_history)
            temporal_features.update(consistency_features)
            
            # Acceleration analysis
            acceleration_features = self._analyze_acceleration(thermal_history)
            temporal_features.update(acceleration_features)
            
            # Fire signature detection
            fire_signature = self._detect_fire_signature(thermal_history)
            temporal_features.update(fire_signature)
            
            return temporal_features
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {str(e)}")
            return {}
    
    def _analyze_temperature_rise(self, thermal_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze temperature rise patterns.
        
        Args:
            thermal_history: List of recent thermal feature dictionaries
            
        Returns:
            Dictionary with temperature rise features
        """
        features = {}
        
        if len(thermal_history) < 2:
            return features
        
        # Extract temperature values
        temps = [reading.get('t_mean', 0.0) for reading in thermal_history]
        max_temps = [reading.get('t_max', 0.0) for reading in thermal_history]
        
        # Calculate recent trends
        recent_temp_diff = temps[-1] - temps[-2] if len(temps) >= 2 else 0.0
        recent_max_diff = max_temps[-1] - max_temps[-2] if len(max_temps) >= 2 else 0.0
        
        features['temp_rise_rate'] = recent_temp_diff
        features['max_temp_rise_rate'] = recent_max_diff
        features['temp_rise_indicator'] = 1.0 if recent_temp_diff > self.rise_threshold else 0.0
        
        # Calculate average rise over window
        if len(temps) >= self.window_size:
            window_temps = temps[-self.window_size:]
            avg_rise = (window_temps[-1] - window_temps[0]) / (len(window_temps) - 1) if len(window_temps) > 1 else 0.0
            features['avg_temp_rise_rate'] = avg_rise
        
        return features
    
    def _analyze_pattern_consistency(self, thermal_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze consistency of temperature patterns.
        
        Args:
            thermal_history: List of recent thermal feature dictionaries
            
        Returns:
            Dictionary with pattern consistency features
        """
        features = {}
        
        if len(thermal_history) < self.window_size:
            return features
        
        # Extract temperature values
        temps = [reading.get('t_mean', 0.0) for reading in thermal_history[-self.window_size:]]
        
        # Calculate consistency metrics
        temp_std = np.std(temps) if len(temps) > 1 else 0.0
        temp_range = max(temps) - min(temps) if temps else 0.0
        
        features['temp_consistency'] = 1.0 / (1.0 + temp_std) if temp_std > 0 else 1.0
        features['temp_range_normalized'] = temp_range / (max(temps) + 1e-8) if max(temps) > 0 else 0.0
        
        return features
    
    def _analyze_acceleration(self, thermal_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze acceleration in temperature changes.
        
        Args:
            thermal_history: List of recent thermal feature dictionaries
            
        Returns:
            Dictionary with acceleration features
        """
        features = {}
        
        if len(thermal_history) < 3:
            return features
        
        # Extract temperature values
        temps = [reading.get('t_mean', 0.0) for reading in thermal_history[-3:]]
        
        # Calculate first derivatives (velocity)
        velocities = [temps[i+1] - temps[i] for i in range(len(temps)-1)]
        
        # Calculate second derivative (acceleration)
        if len(velocities) >= 2:
            acceleration = velocities[-1] - velocities[-2]
            features['temp_acceleration'] = acceleration
            features['positive_acceleration'] = 1.0 if acceleration > 0 else 0.0
        
        return features
    
    def _detect_fire_signature(self, thermal_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Detect fire signature patterns in temporal data.
        
        Args:
            thermal_history: List of recent thermal feature dictionaries
            
        Returns:
            Dictionary with fire signature features
        """
        features = {}
        
        if len(thermal_history) < self.window_size:
            return features
        
        # Extract key features
        temps = [reading.get('t_mean', 0.0) for reading in thermal_history[-self.window_size:]]
        max_temps = [reading.get('t_max', 0.0) for reading in thermal_history[-self.window_size:]]
        
        # Fire signatures typically show:
        # 1. Consistent temperature rise
        # 2. Accelerating temperature increase
        # 3. High maximum temperatures
        
        # Temperature rise consistency
        rises = [temps[i+1] - temps[i] for i in range(len(temps)-1)]
        positive_rises = [r for r in rises if r > 0]
        rise_consistency = len(positive_rises) / len(rises) if rises else 0.0
        
        # High temperature indicator
        avg_max_temp = np.mean(max_temps) if max_temps else 0.0
        high_temp_indicator = min(1.0, avg_max_temp / 60.0)  # Assume 60°C as reference
        
        # Combined fire signature score
        fire_signature_score = 0.5 * rise_consistency + 0.5 * high_temp_indicator
        features['fire_temporal_signature'] = fire_signature_score
        
        return features