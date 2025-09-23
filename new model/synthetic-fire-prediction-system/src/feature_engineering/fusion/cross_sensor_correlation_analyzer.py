"""
Cross-Sensor Correlation Analysis.

Real-time correlation analysis between thermal and gas sensor data.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging
import sys
import os

# Add the src directory to the path to fix relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)


class CrossSensorCorrelationAnalyzer:
    """Analyzes correlation between thermal and gas sensor data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the cross-sensor correlation analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.correlation_window = self.config.get('correlation_window', 5)  # Number of paired readings
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)  # Minimum correlation for fire
        self.time_alignment_window = self.config.get('time_alignment_window', 30)  # Seconds for temporal alignment
        
        logger.info("Initialized Cross-Sensor Correlation Analyzer")
        logger.info(f"Correlation threshold: {self.correlation_threshold}")
    
    def analyze_cross_sensor_correlation(self, 
                                       thermal_history: List[Dict[str, float]], 
                                       gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze correlation between thermal and gas sensor data.
        
        Args:
            thermal_history: List of recent thermal feature dictionaries
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with cross-sensor correlation features
        """
        if len(thermal_history) < 2 or len(gas_history) < 2:
            return {}
        
        try:
            # Calculate correlation features
            correlation_features = {}
            
            # Temporal alignment
            aligned_data = self._align_sensor_data(thermal_history, gas_history)
            if not aligned_data:
                return {}
            
            # Correlation analysis
            correlation_metrics = self._calculate_correlations(aligned_data)
            correlation_features.update(correlation_metrics)
            
            # Synchronization analysis
            sync_features = self._analyze_synchronization(aligned_data)
            correlation_features.update(sync_features)
            
            # Fire correlation signature
            fire_signature = self._detect_fire_correlation_signature(aligned_data)
            correlation_features.update(fire_signature)
            
            return correlation_features
            
        except Exception as e:
            logger.error(f"Error in cross-sensor correlation analysis: {str(e)}")
            return {}
    
    def _align_sensor_data(self, 
                          thermal_history: List[Dict[str, float]], 
                          gas_history: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Align thermal and gas sensor data based on timestamps.
        
        Args:
            thermal_history: List of recent thermal feature dictionaries
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            List of aligned data points
        """
        # For simplicity, we'll assume data is already aligned temporally
        # In a real implementation, you would match timestamps within the alignment window
        
        min_length = min(len(thermal_history), len(gas_history), self.correlation_window)
        
        if min_length < 2:
            return []
        
        # Take the most recent readings
        aligned_data = []
        for i in range(min_length):
            thermal_idx = -(i + 1)
            gas_idx = -(i + 1)
            
            aligned_point = {
                'thermal': thermal_history[thermal_idx],
                'gas': gas_history[gas_idx],
                'timestamp': thermal_history[thermal_idx].get('timestamp', datetime.now().isoformat())
            }
            aligned_data.append(aligned_point)
        
        # Reverse to get chronological order
        return list(reversed(aligned_data))
    
    def _calculate_correlations(self, aligned_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate correlations between thermal and gas features.
        
        Args:
            aligned_data: List of aligned thermal and gas data points
            
        Returns:
            Dictionary with correlation features
        """
        features = {}
        
        if len(aligned_data) < 2:
            return features
        
        # Extract key features for correlation analysis
        thermal_temps = [point['thermal'].get('t_mean', 0.0) for point in aligned_data]
        gas_values = [point['gas'].get('gas_val', 0.0) for point in aligned_data]
        
        thermal_deltas = [point['thermal'].get('tproxy_delta', 0.0) for point in aligned_data]
        gas_deltas = [point['gas'].get('gas_delta', 0.0) for point in aligned_data]
        
        # Calculate correlations
        temp_gas_corr = self._calculate_correlation(thermal_temps, gas_values)
        delta_corr = self._calculate_correlation(thermal_deltas, gas_deltas)
        
        features['temp_co2_correlation'] = temp_gas_corr
        features['delta_correlation'] = delta_corr
        
        # Correlation strength indicators
        features['strong_temp_co2_correlation'] = 1.0 if abs(temp_gas_corr) > self.correlation_threshold else 0.0
        features['strong_delta_correlation'] = 1.0 if abs(delta_corr) > self.correlation_threshold else 0.0
        
        # Positive correlation (both increasing together)
        features['positive_correlation'] = 1.0 if temp_gas_corr > 0 else 0.0
        
        return features
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            x: First series of values
            y: Second series of values
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        # Calculate means
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Calculate numerator and denominators
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denom_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        denom_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        
        denominator = np.sqrt(denom_x * denom_y)
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation
    
    def _analyze_synchronization(self, aligned_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze synchronization between thermal and gas sensors.
        
        Args:
            aligned_data: List of aligned thermal and gas data points
            
        Returns:
            Dictionary with synchronization features
        """
        features = {}
        
        if len(aligned_data) < 2:
            return features
        
        # Extract timing information
        timestamps = [point['timestamp'] for point in aligned_data]
        
        # In a real implementation, you would calculate actual time differences
        # For now, we'll assume perfect synchronization
        features['synchronization_quality'] = 1.0  # Perfect synchronization assumed
        
        # Consistency of correlation over time
        if len(aligned_data) >= 3:
            # Calculate correlation for first and second half
            mid_point = len(aligned_data) // 2
            first_half = aligned_data[:mid_point]
            second_half = aligned_data[mid_point:]
            
            # Extract features for each half
            first_temps = [point['thermal'].get('t_mean', 0.0) for point in first_half]
            first_gas = [point['gas'].get('gas_val', 0.0) for point in first_half]
            second_temps = [point['thermal'].get('t_mean', 0.0) for point in second_half]
            second_gas = [point['gas'].get('gas_val', 0.0) for point in second_half]
            
            first_corr = self._calculate_correlation(first_temps, first_gas)
            second_corr = self._calculate_correlation(second_temps, second_gas)
            
            # Consistency measure
            correlation_consistency = 1.0 - abs(first_corr - second_corr)
            features['correlation_consistency'] = max(0.0, correlation_consistency)
        
        return features
    
    def _detect_fire_correlation_signature(self, aligned_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Detect fire signature in cross-sensor correlations.
        
        Args:
            aligned_data: List of aligned thermal and gas data points
            
        Returns:
            Dictionary with fire correlation signature features
        """
        features = {}
        
        if len(aligned_data) < 2:
            return features
        
        # Extract key features
        thermal_temps = [point['thermal'].get('t_mean', 0.0) for point in aligned_data]
        gas_values = [point['gas'].get('gas_val', 0.0) for point in aligned_data]
        
        thermal_deltas = [point['thermal'].get('tproxy_delta', 0.0) for point in aligned_data]
        gas_deltas = [point['gas'].get('gas_delta', 0.0) for point in aligned_data]
        
        # Fire correlation signatures:
        # 1. Positive correlation between temperature and CO₂
        # 2. Consistent correlation over time
        # 3. Strong correlation magnitude
        
        correlation = self._calculate_correlation(thermal_temps, gas_values)
        
        # Normalize correlation for fire signature (0-1, where 1 indicates strong fire signature)
        fire_correlation_score = max(0.0, correlation)  # Only positive correlations indicate fire
        features['fire_correlation_signature'] = fire_correlation_score
        
        return features