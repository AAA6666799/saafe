"""
CO₂ Accumulation Rate Analysis with Noise Filtering.

Calculates rate of CO₂ change with noise filtering for fire detection.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GasAccumulationAnalyzer:
    """Analyzes CO₂ accumulation rates with noise filtering."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the gas accumulation analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.noise_threshold = self.config.get('noise_threshold', 5.0)  # ppm
        self.accumulation_threshold = self.config.get('accumulation_threshold', 20.0)  # ppm per reading
        self.window_size = self.config.get('window_size', 3)  # Number of readings for smoothing
        
        logger.info("Initialized Gas Accumulation Analyzer")
        logger.info(f"Accumulation threshold: {self.accumulation_threshold} ppm")
    
    def analyze_accumulation_rate(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze CO₂ accumulation rate with noise filtering.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with accumulation rate features
        """
        if len(gas_history) < 2:
            return {}
        
        try:
            # Calculate accumulation features
            accumulation_features = {}
            
            # Basic rate calculation
            rate_features = self._calculate_basic_rate(gas_history)
            accumulation_features.update(rate_features)
            
            # Noise-filtered rate
            filtered_features = self._calculate_filtered_rate(gas_history)
            accumulation_features.update(filtered_features)
            
            # Trend analysis
            trend_features = self._analyze_trends(gas_history)
            accumulation_features.update(trend_features)
            
            # Fire indicator
            fire_indicator = self._calculate_fire_indicator(gas_history)
            accumulation_features.update(fire_indicator)
            
            return accumulation_features
            
        except Exception as e:
            logger.error(f"Error in gas accumulation analysis: {str(e)}")
            return {}
    
    def _calculate_basic_rate(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate basic CO₂ rate of change.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with basic rate features
        """
        features = {}
        
        if len(gas_history) < 2:
            return features
        
        # Get recent values
        current_val = gas_history[-1].get('gas_val', 0.0)
        previous_val = gas_history[-2].get('gas_val', 0.0)
        
        # Calculate basic rate
        basic_rate = current_val - previous_val
        features['co2_basic_rate'] = basic_rate
        features['co2_rate_magnitude'] = abs(basic_rate)
        
        # Rate indicator
        features['rapid_accumulation'] = 1.0 if abs(basic_rate) > self.accumulation_threshold else 0.0
        
        return features
    
    def _calculate_filtered_rate(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate noise-filtered CO₂ rate of change.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with filtered rate features
        """
        features = {}
        
        if len(gas_history) < self.window_size:
            return features
        
        # Extract recent values
        recent_vals = [reading.get('gas_val', 0.0) for reading in gas_history[-self.window_size:]]
        
        # Apply simple moving average to reduce noise
        smoothed_vals = self._simple_moving_average(recent_vals, 2)
        
        if len(smoothed_vals) >= 2:
            # Calculate filtered rate
            filtered_rate = smoothed_vals[-1] - smoothed_vals[-2]
            features['co2_filtered_rate'] = filtered_rate
            features['co2_filtered_magnitude'] = abs(filtered_rate)
            
            # Filtered rate indicator
            features['filtered_rapid_accumulation'] = (
                1.0 if abs(filtered_rate) > self.accumulation_threshold else 0.0
            )
        
        return features
    
    def _simple_moving_average(self, values: List[float], window: int) -> List[float]:
        """
        Calculate simple moving average.
        
        Args:
            values: List of values
            window: Window size for moving average
            
        Returns:
            List of smoothed values
        """
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values) - window + 1):
            window_vals = values[i:i+window]
            avg = sum(window_vals) / len(window_vals)
            smoothed.append(avg)
        
        return smoothed
    
    def _analyze_trends(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze trends in CO₂ accumulation.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with trend features
        """
        features = {}
        
        if len(gas_history) < self.window_size:
            return features
        
        # Extract recent values
        recent_vals = [reading.get('gas_val', 0.0) for reading in gas_history[-self.window_size:]]
        
        # Calculate trend using linear regression (simplified)
        if len(recent_vals) >= 2:
            x = list(range(len(recent_vals)))
            y = recent_vals
            
            # Simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_xx = sum(x[i] * x[i] for i in range(n))
            
            denominator = n * sum_xx - sum_x * sum_x
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                features['co2_trend_slope'] = slope
                features['positive_trend'] = 1.0 if slope > 0 else 0.0
        
        return features
    
    def _calculate_fire_indicator(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate fire indicator based on CO₂ accumulation patterns.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with fire indicator features
        """
        features = {}
        
        if len(gas_history) < self.window_size:
            return features
        
        # Extract recent values
        recent_vals = [reading.get('gas_val', 0.0) for reading in gas_history[-self.window_size:]]
        recent_rates = [reading.get('gas_delta', 0.0) for reading in gas_history[-self.window_size:]]
        
        # Fire indicators:
        # 1. Consistently positive rates
        # 2. Accelerating accumulation
        # 3. High CO₂ levels
        
        # Positive rate consistency
        positive_rates = [r for r in recent_rates if r > 0]
        consistency = len(positive_rates) / len(recent_rates) if recent_rates else 0.0
        
        # Average rate
        avg_rate = sum(recent_rates) / len(recent_rates) if recent_rates else 0.0
        normalized_rate = min(1.0, abs(avg_rate) / self.accumulation_threshold) if self.accumulation_threshold > 0 else 0.0
        
        # Combined fire indicator
        fire_indicator = 0.6 * consistency + 0.4 * normalized_rate
        features['gas_fire_indicator'] = fire_indicator
        
        return features