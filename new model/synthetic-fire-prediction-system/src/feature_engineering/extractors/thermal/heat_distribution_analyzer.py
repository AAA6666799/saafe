"""
Heat Distribution Skewness Analysis.

Statistical measures of temperature distribution to identify fire patterns.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HeatDistributionAnalyzer:
    """Analyzes statistical properties of temperature distribution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the heat distribution analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.skewness_threshold = self.config.get('skewness_threshold', 1.0)
        self.kurtosis_threshold = self.config.get('kurtosis_threshold', 3.0)
        
        logger.info("Initialized Heat Distribution Analyzer")
        logger.info(f"Skewness threshold: {self.skewness_threshold}")
    
    def analyze_heat_distribution(self, thermal_features: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze statistical properties of temperature distribution.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Dictionary with heat distribution features
        """
        try:
            # Extract relevant features for distribution analysis
            t_mean = thermal_features.get('t_mean', 0.0)
            t_std = thermal_features.get('t_std', 0.0)
            t_max = thermal_features.get('t_max', 0.0)
            t_p95 = thermal_features.get('t_p95', 0.0)
            
            # Calculate distribution metrics
            distribution_features = {}
            
            # Skewness estimation (using relationship between mean, median, and mode)
            # For fire detection, we expect positive skewness (long tail to hot side)
            estimated_median = t_mean  # Simplified assumption
            skewness = 3 * (t_mean - estimated_median) / t_std if t_std > 0 else 0.0
            distribution_features['temp_skewness'] = skewness
            distribution_features['skewness_indicator'] = 1.0 if skewness > self.skewness_threshold else 0.0
            
            # Kurtosis estimation (peakedness of distribution)
            # Fire patterns may have high kurtosis (concentrated hot spots)
            kurtosis = self._estimate_kurtosis(t_mean, t_std, t_max, t_p95)
            distribution_features['temp_kurtosis'] = kurtosis
            distribution_features['kurtosis_indicator'] = 1.0 if kurtosis > self.kurtosis_threshold else 0.0
            
            # Distribution shape features
            distribution_features['temp_range'] = t_max - t_mean
            distribution_features['percentile_range'] = t_max - t_p95
            
            # Fire distribution signature
            distribution_features['fire_distribution_score'] = self._calculate_fire_distribution_score(
                thermal_features)
            
            return distribution_features
            
        except Exception as e:
            logger.error(f"Error in heat distribution analysis: {str(e)}")
            return {}
    
    def _estimate_kurtosis(self, t_mean: float, t_std: float, t_max: float, t_p95: float) -> float:
        """
        Estimate kurtosis from available temperature statistics.
        
        Args:
            t_mean: Mean temperature
            t_std: Standard deviation of temperature
            t_max: Maximum temperature
            t_p95: 95th percentile temperature
            
        Returns:
            Estimated kurtosis value
        """
        # Simplified kurtosis estimation based on extreme values
        if t_std > 0:
            # Higher kurtosis when extreme values are further from the mean
            max_deviation = (t_max - t_mean) / t_std
            p95_deviation = (t_p95 - t_mean) / t_std
            kurtosis = (max_deviation + p95_deviation) / 2.0
            return max(0.0, kurtosis)
        return 0.0
    
    def _calculate_fire_distribution_score(self, thermal_features: Dict[str, float]) -> float:
        """
        Calculate fire distribution signature score.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Fire distribution score (0-1)
        """
        t_mean = thermal_features.get('t_mean', 0.0)
        t_max = thermal_features.get('t_max', 0.0)
        t_std = thermal_features.get('t_std', 0.0)
        
        # Fire distributions typically have:
        # 1. High skewness (concentration of heat)
        # 2. High kurtosis (peaked distribution)
        # 3. Large temperature range
        
        # Normalize features
        temp_range = t_max - t_mean
        normalized_range = min(1.0, temp_range / 50.0)  # Assume 50°C as reference range
        normalized_std = min(1.0, t_std / 20.0)  # Assume 20°C as reference std
        
        # Combined score
        fire_score = 0.5 * normalized_range + 0.5 * normalized_std
        return fire_score