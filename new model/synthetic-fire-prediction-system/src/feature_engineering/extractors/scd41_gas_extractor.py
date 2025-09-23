"""
Sensirion SCD41 CO₂ Gas Data Extractor.

Processes gas features from Sensirion SCD41 sensors:
- gas_val: Current CO₂ concentration (ppm)
- gas_delta: Change from previous reading
- gas_vel: Rate of change (same as delta)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Scd41GasExtractor:
    """Extract and process features from Sensirion SCD41 CO₂ gas data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SCD41 gas extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_names = ['gas_val', 'gas_delta', 'gas_vel']
        
        # Thresholds for fire detection
        self.thresholds = {
            'gas_val': 1000.0,    # ppm CO₂
            'gas_delta': 50.0,    # ppm change
            'gas_vel': 50.0       # Rate of change
        }
        
        # Normal CO₂ levels
        self.normal_co2 = 450.0  # Outdoor CO₂ level
        
        logger.info("Initialized Sensirion SCD41 Gas Extractor")
        logger.info(f"Processing {len(self.feature_names)} gas features")
    
    def extract_features(self, gas_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Extract features from SCD41 gas data.
        
        Args:
            gas_data: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with extracted features and derived metrics
        """
        # Validate input data
        if not self._validate_gas_data(gas_data):
            raise ValueError("Invalid SCD41 gas data format")
        
        # Extract base features
        features = {}
        for feature_name in self.feature_names:
            features[feature_name] = gas_data.get(feature_name, 0.0)
        
        # Add derived features
        derived_features = self._compute_derived_features(gas_data)
        features.update(derived_features)
        
        # Add fire indicators
        fire_indicators = self._compute_fire_indicators(gas_data)
        features.update(fire_indicators)
        
        # Add quality metrics
        quality_metrics = self._compute_quality_metrics(gas_data)
        features.update(quality_metrics)
        
        features['timestamp'] = datetime.now().isoformat()
        features['extraction_success'] = True
        
        return features
    
    def _validate_gas_data(self, data: Dict[str, float]) -> bool:
        """Validate SCD41 gas data format."""
        required_fields = self.feature_names
        return all(field in data for field in required_fields)
    
    def _compute_derived_features(self, gas_data: Dict[str, float]) -> Dict[str, float]:
        """Compute derived features from base gas data."""
        derived = {}
        
        # Gas concentration analysis
        gas_val = gas_data.get('gas_val', 0.0)
        derived['gas_concentration_level'] = gas_val / 1000.0  # Normalize to kppm
        
        # Change magnitude (absolute values)
        gas_delta = gas_data.get('gas_delta', 0.0)
        gas_vel = gas_data.get('gas_vel', 0.0)
        derived['gas_change_magnitude'] = abs(gas_delta)
        derived['gas_velocity_magnitude'] = abs(gas_vel)
        
        # Deviation from normal
        derived['gas_deviation_from_normal'] = abs(gas_val - self.normal_co2)
        
        # Relative change
        derived['gas_relative_change'] = gas_delta / gas_val if gas_val > 0 else 0.0
        
        # Acceleration (change in velocity)
        # Note: Since gas_vel and gas_delta are the same, acceleration is 0
        derived['gas_acceleration'] = 0.0
        
        return derived
    
    def _compute_fire_indicators(self, gas_data: Dict[str, float]) -> Dict[str, float]:
        """Compute fire detection indicators from gas data."""
        indicators = {}
        
        # High CO₂ indicator
        gas_val = gas_data.get('gas_val', 0.0)
        indicators['high_co2_indicator'] = 1.0 if gas_val > self.thresholds['gas_val'] else 0.0
        
        # Rapid increase indicator
        gas_delta = gas_data.get('gas_delta', 0.0)
        indicators['rapid_increase_indicator'] = 1.0 if gas_delta > self.thresholds['gas_delta'] else 0.0
        
        # High velocity indicator
        gas_vel = gas_data.get('gas_vel', 0.0)
        indicators['high_velocity_indicator'] = 1.0 if abs(gas_vel) > self.thresholds['gas_vel'] else 0.0
        
        # Combined gas score (0-1)
        gas_indicators = [
            indicators['high_co2_indicator'],
            indicators['rapid_increase_indicator'],
            indicators['high_velocity_indicator']
        ]
        indicators['gas_fire_likelihood_score'] = sum(gas_indicators) / len(gas_indicators)
        
        # CO₂ trend analysis
        if gas_val > self.normal_co2 * 2:  # Significantly above normal
            indicators['co2_elevation_level'] = 'HIGH'
        elif gas_val > self.normal_co2 * 1.5:  # Moderately above normal
            indicators['co2_elevation_level'] = 'MEDIUM'
        else:
            indicators['co2_elevation_level'] = 'LOW'
        
        return indicators
    
    def _compute_quality_metrics(self, gas_data: Dict[str, float]) -> Dict[str, float]:
        """Compute data quality metrics for gas data."""
        quality = {}
        
        # Data completeness
        quality['data_completeness'] = 1.0  # Already validated
        
        # Value ranges check
        gas_val = gas_data.get('gas_val', 0.0)
        # Normal CO₂ range: 300-5000 ppm for indoor environments
        quality['co2_range_valid'] = 1.0 if 300 <= gas_val <= 5000 else 0.0
        
        # Consistency check (gas_vel should equal gas_delta)
        gas_delta = gas_data.get('gas_delta', 0.0)
        gas_vel = gas_data.get('gas_vel', 0.0)
        quality['velocity_consistency'] = 1.0 if abs(gas_delta - gas_vel) < 0.1 else 0.0
        
        # Overall quality score
        quality_scores = [
            quality['data_completeness'],
            quality['co2_range_valid'],
            quality['velocity_consistency']
        ]
        quality['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return quality
    
    def extract_batch_features(self, gas_data_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Extract features from a batch of gas data readings.
        
        Args:
            gas_data_list: List of SCD41 gas data dictionaries
            
        Returns:
            List of extracted feature dictionaries
        """
        results = []
        for gas_data in gas_data_list:
            try:
                features = self.extract_features(gas_data)
                results.append(features)
            except Exception as e:
                logger.error(f"Error processing gas data: {str(e)}")
                # Add error entry
                error_features = {
                    'timestamp': datetime.now().isoformat(),
                    'extraction_success': False,
                    'error': str(e)
                }
                results.append(error_features)
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names produced by this extractor."""
        base_features = self.feature_names.copy()
        
        # Add derived feature names
        derived_names = [
            'gas_concentration_level', 'gas_change_magnitude', 'gas_velocity_magnitude',
            'gas_deviation_from_normal', 'gas_relative_change', 'gas_acceleration'
        ]
        
        # Add indicator names
        indicator_names = [
            'high_co2_indicator', 'rapid_increase_indicator', 'high_velocity_indicator',
            'gas_fire_likelihood_score', 'co2_elevation_level'
        ]
        
        # Add quality metric names
        quality_names = [
            'data_completeness', 'co2_range_valid', 'velocity_consistency',
            'overall_quality_score'
        ]
        
        all_features = base_features + derived_names + indicator_names + quality_names
        all_features.extend(['timestamp', 'extraction_success'])
        
        return all_features


# Convenience function for creating extractor
def create_scd41_gas_extractor(config: Optional[Dict[str, Any]] = None) -> Scd41GasExtractor:
    """
    Create a SCD41 gas extractor with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Scd41GasExtractor instance
    """
    return Scd41GasExtractor(config)