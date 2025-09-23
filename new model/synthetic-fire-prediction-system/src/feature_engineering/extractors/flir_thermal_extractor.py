"""
FLIR Lepton 3.5 Thermal Data Extractor.

Processes thermal features from FLIR Lepton 3.5 cameras:
- t_mean, t_std: Average temperature and variation
- t_max, t_p95: Hottest pixel and 95th percentile
- t_hot_area_pct, t_hot_largest_blob_pct: Hot area percentages
- t_grad_mean, t_grad_std: Gradient sharpness
- t_diff_mean, t_diff_std: Frame-to-frame changes
- flow_mag_mean, flow_mag_std: Optical flow (motion)
- tproxy_val, tproxy_delta, tproxy_vel: Hotspot proxy values
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FlirThermalExtractor:
    """Extract and process features from FLIR Lepton 3.5 thermal data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FLIR thermal extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_names = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel'
        ]
        
        # Thresholds for fire detection
        self.thresholds = {
            't_max': 60.0,           # °C
            't_hot_area_pct': 10.0,   # %
            'tproxy_vel': 2.0,       # Rate of change
            't_grad_mean': 3.0       # Sharpness threshold
        }
        
        logger.info("Intialized FLIR Lepton 3.5 Thermal Extractor")
        logger.info(f"Processing {len(self.feature_names)} thermal features")
    
    def extract_features(self, thermal_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Extract features from FLIR thermal data.
        
        Args:
            thermal_data: Dictionary with FLIR thermal features
            
        Returns:
            Dictionary with extracted features and derived metrics
        """
        # Validate input data
        if not self._validate_thermal_data(thermal_data):
            raise ValueError("Invalid FLIR thermal data format")
        
        # Extract base features
        features = {}
        for feature_name in self.feature_names:
            features[feature_name] = thermal_data.get(feature_name, 0.0)
        
        # Add derived features
        derived_features = self._compute_derived_features(thermal_data)
        features.update(derived_features)
        
        # Add fire indicators
        fire_indicators = self._compute_fire_indicators(thermal_data)
        features.update(fire_indicators)
        
        # Add quality metrics
        quality_metrics = self._compute_quality_metrics(thermal_data)
        features.update(quality_metrics)
        
        features['timestamp'] = datetime.now().isoformat()
        features['extraction_success'] = True
        
        return features
    
    def _validate_thermal_data(self, data: Dict[str, float]) -> bool:
        """Validate FLIR thermal data format."""
        required_fields = self.feature_names
        return all(field in data for field in required_fields)
    
    def _compute_derived_features(self, thermal_data: Dict[str, float]) -> Dict[str, float]:
        """Compute derived features from base thermal data."""
        derived = {}
        
        # Temperature ratios
        t_max = thermal_data.get('t_max', 0.0)
        t_mean = thermal_data.get('t_mean', 0.0)
        t_p95 = thermal_data.get('t_p95', 0.0)
        
        derived['t_max_to_mean_ratio'] = t_max / t_mean if t_mean > 0 else 0.0
        derived['t_p95_to_max_ratio'] = t_p95 / t_max if t_max > 0 else 0.0
        
        # Gradient activity
        t_grad_mean = thermal_data.get('t_grad_mean', 0.0)
        t_grad_std = thermal_data.get('t_grad_std', 0.0)
        derived['t_grad_activity'] = t_grad_mean * t_grad_std
        
        # Motion activity
        flow_mag_mean = thermal_data.get('flow_mag_mean', 0.0)
        flow_mag_std = thermal_data.get('flow_mag_std', 0.0)
        derived['flow_activity'] = flow_mag_mean * flow_mag_std
        
        # Hotspot dynamics
        tproxy_delta = thermal_data.get('tproxy_delta', 0.0)
        tproxy_vel = thermal_data.get('tproxy_vel', 0.0)
        derived['tproxy_acceleration'] = abs(tproxy_vel - tproxy_delta) if tproxy_delta != 0 else 0.0
        
        return derived
    
    def _compute_fire_indicators(self, thermal_data: Dict[str, float]) -> Dict[str, float]:
        """Compute fire detection indicators."""
        indicators = {}
        
        # Hot temperature indicator
        t_max = thermal_data.get('t_max', 0.0)
        indicators['hot_temp_indicator'] = 1.0 if t_max > self.thresholds['t_max'] else 0.0
        
        # Hot area indicator
        t_hot_area_pct = thermal_data.get('t_hot_area_pct', 0.0)
        indicators['hot_area_indicator'] = 1.0 if t_hot_area_pct > self.thresholds['t_hot_area_pct'] else 0.0
        
        # Rapid change indicator
        tproxy_vel = thermal_data.get('tproxy_vel', 0.0)
        indicators['rapid_change_indicator'] = 1.0 if abs(tproxy_vel) > self.thresholds['tproxy_vel'] else 0.0
        
        # Sharp gradient indicator
        t_grad_mean = thermal_data.get('t_grad_mean', 0.0)
        indicators['sharp_gradient_indicator'] = 1.0 if t_grad_mean > self.thresholds['t_grad_mean'] else 0.0
        
        # Combined fire score (0-1)
        fire_indicators = [
            indicators['hot_temp_indicator'],
            indicators['hot_area_indicator'],
            indicators['rapid_change_indicator'],
            indicators['sharp_gradient_indicator']
        ]
        indicators['fire_likelihood_score'] = sum(fire_indicators) / len(fire_indicators)
        
        return indicators
    
    def _compute_quality_metrics(self, thermal_data: Dict[str, float]) -> Dict[str, float]:
        """Compute data quality metrics."""
        quality = {}
        
        # Data completeness (all required fields present)
        quality['data_completeness'] = 1.0  # Already validated
        
        # Value ranges check
        t_mean = thermal_data.get('t_mean', 0.0)
        quality['temp_range_valid'] = 1.0 if -50 <= t_mean <= 150 else 0.0  # Reasonable temp range
        
        t_hot_area_pct = thermal_data.get('t_hot_area_pct', 0.0)
        quality['hot_area_valid'] = 1.0 if 0 <= t_hot_area_pct <= 100 else 0.0
        
        # Consistency checks
        t_max = thermal_data.get('t_max', 0.0)
        t_p95 = thermal_data.get('t_p95', 0.0)
        quality['temp_consistency'] = 1.0 if t_max >= t_p95 >= t_mean else 0.0
        
        # Overall quality score
        quality_scores = [
            quality['data_completeness'],
            quality['temp_range_valid'],
            quality['hot_area_valid'],
            quality['temp_consistency']
        ]
        quality['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return quality
    
    def extract_batch_features(self, thermal_data_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Extract features from a batch of thermal data readings.
        
        Args:
            thermal_data_list: List of FLIR thermal data dictionaries
            
        Returns:
            List of extracted feature dictionaries
        """
        results = []
        for thermal_data in thermal_data_list:
            try:
                features = self.extract_features(thermal_data)
                results.append(features)
            except Exception as e:
                logger.error(f"Error processing thermal data: {str(e)}")
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
            't_max_to_mean_ratio', 't_p95_to_max_ratio', 't_grad_activity',
            'flow_activity', 'tproxy_acceleration'
        ]
        
        # Add indicator names
        indicator_names = [
            'hot_temp_indicator', 'hot_area_indicator', 'rapid_change_indicator',
            'sharp_gradient_indicator', 'fire_likelihood_score'
        ]
        
        # Add quality metric names
        quality_names = [
            'data_completeness', 'temp_range_valid', 'hot_area_valid',
            'temp_consistency', 'overall_quality_score'
        ]
        
        all_features = base_features + derived_names + indicator_names + quality_names
        all_features.extend(['timestamp', 'extraction_success'])
        
        return all_features


# Convenience function for creating extractor
def create_flir_thermal_extractor(config: Optional[Dict[str, Any]] = None) -> FlirThermalExtractor:
    """
    Create a FLIR thermal extractor with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FlirThermalExtractor instance
    """
    return FlirThermalExtractor(config)