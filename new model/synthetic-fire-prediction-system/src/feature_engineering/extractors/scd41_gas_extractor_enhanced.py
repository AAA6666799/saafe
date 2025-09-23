"""
Enhanced Sensirion SCD41 CO₂ Gas Data Extractor.

Processes gas features from Sensirion SCD41 sensors with enhanced analysis:
- CO₂ accumulation rate with noise filtering
- Baseline drift detection
- Advanced fire indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import sys
import os

# Add the src directory to the path to fix relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our new analysis modules
from feature_engineering.extractors.gas.gas_accumulation_analyzer import GasAccumulationAnalyzer
from feature_engineering.extractors.gas.baseline_drift_detector import BaselineDriftDetector

logger = logging.getLogger(__name__)


class Scd41GasExtractorEnhanced:
    """Enhanced extract and process features from Sensirion SCD41 CO₂ gas data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced SCD41 gas extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_names = ['gas_val', 'gas_delta', 'gas_vel']
        
        # Initialize enhanced analysis components
        self.accumulation_analyzer = GasAccumulationAnalyzer(self.config.get('accumulation_analysis', {}))
        self.drift_detector = BaselineDriftDetector(self.config.get('drift_detection', {}))
        
        # Thresholds for fire detection
        self.thresholds = {
            'gas_val': 1000.0,    # ppm CO₂
            'gas_delta': 50.0,    # ppm change
            'gas_vel': 50.0       # Rate of change
        }
        
        # Normal CO₂ levels
        self.normal_co2 = 450.0  # Outdoor CO₂ level
        
        logger.info("Initialized Enhanced Sensirion SCD41 Gas Extractor")
        logger.info(f"Processing {len(self.feature_names)} base gas features")
        logger.info("Enhanced analysis modules loaded")
    
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
        
        # Add derived features from base extractor
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
    
    def extract_features_with_history(self, 
                                    gas_data: Dict[str, float], 
                                    gas_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Extract enhanced features with temporal analysis.
        
        Args:
            gas_data: Current gas data dictionary
            gas_history: List of recent gas data dictionaries
            
        Returns:
            Dictionary with extracted features including temporal analysis
        """
        # Get base features
        features = self.extract_features(gas_data)
        
        # Add enhanced temporal analysis
        if gas_history:
            # Accumulation rate analysis
            accumulation_features = self.accumulation_analyzer.analyze_accumulation_rate(gas_history)
            features.update(accumulation_features)
            
            # Baseline drift detection
            drift_features = self.drift_detector.detect_baseline_drift(gas_history)
            features.update(drift_features)
        
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
        for i, gas_data in enumerate(gas_data_list):
            try:
                # For batch processing, we can use history for temporal analysis
                gas_history = gas_data_list[max(0, i-10):i] if i > 0 else []
                features = self.extract_features_with_history(gas_data, gas_history)
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
        # This is a simplified implementation
        # In a real implementation, you would dynamically generate this list
        base_names = self.feature_names.copy()
        base_names.extend([
            'gas_concentration_level', 'gas_change_magnitude', 'gas_velocity_magnitude',
            'gas_deviation_from_normal', 'gas_relative_change', 'gas_acceleration'
        ])
        return base_names