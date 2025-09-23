"""
Enhanced FLIR Lepton 3.5 Thermal Data Extractor.

Processes thermal features from FLIR Lepton 3.5 cameras with enhanced analysis:
- Multi-scale blob analysis
- Temporal signature pattern recognition
- Edge sharpness metrics
- Heat distribution skewness
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
from feature_engineering.extractors.thermal.blob_analyzer import BlobAnalyzer
from feature_engineering.extractors.thermal.temporal_signature_analyzer import TemporalSignatureAnalyzer
from feature_engineering.extractors.thermal.edge_sharpness_analyzer import EdgeSharpnessAnalyzer
from feature_engineering.extractors.thermal.heat_distribution_analyzer import HeatDistributionAnalyzer

logger = logging.getLogger(__name__)


class FlirThermalExtractorEnhanced:
    """Enhanced extract and process features from FLIR Lepton 3.5 thermal data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced FLIR thermal extractor.
        
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
        
        # Initialize enhanced analysis components
        self.blob_analyzer = BlobAnalyzer(self.config.get('blob_analysis', {}))
        self.temporal_analyzer = TemporalSignatureAnalyzer(self.config.get('temporal_analysis', {}))
        self.edge_analyzer = EdgeSharpnessAnalyzer(self.config.get('edge_analysis', {}))
        self.distribution_analyzer = HeatDistributionAnalyzer(self.config.get('distribution_analysis', {}))
        
        # Thresholds for fire detection
        self.thresholds = {
            't_max': 60.0,           # °C
            't_hot_area_pct': 10.0,   # %
            'tproxy_vel': 2.0,       # Rate of change
            't_grad_mean': 3.0       # Sharpness threshold
        }
        
        logger.info("Initialized Enhanced FLIR Lepton 3.5 Thermal Extractor")
        logger.info(f"Processing {len(self.feature_names)} base thermal features")
        logger.info("Enhanced analysis modules loaded")
    
    def extract_features(self, thermal_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Extract enhanced features from FLIR thermal data.
        
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
        
        # Add derived features from base extractor
        derived_features = self._compute_derived_features(thermal_data)
        features.update(derived_features)
        
        # Add enhanced analysis features
        enhanced_features = self._compute_enhanced_features(thermal_data)
        features.update(enhanced_features)
        
        # Add fire indicators
        fire_indicators = self._compute_fire_indicators(thermal_data)
        features.update(fire_indicators)
        
        # Add quality metrics
        quality_metrics = self._compute_quality_metrics(thermal_data)
        features.update(quality_metrics)
        
        features['timestamp'] = datetime.now().isoformat()
        features['extraction_success'] = True
        
        return features
    
    def extract_features_with_history(self, 
                                    thermal_data: Dict[str, float], 
                                    thermal_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Extract enhanced features with temporal analysis.
        
        Args:
            thermal_data: Current thermal data dictionary
            thermal_history: List of recent thermal data dictionaries
            
        Returns:
            Dictionary with extracted features including temporal analysis
        """
        # Get base features
        features = self.extract_features(thermal_data)
        
        # Add temporal analysis
        if thermal_history:
            temporal_features = self.temporal_analyzer.analyze_temporal_patterns(thermal_history)
            features.update(temporal_features)
        
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
    
    def _compute_enhanced_features(self, thermal_data: Dict[str, float]) -> Dict[str, float]:
        """Compute enhanced features using our new analysis modules."""
        enhanced = {}
        
        # Multi-scale blob analysis
        blob_features = self.blob_analyzer.analyze_blobs(thermal_data)
        enhanced.update(blob_features)
        
        # Edge sharpness analysis
        edge_features = self.edge_analyzer.analyze_edge_sharpness(thermal_data)
        enhanced.update(edge_features)
        
        # Heat distribution analysis
        distribution_features = self.distribution_analyzer.analyze_heat_distribution(thermal_data)
        enhanced.update(distribution_features)
        
        return enhanced
    
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
        for i, thermal_data in enumerate(thermal_data_list):
            try:
                # For batch processing, we can use history for temporal analysis
                thermal_history = thermal_data_list[max(0, i-5):i] if i > 0 else []
                features = self.extract_features_with_history(thermal_data, thermal_history)
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
        # This is a simplified implementation
        # In a real implementation, you would dynamically generate this list
        base_names = self.feature_names.copy()
        base_names.extend([
            't_max_to_mean_ratio', 't_p95_to_max_ratio', 't_grad_activity',
            'flow_activity', 'tproxy_acceleration'
        ])
        return base_names