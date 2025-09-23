"""
Cross-Sensor Fusion Extractor.

Combines features from FLIR thermal and SCD41 gas sensors with advanced fusion techniques.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import sys
import os

# Add the src directory to the path to fix relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our new analysis modules
from src.feature_engineering.fusion.cross_sensor_correlation_analyzer import CrossSensorCorrelationAnalyzer

logger = logging.getLogger(__name__)


class CrossSensorFusionExtractor:
    """Extracts fused features from FLIR thermal and SCD41 gas sensor data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cross-sensor fusion extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize fusion analysis components
        self.correlation_analyzer = CrossSensorCorrelationAnalyzer(
            self.config.get('correlation_analysis', {}))
        
        logger.info("Initialized Cross-Sensor Fusion Extractor")
        logger.info("Fusion analysis modules loaded")
    
    def extract_fused_features(self, 
                             thermal_features: Dict[str, float],
                             gas_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Extract fused features from thermal and gas sensor data.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with fused features
        """
        try:
            # Initialize fused features
            fused_features = {}
            
            # Add timestamp
            fused_features['timestamp'] = datetime.now().isoformat()
            
            # Basic cross-sensor features
            basic_fusion = self._compute_basic_fusion(thermal_features, gas_features)
            fused_features.update(basic_fusion)
            
            # Risk convergence features
            risk_features = self._compute_risk_convergence(thermal_features, gas_features)
            fused_features.update(risk_features)
            
            # False positive discrimination features
            fp_discrimination = self._compute_false_positive_discrimination(
                thermal_features, gas_features)
            fused_features.update(fp_discrimination)
            
            # Fire likelihood fusion
            fire_likelihood = self._compute_fire_likelihood_fusion(thermal_features, gas_features)
            fused_features.update(fire_likelihood)
            
            fused_features['fusion_success'] = True
            
            return fused_features
            
        except Exception as e:
            logger.error(f"Error in cross-sensor fusion: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'fusion_success': False,
                'error': str(e)
            }
    
    def extract_fused_features_with_history(self,
                                          thermal_features: Dict[str, float],
                                          gas_features: Dict[str, float],
                                          thermal_history: List[Dict[str, float]],
                                          gas_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Extract fused features with temporal analysis.
        
        Args:
            thermal_features: Dictionary with current FLIR thermal features
            gas_features: Dictionary with current SCD41 gas features
            thermal_history: List of recent thermal feature dictionaries
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with fused features including temporal analysis
        """
        # Get base fused features
        fused_features = self.extract_fused_features(thermal_features, gas_features)
        
        # Add temporal correlation analysis
        if thermal_history and gas_history:
            correlation_features = self.correlation_analyzer.analyze_cross_sensor_correlation(
                thermal_history, gas_history)
            fused_features.update(correlation_features)
        
        return fused_features
    
    def _compute_basic_fusion(self, 
                            thermal_features: Dict[str, float],
                            gas_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute basic cross-sensor fusion features.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with basic fusion features
        """
        fusion = {}
        
        # Temperature and CO₂ combination
        t_mean = thermal_features.get('t_mean', 0.0)
        gas_val = gas_features.get('gas_val', 0.0)
        fusion['temp_co2_product'] = t_mean * gas_val
        fusion['temp_co2_ratio'] = t_mean / (gas_val + 1e-8)  # Avoid division by zero
        
        # Rate combination
        tproxy_vel = thermal_features.get('tproxy_vel', 0.0)
        gas_vel = gas_features.get('gas_vel', 0.0)
        fusion['thermal_gas_velocity_product'] = tproxy_vel * gas_vel
        fusion['velocity_difference'] = abs(tproxy_vel - gas_vel)
        
        # Hot area and CO₂ combination
        t_hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
        fusion['hot_area_co2_interaction'] = t_hot_area_pct * gas_val / 1000.0
        
        return fusion
    
    def _compute_risk_convergence(self, 
                                thermal_features: Dict[str, float],
                                gas_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute risk convergence features.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with risk convergence features
        """
        convergence = {}
        
        # Extract fire indicators from both sensors
        thermal_fire_score = thermal_features.get('fire_likelihood_score', 0.0)
        gas_fire_score = gas_features.get('gas_fire_likelihood_score', 0.0)
        
        # Risk convergence index
        # Higher values indicate both sensors detecting anomalies simultaneously
        convergence['risk_convergence_index'] = thermal_fire_score * gas_fire_score
        
        # Risk agreement (both sensors agree on fire presence)
        thermal_fire_detected = thermal_fire_score > 0.5
        gas_fire_detected = gas_fire_score > 0.5
        convergence['risk_agreement'] = 1.0 if (thermal_fire_detected and gas_fire_detected) else 0.0
        
        # Risk divergence (sensors disagree - potential false positive)
        convergence['risk_divergence'] = 1.0 if (thermal_fire_detected != gas_fire_detected) else 0.0
        
        return convergence
    
    def _compute_false_positive_discrimination(self, 
                                            thermal_features: Dict[str, float],
                                            gas_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute features that help discriminate false positives.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with false positive discrimination features
        """
        discrimination = {}
        
        # Extract key features
        t_max = thermal_features.get('t_max', 0.0)
        gas_val = gas_features.get('gas_val', 0.0)
        t_hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
        gas_delta = gas_features.get('gas_delta', 0.0)
        
        # Sunlight heating discriminator
        # Sunlight typically causes uniform heating without CO₂ increase
        high_temp = t_max > 40.0
        low_co2_change = abs(gas_delta) < 10.0
        discrimination['sunlight_heating_indicator'] = 1.0 if (high_temp and low_co2_change) else 0.0
        
        # HVAC effect discriminator
        # HVAC typically causes temperature changes without high CO₂
        moderate_temp_change = 5.0 < (t_max - thermal_features.get('t_mean', 0.0)) < 20.0
        normal_co2 = 400.0 <= gas_val <= 1000.0
        discrimination['hvac_effect_indicator'] = 1.0 if (moderate_temp_change and normal_co2) else 0.0
        
        # Cooking discriminator
        # Cooking causes localized heating with some CO₂ but not extreme levels
        localized_heating = t_hot_area_pct < 5.0
        moderate_co2 = 600.0 <= gas_val <= 1500.0
        discrimination['cooking_indicator'] = 1.0 if (localized_heating and moderate_co2) else 0.0
        
        return discrimination
    
    def _compute_fire_likelihood_fusion(self, 
                                      thermal_features: Dict[str, float],
                                      gas_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute fused fire likelihood scores.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with fused fire likelihood features
        """
        fusion = {}
        
        # Extract individual scores
        thermal_score = thermal_features.get('fire_likelihood_score', 0.0)
        gas_score = gas_features.get('gas_fire_likelihood_score', 0.0)
        
        # Weighted combination (thermal gets higher weight as it's more specific)
        fused_score = 0.6 * thermal_score + 0.4 * gas_score
        fusion['fused_fire_likelihood'] = fused_score
        
        # Confidence in fused score
        # Higher confidence when both sensors agree
        agreement = 1.0 - abs(thermal_score - gas_score)
        fusion['fused_fire_confidence'] = agreement
        
        # Early warning score (sensitive to initial fire signs)
        early_thermal = thermal_features.get('rapid_change_indicator', 0.0)
        early_gas = gas_features.get('rapid_increase_indicator', 0.0)
        fusion['early_fire_warning_score'] = 0.5 * early_thermal + 0.5 * early_gas
        
        return fusion