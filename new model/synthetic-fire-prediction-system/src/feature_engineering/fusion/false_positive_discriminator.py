"""
False Positive Discriminator for FLIR+SCD41 Fire Detection.

Identifies and filters out common false positive scenarios.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FalsePositiveDiscriminator:
    """Discriminates false positives in fire detection scenarios."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the false positive discriminator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Thresholds for false positive scenarios
        self.thresholds = {
            'sunlight_temp_threshold': self.config.get('sunlight_temp_threshold', 40.0),  # °C
            'sunlight_co2_threshold': self.config.get('sunlight_co2_threshold', 10.0),   # ppm change
            'hvac_temp_change_range': self.config.get('hvac_temp_change_range', (5.0, 20.0)),  # °C
            'hvac_co2_normal_range': self.config.get('hvac_co2_normal_range', (400.0, 1000.0)),  # ppm
            'cooking_temp_threshold': self.config.get('cooking_temp_threshold', 5.0),   # % hot area
            'cooking_co2_range': self.config.get('cooking_co2_range', (600.0, 1500.0)),  # ppm
            'steam_temp_threshold': self.config.get('steam_temp_threshold', 35.0),       # °C
            'steam_humidity_threshold': self.config.get('steam_humidity_threshold', 70.0),  # %
            'dust_temp_threshold': self.config.get('dust_temp_threshold', 30.0),         # °C
            'dust_co2_threshold': self.config.get('dust_co2_threshold', 50.0),           # ppm
        }
        
        logger.info("Initialized False Positive Discriminator")
    
    def discriminate_false_positives(self, 
                                   thermal_features: Dict[str, float],
                                   gas_features: Dict[str, float],
                                   environmental_features: Dict[str, float] = None) -> Dict[str, float]:
        """
        Discriminate false positives based on sensor data patterns.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            environmental_features: Dictionary with environmental features (optional)
            
        Returns:
            Dictionary with false positive discrimination features
        """
        try:
            discrimination_features = {}
            
            # Sunlight heating discriminator
            sunlight_features = self._discriminate_sunlight_heating(thermal_features, gas_features)
            discrimination_features.update(sunlight_features)
            
            # HVAC effect discriminator
            hvac_features = self._discriminate_hvac_effect(thermal_features, gas_features)
            discrimination_features.update(hvac_features)
            
            # Cooking discriminator
            cooking_features = self._discriminate_cooking(thermal_features, gas_features)
            discrimination_features.update(cooking_features)
            
            # Steam/dust discriminator
            steam_dust_features = self._discriminate_steam_dust(thermal_features, gas_features, environmental_features)
            discrimination_features.update(steam_dust_features)
            
            # Overall false positive likelihood
            overall_discrimination = self._calculate_false_positive_likelihood(discrimination_features)
            discrimination_features.update(overall_discrimination)
            
            # Confidence in discrimination
            confidence_features = self._calculate_discrimination_confidence(discrimination_features)
            discrimination_features.update(confidence_features)
            
            return discrimination_features
            
        except Exception as e:
            logger.error(f"Error in false positive discrimination: {str(e)}")
            return {}
    
    def _discriminate_sunlight_heating(self, 
                                     thermal_features: Dict[str, float],
                                     gas_features: Dict[str, float]) -> Dict[str, float]:
        """
        Discriminate sunlight heating false positives.
        
        Sunlight typically causes:
        - High temperature readings
        - Minimal or no CO₂ increase
        - Uniform heating pattern
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with sunlight heating discrimination features
        """
        features = {}
        
        try:
            t_max = thermal_features.get('t_max', 0.0)
            gas_delta = gas_features.get('gas_delta', 0.0)
            t_hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
            
            # Sunlight heating indicator
            high_temp = t_max > self.thresholds['sunlight_temp_threshold']
            low_co2_change = abs(gas_delta) < self.thresholds['sunlight_co2_threshold']
            uniform_heating = t_hot_area_pct > 20.0  # Sunlight often heats large areas
            
            features['sunlight_heating_indicator'] = 1.0 if (high_temp and low_co2_change and uniform_heating) else 0.0
            features['sunlight_temp_component'] = 1.0 if high_temp else 0.0
            features['sunlight_co2_component'] = 1.0 if low_co2_change else 0.0
            features['sunlight_uniformity_component'] = 1.0 if uniform_heating else 0.0
            
            # Sunlight discrimination score (0-1, where 1 indicates likely sunlight heating)
            sunlight_score = 0.0
            component_count = 0
            
            if high_temp:
                sunlight_score += 0.4
                component_count += 1
            if low_co2_change:
                sunlight_score += 0.4
                component_count += 1
            if uniform_heating:
                sunlight_score += 0.2
                component_count += 1
                
            features['sunlight_discrimination_score'] = sunlight_score if component_count > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Error in sunlight heating discrimination: {str(e)}")
        
        return features
    
    def _discriminate_hvac_effect(self, 
                                thermal_features: Dict[str, float],
                                gas_features: Dict[str, float]) -> Dict[str, float]:
        """
        Discriminate HVAC effect false positives.
        
        HVAC effects typically cause:
        - Moderate temperature changes
        - Normal CO₂ levels (no significant increase)
        - Periodic patterns
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with HVAC effect discrimination features
        """
        features = {}
        
        try:
            t_max = thermal_features.get('t_max', 0.0)
            t_mean = thermal_features.get('t_mean', 0.0)
            gas_val = gas_features.get('gas_val', 0.0)
            
            # Calculate temperature change
            temp_change = t_max - t_mean
            
            # HVAC effect indicator
            moderate_temp_change = (
                self.thresholds['hvac_temp_change_range'][0] < temp_change < self.thresholds['hvac_temp_change_range'][1]
            )
            normal_co2 = (
                self.thresholds['hvac_co2_normal_range'][0] <= gas_val <= self.thresholds['hvac_co2_normal_range'][1]
            )
            
            features['hvac_effect_indicator'] = 1.0 if (moderate_temp_change and normal_co2) else 0.0
            features['hvac_temp_component'] = 1.0 if moderate_temp_change else 0.0
            features['hvac_co2_component'] = 1.0 if normal_co2 else 0.0
            
            # HVAC discrimination score
            hvac_score = 0.0
            if moderate_temp_change:
                hvac_score += 0.6
            if normal_co2:
                hvac_score += 0.4
                
            features['hvac_discrimination_score'] = hvac_score
        
        except Exception as e:
            logger.warning(f"Error in HVAC effect discrimination: {str(e)}")
        
        return features
    
    def _discriminate_cooking(self, 
                            thermal_features: Dict[str, float],
                            gas_features: Dict[str, float]) -> Dict[str, float]:
        """
        Discriminate cooking false positives.
        
        Cooking typically causes:
        - Localized heating (small hot areas)
        - Moderate CO₂ levels
        - Occurs in kitchen areas
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            
        Returns:
            Dictionary with cooking discrimination features
        """
        features = {}
        
        try:
            t_hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
            gas_val = gas_features.get('gas_val', 0.0)
            
            # Cooking indicator
            localized_heating = t_hot_area_pct < self.thresholds['cooking_temp_threshold']
            moderate_co2 = (
                self.thresholds['cooking_co2_range'][0] <= gas_val <= self.thresholds['cooking_co2_range'][1]
            )
            
            features['cooking_indicator'] = 1.0 if (localized_heating and moderate_co2) else 0.0
            features['cooking_temp_component'] = 1.0 if localized_heating else 0.0
            features['cooking_co2_component'] = 1.0 if moderate_co2 else 0.0
            
            # Cooking discrimination score
            cooking_score = 0.0
            if localized_heating:
                cooking_score += 0.5
            if moderate_co2:
                cooking_score += 0.5
                
            features['cooking_discrimination_score'] = cooking_score
        
        except Exception as e:
            logger.warning(f"Error in cooking discrimination: {str(e)}")
        
        return features
    
    def _discriminate_steam_dust(self, 
                               thermal_features: Dict[str, float],
                               gas_features: Dict[str, float],
                               environmental_features: Dict[str, float]) -> Dict[str, float]:
        """
        Discriminate steam/dust false positives.
        
        Steam/dust typically causes:
        - Moderate temperature increases
        - May cause CO₂ fluctuations
        - Associated with high humidity (for steam)
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            gas_features: Dictionary with SCD41 gas features
            environmental_features: Dictionary with environmental features
            
        Returns:
            Dictionary with steam/dust discrimination features
        """
        features = {}
        
        try:
            t_max = thermal_features.get('t_max', 0.0)
            gas_val = gas_features.get('gas_val', 0.0)
            
            # Steam discriminator (if environmental data available)
            if environmental_features:
                humidity = environmental_features.get('humidity', 0.0)
                steam_temp = t_max > self.thresholds['steam_temp_threshold']
                high_humidity = humidity > self.thresholds['steam_humidity_threshold']
                
                features['steam_indicator'] = 1.0 if (steam_temp and high_humidity) else 0.0
                features['steam_temp_component'] = 1.0 if steam_temp else 0.0
                features['steam_humidity_component'] = 1.0 if high_humidity else 0.0
                
                steam_score = 0.0
                if steam_temp:
                    steam_score += 0.5
                if high_humidity:
                    steam_score += 0.5
                features['steam_discrimination_score'] = steam_score
            
            # Dust discriminator
            dust_temp = t_max > self.thresholds['dust_temp_threshold']
            low_co2 = gas_val < self.thresholds['dust_co2_threshold']
            
            features['dust_indicator'] = 1.0 if (dust_temp and low_co2) else 0.0
            features['dust_temp_component'] = 1.0 if dust_temp else 0.0
            features['dust_co2_component'] = 1.0 if low_co2 else 0.0
            
            dust_score = 0.0
            if dust_temp:
                dust_score += 0.6
            if low_co2:
                dust_score += 0.4
            features['dust_discrimination_score'] = dust_score
        
        except Exception as e:
            logger.warning(f"Error in steam/dust discrimination: {str(e)}")
        
        return features
    
    def _calculate_false_positive_likelihood(self, discrimination_features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate overall false positive likelihood.
        
        Args:
            discrimination_features: Dictionary with all discrimination features
            
        Returns:
            Dictionary with overall false positive features
        """
        features = {}
        
        try:
            # Combine individual discrimination scores
            scores = [
                discrimination_features.get('sunlight_discrimination_score', 0.0),
                discrimination_features.get('hvac_discrimination_score', 0.0),
                discrimination_features.get('cooking_discrimination_score', 0.0),
                discrimination_features.get('steam_discrimination_score', 0.0),
                discrimination_features.get('dust_discrimination_score', 0.0)
            ]
            
            # Remove zero scores to avoid biasing the average
            non_zero_scores = [score for score in scores if score > 0]
            
            if non_zero_scores:
                # Weighted average (higher weights for more confident discriminators)
                weights = [0.25, 0.2, 0.2, 0.2, 0.15]  # Sunlight, HVAC, Cooking, Steam, Dust
                non_zero_weights = [weights[i] for i, score in enumerate(scores) if score > 0]
                
                if len(non_zero_scores) == len(non_zero_weights):
                    weighted_sum = sum(score * weight for score, weight in zip(non_zero_scores, non_zero_weights))
                    total_weight = sum(non_zero_weights)
                    overall_likelihood = weighted_sum / total_weight if total_weight > 0 else 0.0
                else:
                    overall_likelihood = np.mean(non_zero_scores)
                
                features['false_positive_likelihood'] = overall_likelihood
                features['high_false_positive_risk'] = 1.0 if overall_likelihood > 0.6 else 0.0
            else:
                features['false_positive_likelihood'] = 0.0
                features['high_false_positive_risk'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error calculating false positive likelihood: {str(e)}")
        
        return features
    
    def _calculate_discrimination_confidence(self, discrimination_features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate confidence in discrimination results.
        
        Args:
            discrimination_features: Dictionary with all discrimination features
            
        Returns:
            Dictionary with discrimination confidence features
        """
        features = {}
        
        try:
            # Count how many discriminators have activated
            discriminator_indicators = [
                discrimination_features.get('sunlight_heating_indicator', 0.0),
                discrimination_features.get('hvac_effect_indicator', 0.0),
                discrimination_features.get('cooking_indicator', 0.0),
                discrimination_features.get('steam_indicator', 0.0),
                discrimination_features.get('dust_indicator', 0.0)
            ]
            
            active_discriminators = sum(discriminator_indicators)
            total_discriminators = len(discriminator_indicators)
            
            # Confidence is higher when more discriminators agree
            features['discrimination_confidence'] = active_discriminators / total_discriminators if total_discriminators > 0 else 0.0
            
            # Strong discrimination indicator
            features['strong_discrimination'] = 1.0 if active_discriminators >= 2 else 0.0
        
        except Exception as e:
            logger.warning(f"Error calculating discrimination confidence: {str(e)}")
        
        return features