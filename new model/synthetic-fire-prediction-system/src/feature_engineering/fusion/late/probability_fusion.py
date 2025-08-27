"""
Probability fusion implementation for the synthetic fire prediction system.

This module provides an implementation of probability fusion, which combines
probability outputs from different models to create a unified probability.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict

from ...base import FeatureFusion


class ProbabilityFusion(FeatureFusion):
    """
    Implementation of probability fusion.
    
    This class combines probability outputs from different models to create a unified
    probability estimate. It operates on the probability outputs of models trained on
    different feature sets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the probability fusion component.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ['fusion_method', 'decision_threshold']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate fusion method
        valid_methods = ['average', 'weighted_average', 'product', 'max', 'min']
        if self.config['fusion_method'] not in valid_methods:
            raise ValueError(f"Invalid fusion method: {self.config['fusion_method']}. "
                           f"Must be one of {valid_methods}")
        
        # Validate decision threshold
        if not isinstance(self.config['decision_threshold'], (int, float)) or \
           not 0 <= self.config['decision_threshold'] <= 1:
            raise ValueError("'decision_threshold' must be a number between 0 and 1")
        
        # Set default values for optional parameters
        if 'model_weights' not in self.config:
            self.config['model_weights'] = {
                'thermal': 1.0,
                'gas': 1.0,
                'environmental': 1.0
            }
        
        if 'calibration' not in self.config:
            self.config['calibration'] = False
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors at the probability level.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing probability fusion")
        
        # Extract model probabilities
        thermal_prob = self._extract_probability(thermal_features, 'thermal')
        gas_prob = self._extract_probability(gas_features, 'gas')
        env_prob = self._extract_probability(environmental_features, 'environmental')
        
        # Apply calibration if configured
        if self.config.get('calibration', False):
            thermal_prob = self._calibrate_probability(thermal_prob, 'thermal')
            gas_prob = self._calibrate_probability(gas_prob, 'gas')
            env_prob = self._calibrate_probability(env_prob, 'environmental')
        
        # Combine probabilities based on the selected fusion method
        fusion_method = self.config['fusion_method']
        
        if fusion_method == 'average':
            fused_prob = self._average_probabilities([thermal_prob, gas_prob, env_prob])
        
        elif fusion_method == 'weighted_average':
            fused_prob = self._weighted_average_probabilities(
                [thermal_prob, gas_prob, env_prob],
                ['thermal', 'gas', 'environmental']
            )
        
        elif fusion_method == 'product':
            fused_prob = self._product_probabilities([thermal_prob, gas_prob, env_prob])
        
        elif fusion_method == 'max':
            fused_prob = self._max_probability([thermal_prob, gas_prob, env_prob])
        
        elif fusion_method == 'min':
            fused_prob = self._min_probability([thermal_prob, gas_prob, env_prob])
        
        else:
            self.logger.warning(f"Unknown fusion method: {fusion_method}, using average")
            fused_prob = self._average_probabilities([thermal_prob, gas_prob, env_prob])
        
        # Determine decision based on threshold
        threshold = self.config['decision_threshold']
        fused_decision = fused_prob >= threshold
        
        # Create fused features dictionary
        fused_features = {
            'fusion_method': fusion_method,
            'fusion_time': datetime.now().isoformat(),
            'model_probabilities': {
                'thermal': float(thermal_prob),
                'gas': float(gas_prob),
                'environmental': float(env_prob)
            },
            'fused_probability': float(fused_prob),
            'fused_decision': int(fused_decision),
            'decision_threshold': threshold,
            'calibration_applied': self.config.get('calibration', False)
        }
        
        self.logger.info(f"Probability fusion completed with probability: {fused_prob}, decision: {fused_decision}")
        return fused_features
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # For probability fusion, the fused probability directly represents the risk score
        fused_prob = fused_features.get('fused_probability', 0.0)
        
        # Ensure the risk score is a float between 0 and 1
        risk_score = min(1.0, max(0.0, float(fused_prob)))
        
        self.logger.info(f"Calculated risk score: {risk_score}")
        return risk_score
    
    def _extract_probability(self, features: Dict[str, Any], source: str) -> float:
        """
        Extract probability from features.
        
        Args:
            features: Features dictionary
            source: Source of the features (thermal, gas, environmental)
            
        Returns:
            Probability value
        """
        # Look for probability in the features
        probability = None
        
        # Check for explicit probability
        if 'probability' in features:
            probability = features['probability']
        
        # If not found, check for confidence or risk score
        if probability is None and 'confidence' in features:
            probability = features['confidence']
        
        if probability is None and 'risk_score' in features:
            probability = features['risk_score']
        
        # If still not found, try to infer from other features
        if probability is None:
            # For thermal features, check for high temperatures or hotspots
            if source == 'thermal':
                max_temp = None
                for key, value in features.items():
                    if 'max_temperature' in key and isinstance(value, (int, float)):
                        max_temp = value
                        break
                
                if max_temp is not None:
                    # Convert temperature to probability using a sigmoid-like function
                    threshold = 100  # Example threshold
                    probability = 1.0 / (1.0 + np.exp(-(max_temp - threshold) / 10))
            
            # For gas features, check for high concentrations
            elif source == 'gas':
                max_conc = None
                for key, value in features.items():
                    if 'concentration' in key and isinstance(value, (int, float)):
                        max_conc = value
                        break
                
                if max_conc is not None:
                    # Convert concentration to probability using a sigmoid-like function
                    threshold = 50  # Example threshold
                    probability = 1.0 / (1.0 + np.exp(-(max_conc - threshold) / 5))
            
            # For environmental features, check for temperature rise
            elif source == 'environmental':
                temp_rise = None
                for key, value in features.items():
                    if 'temperature_rise' in key and isinstance(value, (int, float)):
                        temp_rise = value
                        break
                
                if temp_rise is not None:
                    # Convert temperature rise to probability using a sigmoid-like function
                    threshold = 5  # Example threshold
                    probability = 1.0 / (1.0 + np.exp(-(temp_rise - threshold) / 2))
        
        # If still not found, use default value
        if probability is None:
            probability = 0.5
            self.logger.warning(f"Could not extract probability from {source} features, using default")
        
        # Ensure probability is a float between 0 and 1
        if not isinstance(probability, (int, float)):
            probability = 0.5
        else:
            probability = min(1.0, max(0.0, float(probability)))
        
        return probability
    
    def _calibrate_probability(self, probability: float, source: str) -> float:
        """
        Calibrate probability using the specified calibration method.
        
        Args:
            probability: Raw probability value
            source: Source of the probability (thermal, gas, environmental)
            
        Returns:
            Calibrated probability value
        """
        # Get calibration parameters for this source
        calibration_params = self.config.get('calibration_params', {}).get(source, {})
        
        if not calibration_params:
            return probability
        
        # Apply calibration based on the method
        method = calibration_params.get('method', 'platt')
        
        if method == 'platt':
            # Platt scaling: sigmoid transformation with parameters A and B
            A = calibration_params.get('A', 1.0)
            B = calibration_params.get('B', 0.0)
            
            # Apply sigmoid transformation
            calibrated = 1.0 / (1.0 + np.exp(-(A * probability + B)))
        
        elif method == 'isotonic':
            # Isotonic regression: piecewise linear transformation
            # This is a simplified version using predefined breakpoints
            breakpoints = calibration_params.get('breakpoints', [0.0, 0.5, 1.0])
            outputs = calibration_params.get('outputs', [0.0, 0.5, 1.0])
            
            # Find the appropriate segment
            for i in range(len(breakpoints) - 1):
                if breakpoints[i] <= probability <= breakpoints[i + 1]:
                    # Linear interpolation
                    alpha = (probability - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
                    calibrated = outputs[i] + alpha * (outputs[i + 1] - outputs[i])
                    break
            else:
                calibrated = probability
        
        elif method == 'temperature':
            # Temperature scaling: softmax with temperature parameter
            T = calibration_params.get('temperature', 1.0)
            
            # Apply temperature scaling
            logit = np.log(probability / (1 - probability))
            scaled_logit = logit / T
            calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))
        
        else:
            self.logger.warning(f"Unknown calibration method: {method}, using raw probability")
            calibrated = probability
        
        # Ensure calibrated probability is a float between 0 and 1
        calibrated = min(1.0, max(0.0, float(calibrated)))
        
        return calibrated
    
    def _average_probabilities(self, probabilities: List[float]) -> float:
        """
        Combine probabilities by averaging.
        
        Args:
            probabilities: List of probability values
            
        Returns:
            Averaged probability
        """
        if not probabilities:
            return 0.5
        
        return sum(probabilities) / len(probabilities)
    
    def _weighted_average_probabilities(self, probabilities: List[float], sources: List[str]) -> float:
        """
        Combine probabilities by weighted averaging.
        
        Args:
            probabilities: List of probability values
            sources: List of source names for each probability
            
        Returns:
            Weighted averaged probability
        """
        if not probabilities:
            return 0.5
        
        # Get weights for each source
        weights = [self.config['model_weights'].get(source, 1.0) for source in sources]
        
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_avg = sum(prob * weight for prob, weight in zip(probabilities, weights)) / total_weight
        else:
            weighted_avg = sum(probabilities) / len(probabilities)
        
        return weighted_avg
    
    def _product_probabilities(self, probabilities: List[float]) -> float:
        """
        Combine probabilities by product rule (assuming independence).
        
        Args:
            probabilities: List of probability values
            
        Returns:
            Combined probability
        """
        if not probabilities:
            return 0.5
        
        # Calculate product of probabilities
        pos_prob = np.prod(probabilities)
        
        # Calculate product of complementary probabilities
        neg_prob = np.prod([1.0 - p for p in probabilities])
        
        # Normalize to get final probability
        if pos_prob + neg_prob > 0:
            return pos_prob / (pos_prob + neg_prob)
        else:
            return 0.5
    
    def _max_probability(self, probabilities: List[float]) -> float:
        """
        Select the maximum probability.
        
        Args:
            probabilities: List of probability values
            
        Returns:
            Maximum probability
        """
        if not probabilities:
            return 0.5
        
        return max(probabilities)
    
    def _min_probability(self, probabilities: List[float]) -> float:
        """
        Select the minimum probability.
        
        Args:
            probabilities: List of probability values
            
        Returns:
            Minimum probability
        """
        if not probabilities:
            return 0.5
        
        return min(probabilities)