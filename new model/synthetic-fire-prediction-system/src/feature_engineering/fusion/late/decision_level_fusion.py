"""
Decision-level fusion implementation for the synthetic fire prediction system.

This module provides an implementation of decision-level fusion, which combines
decisions from different models to create a unified decision.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict

from ...base import FeatureFusion


class DecisionLevelFusion(FeatureFusion):
    """
    Implementation of decision-level fusion.
    
    This class combines decisions from different models to create a unified decision.
    It operates on the outputs of models trained on different feature sets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the decision-level fusion component.
        
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
        valid_methods = ['majority_vote', 'weighted_vote', 'max_confidence', 'average_confidence']
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
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors at the decision level.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing decision-level fusion")
        
        # Extract model decisions and confidences
        thermal_decision, thermal_confidence = self._extract_decision(thermal_features, 'thermal')
        gas_decision, gas_confidence = self._extract_decision(gas_features, 'gas')
        env_decision, env_confidence = self._extract_decision(environmental_features, 'environmental')
        
        # Combine decisions based on the selected fusion method
        fusion_method = self.config['fusion_method']
        
        if fusion_method == 'majority_vote':
            fused_decision, fused_confidence = self._majority_vote(
                [thermal_decision, gas_decision, env_decision],
                [thermal_confidence, gas_confidence, env_confidence]
            )
        
        elif fusion_method == 'weighted_vote':
            fused_decision, fused_confidence = self._weighted_vote(
                [thermal_decision, gas_decision, env_decision],
                [thermal_confidence, gas_confidence, env_confidence],
                ['thermal', 'gas', 'environmental']
            )
        
        elif fusion_method == 'max_confidence':
            fused_decision, fused_confidence = self._max_confidence(
                [thermal_decision, gas_decision, env_decision],
                [thermal_confidence, gas_confidence, env_confidence]
            )
        
        elif fusion_method == 'average_confidence':
            fused_decision, fused_confidence = self._average_confidence(
                [thermal_decision, gas_decision, env_decision],
                [thermal_confidence, gas_confidence, env_confidence]
            )
        
        else:
            self.logger.warning(f"Unknown fusion method: {fusion_method}, using majority vote")
            fused_decision, fused_confidence = self._majority_vote(
                [thermal_decision, gas_decision, env_decision],
                [thermal_confidence, gas_confidence, env_confidence]
            )
        
        # Create fused features dictionary
        fused_features = {
            'fusion_method': fusion_method,
            'fusion_time': datetime.now().isoformat(),
            'model_decisions': {
                'thermal': {
                    'decision': int(thermal_decision),
                    'confidence': float(thermal_confidence)
                },
                'gas': {
                    'decision': int(gas_decision),
                    'confidence': float(gas_confidence)
                },
                'environmental': {
                    'decision': int(env_decision),
                    'confidence': float(env_confidence)
                }
            },
            'fused_decision': int(fused_decision),
            'fused_confidence': float(fused_confidence),
            'decision_threshold': self.config['decision_threshold']
        }
        
        self.logger.info(f"Decision-level fusion completed with decision: {fused_decision}, confidence: {fused_confidence}")
        return fused_features
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # Extract fused decision and confidence
        fused_decision = fused_features.get('fused_decision', 0)
        fused_confidence = fused_features.get('fused_confidence', 0.0)
        
        # Calculate risk score based on decision and confidence
        if fused_decision == 1:
            # Positive decision (fire detected)
            risk_score = fused_confidence
        else:
            # Negative decision (no fire detected)
            # Still assign a small risk based on confidence
            # Lower confidence in negative decision means higher risk
            risk_score = 0.1 * (1.0 - fused_confidence)
        
        self.logger.info(f"Calculated risk score: {risk_score}")
        return risk_score
    
    def _extract_decision(self, features: Dict[str, Any], source: str) -> Tuple[bool, float]:
        """
        Extract decision and confidence from features.
        
        Args:
            features: Features dictionary
            source: Source of the features (thermal, gas, environmental)
            
        Returns:
            Tuple of (decision, confidence)
        """
        # Look for decision and confidence in the features
        decision = None
        confidence = None
        
        # Check for explicit decision and confidence
        if 'decision' in features:
            decision = features['decision']
        
        if 'confidence' in features:
            confidence = features['confidence']
        
        # If not found, check for risk score or similar metrics
        if decision is None and 'risk_score' in features:
            threshold = self.config['decision_threshold']
            decision = features['risk_score'] >= threshold
            confidence = abs(features['risk_score'] - 0.5) * 2  # Scale to [0, 1]
        
        # If still not found, try to infer from other features
        if decision is None:
            # For thermal features, check for high temperatures or hotspots
            if source == 'thermal':
                max_temp = None
                for key, value in features.items():
                    if 'max_temperature' in key and isinstance(value, (int, float)):
                        max_temp = value
                        break
                
                if max_temp is not None:
                    threshold = 100  # Example threshold
                    decision = max_temp >= threshold
                    confidence = min(1.0, max(0.0, (max_temp - threshold + 20) / 40))
            
            # For gas features, check for high concentrations
            elif source == 'gas':
                max_conc = None
                for key, value in features.items():
                    if 'concentration' in key and isinstance(value, (int, float)):
                        max_conc = value
                        break
                
                if max_conc is not None:
                    threshold = 50  # Example threshold
                    decision = max_conc >= threshold
                    confidence = min(1.0, max(0.0, (max_conc - threshold + 10) / 20))
            
            # For environmental features, check for temperature rise
            elif source == 'environmental':
                temp_rise = None
                for key, value in features.items():
                    if 'temperature_rise' in key and isinstance(value, (int, float)):
                        temp_rise = value
                        break
                
                if temp_rise is not None:
                    threshold = 5  # Example threshold
                    decision = temp_rise >= threshold
                    confidence = min(1.0, max(0.0, (temp_rise - threshold + 2) / 5))
        
        # If still not found, use default values
        if decision is None:
            decision = False
            confidence = 0.5
            self.logger.warning(f"Could not extract decision from {source} features, using default")
        
        # Ensure confidence is a float between 0 and 1
        if confidence is None or not isinstance(confidence, (int, float)):
            confidence = 0.5
        else:
            confidence = min(1.0, max(0.0, float(confidence)))
        
        return bool(decision), confidence
    
    def _majority_vote(self, decisions: List[bool], confidences: List[float]) -> Tuple[bool, float]:
        """
        Combine decisions using majority voting.
        
        Args:
            decisions: List of decisions from different models
            confidences: List of confidence values for each decision
            
        Returns:
            Tuple of (fused_decision, fused_confidence)
        """
        # Count positive and negative decisions
        positive_count = sum(1 for d in decisions if d)
        negative_count = len(decisions) - positive_count
        
        # Determine majority decision
        fused_decision = positive_count >= negative_count
        
        # Calculate confidence based on vote margin and individual confidences
        vote_margin = abs(positive_count - negative_count) / len(decisions)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Combine vote margin and average confidence
        fused_confidence = 0.5 + (vote_margin * 0.25) + (avg_confidence * 0.25)
        fused_confidence = min(1.0, max(0.0, fused_confidence))
        
        return fused_decision, fused_confidence
    
    def _weighted_vote(self, decisions: List[bool], confidences: List[float], sources: List[str]) -> Tuple[bool, float]:
        """
        Combine decisions using weighted voting.
        
        Args:
            decisions: List of decisions from different models
            confidences: List of confidence values for each decision
            sources: List of source names for each decision
            
        Returns:
            Tuple of (fused_decision, fused_confidence)
        """
        # Get weights for each source
        weights = [self.config['model_weights'].get(source, 1.0) for source in sources]
        
        # Calculate weighted votes
        positive_weight = sum(weight for decision, weight in zip(decisions, weights) if decision)
        negative_weight = sum(weight for decision, weight in zip(decisions, weights) if not decision)
        
        # Determine weighted majority decision
        fused_decision = positive_weight >= negative_weight
        
        # Calculate confidence based on weight margin and individual confidences
        total_weight = sum(weights)
        if total_weight > 0:
            weight_margin = abs(positive_weight - negative_weight) / total_weight
            weighted_confidence = sum(conf * weight for conf, weight in zip(confidences, weights)) / total_weight
            
            # Combine weight margin and weighted confidence
            fused_confidence = 0.5 + (weight_margin * 0.25) + (weighted_confidence * 0.25)
            fused_confidence = min(1.0, max(0.0, fused_confidence))
        else:
            fused_confidence = 0.5
        
        return fused_decision, fused_confidence
    
    def _max_confidence(self, decisions: List[bool], confidences: List[float]) -> Tuple[bool, float]:
        """
        Select the decision with the highest confidence.
        
        Args:
            decisions: List of decisions from different models
            confidences: List of confidence values for each decision
            
        Returns:
            Tuple of (fused_decision, fused_confidence)
        """
        if not decisions:
            return False, 0.5
        
        # Find the index of the decision with the highest confidence
        max_index = np.argmax(confidences)
        
        # Select the decision and confidence
        fused_decision = decisions[max_index]
        fused_confidence = confidences[max_index]
        
        return fused_decision, fused_confidence
    
    def _average_confidence(self, decisions: List[bool], confidences: List[float]) -> Tuple[bool, float]:
        """
        Combine decisions by averaging confidences.
        
        Args:
            decisions: List of decisions from different models
            confidences: List of confidence values for each decision
            
        Returns:
            Tuple of (fused_decision, fused_confidence)
        """
        if not decisions:
            return False, 0.5
        
        # Calculate average confidence for positive and negative decisions
        positive_confidences = [conf for decision, conf in zip(decisions, confidences) if decision]
        negative_confidences = [conf for decision, conf in zip(decisions, confidences) if not decision]
        
        # Calculate average confidences
        avg_positive = sum(positive_confidences) / len(positive_confidences) if positive_confidences else 0.0
        avg_negative = sum(negative_confidences) / len(negative_confidences) if negative_confidences else 0.0
        
        # Determine decision based on which class has higher average confidence
        if avg_positive >= avg_negative:
            return True, avg_positive
        else:
            return False, avg_negative