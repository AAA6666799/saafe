"""
Voting fusion implementation for the synthetic fire prediction system.

This module provides an implementation of voting fusion, which combines
decisions from different models using voting mechanisms.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict, Counter

from ...base import FeatureFusion


class VotingFusion(FeatureFusion):
    """
    Implementation of voting fusion.
    
    This class combines decisions from different models using voting mechanisms
    to create a unified decision.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the voting fusion component.
        
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
        required_params = ['voting_method', 'decision_threshold']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate voting method
        valid_methods = ['hard', 'soft', 'weighted', 'dynamic']
        if self.config['voting_method'] not in valid_methods:
            raise ValueError(f"Invalid voting method: {self.config['voting_method']}. "
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
        
        if 'dynamic_weight_method' not in self.config:
            self.config['dynamic_weight_method'] = 'confidence'
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors using voting mechanisms.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing voting fusion")
        
        # Extract model decisions and confidences
        thermal_decision, thermal_confidence = self._extract_decision(thermal_features, 'thermal')
        gas_decision, gas_confidence = self._extract_decision(gas_features, 'gas')
        env_decision, env_confidence = self._extract_decision(environmental_features, 'environmental')
        
        # Combine decisions based on the selected voting method
        voting_method = self.config['voting_method']
        
        if voting_method == 'hard':
            fused_decision, fused_confidence = self._hard_voting(
                [thermal_decision, gas_decision, env_decision]
            )
        
        elif voting_method == 'soft':
            fused_decision, fused_confidence = self._soft_voting(
                [thermal_decision, gas_decision, env_decision],
                [thermal_confidence, gas_confidence, env_confidence]
            )
        
        elif voting_method == 'weighted':
            fused_decision, fused_confidence = self._weighted_voting(
                [thermal_decision, gas_decision, env_decision],
                [thermal_confidence, gas_confidence, env_confidence],
                ['thermal', 'gas', 'environmental']
            )
        
        elif voting_method == 'dynamic':
            fused_decision, fused_confidence = self._dynamic_voting(
                [thermal_decision, gas_decision, env_decision],
                [thermal_confidence, gas_confidence, env_confidence],
                [thermal_features, gas_features, environmental_features]
            )
        
        else:
            self.logger.warning(f"Unknown voting method: {voting_method}, using hard voting")
            fused_decision, fused_confidence = self._hard_voting(
                [thermal_decision, gas_decision, env_decision]
            )
        
        # Create fused features dictionary
        fused_features = {
            'voting_method': voting_method,
            'fusion_time': datetime.now().isoformat(),
            'model_votes': {
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
        
        self.logger.info(f"Voting fusion completed with decision: {fused_decision}, confidence: {fused_confidence}")
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
        
        # If not found, check for probability
        if decision is None and 'probability' in features:
            threshold = self.config['decision_threshold']
            probability = features['probability']
            decision = probability >= threshold
            confidence = abs(probability - 0.5) * 2  # Scale to [0, 1]
        
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
    
    def _hard_voting(self, decisions: List[bool]) -> Tuple[bool, float]:
        """
        Combine decisions using hard voting (majority vote).
        
        Args:
            decisions: List of decisions from different models
            
        Returns:
            Tuple of (fused_decision, fused_confidence)
        """
        if not decisions:
            return False, 0.5
        
        # Count votes
        vote_counter = Counter(decisions)
        
        # Get majority vote
        majority_vote = vote_counter.most_common(1)[0][0]
        
        # Calculate confidence based on vote margin
        total_votes = len(decisions)
        majority_count = vote_counter[majority_vote]
        vote_margin = (majority_count / total_votes) - 0.5
        
        # Scale confidence to [0.5, 1.0] range
        fused_confidence = 0.5 + vote_margin
        
        return majority_vote, fused_confidence
    
    def _soft_voting(self, decisions: List[bool], confidences: List[float]) -> Tuple[bool, float]:
        """
        Combine decisions using soft voting (weighted by confidence).
        
        Args:
            decisions: List of decisions from different models
            confidences: List of confidence values for each decision
            
        Returns:
            Tuple of (fused_decision, fused_confidence)
        """
        if not decisions:
            return False, 0.5
        
        # Calculate weighted votes for each class
        positive_weight = sum(conf if decision else 0.0 for decision, conf in zip(decisions, confidences))
        negative_weight = sum(conf if not decision else 0.0 for decision, conf in zip(decisions, confidences))
        
        # Determine decision based on weighted votes
        fused_decision = positive_weight >= negative_weight
        
        # Calculate confidence based on vote margin
        total_weight = positive_weight + negative_weight
        if total_weight > 0:
            vote_margin = abs(positive_weight - negative_weight) / total_weight
            fused_confidence = 0.5 + (vote_margin / 2)
        else:
            fused_confidence = 0.5
        
        return fused_decision, fused_confidence
    
    def _weighted_voting(self, decisions: List[bool], confidences: List[float], sources: List[str]) -> Tuple[bool, float]:
        """
        Combine decisions using weighted voting (weighted by model weights).
        
        Args:
            decisions: List of decisions from different models
            confidences: List of confidence values for each decision
            sources: List of source names for each decision
            
        Returns:
            Tuple of (fused_decision, fused_confidence)
        """
        if not decisions:
            return False, 0.5
        
        # Get weights for each source
        weights = [self.config['model_weights'].get(source, 1.0) for source in sources]
        
        # Calculate weighted votes for each class
        positive_weight = sum(weight * conf if decision else 0.0 
                             for decision, conf, weight in zip(decisions, confidences, weights))
        negative_weight = sum(weight * conf if not decision else 0.0 
                             for decision, conf, weight in zip(decisions, confidences, weights))
        
        # Determine decision based on weighted votes
        fused_decision = positive_weight >= negative_weight
        
        # Calculate confidence based on vote margin
        total_weight = positive_weight + negative_weight
        if total_weight > 0:
            vote_margin = abs(positive_weight - negative_weight) / total_weight
            fused_confidence = 0.5 + (vote_margin / 2)
        else:
            fused_confidence = 0.5
        
        return fused_decision, fused_confidence
    
    def _dynamic_voting(self, decisions: List[bool], confidences: List[float], 
                      features_list: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Combine decisions using dynamic voting (weights determined dynamically).
        
        Args:
            decisions: List of decisions from different models
            confidences: List of confidence values for each decision
            features_list: List of feature dictionaries for each model
            
        Returns:
            Tuple of (fused_decision, fused_confidence)
        """
        if not decisions:
            return False, 0.5
        
        # Determine dynamic weights based on the selected method
        method = self.config.get('dynamic_weight_method', 'confidence')
        
        if method == 'confidence':
            # Use confidence as weight
            weights = confidences
        
        elif method == 'reliability':
            # Use historical reliability as weight
            # This is a simplified implementation; in a real system, you would
            # track the historical performance of each model
            weights = [0.8, 0.7, 0.9]  # Example weights
            
            # Ensure we have enough weights
            if len(weights) < len(decisions):
                weights = weights + [0.5] * (len(decisions) - len(weights))
            elif len(weights) > len(decisions):
                weights = weights[:len(decisions)]
        
        elif method == 'feature_quality':
            # Use feature quality as weight
            weights = []
            for features in features_list:
                # Calculate a quality score based on feature completeness
                quality = self._calculate_feature_quality(features)
                weights.append(quality)
        
        else:
            self.logger.warning(f"Unknown dynamic weight method: {method}, using equal weights")
            weights = [1.0] * len(decisions)
        
        # Calculate weighted votes for each class
        positive_weight = sum(weight if decision else 0.0 for decision, weight in zip(decisions, weights))
        negative_weight = sum(weight if not decision else 0.0 for decision, weight in zip(decisions, weights))
        
        # Determine decision based on weighted votes
        fused_decision = positive_weight >= negative_weight
        
        # Calculate confidence based on vote margin
        total_weight = positive_weight + negative_weight
        if total_weight > 0:
            vote_margin = abs(positive_weight - negative_weight) / total_weight
            fused_confidence = 0.5 + (vote_margin / 2)
        else:
            fused_confidence = 0.5
        
        return fused_decision, fused_confidence
    
    def _calculate_feature_quality(self, features: Dict[str, Any]) -> float:
        """
        Calculate a quality score for features.
        
        Args:
            features: Features dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        # This is a simplified implementation; in a real system, you would
        # use more sophisticated methods to assess feature quality
        
        # Count the number of non-null features
        non_null_count = 0
        total_count = 0
        
        def count_non_null(d, parent_key=''):
            nonlocal non_null_count, total_count
            
            for k, v in d.items():
                if isinstance(v, dict):
                    count_non_null(v, f"{parent_key}{k}_")
                else:
                    total_count += 1
                    if v is not None:
                        non_null_count += 1
        
        count_non_null(features)
        
        # Calculate quality score
        if total_count > 0:
            return non_null_count / total_count
        else:
            return 0.5