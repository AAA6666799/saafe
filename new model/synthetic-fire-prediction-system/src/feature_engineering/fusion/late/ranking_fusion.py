"""
Ranking fusion implementation for the synthetic fire prediction system.

This module provides an implementation of ranking fusion, which combines
rankings from different models to create a unified ranking.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict

from ...base import FeatureFusion


class RankingFusion(FeatureFusion):
    """
    Implementation of ranking fusion.
    
    This class combines rankings from different models to create a unified ranking.
    It operates on the ranking outputs of models trained on different feature sets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ranking fusion component.
        
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
        valid_methods = ['borda_count', 'reciprocal_rank', 'median_rank', 'highest_rank']
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
        
        if 'num_classes' not in self.config:
            self.config['num_classes'] = 2  # Binary classification by default
    
    def fuse_features(self,
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from different extractors at the ranking level.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            
        Returns:
            Dictionary containing the fused features
        """
        self.logger.info("Performing ranking fusion")
        
        # Extract model rankings
        thermal_rankings = self._extract_rankings(thermal_features, 'thermal')
        gas_rankings = self._extract_rankings(gas_features, 'gas')
        env_rankings = self._extract_rankings(environmental_features, 'environmental')
        
        # Combine rankings based on the selected fusion method
        fusion_method = self.config['fusion_method']
        
        if fusion_method == 'borda_count':
            fused_rankings, fused_scores = self._borda_count(
                [thermal_rankings, gas_rankings, env_rankings]
            )
        
        elif fusion_method == 'reciprocal_rank':
            fused_rankings, fused_scores = self._reciprocal_rank(
                [thermal_rankings, gas_rankings, env_rankings]
            )
        
        elif fusion_method == 'median_rank':
            fused_rankings, fused_scores = self._median_rank(
                [thermal_rankings, gas_rankings, env_rankings]
            )
        
        elif fusion_method == 'highest_rank':
            fused_rankings, fused_scores = self._highest_rank(
                [thermal_rankings, gas_rankings, env_rankings]
            )
        
        else:
            self.logger.warning(f"Unknown fusion method: {fusion_method}, using Borda count")
            fused_rankings, fused_scores = self._borda_count(
                [thermal_rankings, gas_rankings, env_rankings]
            )
        
        # Determine decision based on threshold
        # For ranking fusion, we convert the top-ranked class score to a probability
        num_classes = self.config['num_classes']
        top_class = fused_rankings[0]
        top_score = fused_scores[0]
        
        # Normalize score to [0, 1] range
        max_possible_score = self._get_max_possible_score(fusion_method, 3)  # 3 models
        normalized_score = top_score / max_possible_score if max_possible_score > 0 else 0.5
        
        # Determine if the top class is the positive class (class 1)
        fused_decision = top_class == 1
        
        # Create fused features dictionary
        fused_features = {
            'fusion_method': fusion_method,
            'fusion_time': datetime.now().isoformat(),
            'model_rankings': {
                'thermal': thermal_rankings,
                'gas': gas_rankings,
                'environmental': env_rankings
            },
            'fused_rankings': fused_rankings.tolist(),
            'fused_scores': fused_scores.tolist(),
            'top_class': int(top_class),
            'top_score': float(top_score),
            'normalized_score': float(normalized_score),
            'fused_decision': int(fused_decision),
            'decision_threshold': self.config['decision_threshold']
        }
        
        self.logger.info(f"Ranking fusion completed with top class: {top_class}, score: {normalized_score}")
        return fused_features
    
    def calculate_risk_score(self, fused_features: Dict[str, Any]) -> float:
        """
        Calculate a risk score from the fused features.
        
        Args:
            fused_features: Fused features from the fuse_features method
            
        Returns:
            Risk score as a float between 0 and 1
        """
        # Extract normalized score and top class
        normalized_score = fused_features.get('normalized_score', 0.5)
        top_class = fused_features.get('top_class', 0)
        
        # Calculate risk score based on top class and normalized score
        if top_class == 1:  # Positive class (fire detected)
            risk_score = normalized_score
        else:  # Negative class (no fire detected)
            # Still assign a small risk based on the normalized score
            # Lower confidence in negative decision means higher risk
            risk_score = 0.1 * normalized_score
        
        self.logger.info(f"Calculated risk score: {risk_score}")
        return risk_score
    
    def _extract_rankings(self, features: Dict[str, Any], source: str) -> np.ndarray:
        """
        Extract rankings from features.
        
        Args:
            features: Features dictionary
            source: Source of the features (thermal, gas, environmental)
            
        Returns:
            Array of class rankings (indices sorted by decreasing probability)
        """
        num_classes = self.config['num_classes']
        
        # Look for class probabilities in the features
        probabilities = None
        
        # Check for explicit class probabilities
        if 'class_probabilities' in features:
            probabilities = features['class_probabilities']
        
        # If not found, check for probability or confidence
        if probabilities is None and 'probability' in features:
            # Convert single probability to binary class probabilities
            prob = features['probability']
            probabilities = [1.0 - prob, prob]  # [P(class=0), P(class=1)]
        
        if probabilities is None and 'confidence' in features:
            # Convert confidence to binary class probabilities
            conf = features['confidence']
            decision = features.get('decision', False)
            
            if decision:
                probabilities = [1.0 - conf, conf]  # [P(class=0), P(class=1)]
            else:
                probabilities = [conf, 1.0 - conf]  # [P(class=0), P(class=1)]
        
        # If still not found, try to infer from other features
        if probabilities is None:
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
                    prob = 1.0 / (1.0 + np.exp(-(max_temp - threshold) / 10))
                    probabilities = [1.0 - prob, prob]  # [P(class=0), P(class=1)]
            
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
                    prob = 1.0 / (1.0 + np.exp(-(max_conc - threshold) / 5))
                    probabilities = [1.0 - prob, prob]  # [P(class=0), P(class=1)]
            
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
                    prob = 1.0 / (1.0 + np.exp(-(temp_rise - threshold) / 2))
                    probabilities = [1.0 - prob, prob]  # [P(class=0), P(class=1)]
        
        # If still not found, use default values
        if probabilities is None:
            probabilities = [0.5] * num_classes
            self.logger.warning(f"Could not extract probabilities from {source} features, using default")
        
        # Ensure probabilities is a list or array with the correct length
        if len(probabilities) != num_classes:
            self.logger.warning(f"Expected {num_classes} classes, but got {len(probabilities)}. Adjusting...")
            if len(probabilities) < num_classes:
                # Pad with zeros
                probabilities = list(probabilities) + [0.0] * (num_classes - len(probabilities))
            else:
                # Truncate
                probabilities = probabilities[:num_classes]
        
        # Convert to numpy array
        probabilities = np.array(probabilities)
        
        # Get rankings (indices sorted by decreasing probability)
        rankings = np.argsort(-probabilities)
        
        return rankings
    
    def _borda_count(self, rankings_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine rankings using Borda count.
        
        Args:
            rankings_list: List of rankings from different models
            
        Returns:
            Tuple of (fused_rankings, fused_scores)
        """
        num_classes = self.config['num_classes']
        
        # Initialize scores for each class
        scores = np.zeros(num_classes)
        
        # Calculate Borda count for each class
        for rankings in rankings_list:
            for i, rank in enumerate(rankings):
                # Assign points based on rank (higher rank = more points)
                scores[rank] += num_classes - i - 1
        
        # Sort classes by decreasing score
        fused_rankings = np.argsort(-scores)
        fused_scores = scores[fused_rankings]
        
        return fused_rankings, fused_scores
    
    def _reciprocal_rank(self, rankings_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine rankings using reciprocal rank fusion.
        
        Args:
            rankings_list: List of rankings from different models
            
        Returns:
            Tuple of (fused_rankings, fused_scores)
        """
        num_classes = self.config['num_classes']
        
        # Initialize scores for each class
        scores = np.zeros(num_classes)
        
        # Calculate reciprocal rank for each class
        for rankings in rankings_list:
            for i, rank in enumerate(rankings):
                # Add reciprocal of (rank + k), where k is a constant (typically k=60)
                k = 60
                scores[rank] += 1.0 / (i + k)
        
        # Sort classes by decreasing score
        fused_rankings = np.argsort(-scores)
        fused_scores = scores[fused_rankings]
        
        return fused_rankings, fused_scores
    
    def _median_rank(self, rankings_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine rankings using median rank.
        
        Args:
            rankings_list: List of rankings from different models
            
        Returns:
            Tuple of (fused_rankings, fused_scores)
        """
        num_classes = self.config['num_classes']
        
        # Initialize rank matrix (rows=models, columns=classes)
        rank_matrix = np.zeros((len(rankings_list), num_classes))
        
        # Fill rank matrix
        for i, rankings in enumerate(rankings_list):
            for j, rank in enumerate(rankings):
                rank_matrix[i, rank] = j
        
        # Calculate median rank for each class
        median_ranks = np.median(rank_matrix, axis=0)
        
        # Sort classes by increasing median rank (lower rank is better)
        fused_rankings = np.argsort(median_ranks)
        
        # Convert ranks to scores (lower rank = higher score)
        fused_scores = num_classes - median_ranks[fused_rankings]
        
        return fused_rankings, fused_scores
    
    def _highest_rank(self, rankings_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine rankings using highest rank (minimum rank).
        
        Args:
            rankings_list: List of rankings from different models
            
        Returns:
            Tuple of (fused_rankings, fused_scores)
        """
        num_classes = self.config['num_classes']
        
        # Initialize rank matrix (rows=models, columns=classes)
        rank_matrix = np.zeros((len(rankings_list), num_classes))
        
        # Fill rank matrix
        for i, rankings in enumerate(rankings_list):
            for j, rank in enumerate(rankings):
                rank_matrix[i, rank] = j
        
        # Calculate minimum rank for each class
        min_ranks = np.min(rank_matrix, axis=0)
        
        # Sort classes by increasing minimum rank (lower rank is better)
        fused_rankings = np.argsort(min_ranks)
        
        # Convert ranks to scores (lower rank = higher score)
        fused_scores = num_classes - min_ranks[fused_rankings]
        
        return fused_rankings, fused_scores
    
    def _get_max_possible_score(self, fusion_method: str, num_models: int) -> float:
        """
        Get the maximum possible score for a fusion method.
        
        Args:
            fusion_method: Fusion method name
            num_models: Number of models
            
        Returns:
            Maximum possible score
        """
        num_classes = self.config['num_classes']
        
        if fusion_method == 'borda_count':
            # Max score is when a class is ranked first by all models
            return num_models * (num_classes - 1)
        
        elif fusion_method == 'reciprocal_rank':
            # Max score is when a class is ranked first by all models
            k = 60  # Same constant as in _reciprocal_rank
            return num_models * (1.0 / (0 + k))
        
        elif fusion_method == 'median_rank' or fusion_method == 'highest_rank':
            # Max score is when a class is ranked first by all models
            return num_classes - 0  # 0 is the best rank
        
        else:
            return 1.0  # Default