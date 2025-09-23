"""
Edge Sharpness Metrics for Flame Front Detection.

Measures sharpness of thermal gradients that indicate flame fronts.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EdgeSharpnessAnalyzer:
    """Analyzes edge sharpness in thermal data to detect flame fronts."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the edge sharpness analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sharpness_threshold = self.config.get('sharpness_threshold', 5.0)
        self.edge_density_threshold = self.config.get('edge_density_threshold', 0.1)
        
        logger.info("Initialized Edge Sharpness Analyzer")
        logger.info(f"Sharpness threshold: {self.sharpness_threshold}")
    
    def analyze_edge_sharpness(self, thermal_features: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze edge sharpness metrics from thermal gradient features.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Dictionary with edge sharpness features
        """
        try:
            # Extract gradient features
            t_grad_mean = thermal_features.get('t_grad_mean', 0.0)
            t_grad_std = thermal_features.get('t_grad_std', 0.0)
            
            # Calculate edge sharpness metrics
            edge_features = {}
            
            # Sharpness indicators
            edge_features['edge_sharpness_mean'] = t_grad_mean
            edge_features['edge_sharpness_std'] = t_grad_std
            edge_features['edge_sharpness_combined'] = t_grad_mean * t_grad_std
            
            # Sharp edge detection
            edge_features['sharp_edge_indicator'] = 1.0 if t_grad_mean > self.sharpness_threshold else 0.0
            
            # Edge density estimation (based on gradient activity)
            edge_features['edge_density'] = self._estimate_edge_density(thermal_features)
            
            # Flame front likelihood
            edge_features['flame_front_likelihood'] = self._calculate_flame_front_likelihood(thermal_features)
            
            return edge_features
            
        except Exception as e:
            logger.error(f"Error in edge sharpness analysis: {str(e)}")
            return {}
    
    def _estimate_edge_density(self, thermal_features: Dict[str, float]) -> float:
        """
        Estimate edge density from gradient features.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Edge density metric
        """
        t_grad_mean = thermal_features.get('t_grad_mean', 0.0)
        t_grad_std = thermal_features.get('t_grad_std', 0.0)
        
        # Edge density is proportional to both mean and standard deviation of gradients
        edge_density = (t_grad_mean * t_grad_std) / 100.0  # Normalize
        return min(1.0, edge_density)  # Clamp to [0,1]
    
    def _calculate_flame_front_likelihood(self, thermal_features: Dict[str, float]) -> float:
        """
        Calculate likelihood of flame front presence based on edge features.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Flame front likelihood score (0-1)
        """
        t_grad_mean = thermal_features.get('t_grad_mean', 0.0)
        t_grad_std = thermal_features.get('t_grad_std', 0.0)
        
        # Flame fronts typically have:
        # 1. High mean gradient (sharp edges)
        # 2. High gradient variation (complex edge structure)
        
        # Normalize features
        normalized_mean = min(1.0, t_grad_mean / self.sharpness_threshold) if self.sharpness_threshold > 0 else 0.0
        normalized_std = min(1.0, t_grad_std / 10.0)  # Assume 10.0 as reference for std
        
        # Combined score (weighted average)
        flame_likelihood = 0.6 * normalized_mean + 0.4 * normalized_std
        return flame_likelihood