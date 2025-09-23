"""
Multi-scale Blob Analysis for FLIR Thermal Data.

Analyzes hotspots at different spatial scales to identify fire signatures.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BlobAnalyzer:
    """Analyzes hotspots at different spatial scales to identify fire signatures."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the blob analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scales = self.config.get('scales', [1, 2, 4, 8])  # Different analysis scales
        self.temperature_threshold = self.config.get('temperature_threshold', 50.0)  # °C
        
        logger.info("Initialized Multi-scale Blob Analyzer")
        logger.info(f"Analyzing at {len(self.scales)} scales: {self.scales}")
    
    def analyze_blobs(self, thermal_features: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze blobs at multiple scales using existing thermal features.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Dictionary with multi-scale blob analysis features
        """
        try:
            # Extract relevant features
            t_max = thermal_features.get('t_max', 0.0)
            t_mean = thermal_features.get('t_mean', 0.0)
            t_hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
            t_hot_largest_blob_pct = thermal_features.get('t_hot_largest_blob_pct', 0.0)
            
            # Multi-scale blob features
            blob_features = {}
            
            # Scale-based analysis
            for scale in self.scales:
                # Blob size variation at different scales
                blob_features[f'blob_size_scale_{scale}'] = t_hot_largest_blob_pct / scale
                
                # Blob density (hot area vs largest blob)
                blob_features[f'blob_density_scale_{scale}'] = (
                    t_hot_area_pct / t_hot_largest_blob_pct if t_hot_largest_blob_pct > 0 else 0.0
                ) / scale
                
                # Temperature concentration at scale
                blob_features[f'temp_concentration_scale_{scale}'] = (
                    (t_max - t_mean) / scale if scale > 0 else 0.0
                )
            
            # Blob growth patterns
            blob_features['blob_growth_rate'] = self._calculate_growth_rate(thermal_features)
            
            # Blob distribution metrics
            blob_features['blob_uniformity'] = self._calculate_uniformity(thermal_features)
            
            # Fire signature indicators
            blob_features['fire_blob_pattern'] = self._detect_fire_pattern(thermal_features)
            
            return blob_features
            
        except Exception as e:
            logger.error(f"Error in blob analysis: {str(e)}")
            return {}
    
    def _calculate_growth_rate(self, thermal_features: Dict[str, float]) -> float:
        """
        Calculate blob growth rate based on temperature changes.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Growth rate metric
        """
        tproxy_vel = thermal_features.get('tproxy_vel', 0.0)
        tproxy_delta = thermal_features.get('tproxy_delta', 0.0)
        
        # Growth rate is proportional to velocity and delta
        growth_rate = abs(tproxy_vel) * abs(tproxy_delta)
        return growth_rate
    
    def _calculate_uniformity(self, thermal_features: Dict[str, float]) -> float:
        """
        Calculate blob distribution uniformity.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Uniformity metric (0-1, where 1 is perfectly uniform)
        """
        t_hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
        t_hot_largest_blob_pct = thermal_features.get('t_hot_largest_blob_pct', 0.0)
        
        # Uniformity is higher when hot area is distributed across many small blobs
        # rather than concentrated in one large blob
        if t_hot_area_pct > 0:
            uniformity = 1.0 - (t_hot_largest_blob_pct / t_hot_area_pct)
            return max(0.0, min(1.0, uniformity))  # Clamp to [0,1]
        return 0.0
    
    def _detect_fire_pattern(self, thermal_features: Dict[str, float]) -> float:
        """
        Detect fire signature patterns in blob distribution.
        
        Args:
            thermal_features: Dictionary with FLIR thermal features
            
        Returns:
            Fire pattern indicator (0-1, where 1 indicates strong fire pattern)
        """
        t_max = thermal_features.get('t_max', 0.0)
        t_hot_largest_blob_pct = thermal_features.get('t_hot_largest_blob_pct', 0.0)
        
        # Fire patterns typically have:
        # 1. High maximum temperature
        # 2. Concentrated hot area in largest blob
        temp_score = min(1.0, t_max / self.temperature_threshold) if self.temperature_threshold > 0 else 0.0
        concentration_score = min(1.0, t_hot_largest_blob_pct / 50.0) if t_hot_largest_blob_pct > 0 else 0.0
        
        # Combine scores (weighted average)
        fire_pattern = 0.6 * temp_score + 0.4 * concentration_score
        return fire_pattern