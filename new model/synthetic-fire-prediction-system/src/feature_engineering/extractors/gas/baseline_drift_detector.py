"""
Baseline Drift Detection for Gas Sensors.

Identifies gradual changes vs. sudden spikes in CO₂ measurements.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaselineDriftDetector:
    """Detects baseline drift vs. sudden spikes in gas measurements."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the baseline drift detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.drift_threshold = self.config.get('drift_threshold', 2.0)  # ppm per hour
        self.spike_threshold = self.config.get('spike_threshold', 50.0)  # ppm sudden change
        self.window_size = self.config.get('window_size', 10)  # Readings for baseline calculation
        
        logger.info("Initialized Baseline Drift Detector")
        logger.info(f"Drift threshold: {self.drift_threshold} ppm/hour")
    
    def detect_baseline_drift(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Detect baseline drift vs. sudden spikes.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with baseline drift features
        """
        if len(gas_history) < self.window_size:
            return {}
        
        try:
            # Calculate drift detection features
            drift_features = {}
            
            # Baseline calculation
            baseline_features = self._calculate_baseline(gas_history)
            drift_features.update(baseline_features)
            
            # Drift vs. spike classification
            classification_features = self._classify_changes(gas_history)
            drift_features.update(classification_features)
            
            # Drift rate analysis
            drift_rate_features = self._analyze_drift_rates(gas_history)
            drift_features.update(drift_rate_features)
            
            # Fire-related drift patterns
            fire_drift_features = self._detect_fire_drift_patterns(gas_history)
            drift_features.update(fire_drift_features)
            
            return drift_features
            
        except Exception as e:
            logger.error(f"Error in baseline drift detection: {str(e)}")
            return {}
    
    def _calculate_baseline(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate baseline CO₂ levels.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with baseline features
        """
        features = {}
        
        if len(gas_history) < self.window_size:
            return features
        
        # Extract baseline values (older readings)
        baseline_vals = [reading.get('gas_val', 0.0) for reading in gas_history[-self.window_size:-2]]
        recent_vals = [reading.get('gas_val', 0.0) for reading in gas_history[-2:]]
        
        if baseline_vals:
            # Calculate baseline statistics
            baseline_mean = np.mean(baseline_vals)
            baseline_std = np.std(baseline_vals) if len(baseline_vals) > 1 else 0.0
            
            features['baseline_co2_mean'] = baseline_mean
            features['baseline_co2_std'] = baseline_std
            
            # Compare recent values to baseline
            if recent_vals:
                recent_mean = np.mean(recent_vals)
                deviation_from_baseline = recent_mean - baseline_mean
                features['deviation_from_baseline'] = deviation_from_baseline
                features['normalized_deviation'] = (
                    deviation_from_baseline / (baseline_std + 1e-8) if baseline_std > 0 else 0.0
                )
        
        return features
    
    def _classify_changes(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Classify changes as drift or spikes.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with change classification features
        """
        features = {}
        
        if len(gas_history) < 3:
            return features
        
        # Get recent changes
        recent_deltas = [reading.get('gas_delta', 0.0) for reading in gas_history[-3:]]
        
        # Classify each change
        drift_count = 0
        spike_count = 0
        
        for delta in recent_deltas:
            abs_delta = abs(delta)
            if abs_delta > self.spike_threshold:
                spike_count += 1
            elif abs_delta > 0.1:  # Small but consistent changes indicate drift
                drift_count += 1
        
        features['drift_changes'] = drift_count
        features['spike_changes'] = spike_count
        features['drift_to_spike_ratio'] = spike_count / (drift_count + 1e-8) if drift_count > 0 else 0.0
        
        # Dominant change type
        if spike_count > drift_count:
            features['dominant_change_type'] = 'spike'
        elif drift_count > 0:
            features['dominant_change_type'] = 'drift'
        else:
            features['dominant_change_type'] = 'stable'
        
        return features
    
    def _analyze_drift_rates(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze drift rates over time.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with drift rate features
        """
        features = {}
        
        if len(gas_history) < self.window_size:
            return features
        
        # Extract values for trend analysis
        values = [reading.get('gas_val', 0.0) for reading in gas_history[-self.window_size:]]
        timestamps = [reading.get('timestamp', datetime.now().isoformat()) for reading in gas_history[-self.window_size:]]
        
        # Calculate drift rate (simplified linear trend)
        if len(values) >= 2:
            x = list(range(len(values)))
            y = values
            
            # Linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_xx = sum(x[i] * x[i] for i in range(n))
            
            denominator = n * sum_xx - sum_x * sum_x
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                features['drift_rate'] = slope  # ppm per reading
                
                # Convert to ppm per hour (assuming 1 reading per minute)
                features['drift_rate_hourly'] = slope * 60
                
                # Drift indicator
                features['significant_drift'] = 1.0 if abs(slope) > self.drift_threshold else 0.0
        
        return features
    
    def _detect_fire_drift_patterns(self, gas_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Detect fire-related drift patterns.
        
        Args:
            gas_history: List of recent gas feature dictionaries
            
        Returns:
            Dictionary with fire drift pattern features
        """
        features = {}
        
        if len(gas_history) < self.window_size:
            return features
        
        # Extract recent values
        recent_vals = [reading.get('gas_val', 0.0) for reading in gas_history[-5:]]
        recent_deltas = [reading.get('gas_delta', 0.0) for reading in gas_history[-5:]]
        
        # Fire drift patterns:
        # 1. Consistently positive drift
        # 2. Accelerating drift rate
        # 3. Deviation from normal baseline
        
        # Positive drift consistency
        positive_drifts = [d for d in recent_deltas if d > 0]
        consistency = len(positive_drifts) / len(recent_deltas) if recent_deltas else 0.0
        
        # Acceleration (change in delta)
        if len(recent_deltas) >= 3:
            acceleration = recent_deltas[-1] - recent_deltas[-2]
            features['drift_acceleration'] = acceleration
            features['positive_acceleration'] = 1.0 if acceleration > 0 else 0.0
        
        # Combined fire drift score
        fire_drift_score = consistency  # Simplified for now
        features['fire_drift_score'] = fire_drift_score
        
        return features