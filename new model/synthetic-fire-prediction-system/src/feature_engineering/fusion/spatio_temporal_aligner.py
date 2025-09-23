"""
Spatio-Temporal Alignment Features for FLIR+SCD41 Sensors.

Aligns thermal and gas sensor data in both space and time domains.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SpatioTemporalAligner:
    """Aligns thermal and gas sensor data in space and time domains."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the spatio-temporal aligner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.time_alignment_window = self.config.get('time_alignment_window', 30)  # seconds
        self.spatial_proximity_threshold = self.config.get('spatial_proximity_threshold', 5.0)  # meters
        self.temporal_correlation_window = self.config.get('temporal_correlation_window', 5)  # readings
        
        logger.info("Initialized Spatio-Temporal Aligner")
        logger.info(f"Time alignment window: {self.time_alignment_window} seconds")
    
    def align_sensor_data(self, 
                         thermal_data: List[Dict[str, Any]], 
                         gas_data: List[Dict[str, Any]],
                         sensor_positions: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Align thermal and gas sensor data in spatio-temporal domain.
        
        Args:
            thermal_data: List of thermal feature dictionaries with timestamps
            gas_data: List of gas feature dictionaries with timestamps
            sensor_positions: Dictionary mapping sensor IDs to (x, y) positions
            
        Returns:
            Dictionary with spatio-temporal alignment features
        """
        try:
            alignment_features = {}
            
            # Time alignment analysis
            time_alignment = self._analyze_time_alignment(thermal_data, gas_data)
            alignment_features.update(time_alignment)
            
            # Spatial alignment analysis (if positions provided)
            if sensor_positions:
                spatial_alignment = self._analyze_spatial_alignment(sensor_positions)
                alignment_features.update(spatial_alignment)
            
            # Cross-sensor temporal correlation
            temporal_correlation = self._analyze_temporal_correlation(thermal_data, gas_data)
            alignment_features.update(temporal_correlation)
            
            # Convergence timing analysis
            convergence_timing = self._analyze_convergence_timing(thermal_data, gas_data)
            alignment_features.update(convergence_timing)
            
            # Alignment quality metrics
            quality_metrics = self._calculate_alignment_quality(thermal_data, gas_data)
            alignment_features.update(quality_metrics)
            
            return alignment_features
            
        except Exception as e:
            logger.error(f"Error in spatio-temporal alignment: {str(e)}")
            return {}
    
    def _analyze_time_alignment(self, 
                              thermal_data: List[Dict[str, Any]], 
                              gas_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze temporal alignment between thermal and gas sensors.
        
        Args:
            thermal_data: List of thermal feature dictionaries with timestamps
            gas_data: List of gas feature dictionaries with timestamps
            
        Returns:
            Dictionary with time alignment features
        """
        features = {}
        
        if not thermal_data or not gas_data:
            return features
        
        # Extract timestamps (assuming ISO format)
        try:
            thermal_timestamps = [
                datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
                for data in thermal_data[-self.temporal_correlation_window:]
            ]
            gas_timestamps = [
                datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
                for data in gas_data[-self.temporal_correlation_window:]
            ]
            
            # Calculate time differences
            if thermal_timestamps and gas_timestamps:
                # Time difference between most recent readings
                time_diff = abs((thermal_timestamps[-1] - gas_timestamps[-1]).total_seconds())
                features['sensor_time_difference'] = time_diff
                features['synchronized_reading'] = 1.0 if time_diff <= self.time_alignment_window else 0.0
                
                # Average time difference over window
                if len(thermal_timestamps) > 1 and len(gas_timestamps) > 1:
                    avg_diffs = []
                    min_len = min(len(thermal_timestamps), len(gas_timestamps))
                    for i in range(min_len):
                        diff = abs((
                            thermal_timestamps[-(i+1)] - gas_timestamps[-(i+1)]
                        ).total_seconds())
                        avg_diffs.append(diff)
                    
                    if avg_diffs:
                        features['avg_time_difference'] = np.mean(avg_diffs)
                        features['time_difference_std'] = np.std(avg_diffs) if len(avg_diffs) > 1 else 0.0
        
        except Exception as e:
            logger.warning(f"Error parsing timestamps for time alignment: {str(e)}")
        
        return features
    
    def _analyze_spatial_alignment(self, sensor_positions: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Analyze spatial alignment between sensors.
        
        Args:
            sensor_positions: Dictionary mapping sensor IDs to (x, y) positions
            
        Returns:
            Dictionary with spatial alignment features
        """
        features = {}
        
        try:
            # Extract positions
            positions = list(sensor_positions.values())
            if len(positions) >= 2:
                # Calculate distances between sensors
                distances = []
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        pos1, pos2 = positions[i], positions[j]
                        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        distances.append(distance)
                
                if distances:
                    features['sensor_separation_distance'] = np.mean(distances)
                    features['proximity_indicator'] = (
                        1.0 if np.mean(distances) <= self.spatial_proximity_threshold else 0.0
                    )
                    features['distance_std'] = np.std(distances) if len(distances) > 1 else 0.0
        
        except Exception as e:
            logger.warning(f"Error in spatial alignment analysis: {str(e)}")
        
        return features
    
    def _analyze_temporal_correlation(self, 
                                    thermal_data: List[Dict[str, Any]], 
                                    gas_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze temporal correlation between thermal and gas sensors.
        
        Args:
            thermal_data: List of thermal feature dictionaries
            gas_data: List of gas feature dictionaries
            
        Returns:
            Dictionary with temporal correlation features
        """
        features = {}
        
        if len(thermal_data) < 2 or len(gas_data) < 2:
            return features
        
        try:
            # Extract key features for correlation
            thermal_temps = [data.get('t_mean', 0.0) for data in thermal_data[-self.temporal_correlation_window:]]
            gas_values = [data.get('gas_val', 0.0) for data in gas_data[-self.temporal_correlation_window:]]
            
            # Align to same length
            min_len = min(len(thermal_temps), len(gas_values))
            if min_len >= 2:
                thermal_aligned = thermal_temps[-min_len:]
                gas_aligned = gas_values[-min_len:]
                
                # Calculate correlation
                correlation = self._calculate_correlation(thermal_aligned, gas_aligned)
                features['temporal_correlation'] = correlation
                features['strong_temporal_correlation'] = 1.0 if abs(correlation) > 0.5 else 0.0
        
        except Exception as e:
            logger.warning(f"Error in temporal correlation analysis: {str(e)}")
        
        return features
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            x: First series of values
            y: Second series of values
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        # Calculate means
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Calculate numerator and denominators
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denom_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        denom_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        
        denominator = np.sqrt(denom_x * denom_y)
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation
    
    def _analyze_convergence_timing(self, 
                                  thermal_data: List[Dict[str, Any]], 
                                  gas_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze timing of fire signature convergence between sensors.
        
        Args:
            thermal_data: List of thermal feature dictionaries
            gas_data: List of gas feature dictionaries
            
        Returns:
            Dictionary with convergence timing features
        """
        features = {}
        
        try:
            # Extract fire indicators
            thermal_fire_scores = [data.get('fire_likelihood_score', 0.0) for data in thermal_data[-self.temporal_correlation_window:]]
            gas_fire_scores = [data.get('gas_fire_likelihood_score', 0.0) for data in gas_data[-self.temporal_correlation_window:]]
            
            # Find when each sensor first detected fire (crossing 0.5 threshold)
            thermal_fire_start = None
            gas_fire_start = None
            
            for i, score in enumerate(reversed(thermal_fire_scores)):
                if score > 0.5:
                    thermal_fire_start = len(thermal_fire_scores) - 1 - i
                    break
            
            for i, score in enumerate(reversed(gas_fire_scores)):
                if score > 0.5:
                    gas_fire_start = len(gas_fire_scores) - 1 - i
                    break
            
            # Calculate timing difference
            if thermal_fire_start is not None and gas_fire_start is not None:
                timing_diff = abs(thermal_fire_start - gas_fire_start)
                features['fire_detection_timing_diff'] = timing_diff
                features['synchronized_fire_detection'] = 1.0 if timing_diff <= 1 else 0.0  # Within 1 reading
            else:
                features['fire_detection_timing_diff'] = -1  # Not detected by both
                features['synchronized_fire_detection'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error in convergence timing analysis: {str(e)}")
        
        return features
    
    def _calculate_alignment_quality(self, 
                                   thermal_data: List[Dict[str, Any]], 
                                   gas_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall alignment quality metrics.
        
        Args:
            thermal_data: List of thermal feature dictionaries
            gas_data: List of gas feature dictionaries
            
        Returns:
            Dictionary with alignment quality features
        """
        features = {}
        
        try:
            # Data completeness
            thermal_completeness = len(thermal_data) / self.temporal_correlation_window
            gas_completeness = len(gas_data) / self.temporal_correlation_window
            features['data_completeness_ratio'] = min(thermal_completeness, gas_completeness)
            
            # Temporal resolution consistency
            if len(thermal_data) >= 2:
                thermal_intervals = []
                for i in range(1, min(len(thermal_data), self.temporal_correlation_window)):
                    try:
                        t1 = datetime.fromisoformat(thermal_data[-(i+1)].get('timestamp'))
                        t2 = datetime.fromisoformat(thermal_data[-i].get('timestamp'))
                        interval = (t2 - t1).total_seconds()
                        thermal_intervals.append(interval)
                    except:
                        pass
                
                if thermal_intervals:
                    features['thermal_sampling_consistency'] = 1.0 / (1.0 + np.std(thermal_intervals))
            
            if len(gas_data) >= 2:
                gas_intervals = []
                for i in range(1, min(len(gas_data), self.temporal_correlation_window)):
                    try:
                        t1 = datetime.fromisoformat(gas_data[-(i+1)].get('timestamp'))
                        t2 = datetime.fromisoformat(gas_data[-i].get('timestamp'))
                        interval = (t2 - t1).total_seconds()
                        gas_intervals.append(interval)
                    except:
                        pass
                
                if gas_intervals:
                    features['gas_sampling_consistency'] = 1.0 / (1.0 + np.std(gas_intervals))
        
        except Exception as e:
            logger.warning(f"Error in alignment quality calculation: {str(e)}")
        
        return features