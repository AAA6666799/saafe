"""
Detection Time Tracker for FLIR+SCD41 Fire Detection System.

This module implements time-to-detection tracking, early warning generation,
and performance metrics for fire detection speed.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import os

logger = logging.getLogger(__name__)

class DetectionTimeMetrics:
    """Container for detection time metrics."""
    
    def __init__(self):
        self.average_detection_time = 0.0
        self.median_detection_time = 0.0
        self.min_detection_time = 0.0
        self.max_detection_time = 0.0
        self.detection_rate = 0.0  # Percentage of fires detected
        self.early_warning_rate = 0.0  # Percentage of early warnings generated
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'average_detection_time': self.average_detection_time,
            'median_detection_time': self.median_detection_time,
            'min_detection_time': self.min_detection_time,
            'max_detection_time': self.max_detection_time,
            'detection_rate': self.detection_rate,
            'early_warning_rate': self.early_warning_rate,
            'timestamp': self.timestamp.isoformat()
        }

class DetectionTimeTracker:
    """Tracks and analyzes fire detection timing performance."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize detection time tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.baseline_detection_time = self.config.get('baseline_detection_time', 45.0)  # 45 seconds baseline
        self.target_improvement = self.config.get('target_improvement', 0.378)  # 37.8% improvement target
        self.early_warning_threshold = self.config.get('early_warning_threshold', 0.5)  # 50% of detection time
        self.detection_history = deque(maxlen=1000)
        self.scenario_types = [
            'rapid_flame_spread',
            'smoldering_fire',
            'flashover',
            'backdraft',
            'other'
        ]
        
        # Load historical data if file exists
        self.history_file = self.config.get('history_file', 'detection_time_history.json')
        self._load_history()
        
        logger.info("Detection Time Tracker initialized")
    
    def _load_history(self):
        """Load detection history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    for entry_dict in data.get('detections', []):
                        self.detection_history.append({
                            'detection_time': entry_dict['detection_time'],
                            'scenario_type': entry_dict['scenario_type'],
                            'early_warning': entry_dict['early_warning'],
                            'timestamp': datetime.fromisoformat(entry_dict['timestamp']),
                            'confidence': entry_dict.get('confidence', 0.0)
                        })
                logger.info(f"Loaded {len(self.detection_history)} historical detections")
            except Exception as e:
                logger.warning(f"Failed to load detection history: {e}")
    
    def _save_history(self):
        """Save detection history to file."""
        try:
            data = {
                'detections': [
                    {
                        'detection_time': entry['detection_time'],
                        'scenario_type': entry['scenario_type'],
                        'early_warning': entry['early_warning'],
                        'timestamp': entry['timestamp'].isoformat(),
                        'confidence': entry.get('confidence', 0.0)
                    }
                    for entry in self.detection_history
                ],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save detection history: {e}")
    
    def record_detection(self, detection_time: float, scenario_type: str = 'other',
                        early_warning: bool = False, confidence: float = 0.0):
        """
        Record a fire detection event.
        
        Args:
            detection_time: Time from fire start to detection (seconds)
            scenario_type: Type of fire scenario
            early_warning: Whether early warning was generated
            confidence: Detection confidence score
        """
        detection_record = {
            'detection_time': detection_time,
            'scenario_type': scenario_type,
            'early_warning': early_warning,
            'timestamp': datetime.now(),
            'confidence': confidence
        }
        
        self.detection_history.append(detection_record)
        self._save_history()
        
        logger.info(f"Recorded detection: {detection_time:.1f}s for {scenario_type} scenario")
    
    def analyze_detection_times(self) -> DetectionTimeMetrics:
        """
        Analyze detection time performance.
        
        Returns:
            DetectionTimeMetrics object
        """
        metrics = DetectionTimeMetrics()
        
        if len(self.detection_history) == 0:
            return metrics
        
        try:
            # Extract detection times
            detection_times = [entry['detection_time'] for entry in self.detection_history]
            
            # Calculate basic statistics
            metrics.average_detection_time = np.mean(detection_times)
            metrics.median_detection_time = np.median(detection_times)
            metrics.min_detection_time = np.min(detection_times)
            metrics.max_detection_time = np.max(detection_times)
            
            # Calculate detection rate (assuming all recorded events were actual fires)
            total_events = len(self.detection_history)
            metrics.detection_rate = 1.0  # By definition, since we're recording detections
            
            # Calculate early warning rate
            early_warnings = sum(1 for entry in self.detection_history if entry['early_warning'])
            if total_events > 0:
                metrics.early_warning_rate = early_warnings / total_events
            
        except Exception as e:
            logger.warning(f"Error analyzing detection times: {e}")
        
        return metrics
    
    def get_scenario_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze detection performance by scenario type.
        
        Returns:
            Dictionary with scenario analysis
        """
        scenario_analysis = {}
        
        try:
            # Group by scenario type
            scenario_groups = defaultdict(list)
            for entry in self.detection_history:
                scenario_groups[entry['scenario_type']].append(entry)
            
            # Analyze each scenario type
            for scenario_type, entries in scenario_groups.items():
                if len(entries) == 0:
                    continue
                
                detection_times = [entry['detection_time'] for entry in entries]
                early_warnings = sum(1 for entry in entries if entry['early_warning'])
                
                scenario_analysis[scenario_type] = {
                    'count': len(entries),
                    'average_detection_time': np.mean(detection_times),
                    'median_detection_time': np.median(detection_times),
                    'min_detection_time': np.min(detection_times),
                    'max_detection_time': np.max(detection_times),
                    'early_warning_rate': early_warnings / len(entries) if entries else 0.0,
                    'improvement': (self.baseline_detection_time - np.mean(detection_times)) / self.baseline_detection_time * 100
                }
                
        except Exception as e:
            logger.warning(f"Error in scenario analysis: {e}")
        
        return scenario_analysis
    
    def get_improvement_metrics(self) -> Dict[str, Any]:
        """
        Calculate detection time improvement metrics.
        
        Returns:
            Dictionary with improvement metrics
        """
        if len(self.detection_history) == 0:
            return {'status': 'no_data'}
        
        # Get current average detection time
        detection_times = [entry['detection_time'] for entry in self.detection_history]
        current_avg_time = np.mean(detection_times)
        
        # Calculate improvement
        improvement_percentage = (self.baseline_detection_time - current_avg_time) / self.baseline_detection_time * 100
        
        # Check if target achieved
        target_achieved = improvement_percentage >= (self.target_improvement * 100)
        
        return {
            'status': 'success',
            'baseline_detection_time': self.baseline_detection_time,
            'current_average_time': current_avg_time,
            'improvement_percentage': improvement_percentage,
            'target_improvement': self.target_improvement * 100,
            'target_achieved': target_achieved,
            'time_saved_per_detection': self.baseline_detection_time - current_avg_time,
            'total_detections_analyzed': len(self.detection_history)
        }
    
    def generate_early_warning(self, current_features: pd.DataFrame, 
                              prediction_proba: np.ndarray,
                              time_series_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate early warning based on current data and predictions.
        
        Args:
            current_features: Current sensor features
            prediction_proba: Prediction probabilities
            time_series_data: Historical time series data
            
        Returns:
            Dictionary with early warning information
        """
        early_warning = {
            'should_warn': False,
            'confidence': 0.0,
            'warning_level': 'normal',
            'indicators': {},
            'estimated_time_to_fire': None
        }
        
        try:
            # Get fire probability
            fire_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba
            
            # Check if probability exceeds early warning threshold
            if fire_probability >= self.early_warning_threshold:
                early_warning['should_warn'] = True
                early_warning['confidence'] = fire_probability
                early_warning['warning_level'] = 'warning' if fire_probability < 0.8 else 'critical'
            
            # Analyze temporal trends for early warning
            if time_series_data is not None and len(time_series_data) >= 5:
                temporal_indicators = self._analyze_temporal_indicators(time_series_data)
                early_warning['indicators'].update(temporal_indicators)
                
                # If temporal trends indicate fire buildup, enhance warning
                increasing_trends = sum(1 for ind in temporal_indicators.values() if ind.get('is_increasing', False))
                if increasing_trends >= 2 and fire_probability >= 0.3:
                    early_warning['should_warn'] = True
                    early_warning['confidence'] = max(early_warning['confidence'], 0.6)
                    if early_warning['warning_level'] == 'normal':
                        early_warning['warning_level'] = 'warning'
            
            # Estimate time to fire based on trends
            if early_warning['should_warn']:
                early_warning['estimated_time_to_fire'] = self._estimate_time_to_fire(
                    current_features, time_series_data
                )
                
        except Exception as e:
            logger.warning(f"Error generating early warning: {e}")
        
        return early_warning
    
    def _analyze_temporal_indicators(self, time_series_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal indicators for early warning.
        
        Args:
            time_series_data: Historical time series data
            
        Returns:
            Dictionary with temporal indicators
        """
        indicators = {}
        
        try:
            # Analyze recent trends (last 5 points)
            recent_data = time_series_data.tail(5)
            
            # Temperature trend
            if 't_max' in recent_data.columns:
                temp_values = recent_data['t_max'].values
                if len(temp_values) >= 2:
                    temp_slope = np.polyfit(np.arange(len(temp_values)), temp_values, 1)[0]
                    indicators['temperature_trend'] = {
                        'slope': temp_slope,
                        'is_increasing': temp_slope > 1.0,  # Increasing by more than 1°C per time step
                        'magnitude': abs(temp_slope)
                    }
            
            # CO2 trend
            if 'gas_val' in recent_data.columns:
                co2_values = recent_data['gas_val'].values
                if len(co2_values) >= 2:
                    co2_slope = np.polyfit(np.arange(len(co2_values)), co2_values, 1)[0]
                    indicators['co2_trend'] = {
                        'slope': co2_slope,
                        'is_increasing': co2_slope > 10.0,  # Increasing by more than 10ppm per time step
                        'magnitude': abs(co2_slope)
                    }
            
            # Hot area trend
            if 't_hot_area_pct' in recent_data.columns:
                hot_area_values = recent_data['t_hot_area_pct'].values
                if len(hot_area_values) >= 2:
                    hot_area_slope = np.polyfit(np.arange(len(hot_area_values)), hot_area_values, 1)[0]
                    indicators['hot_area_trend'] = {
                        'slope': hot_area_slope,
                        'is_increasing': hot_area_slope > 0.5,  # Increasing by more than 0.5% per time step
                        'magnitude': abs(hot_area_slope)
                    }
                    
        except Exception as e:
            logger.warning(f"Error analyzing temporal indicators: {e}")
        
        return indicators
    
    def _estimate_time_to_fire(self, current_features: pd.DataFrame, 
                              time_series_data: pd.DataFrame = None) -> Optional[float]:
        """
        Estimate time to fire based on current trends.
        
        Args:
            current_features: Current sensor features
            time_series_data: Historical time series data
            
        Returns:
            Estimated time to fire in seconds, or None if cannot estimate
        """
        try:
            # This is a simplified estimation - in a real system, this would be more sophisticated
            if time_series_data is None or len(time_series_data) < 3:
                return None
            
            # Get recent trend data
            recent_data = time_series_data.tail(3)
            
            # Estimate based on temperature trend
            if 't_max' in recent_data.columns:
                temp_values = recent_data['t_max'].values
                if len(temp_values) >= 2:
                    temp_slope = np.polyfit(np.arange(len(temp_values)), temp_values, 1)[0]
                    current_temp = current_features['t_max'].iloc[0] if len(current_features) > 0 else temp_values[-1]
                    
                    # Estimate time to reach critical temperature (e.g., 80°C)
                    critical_temp = 80.0
                    if temp_slope > 0 and current_temp < critical_temp:
                        estimated_time = (critical_temp - current_temp) / temp_slope
                        return max(0.0, estimated_time)
            
        except Exception as e:
            logger.warning(f"Error estimating time to fire: {e}")
        
        return None
    
    def generate_detection_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive detection time analysis report.
        
        Returns:
            Dictionary with detection time report
        """
        if len(self.detection_history) == 0:
            return {'status': 'no_data'}
        
        # Get current metrics
        current_metrics = self.analyze_detection_times()
        
        # Get scenario analysis
        scenario_analysis = self.get_scenario_analysis()
        
        # Get improvement metrics
        improvement_metrics = self.get_improvement_metrics()
        
        # Get historical statistics
        all_detection_times = [entry['detection_time'] for entry in self.detection_history]
        
        report = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(self.detection_history),
            'current_metrics': current_metrics.to_dict(),
            'scenario_analysis': scenario_analysis,
            'improvement_metrics': improvement_metrics,
            'historical_statistics': {
                'min_detection_time': np.min(all_detection_times) if all_detection_times else 0,
                'max_detection_time': np.max(all_detection_times) if all_detection_times else 0,
                'mean_detection_time': np.mean(all_detection_times) if all_detection_times else 0,
                'std_detection_time': np.std(all_detection_times) if all_detection_times else 0,
                'percentile_25': np.percentile(all_detection_times, 25) if all_detection_times else 0,
                'percentile_75': np.percentile(all_detection_times, 75) if all_detection_times else 0
            }
        }
        
        return report

# Convenience function
def create_detection_time_tracker(config: Dict[str, Any] = None) -> DetectionTimeTracker:
    """
    Create a detection time tracker instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DetectionTimeTracker instance
    """
    return DetectionTimeTracker(config)

__all__ = ['DetectionTimeMetrics', 'DetectionTimeTracker', 'create_detection_time_tracker']