"""
Change Point Detector for time-series data analysis.

This module provides a feature extractor that detects and characterizes change points in time-series data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal
import matplotlib.pyplot as plt

from ...base_temporal import TemporalFeatureExtractor


class ChangePointDetector(TemporalFeatureExtractor):
    """
    Feature extractor for change point detection in time-series data.
    
    This class analyzes time-series data to identify and characterize change points,
    including level shifts, trend changes, and variance changes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the change point detector.
        
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
        # Set default values for optional parameters
        if 'time_series_column' not in self.config:
            self.config['time_series_column'] = 'value'
        
        if 'timestamp_column' not in self.config:
            self.config['timestamp_column'] = 'timestamp'
        
        if 'change_point_detection_method' not in self.config:
            self.config['change_point_detection_method'] = 'pelt'  # Options: 'pelt', 'binary_segmentation', 'window_based'
        
        if 'cost_function' not in self.config:
            self.config['cost_function'] = 'l2'  # Options: 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'
        
        if 'penalty' not in self.config:
            self.config['penalty'] = 'default'  # Penalty value for change point detection
        
        if 'min_size' not in self.config:
            self.config['min_size'] = 2  # Minimum segment size
        
        if 'jump' not in self.config:
            self.config['jump'] = 5  # Jump size for binary segmentation
        
        if 'window_size' not in self.config:
            self.config['window_size'] = 10  # Window size for window-based detection
        
        if 'significance_threshold' not in self.config:
            self.config['significance_threshold'] = 0.05  # Threshold for statistical significance
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 3
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract change point features from time-series data.
        
        Args:
            data: Input time-series data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting change point features")
        
        # Check if data is a dictionary or DataFrame
        if isinstance(data, dict):
            # Convert dictionary to DataFrame if needed
            if 'timestamps' in data and 'values' in data:
                timestamps = data['timestamps']
                values = data['values']
                
                # Create DataFrame
                df = pd.DataFrame({
                    self.config['timestamp_column']: timestamps,
                    self.config['time_series_column']: values
                })
            else:
                self.logger.warning("Invalid data format for dictionary input")
                return {}
        else:
            df = data
        
        # Check if DataFrame has required columns
        time_series_column = self.config['time_series_column']
        timestamp_column = self.config['timestamp_column']
        
        if time_series_column not in df.columns:
            self.logger.warning(f"Missing time series column '{time_series_column}' in data")
            return {}
        
        # Apply smoothing if configured
        if self.config.get('apply_smoothing', True):
            df_processed = self._apply_smoothing(df, [time_series_column])
        else:
            df_processed = df
        
        # Extract change point features
        change_point_features = self._extract_change_point_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'change_point_features': change_point_features
        }
        
        self.logger.info(f"Extracted change point features from {len(df)} samples")
        return features
    
    def to_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert extracted features to a pandas DataFrame.
        
        Args:
            features: Extracted features from the extract_features method
            
        Returns:
            DataFrame containing the features in a structured format
        """
        # Flatten the nested feature dictionary
        flat_features = {}
        
        # Add change point features
        change_point_features = features.get('change_point_features', {})
        for feature_name, feature_value in change_point_features.items():
            if isinstance(feature_value, list):
                # Skip time series data in the flattened representation
                if len(feature_value) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_value):
                        flat_features[f"change_point_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"change_point_{feature_name}_count"] = len(feature_value)
            elif isinstance(feature_value, dict):
                for sub_name, sub_value in feature_value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 20:
                        # For large lists, just include summary statistics
                        flat_features[f"change_point_{feature_name}_{sub_name}_count"] = len(sub_value)
                    else:
                        flat_features[f"change_point_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"change_point_{feature_name}"] = feature_value
        
        # Create DataFrame with a single row
        df = pd.DataFrame([flat_features])
        
        return df
    
    def save(self, features: Dict[str, Any], filepath: str) -> None:
        """
        Save extracted features to a file.
        
        Args:
            features: Extracted features from the extract_features method
            filepath: Path to save the features
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(features, f, indent=2)
        
        self.logger.info(f"Saved change point features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'change_points',
            'change_point_count',
            'change_point_locations',
            'change_point_types',
            'change_point_magnitudes',
            'change_point_significance',
            'change_point_segments',
            'change_point_segment_count',
            'change_point_segment_statistics',
            'change_point_segment_trends'
        ]
    
    def _apply_smoothing(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply smoothing to time-series data.
        
        Args:
            df: DataFrame containing time-series data
            columns: List of columns to smooth
            
        Returns:
            DataFrame with smoothed values
        """
        # Create a copy of the DataFrame
        df_smoothed = df.copy()
        
        # Get smoothing window size
        window = self.config.get('smoothing_window', 3)
        
        # Apply smoothing to each column
        for col in columns:
            if col in df.columns:
                # Apply moving average smoothing
                df_smoothed[col] = df[col].rolling(window=window, center=True).mean()
                
                # Fill NaN values at the edges
                df_smoothed[col] = df_smoothed[col].fillna(df[col])
        
        return df_smoothed
    
    def _extract_change_point_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract change point features from time-series data.
        
        Args:
            df: DataFrame containing time-series data
            
        Returns:
            Dictionary containing change point features
        """
        # Get time series column
        time_series_column = self.config['time_series_column']
        timestamp_column = self.config['timestamp_column']
        
        # Check if DataFrame has the required column
        if time_series_column not in df.columns:
            self.logger.warning(f"Column '{time_series_column}' not found in time series data")
            return {}
        
        # Get time series values
        values = df[time_series_column].values
        
        # Check if we have enough data
        if len(values) < 10:
            self.logger.warning("Not enough data for change point detection (need at least 10 points)")
            return {
                'change_points': [],
                'change_point_count': 0
            }
        
        # Get change point detection method
        method = self.config.get('change_point_detection_method', 'pelt')
        
        # Extract change points using the configured method
        if method == 'pelt':
            change_points = self._detect_change_points_pelt(values)
        elif method == 'binary_segmentation':
            change_points = self._detect_change_points_binary_segmentation(values)
        elif method == 'window_based':
            change_points = self._detect_change_points_window_based(values)
        else:
            self.logger.warning(f"Unknown change point detection method: {method}, using pelt")
            change_points = self._detect_change_points_pelt(values)
        
        # Extract change point types
        change_point_types = self._classify_change_points(values, change_points)
        
        # Extract change point segments
        segments = self._extract_segments(values, change_points)
        
        # Extract segment statistics
        segment_statistics = self._extract_segment_statistics(values, segments)
        
        # Extract segment trends
        segment_trends = self._extract_segment_trends(values, segments)
        
        # Prepare result
        change_point_features = {
            'change_points': change_points,
            'change_point_count': len(change_points),
            'change_point_locations': [cp['index'] for cp in change_points],
            'change_point_types': change_point_types,
            'change_point_magnitudes': [cp['magnitude'] for cp in change_points],
            'change_point_significance': [cp['significance'] for cp in change_points],
            'change_point_segments': segments,
            'change_point_segment_count': len(segments),
            'change_point_segment_statistics': segment_statistics,
            'change_point_segment_trends': segment_trends
        }
        
        return change_point_features
    
    def _detect_change_points_pelt(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect change points using Pruned Exact Linear Time (PELT) algorithm.
        
        Args:
            values: Time series values
            
        Returns:
            List of dictionaries containing change point information
        """
        try:
            # Try to import ruptures
            import ruptures as rpt
            
            # Get cost function
            cost_function = self.config.get('cost_function', 'l2')
            
            # Get penalty value
            penalty = self.config.get('penalty', 'default')
            if penalty == 'default':
                penalty = 1.0
            
            # Get minimum segment size
            min_size = self.config.get('min_size', 2)
            
            # Create change point detection model
            algo = rpt.Pelt(model=cost_function, min_size=min_size).fit(values.reshape(-1, 1))
            
            # Find change points
            change_points = algo.predict(pen=penalty)
            
            # Remove the last change point if it's the end of the series
            if change_points and change_points[-1] == len(values):
                change_points = change_points[:-1]
            
            # Create change point information
            change_point_info = []
            for cp in change_points:
                # Calculate change metrics
                if cp > 1 and cp < len(values) - 1:
                    before_mean = np.mean(values[max(0, cp-5):cp])
                    after_mean = np.mean(values[cp:min(len(values), cp+5)])
                    change_magnitude = abs(after_mean - before_mean)
                    
                    # Calculate significance
                    # (Based on t-test for difference in means)
                    before_values = values[max(0, cp-5):cp]
                    after_values = values[cp:min(len(values), cp+5)]
                    
                    if len(before_values) >= 2 and len(after_values) >= 2:
                        t_stat, p_value = stats.ttest_ind(before_values, after_values, equal_var=False)
                        significance = 1.0 - p_value
                    else:
                        significance = 0.0
                    
                    change_point_info.append({
                        'index': int(cp),
                        'value': float(values[cp]),
                        'magnitude': float(change_magnitude),
                        'significance': float(significance),
                        'before_mean': float(before_mean),
                        'after_mean': float(after_mean)
                    })
            
            return change_point_info
        
        except ImportError:
            self.logger.warning("ruptures package not available, using simple change point detection")
            return self._detect_change_points_window_based(values)
    
    def _detect_change_points_binary_segmentation(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect change points using Binary Segmentation algorithm.
        
        Args:
            values: Time series values
            
        Returns:
            List of dictionaries containing change point information
        """
        try:
            # Try to import ruptures
            import ruptures as rpt
            
            # Get cost function
            cost_function = self.config.get('cost_function', 'l2')
            
            # Get penalty value
            penalty = self.config.get('penalty', 'default')
            if penalty == 'default':
                penalty = 1.0
            
            # Get minimum segment size
            min_size = self.config.get('min_size', 2)
            
            # Get jump size
            jump = self.config.get('jump', 5)
            
            # Create change point detection model
            algo = rpt.Binseg(model=cost_function, min_size=min_size, jump=jump).fit(values.reshape(-1, 1))
            
            # Find change points
            change_points = algo.predict(pen=penalty)
            
            # Remove the last change point if it's the end of the series
            if change_points and change_points[-1] == len(values):
                change_points = change_points[:-1]
            
            # Create change point information
            change_point_info = []
            for cp in change_points:
                # Calculate change metrics
                if cp > 1 and cp < len(values) - 1:
                    before_mean = np.mean(values[max(0, cp-5):cp])
                    after_mean = np.mean(values[cp:min(len(values), cp+5)])
                    change_magnitude = abs(after_mean - before_mean)
                    
                    # Calculate significance
                    # (Based on t-test for difference in means)
                    before_values = values[max(0, cp-5):cp]
                    after_values = values[cp:min(len(values), cp+5)]
                    
                    if len(before_values) >= 2 and len(after_values) >= 2:
                        t_stat, p_value = stats.ttest_ind(before_values, after_values, equal_var=False)
                        significance = 1.0 - p_value
                    else:
                        significance = 0.0
                    
                    change_point_info.append({
                        'index': int(cp),
                        'value': float(values[cp]),
                        'magnitude': float(change_magnitude),
                        'significance': float(significance),
                        'before_mean': float(before_mean),
                        'after_mean': float(after_mean)
                    })
            
            return change_point_info
        
        except ImportError:
            self.logger.warning("ruptures package not available, using simple change point detection")
            return self._detect_change_points_window_based(values)
    
    def _detect_change_points_window_based(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect change points using a window-based approach.
        
        Args:
            values: Time series values
            
        Returns:
            List of dictionaries containing change point information
        """
        # Get window size
        window_size = self.config.get('window_size', 10)
        
        # Get significance threshold
        significance_threshold = self.config.get('significance_threshold', 0.05)
        
        # Check if we have enough data
        if len(values) <= window_size * 2:
            return []
        
        # Calculate rolling statistics
        change_points = []
        
        for i in range(window_size, len(values) - window_size):
            # Get windows
            before_window = values[i-window_size:i]
            after_window = values[i:i+window_size]
            
            # Calculate window statistics
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            before_std = np.std(before_window)
            after_std = np.std(after_window)
            
            # Calculate change magnitude
            change_magnitude = abs(after_mean - before_mean)
            
            # Calculate significance
            # (Based on t-test for difference in means)
            t_stat, p_value = stats.ttest_ind(before_window, after_window, equal_var=False)
            
            # Check if change is significant
            if p_value < significance_threshold:
                change_points.append({
                    'index': int(i),
                    'value': float(values[i]),
                    'magnitude': float(change_magnitude),
                    'significance': float(1.0 - p_value),
                    'before_mean': float(before_mean),
                    'after_mean': float(after_mean),
                    'before_std': float(before_std),
                    'after_std': float(after_std),
                    'p_value': float(p_value)
                })
        
        # Filter out overlapping change points
        filtered_change_points = []
        
        if change_points:
            # Sort change points by significance
            sorted_change_points = sorted(change_points, key=lambda x: x['significance'], reverse=True)
            
            # Add the most significant change point
            filtered_change_points.append(sorted_change_points[0])
            
            # Add non-overlapping change points
            for cp in sorted_change_points[1:]:
                # Check if change point is far enough from existing change points
                is_far_enough = True
                
                for existing_cp in filtered_change_points:
                    if abs(cp['index'] - existing_cp['index']) < window_size:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    filtered_change_points.append(cp)
        
        # Sort change points by index
        filtered_change_points = sorted(filtered_change_points, key=lambda x: x['index'])
        
        return filtered_change_points
    
    def _classify_change_points(self, values: np.ndarray, change_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify change points by type.
        
        Args:
            values: Time series values
            change_points: List of change points
            
        Returns:
            List of dictionaries containing change point type information
        """
        change_point_types = []
        
        for cp in change_points:
            cp_index = cp['index']
            
            # Check if we have enough data before and after the change point
            if cp_index < 5 or cp_index >= len(values) - 5:
                change_point_types.append({
                    'index': cp_index,
                    'type': 'unknown',
                    'confidence': 0.0
                })
                continue
            
            # Get windows
            before_window = values[cp_index-5:cp_index]
            after_window = values[cp_index:cp_index+5]
            
            # Calculate window statistics
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            before_std = np.std(before_window)
            after_std = np.std(after_window)
            
            # Calculate slopes
            before_x = np.arange(5)
            after_x = np.arange(5)
            
            before_slope, _, _, _, _ = stats.linregress(before_x, before_window)
            after_slope, _, _, _, _ = stats.linregress(after_x, after_window)
            
            # Classify change point
            mean_change = abs(after_mean - before_mean)
            std_change = abs(after_std - before_std)
            slope_change = abs(after_slope - before_slope)
            
            # Normalize changes
            mean_range = max(abs(before_mean), abs(after_mean)) + 1e-10
            std_range = max(before_std, after_std) + 1e-10
            slope_range = max(abs(before_slope), abs(after_slope)) + 1e-10
            
            norm_mean_change = mean_change / mean_range
            norm_std_change = std_change / std_range
            norm_slope_change = slope_change / slope_range
            
            # Determine change type
            if norm_mean_change > norm_std_change and norm_mean_change > norm_slope_change:
                change_type = 'level_shift'
                confidence = norm_mean_change
            elif norm_std_change > norm_mean_change and norm_std_change > norm_slope_change:
                change_type = 'variance_change'
                confidence = norm_std_change
            elif norm_slope_change > norm_mean_change and norm_slope_change > norm_std_change:
                change_type = 'trend_change'
                confidence = norm_slope_change
            else:
                change_type = 'complex'
                confidence = max(norm_mean_change, norm_std_change, norm_slope_change)
            
            change_point_types.append({
                'index': cp_index,
                'type': change_type,
                'confidence': float(confidence),
                'mean_change': float(norm_mean_change),
                'std_change': float(norm_std_change),
                'slope_change': float(norm_slope_change)
            })
        
        return change_point_types
    
    def _extract_segments(self, values: np.ndarray, change_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract segments between change points.
        
        Args:
            values: Time series values
            change_points: List of change points
            
        Returns:
            List of dictionaries containing segment information
        """
        segments = []
        
        # Add start point
        change_point_indices = [0] + [cp['index'] for cp in change_points] + [len(values)]
        
        # Create segments
        for i in range(len(change_point_indices) - 1):
            start_idx = change_point_indices[i]
            end_idx = change_point_indices[i + 1]
            
            # Get segment values
            segment_values = values[start_idx:end_idx]
            
            segments.append({
                'start_index': int(start_idx),
                'end_index': int(end_idx),
                'length': int(end_idx - start_idx)
            })
        
        return segments
    
    def _extract_segment_statistics(self, values: np.ndarray, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract statistics for each segment.
        
        Args:
            values: Time series values
            segments: List of segments
            
        Returns:
            List of dictionaries containing segment statistics
        """
        segment_statistics = []
        
        for segment in segments:
            start_idx = segment['start_index']
            end_idx = segment['end_index']
            
            # Get segment values
            segment_values = values[start_idx:end_idx]
            
            # Calculate segment statistics
            segment_mean = float(np.mean(segment_values))
            segment_std = float(np.std(segment_values))
            segment_min = float(np.min(segment_values))
            segment_max = float(np.max(segment_values))
            segment_range = float(segment_max - segment_min)
            
            # Calculate additional statistics
            if len(segment_values) >= 2:
                segment_skew = float(stats.skew(segment_values))
                segment_kurtosis = float(stats.kurtosis(segment_values))
            else:
                segment_skew = 0.0
                segment_kurtosis = 0.0
            
            segment_statistics.append({
                'start_index': int(start_idx),
                'end_index': int(end_idx),
                'length': int(end_idx - start_idx),
                'mean': segment_mean,
                'std': segment_std,
                'min': segment_min,
                'max': segment_max,
                'range': segment_range,
                'skew': segment_skew,
                'kurtosis': segment_kurtosis
            })
        
        return segment_statistics
    
    def _extract_segment_trends(self, values: np.ndarray, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract trends for each segment.
        
        Args:
            values: Time series values
            segments: List of segments
            
        Returns:
            List of dictionaries containing segment trends
        """
        segment_trends = []
        
        for segment in segments:
            start_idx = segment['start_index']
            end_idx = segment['end_index']
            
            # Get segment values
            segment_values = values[start_idx:end_idx]
            
            # Calculate segment trend
            if len(segment_values) >= 2:
                x = np.arange(len(segment_values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, segment_values)
                
                # Determine trend direction
                if slope > 0.01:
                    direction = 'rising'
                elif slope < -0.01:
                    direction = 'falling'
                else:
                    direction = 'stable'
                
                # Calculate trend strength (R-squared)
                trend_strength = float(r_value ** 2)
                
                segment_trends.append({
                    'start_index': int(start_idx),
                    'end_index': int(end_idx),
                    'length': int(end_idx - start_idx),
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'std_err': float(std_err),
                    'direction': direction,
                    'strength': trend_strength
                })
            else:
                segment_trends.append({
                    'start_index': int(start_idx),
                    'end_index': int(end_idx),
                    'length': int(end_idx - start_idx),
                    'slope': 0.0,
                    'intercept': float(segment_values[0]) if len(segment_values) > 0 else 0.0,
                    'r_squared': 0.0,
                    'p_value': 1.0,
                    'std_err': 0.0,
                    'direction': 'stable',
                    'strength': 0.0
                })
        
        return segment_trends
    
    def plot_change_points(self, values: np.ndarray, change_points: List[Dict[str, Any]], filepath: Optional[str] = None) -> None:
        """
        Plot time series with change points.
        
        Args:
            values: Time series values
            change_points: List of change points
            filepath: Optional path to save the plot
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot time series
            plt.plot(values, label='Time Series')
            
            # Plot change points
            for cp in change_points:
                plt.axvline(x=cp['index'], color='r', linestyle='--', alpha=0.7)
                plt.text(cp['index'], np.max(values), f"CP: {cp['index']}", rotation=90, verticalalignment='top')
            
            # Add labels and title
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Time Series with Change Points')
            plt.legend()
            
            # Save or show plot
            if filepath:
                plt.savefig(filepath)
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.warning(f"Error plotting change points: {str(e)}")
    
    def plot_segments(self, values: np.ndarray, segments: List[Dict[str, Any]], segment_trends: List[Dict[str, Any]], filepath: Optional[str] = None) -> None:
        """
        Plot time series with segments and trends.
        
        Args:
            values: Time series values
            segments: List of segments
            segment_trends: List of segment trends
            filepath: Optional path to save the plot
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot time series
            plt.plot(values, label='Time Series')
            
            # Plot segments and trends
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            
            for i, (segment, trend) in enumerate(zip(segments, segment_trends)):
                start_idx = segment['start_index']
                