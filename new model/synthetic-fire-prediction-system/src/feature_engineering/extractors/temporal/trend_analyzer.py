"""
Trend Analyzer for time-series data analysis.

This module provides a feature extractor that analyzes trends in time-series data.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ...base_temporal import TemporalFeatureExtractor


class TrendAnalyzer(TemporalFeatureExtractor):
    """
    Feature extractor for trend analysis in time-series data.
    
    This class analyzes time-series data to identify and characterize trends,
    including linear and non-linear trends, trend changes, and trend strength.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trend analyzer.
        
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
        
        if 'trend_detection_method' not in self.config:
            self.config['trend_detection_method'] = 'linear'  # Options: 'linear', 'polynomial', 'moving_average'
        
        if 'polynomial_degree' not in self.config:
            self.config['polynomial_degree'] = 2  # Degree for polynomial trend detection
        
        if 'window_size' not in self.config:
            self.config['window_size'] = 10  # Window size for moving average trend detection
        
        if 'segment_count' not in self.config:
            self.config['segment_count'] = 3  # Number of segments for piecewise trend detection
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 3
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract trend features from time-series data.
        
        Args:
            data: Input time-series data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting trend features")
        
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
        
        # Extract trend features
        trend_features = self._extract_trend_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'trend_features': trend_features
        }
        
        self.logger.info(f"Extracted trend features from {len(df)} samples")
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
        
        # Add trend features
        trend_features = features.get('trend_features', {})
        for feature_name, feature_value in trend_features.items():
            if isinstance(feature_value, list):
                # Skip time series data in the flattened representation
                if len(feature_value) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_value):
                        flat_features[f"trend_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"trend_{feature_name}_mean"] = np.mean(feature_value)
                    flat_features[f"trend_{feature_name}_max"] = np.max(feature_value)
                    flat_features[f"trend_{feature_name}_min"] = np.min(feature_value)
            elif isinstance(feature_value, dict):
                for sub_name, sub_value in feature_value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 20:
                        # For large lists, just include summary statistics
                        flat_features[f"trend_{feature_name}_{sub_name}_count"] = len(sub_value)
                    else:
                        flat_features[f"trend_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"trend_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved trend features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'overall_trend',
            'trend_strength',
            'trend_significance',
            'trend_direction',
            'trend_slope',
            'trend_intercept',
            'trend_r_squared',
            'trend_p_value',
            'trend_std_error',
            'trend_change_points',
            'trend_segments',
            'trend_acceleration',
            'trend_volatility',
            'trend_consistency',
            'trend_forecast'
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
    
    def _extract_trend_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract trend features from time-series data.
        
        Args:
            df: DataFrame containing time-series data
            
        Returns:
            Dictionary containing trend features
        """
        # Get time series column
        time_series_column = self.config['time_series_column']
        
        # Check if DataFrame has the required column
        if time_series_column not in df.columns:
            self.logger.warning(f"Column '{time_series_column}' not found in time series data")
            return {}
        
        # Get time series values
        values = df[time_series_column].values
        
        # Check if we have enough data
        if len(values) < 3:
            self.logger.warning("Not enough data for trend analysis (need at least 3 points)")
            return {}
        
        # Get trend detection method
        method = self.config.get('trend_detection_method', 'linear')
        
        # Extract trend features using the configured method
        if method == 'linear':
            trend_features = self._extract_linear_trend(values)
        elif method == 'polynomial':
            trend_features = self._extract_polynomial_trend(values)
        elif method == 'moving_average':
            trend_features = self._extract_moving_average_trend(values)
        else:
            self.logger.warning(f"Unknown trend detection method: {method}, using linear")
            trend_features = self._extract_linear_trend(values)
        
        # Extract additional trend features
        trend_features.update(self._extract_trend_change_points(values))
        trend_features.update(self._extract_trend_segments(values))
        trend_features.update(self._extract_trend_volatility(values))
        trend_features.update(self._extract_trend_forecast(values))
        
        return trend_features
    
    def _extract_linear_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract linear trend features from time-series data.
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing linear trend features
        """
        # Create time index
        x = np.arange(len(values)).reshape(-1, 1)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(x, values)
        
        # Get trend parameters
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        # Calculate predicted values
        y_pred = model.predict(x)
        
        # Calculate R-squared
        ss_total = np.sum((values - np.mean(values)) ** 2)
        ss_residual = np.sum((values - y_pred) ** 2)
        r_squared = float(1 - ss_residual / ss_total) if ss_total > 0 else 0.0
        
        # Calculate p-value and standard error
        n = len(values)
        if n > 2:
            # Calculate standard error
            std_err = float(np.sqrt(ss_residual / (n - 2)) / np.sqrt(np.sum((x - np.mean(x)) ** 2)))
            
            # Calculate t-statistic
            t_stat = slope / std_err if std_err > 0 else 0.0
            
            # Calculate p-value
            from scipy import stats
            p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), n - 2)))
        else:
            std_err = 0.0
            p_value = 1.0
        
        # Determine trend direction
        if slope > 0.01:
            direction = 'rising'
        elif slope < -0.01:
            direction = 'falling'
        else:
            direction = 'stable'
        
        # Calculate trend strength (R-squared)
        trend_strength = r_squared
        
        # Calculate trend significance (1 - p_value)
        trend_significance = float(1.0 - p_value)
        
        return {
            'overall_trend': 'linear',
            'trend_strength': trend_strength,
            'trend_significance': trend_significance,
            'trend_direction': direction,
            'trend_slope': slope,
            'trend_intercept': intercept,
            'trend_r_squared': r_squared,
            'trend_p_value': p_value,
            'trend_std_error': std_err,
            'trend_fitted_values': y_pred.tolist()
        }
    
    def _extract_polynomial_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract polynomial trend features from time-series data.
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing polynomial trend features
        """
        # Create time index
        x = np.arange(len(values)).reshape(-1, 1)
        
        # Get polynomial degree
        degree = self.config.get('polynomial_degree', 2)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x)
        
        # Fit polynomial regression
        model = LinearRegression()
        model.fit(x_poly, values)
        
        # Get trend parameters
        coefficients = model.coef_.tolist()
        intercept = float(model.intercept_)
        
        # Calculate predicted values
        y_pred = model.predict(x_poly)
        
        # Calculate R-squared
        ss_total = np.sum((values - np.mean(values)) ** 2)
        ss_residual = np.sum((values - y_pred) ** 2)
        r_squared = float(1 - ss_residual / ss_total) if ss_total > 0 else 0.0
        
        # Calculate trend strength (R-squared)
        trend_strength = r_squared
        
        # Calculate trend significance
        # For polynomial regression, we use F-test
        n = len(values)
        p = degree + 1  # Number of parameters
        
        if n > p and ss_residual > 0:
            # Calculate F-statistic
            f_stat = (ss_total - ss_residual) / p / (ss_residual / (n - p))
            
            # Calculate p-value
            from scipy import stats
            p_value = float(1 - stats.f.cdf(f_stat, p, n - p))
            
            # Calculate trend significance (1 - p_value)
            trend_significance = float(1.0 - p_value)
        else:
            p_value = 1.0
            trend_significance = 0.0
        
        # Determine trend direction based on the derivative at the end
        # For polynomial: a_n * n * x^(n-1) + ... + a_1
        derivative_coeffs = [coefficients[i] * i for i in range(1, len(coefficients))]
        x_end = len(values) - 1
        
        # Calculate derivative at the end
        derivative_end = sum(coef * (x_end ** (i)) for i, coef in enumerate(derivative_coeffs))
        
        if derivative_end > 0.01:
            direction = 'rising'
        elif derivative_end < -0.01:
            direction = 'falling'
        else:
            direction = 'stable'
        
        # Calculate trend acceleration (second derivative)
        if degree >= 2:
            # For polynomial: a_n * n * (n-1) * x^(n-2) + ... + a_2 * 2
            accel_coeffs = [derivative_coeffs[i] * i for i in range(1, len(derivative_coeffs))]
            
            # Calculate acceleration at the end
            accel_end = sum(coef * (x_end ** (i)) for i, coef in enumerate(accel_coeffs))
            
            if accel_end > 0.01:
                acceleration = 'accelerating'
            elif accel_end < -0.01:
                acceleration = 'decelerating'
            else:
                acceleration = 'constant'
        else:
            acceleration = 'constant'
            accel_end = 0.0
        
        return {
            'overall_trend': 'polynomial',
            'trend_strength': trend_strength,
            'trend_significance': trend_significance,
            'trend_direction': direction,
            'trend_coefficients': coefficients,
            'trend_intercept': intercept,
            'trend_r_squared': r_squared,
            'trend_p_value': p_value,
            'trend_degree': degree,
            'trend_acceleration': acceleration,
            'trend_acceleration_value': float(accel_end),
            'trend_fitted_values': y_pred.tolist()
        }
    
    def _extract_moving_average_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract moving average trend features from time-series data.
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing moving average trend features
        """
        # Get window size
        window_size = self.config.get('window_size', 10)
        
        # Check if we have enough data
        if len(values) < window_size:
            self.logger.warning(f"Not enough data for moving average trend (need at least {window_size} points)")
            return {
                'overall_trend': 'unknown',
                'trend_strength': 0.0,
                'trend_significance': 0.0,
                'trend_direction': 'unknown',
                'trend_slope': 0.0,
                'trend_consistency': 0.0
            }
        
        # Calculate moving average
        ma = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        
        # Calculate trend direction
        if len(ma) >= 2:
            # Calculate slope of moving average
            ma_slope = (ma[-1] - ma[0]) / (len(ma) - 1)
            
            if ma_slope > 0.01:
                direction = 'rising'
            elif ma_slope < -0.01:
                direction = 'falling'
            else:
                direction = 'stable'
            
            # Calculate trend consistency
            # (Percentage of consecutive increases or decreases)
            ma_diff = np.diff(ma)
            if ma_slope > 0:
                consistency = float(np.sum(ma_diff > 0) / len(ma_diff))
            elif ma_slope < 0:
                consistency = float(np.sum(ma_diff < 0) / len(ma_diff))
            else:
                consistency = float(np.sum(ma_diff == 0) / len(ma_diff))
            
            # Calculate trend strength
            # (Ratio of moving average variance to original variance)
            var_original = np.var(values)
            var_ma = np.var(ma)
            
            trend_strength = float(var_ma / var_original) if var_original > 0 else 0.0
            
            # Calculate trend significance
            # (Based on Mann-Kendall test)
            try:
                from scipy import stats
                
                # Calculate Mann-Kendall statistic
                n = len(ma)
                s = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        s += np.sign(ma[j] - ma[i])
                
                # Calculate variance of s
                var_s = (n * (n - 1) * (2 * n + 5)) / 18
                
                # Calculate z-statistic
                if s > 0:
                    z = (s - 1) / np.sqrt(var_s)
                elif s < 0:
                    z = (s + 1) / np.sqrt(var_s)
                else:
                    z = 0
                
                # Calculate p-value
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                
                # Calculate trend significance (1 - p_value)
                trend_significance = float(1.0 - p_value)
            except Exception as e:
                self.logger.warning(f"Error calculating Mann-Kendall test: {str(e)}")
                trend_significance = float(consistency)  # Fallback to consistency
        else:
            direction = 'unknown'
            ma_slope = 0.0
            consistency = 0.0
            trend_strength = 0.0
            trend_significance = 0.0
        
        return {
            'overall_trend': 'moving_average',
            'trend_strength': trend_strength,
            'trend_significance': trend_significance,
            'trend_direction': direction,
            'trend_slope': float(ma_slope),
            'trend_consistency': consistency,
            'trend_moving_average': ma.tolist()
        }
    
    def _extract_trend_change_points(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract trend change points from time-series data.
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing trend change point features
        """
        # Check if we have enough data
        if len(values) < 10:
            return {
                'trend_change_points': [],
                'trend_change_point_count': 0
            }
        
        try:
            # Use ruptures for change point detection
            import ruptures as rpt
            
            # Create change point detection model
            model = "l2"  # L2 norm
            algo = rpt.Pelt(model=model).fit(values.reshape(-1, 1))
            
            # Find change points
            change_points = algo.predict(pen=1.0)
            
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
                    
                    # Calculate before and after slopes
                    before_x = np.arange(max(0, cp-5), cp)
                    before_y = values[max(0, cp-5):cp]
                    after_x = np.arange(cp, min(len(values), cp+5))
                    after_y = values[cp:min(len(values), cp+5)]
                    
                    if len(before_x) >= 2 and len(after_x) >= 2:
                        before_slope, _, _, _, _ = stats.linregress(before_x, before_y)
                        after_slope, _, _, _, _ = stats.linregress(after_x, after_y)
                        slope_change = after_slope - before_slope
                    else:
                        before_slope = 0.0
                        after_slope = 0.0
                        slope_change = 0.0
                    
                    change_point_info.append({
                        'index': int(cp),
                        'value': float(values[cp]),
                        'change_magnitude': float(change_magnitude),
                        'before_mean': float(before_mean),
                        'after_mean': float(after_mean),
                        'before_slope': float(before_slope),
                        'after_slope': float(after_slope),
                        'slope_change': float(slope_change)
                    })
            
            return {
                'trend_change_points': change_point_info,
                'trend_change_point_count': len(change_point_info)
            }
        
        except ImportError:
            self.logger.warning("ruptures package not available, using simple change point detection")
            
            # Simple change point detection using rolling statistics
            window_size = 5
            change_points = []
            
            if len(values) <= window_size * 2:
                return {
                    'trend_change_points': [],
                    'trend_change_point_count': 0
                }
            
            # Calculate rolling mean
            rolling_mean = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            
            # Calculate rolling slope
            rolling_slope = np.zeros_like(rolling_mean)
            
            for i in range(len(rolling_mean) - window_size):
                x = np.arange(window_size)
                y = rolling_mean[i:i+window_size]
                slope, _, _, _, _ = stats.linregress(x, y)
                rolling_slope[i] = slope
            
            # Calculate differences between consecutive slopes
            slope_diff = np.abs(np.diff(rolling_slope))
            
            # Find points where difference exceeds threshold
            threshold = np.std(slope_diff) * 2
            cp_indices = np.where(slope_diff > threshold)[0]
            
            # Create change point information
            change_point_info = []
            for cp in cp_indices:
                # Adjust index to account for rolling window
                adjusted_cp = cp + window_size
                
                # Calculate change metrics
                if adjusted_cp > window_size and adjusted_cp < len(values) - window_size:
                    before_mean = np.mean(values[adjusted_cp-window_size:adjusted_cp])
                    after_mean = np.mean(values[adjusted_cp:adjusted_cp+window_size])
                    change_magnitude = abs(after_mean - before_mean)
                    
                    # Calculate before and after slopes
                    before_x = np.arange(window_size)
                    before_y = values[adjusted_cp-window_size:adjusted_cp]
                    after_x = np.arange(window_size)
                    after_y = values[adjusted_cp:adjusted_cp+window_size]
                    
                    before_slope, _, _, _, _ = stats.linregress(before_x, before_y)
                    after_slope, _, _, _, _ = stats.linregress(after_x, after_y)
                    slope_change = after_slope - before_slope
                    
                    change_point_info.append({
                        'index': int(adjusted_cp),
                        'value': float(values[adjusted_cp]),
                        'change_magnitude': float(change_magnitude),
                        'before_mean': float(before_mean),
                        'after_mean': float(after_mean),
                        'before_slope': float(before_slope),
                        'after_slope': float(after_slope),
                        'slope_change': float(slope_change)
                    })
            
            return {
                'trend_change_points': change_point_info,
                'trend_change_point_count': len(change_point_info)
            }
    
    def _extract_trend_segments(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract trend segments from time-series data.
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing trend segment features
        """
        # Get segment count
        segment_count = self.config.get('segment_count', 3)
        
        # Check if we have enough data
        if len(values) < segment_count * 2:
            return {
                'trend_segments': [],
                'trend_segment_count': 0
            }
        
        # Calculate segment size
        segment_size = len(values) // segment_count
        
        # Create segments
        segments = []
        
        for i in range(segment_count):
            # Calculate segment indices
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < segment_count - 1 else len(values)
            
            # Get segment values
            segment_values = values[start_idx:end_idx]
            
            # Calculate segment statistics
            segment_mean = float(np.mean(segment_values))
            segment_std = float(np.std(segment_values))
            segment_min = float(np.min(segment_values))
            segment_max = float(np.max(segment_values))
            
            # Calculate segment trend
            x = np.arange(len(segment_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, segment_values)
            
            # Determine segment trend direction
            if slope > 0.01:
                direction = 'rising'
            elif slope < -0.01:
                direction = 'falling'
            else:
                direction = 'stable'
            
            segments.append({
                'start_index': int(start_idx),
                'end_index': int(end_idx),
                'length': int(end_idx - start_idx),
                'mean': segment_mean,
                'std': segment_std,
                'min': segment_min,
                'max': segment_max,
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'std_err': float(std_err),
                'direction': direction
            })
        
        return {
            'trend_segments': segments,
            'trend_segment_count': len(segments)
        }
    
    def _extract_trend_volatility(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract trend volatility features from time-series data.
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing trend volatility features
        """
        # Check if we have enough data
        if len(values) < 3:
            return {
                'trend_volatility': 0.0,
                'trend_consistency': 0.0
            }
        
        # Calculate differences
        diffs = np.diff(values)
        
        # Calculate volatility (standard deviation of differences)
        volatility = float(np.std(diffs))
        
        # Calculate normalized volatility
        mean_abs_value = np.mean(np.abs(values))
        normalized_volatility = float(volatility / mean_abs_value) if mean_abs_value > 0 else 0.0
        
        # Calculate consistency
        # (Percentage of consecutive changes in the same direction)
        sign_changes = np.diff(np.sign(diffs))
        consistency = float(1.0 - np.sum(sign_changes != 0) / len(sign_changes)) if len(sign_changes) > 0 else 1.0
        
        return {
            'trend_volatility': normalized_volatility,
            'trend_consistency': consistency
        }
    
    def _extract_trend_forecast(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract trend forecast features from time-series data.
        
        Args:
            values: Time series values
            