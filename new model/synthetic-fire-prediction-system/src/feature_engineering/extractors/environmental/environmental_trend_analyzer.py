"""
Environmental Trend Analyzer for environmental data analysis.

This module provides a feature extractor that analyzes trends in environmental data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal
from statsmodels.tsa.seasonal import STL

from ...base import EnvironmentalFeatureExtractor


class EnvironmentalTrendAnalyzer(EnvironmentalFeatureExtractor):
    """
    Feature extractor for analyzing trends in environmental data.
    
    This class analyzes environmental data to extract features related to
    trends, seasonality, and long-term patterns in environmental parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environmental trend analyzer.
        
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
        if 'parameters' not in self.config:
            self.config['parameters'] = ['temperature', 'humidity', 'pressure']
        
        if 'time_windows' not in self.config:
            self.config['time_windows'] = [10, 30, 60, 120]  # Window sizes for trend analysis
        
        if 'seasonality_period' not in self.config:
            self.config['seasonality_period'] = 24  # Default period for daily seasonality
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 5
        
        if 'trend_threshold' not in self.config:
            self.config['trend_threshold'] = 0.01  # Threshold for significant trend
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract trend features from environmental data.
        
        Args:
            data: Input environmental data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting environmental trend features")
        
        # Check if data is a dictionary or DataFrame
        if isinstance(data, dict):
            # Convert dictionary to DataFrame if needed
            if 'timestamps' in data and 'readings' in data:
                timestamps = data['timestamps']
                readings = data['readings']
                
                # Create DataFrame
                df = pd.DataFrame(readings)
                if len(timestamps) == len(readings):
                    df['timestamp'] = timestamps
            else:
                self.logger.warning("Invalid data format for dictionary input")
                return {}
        else:
            df = data
        
        # Check if timestamp column exists
        if 'timestamp' not in df.columns:
            # Create a dummy timestamp column
            self.logger.warning("No timestamp column found, creating dummy timestamps")
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='S')
        
        # Convert timestamps to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                self.logger.warning("Failed to convert timestamps to datetime")
        
        # Get parameters to analyze
        parameters = self.config['parameters']
        available_params = [param for param in parameters if param in df.columns]
        
        if not available_params:
            self.logger.warning(f"None of the specified parameters {parameters} found in data")
            return {}
        
        # Apply smoothing if configured
        if self.config.get('apply_smoothing', True):
            df_processed = self._apply_smoothing(df, available_params)
        else:
            df_processed = df
        
        # Extract trend features
        trend_features = self._extract_trend_features(df_processed, available_params)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'parameters_analyzed': available_params,
            'trend_features': trend_features
        }
        
        self.logger.info(f"Extracted environmental trend features from {len(df)} samples")
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
        for param_name, param_features in trend_features.items():
            for feature_name, feature_value in param_features.items():
                if isinstance(feature_value, dict):
                    for sub_name, sub_value in feature_value.items():
                        flat_features[f"{param_name}_trend_{feature_name}_{sub_name}"] = sub_value
                elif isinstance(feature_value, list):
                    # Skip time series data in the flattened representation
                    if len(feature_value) <= 20:  # Only include reasonably sized lists
                        for i, value in enumerate(feature_value):
                            flat_features[f"{param_name}_trend_{feature_name}_{i}"] = value
                    else:
                        # For large lists, just include summary statistics
                        flat_features[f"{param_name}_trend_{feature_name}_mean"] = np.mean(feature_value)
                        flat_features[f"{param_name}_trend_{feature_name}_max"] = np.max(feature_value)
                        flat_features[f"{param_name}_trend_{feature_name}_min"] = np.min(feature_value)
                else:
                    flat_features[f"{param_name}_trend_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved environmental trend features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        # Base feature names
        base_features = [
            'trend_slope',
            'trend_strength',
            'trend_direction',
            'trend_change_points',
            'seasonality_strength',
            'seasonality_period',
            'residual_strength',
            'long_term_trend',
            'short_term_trend',
            'trend_acceleration',
            'trend_stability',
            'trend_reversal_count',
            'trend_duration'
        ]
        
        # Generate feature names for each parameter
        feature_names = []
        for param in self.config.get('parameters', []):
            for feature in base_features:
                feature_names.append(f"{param}_trend_{feature}")
        
        return feature_names
    
    def calculate_environmental_statistics(self,
                                         env_readings: pd.DataFrame,
                                         parameters: List[str],
                                         window_size: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for environmental parameters.
        
        Args:
            env_readings: DataFrame containing environmental readings
            parameters: List of environmental parameters to analyze
            window_size: Window size (in samples) for statistics calculation
            
        Returns:
            Nested dictionary mapping parameter to statistics
        """
        result = {}
        
        for param in parameters:
            if param not in env_readings.columns:
                continue
            
            # Get parameter values
            values = env_readings[param].values
            
            # Calculate basic statistics
            result[param] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values))
            }
            
            # Calculate rolling statistics if we have enough data
            if len(values) > window_size:
                rolling_mean = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                rolling_std = np.array([np.std(values[i:i+window_size]) for i in range(len(values)-window_size+1)])
                
                result[param].update({
                    'rolling_mean_std': float(np.std(rolling_mean)),
                    'rolling_std_mean': float(np.mean(rolling_std))
                })
        
        return result
    
    def calculate_dew_point(self,
                           temperature: float,
                           humidity: float) -> float:
        """
        Calculate dew point from temperature and humidity.
        
        Args:
            temperature: Temperature in degrees Celsius
            humidity: Relative humidity as a percentage
            
        Returns:
            Dew point in degrees Celsius
        """
        # Constants for Magnus formula
        a = 17.27
        b = 237.7
        
        # Calculate dew point
        alpha = ((a * temperature) / (b + temperature)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return float(dew_point)
    
    def _apply_smoothing(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply smoothing to environmental data.
        
        Args:
            df: DataFrame containing environmental readings
            columns: List of columns to smooth
            
        Returns:
            DataFrame with smoothed values
        """
        # Create a copy of the DataFrame
        df_smoothed = df.copy()
        
        # Get smoothing window size
        window = self.config.get('smoothing_window', 5)
        
        # Apply smoothing to each column
        for col in columns:
            if col in df.columns:
                # Apply moving average smoothing
                df_smoothed[col] = df[col].rolling(window=window, center=True).mean()
                
                # Fill NaN values at the edges
                df_smoothed[col] = df_smoothed[col].fillna(df[col])
        
        return df_smoothed
    
    def _calculate_trend(self, values: np.ndarray, window_size: int = None) -> Dict[str, Any]:
        """
        Calculate trend for a time series.
        
        Args:
            values: Time series values
            window_size: Optional window size for trend calculation
            
        Returns:
            Dictionary containing trend features
        """
        # Check if we have enough data
        if len(values) < 3:
            return {
                'slope': 0.0,
                'intercept': 0.0,
                'r_value': 0.0,
                'p_value': 1.0,
                'std_err': 0.0,
                'direction': 'stable'
            }
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        
        if window_size is not None and window_size < len(values):
            # Use only the last window_size points
            x = x[-window_size:]
            values = values[-window_size:]
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        trend_threshold = self.config.get('trend_threshold', 0.01)
        
        if slope > trend_threshold:
            direction = 'rising'
        elif slope < -trend_threshold:
            direction = 'falling'
        else:
            direction = 'stable'
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'direction': direction
        }
    
    def _calculate_rolling_trends(self, values: np.ndarray, window_sizes: List[int]) -> Dict[str, List[float]]:
        """
        Calculate rolling trends for different window sizes.
        
        Args:
            values: Time series values
            window_sizes: List of window sizes for rolling trend calculation
            
        Returns:
            Dictionary mapping window size to list of trend slopes
        """
        rolling_trends = {}
        
        for window in window_sizes:
            if window < len(values):
                # Calculate rolling trends
                slopes = []
                for i in range(len(values) - window + 1):
                    window_values = values[i:i+window]
                    x = np.arange(window)
                    
                    # Calculate linear regression
                    slope, _, _, _, _ = stats.linregress(x, window_values)
                    slopes.append(slope)
                
                rolling_trends[str(window)] = slopes
            else:
                rolling_trends[str(window)] = []
        
        return rolling_trends
    
    def _detect_trend_change_points(self, values: np.ndarray, window_size: int = 10) -> List[int]:
        """
        Detect change points in trend direction.
        
        Args:
            values: Time series values
            window_size: Window size for trend calculation
            
        Returns:
            List of indices where trend direction changes
        """
        # Check if we have enough data
        if len(values) < window_size * 2:
            return []
        
        # Calculate rolling trends
        slopes = []
        for i in range(len(values) - window_size + 1):
            window_values = values[i:i+window_size]
            x = np.arange(window_size)
            
            # Calculate linear regression
            slope, _, _, _, _ = stats.linregress(x, window_values)
            slopes.append(slope)
        
        # Detect changes in trend direction
        change_points = []
        prev_direction = 0
        
        for i in range(1, len(slopes)):
            # Determine direction
            direction = 1 if slopes[i] > 0 else (-1 if slopes[i] < 0 else 0)
            
            # Check if direction changed
            if prev_direction != 0 and direction != 0 and direction != prev_direction:
                change_points.append(i + window_size // 2)  # Adjust index to original time series
            
            prev_direction = direction
        
        return change_points
    
    def _calculate_trend_acceleration(self, values: np.ndarray, window_size: int = 10) -> float:
        """
        Calculate trend acceleration (change in trend slope).
        
        Args:
            values: Time series values
            window_size: Window size for trend calculation
            
        Returns:
            Trend acceleration value
        """
        # Check if we have enough data
        if len(values) < window_size * 2:
            return 0.0
        
        # Calculate rolling trends
        slopes = []
        for i in range(len(values) - window_size + 1):
            window_values = values[i:i+window_size]
            x = np.arange(window_size)
            
            # Calculate linear regression
            slope, _, _, _, _ = stats.linregress(x, window_values)
            slopes.append(slope)
        
        # Calculate acceleration as the trend of the slopes
        if len(slopes) > 2:
            x = np.arange(len(slopes))
            accel_slope, _, _, _, _ = stats.linregress(x, slopes)
            return float(accel_slope)
        else:
            return 0.0
    
    def _calculate_trend_stability(self, values: np.ndarray, window_size: int = 10) -> float:
        """
        Calculate trend stability (consistency of trend direction).
        
        Args:
            values: Time series values
            window_size: Window size for trend calculation
            
        Returns:
            Trend stability value between 0 and 1
        """
        # Check if we have enough data
        if len(values) < window_size * 2:
            return 1.0
        
        # Calculate rolling trends
        slopes = []
        for i in range(len(values) - window_size + 1):
            window_values = values[i:i+window_size]
            x = np.arange(window_size)
            
            # Calculate linear regression
            slope, _, _, _, _ = stats.linregress(x, window_values)
            slopes.append(slope)
        
        # Calculate stability as the inverse of the normalized standard deviation of slopes
        mean_slope = np.mean(slopes)
        if mean_slope != 0:
            stability = 1.0 - min(1.0, np.std(slopes) / abs(mean_slope))
        else:
            # If mean slope is zero, use absolute mean of slopes as denominator
            abs_mean = np.mean(np.abs(slopes))
            if abs_mean > 0:
                stability = 1.0 - min(1.0, np.std(slopes) / abs_mean)
            else:
                stability = 1.0  # If all slopes are zero, trend is perfectly stable
        
        return float(stability)
    
    def _calculate_seasonality_features(self, values: np.ndarray, period: int = None) -> Dict[str, Any]:
        """
        Calculate seasonality features for a time series.
        
        Args:
            values: Time series values
            period: Optional seasonality period
            
        Returns:
            Dictionary containing seasonality features
        """
        # Check if we have enough data
        if period is None:
            period = self.config.get('seasonality_period', 24)
        
        if len(values) < period * 2:
            return {
                'strength': 0.0,
                'detected_period': 0
            }
        
        try:
            # Apply STL decomposition
            stl = STL(pd.Series(values), period=period, robust=True)
            result = stl.fit()
            
            # Extract components
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            # Calculate seasonality strength
            var_trend_resid = np.var(trend + residual)
            var_seas_resid = np.var(seasonal + residual)
            var_resid = np.var(residual)
            
            if var_trend_resid > 0:
                seasonality_strength = max(0, min(1, 1 - var_resid / var_seas_resid))
            else:
                seasonality_strength = 0.0
            
            # Detect period using autocorrelation
            autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr, height=0.1)
            
            # Find first significant peak (excluding lag 0)
            detected_period = 0
            if len(peaks) > 0 and peaks[0] > 0:
                detected_period = int(peaks[0])
            
            return {
                'strength': float(seasonality_strength),
                'detected_period': detected_period
            }
        
        except Exception as e:
            self.logger.warning(f"Error calculating seasonality features: {str(e)}")
            return {
                'strength': 0.0,
                'detected_period': 0
            }
    
    def _extract_trend_features(self, df: pd.DataFrame, parameters: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract trend features for each parameter.
        
        Args:
            df: DataFrame containing environmental readings
            parameters: List of parameters to analyze
            
        Returns:
            Dictionary mapping parameter to trend features
        """
        # Get window sizes for trend analysis
        window_sizes = self.config['time_windows']
        
        # Initialize results
        features = {}
        
        for param in parameters:
            if param not in df.columns:
                continue
            
            # Get parameter values
            values = df[param].values
            
            # Calculate overall trend
            overall_trend = self._calculate_trend(values)
            
            # Calculate short-term trend (using the smallest window size)
            short_term_window = min(window_sizes) if window_sizes else len(values) // 4
            short_term_trend = self._calculate_trend(values, short_term_window)
            
            # Calculate long-term trend (using the largest window size)
            long_term_window = max(window_sizes) if window_sizes else len(values)
            long_term_trend = self._calculate_trend(values, long_term_window)
            
            # Calculate rolling trends
            rolling_trends = self._calculate_rolling_trends(values, window_sizes)
            
            # Detect trend change points
            change_points = self._detect_trend_change_points(values)
            
            # Calculate trend acceleration
            trend_acceleration = self._calculate_trend_acceleration(values)
            
            # Calculate trend stability
            trend_stability = self._calculate_trend_stability(values)
            
            # Calculate seasonality features
            seasonality_period = self.config.get('seasonality_period', 24)
            seasonality = self._calculate_seasonality_features(values, seasonality_period)
            
            # Calculate trend duration (how long the current trend has been maintained)
            if change_points:
                trend_duration = len(values) - change_points[-1]
            else:
                trend_duration = len(values)
            
            # Calculate residual strength
            try:
                # Apply STL decomposition
                if len(values) >= seasonality_period * 2:
                    stl = STL(pd.Series(values), period=seasonality_period, robust=True)
                    result = stl.fit()
                    
                    # Extract components
                    trend_component = result.trend
                    seasonal = result.seasonal
                    residual = result.resid
                    
                    # Calculate residual strength
                    var_data = np.var(values)
                    var_resid = np.var(residual)
                    
                    if var_data > 0:
                        residual_strength = min(1.0, var_resid / var_data)
                    else:
                        residual_strength = 0.0
                else:
                    residual_strength = 0.0
            except Exception as e:
                self.logger.warning(f"Error calculating residual strength: {str(e)}")
                residual_strength = 0.0
            
            # Store features
            features[param] = {
                'trend_slope': overall_trend['slope'],
                'trend_direction': overall_trend['direction'],
                'trend_r_value': overall_trend['r_value'],
                'trend_p_value': overall_trend['p_value'],
                'trend_change_points': change_points,
                'trend_change_count': len(change_points),
                'short_term_trend': short_term_trend['slope'],
                'short_term_direction': short_term_trend['direction'],
                'long_term_trend': long_term_trend['slope'],
                'long_term_direction': long_term_trend['direction'],
                'trend_acceleration': trend_acceleration,
                'trend_stability': trend_stability,
                'trend_duration': trend_duration,
                'seasonality_strength': seasonality['strength'],
                'seasonality_period': seasonality['detected_period'],
                'residual_strength': float(residual_strength),
                'rolling_trends': rolling_trends
            }
        
        return features