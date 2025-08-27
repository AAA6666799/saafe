"""
Humidity Correlation Analyzer for environmental data analysis.

This module provides a feature extractor that analyzes correlations with humidity.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats

from ...base import EnvironmentalFeatureExtractor


class HumidityCorrelationAnalyzer(EnvironmentalFeatureExtractor):
    """
    Feature extractor for analyzing correlations with humidity.
    
    This class analyzes environmental data to extract features related to
    humidity correlations with temperature, pressure, and other environmental parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the humidity correlation analyzer.
        
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
        if 'humidity_column' not in self.config:
            self.config['humidity_column'] = 'humidity'
        
        if 'temperature_column' not in self.config:
            self.config['temperature_column'] = 'temperature'
        
        if 'pressure_column' not in self.config:
            self.config['pressure_column'] = 'pressure'
        
        if 'additional_parameters' not in self.config:
            self.config['additional_parameters'] = []
        
        if 'window_sizes' not in self.config:
            self.config['window_sizes'] = [10, 30, 60]  # Window sizes for rolling correlation
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 5
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract humidity correlation features from environmental data.
        
        Args:
            data: Input environmental data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting humidity correlation features")
        
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
        
        # Check if DataFrame has required columns
        humidity_column = self.config['humidity_column']
        if humidity_column not in df.columns:
            self.logger.warning(f"Missing humidity column '{humidity_column}' in data")
            return {}
        
        # Apply smoothing if configured
        if self.config.get('apply_smoothing', True):
            columns_to_smooth = [humidity_column]
            
            # Add temperature column if available
            temp_column = self.config['temperature_column']
            if temp_column in df.columns:
                columns_to_smooth.append(temp_column)
            
            # Add pressure column if available
            pressure_column = self.config['pressure_column']
            if pressure_column in df.columns:
                columns_to_smooth.append(pressure_column)
            
            # Add additional parameters if available
            additional_params = self.config['additional_parameters']
            for param in additional_params:
                if param in df.columns:
                    columns_to_smooth.append(param)
            
            df_processed = self._apply_smoothing(df, columns_to_smooth)
        else:
            df_processed = df
        
        # Extract correlation features
        correlation_features = self._extract_correlation_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'humidity_correlation_features': correlation_features
        }
        
        self.logger.info(f"Extracted humidity correlation features from {len(df)} samples")
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
        
        # Add correlation features
        correlation_features = features.get('humidity_correlation_features', {})
        for feature_name, feature_value in correlation_features.items():
            if isinstance(feature_value, dict):
                for sub_name, sub_value in feature_value.items():
                    if isinstance(sub_value, dict):
                        for sub_sub_name, sub_sub_value in sub_value.items():
                            flat_features[f"humidity_corr_{feature_name}_{sub_name}_{sub_sub_name}"] = sub_sub_value
                    else:
                        flat_features[f"humidity_corr_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"humidity_corr_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved humidity correlation features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        # Base feature names
        base_features = [
            'temperature_correlation',
            'temperature_correlation_pvalue',
            'pressure_correlation',
            'pressure_correlation_pvalue',
            'dew_point_correlation',
            'heat_index_correlation',
            'temperature_humidity_ratio',
            'rolling_correlation_temperature',
            'rolling_correlation_pressure',
            'correlation_stability',
            'lag_correlation_temperature',
            'lag_correlation_pressure',
            'partial_correlation_temperature',
            'partial_correlation_pressure'
        ]
        
        # Add features for additional parameters
        additional_params = self.config.get('additional_parameters', [])
        for param in additional_params:
            base_features.append(f"{param}_correlation")
            base_features.append(f"{param}_correlation_pvalue")
            base_features.append(f"rolling_correlation_{param}")
            base_features.append(f"lag_correlation_{param}")
        
        return base_features
    
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
    
    def _calculate_heat_index(self, temperature: float, humidity: float) -> float:
        """
        Calculate heat index from temperature and humidity.
        
        Args:
            temperature: Temperature in degrees Celsius
            humidity: Relative humidity as a percentage
            
        Returns:
            Heat index in degrees Celsius
        """
        # Convert temperature to Fahrenheit for the standard heat index formula
        temp_f = temperature * 9/5 + 32
        
        # Calculate heat index in Fahrenheit
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
        
        # For higher temperatures, use the full regression formula
        if temp_f >= 80:
            hi = -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
            hi = hi - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2
            hi = hi - 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity
            hi = hi + 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2
        
        # Convert back to Celsius
        hi_c = (hi - 32) * 5/9
        
        return float(hi_c)
    
    def _calculate_correlation(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Pearson correlation coefficient and p-value.
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Check if we have enough data
        if len(x_clean) < 2:
            return 0.0, 1.0
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(x_clean, y_clean)
        
        return float(correlation), float(p_value)
    
    def _calculate_rolling_correlation(self, x: np.ndarray, y: np.ndarray, window_size: int) -> np.ndarray:
        """
        Calculate rolling correlation between two arrays.
        
        Args:
            x: First array
            y: Second array
            window_size: Window size for rolling correlation
            
        Returns:
            Array of rolling correlation coefficients
        """
        # Check if we have enough data
        if len(x) < window_size or len(y) < window_size:
            return np.array([])
        
        # Calculate rolling correlation
        rolling_corr = []
        for i in range(len(x) - window_size + 1):
            x_window = x[i:i+window_size]
            y_window = y[i:i+window_size]
            
            # Remove NaN values
            mask = ~np.isnan(x_window) & ~np.isnan(y_window)
            x_clean = x_window[mask]
            y_clean = y_window[mask]
            
            # Calculate correlation if we have enough data
            if len(x_clean) > 1:
                corr, _ = stats.pearsonr(x_clean, y_clean)
                rolling_corr.append(corr)
            else:
                rolling_corr.append(0.0)
        
        return np.array(rolling_corr)
    
    def _calculate_lag_correlation(self, x: np.ndarray, y: np.ndarray, max_lag: int) -> Dict[str, Any]:
        """
        Calculate correlation between two arrays with different lags.
        
        Args:
            x: First array
            y: Second array
            max_lag: Maximum lag to consider
            
        Returns:
            Dictionary containing lag correlation results
        """
        # Check if we have enough data
        if len(x) < max_lag + 1 or len(y) < max_lag + 1:
            return {
                'max_correlation': 0.0,
                'optimal_lag': 0,
                'lag_correlations': []
            }
        
        # Calculate correlation for different lags
        lag_correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # y is lagged behind x
                x_subset = x[:lag]
                y_subset = y[-lag:]
            elif lag > 0:
                # x is lagged behind y
                x_subset = x[lag:]
                y_subset = y[:-lag]
            else:
                # No lag
                x_subset = x
                y_subset = y
            
            # Calculate correlation
            corr, _ = self._calculate_correlation(x_subset, y_subset)
            lag_correlations.append((lag, corr))
        
        # Find lag with maximum correlation
        max_corr_lag, max_corr = max(lag_correlations, key=lambda x: abs(x[1]))
        
        return {
            'max_correlation': float(max_corr),
            'optimal_lag': int(max_corr_lag),
            'lag_correlations': [(int(lag), float(corr)) for lag, corr in lag_correlations]
        }
    
    def _calculate_partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """
        Calculate partial correlation between x and y, controlling for z.
        
        Args:
            x: First array
            y: Second array
            z: Control array
            
        Returns:
            Partial correlation coefficient
        """
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        x_clean = x[mask]
        y_clean = y[mask]
        z_clean = z[mask]
        
        # Check if we have enough data
        if len(x_clean) < 3:
            return 0.0
        
        try:
            # Calculate correlations
            r_xy, _ = stats.pearsonr(x_clean, y_clean)
            r_xz, _ = stats.pearsonr(x_clean, z_clean)
            r_yz, _ = stats.pearsonr(y_clean, z_clean)
            
            # Calculate partial correlation
            numerator = r_xy - r_xz * r_yz
            denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            
            if denominator != 0:
                partial_corr = numerator / denominator
            else:
                partial_corr = 0.0
            
            return float(partial_corr)
        
        except Exception as e:
            self.logger.warning(f"Error calculating partial correlation: {str(e)}")
            return 0.0
    
    def _extract_correlation_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract humidity correlation features.
        
        Args:
            df: DataFrame containing environmental readings
            
        Returns:
            Dictionary containing correlation features
        """
        # Get column names
        humidity_column = self.config['humidity_column']
        temp_column = self.config['temperature_column']
        pressure_column = self.config['pressure_column']
        additional_params = self.config['additional_parameters']
        
        # Get window sizes for rolling correlation
        window_sizes = self.config['window_sizes']
        
        # Initialize results
        features = {}
        
        # Get humidity values
        humidity = df[humidity_column].values if humidity_column in df.columns else None
        
        if humidity is None:
            self.logger.warning(f"Humidity column '{humidity_column}' not found in data")
            return {}
        
        # Calculate temperature correlation
        if temp_column in df.columns:
            temperature = df[temp_column].values
            
            # Calculate basic correlation
            temp_corr, temp_pvalue = self._calculate_correlation(humidity, temperature)
            
            # Calculate rolling correlation for different window sizes
            rolling_temp_corr = {}
            for window in window_sizes:
                if window < len(df):
                    rolling_corr = self._calculate_rolling_correlation(humidity, temperature, window)
                    rolling_temp_corr[str(window)] = {
                        'mean': float(np.mean(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                        'std': float(np.std(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                        'min': float(np.min(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                        'max': float(np.max(rolling_corr)) if len(rolling_corr) > 0 else 0.0
                    }
            
            # Calculate lag correlation
            max_lag = min(24, len(df) // 4)  # Use at most 1/4 of the data length
            lag_temp_corr = self._calculate_lag_correlation(humidity, temperature, max_lag)
            
            # Calculate temperature-humidity ratio
            temp_humidity_ratio = np.mean(temperature) / np.mean(humidity) if np.mean(humidity) > 0 else 0.0
            
            # Calculate dew point
            dew_points = []
            for t, h in zip(temperature, humidity):
                if not np.isnan(t) and not np.isnan(h):
                    dew_points.append(self.calculate_dew_point(t, h))
            
            dew_point_corr, dew_point_pvalue = self._calculate_correlation(
                humidity, np.array(dew_points) if dew_points else np.array([]))
            
            # Calculate heat index
            heat_indices = []
            for t, h in zip(temperature, humidity):
                if not np.isnan(t) and not np.isnan(h):
                    heat_indices.append(self._calculate_heat_index(t, h))
            
            heat_index_corr, heat_index_pvalue = self._calculate_correlation(
                humidity, np.array(heat_indices) if heat_indices else np.array([]))
            
            # Store temperature correlation features
            features['temperature'] = {
                'correlation': temp_corr,
                'p_value': temp_pvalue,
                'rolling_correlation': rolling_temp_corr,
                'lag_correlation': lag_temp_corr,
                'temperature_humidity_ratio': float(temp_humidity_ratio),
                'dew_point_correlation': dew_point_corr,
                'heat_index_correlation': heat_index_corr
            }
        
        # Calculate pressure correlation
        if pressure_column in df.columns:
            pressure = df[pressure_column].values
            
            # Calculate basic correlation
            pressure_corr, pressure_pvalue = self._calculate_correlation(humidity, pressure)
            
            # Calculate rolling correlation for different window sizes
            rolling_pressure_corr = {}
            for window in window_sizes:
                if window < len(df):
                    rolling_corr = self._calculate_rolling_correlation(humidity, pressure, window)
                    rolling_pressure_corr[str(window)] = {
                        'mean': float(np.mean(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                        'std': float(np.std(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                        'min': float(np.min(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                        'max': float(np.max(rolling_corr)) if len(rolling_corr) > 0 else 0.0
                    }
            
            # Calculate lag correlation
            max_lag = min(24, len(df) // 4)  # Use at most 1/4 of the data length
            lag_pressure_corr = self._calculate_lag_correlation(humidity, pressure, max_lag)
            
            # Store pressure correlation features
            features['pressure'] = {
                'correlation': pressure_corr,
                'p_value': pressure_pvalue,
                'rolling_correlation': rolling_pressure_corr,
                'lag_correlation': lag_pressure_corr
            }
        
        # Calculate partial correlation if both temperature and pressure are available
        if temp_column in df.columns and pressure_column in df.columns:
            temperature = df[temp_column].values
            pressure = df[pressure_column].values
            
            # Calculate partial correlations
            partial_temp_corr = self._calculate_partial_correlation(humidity, temperature, pressure)
            partial_pressure_corr = self._calculate_partial_correlation(humidity, pressure, temperature)
            
            # Store partial correlation features
            features['partial_correlation'] = {
                'temperature': partial_temp_corr,
                'pressure': partial_pressure_corr
            }
        
        # Calculate correlation with additional parameters
        for param in additional_params:
            if param in df.columns:
                param_values = df[param].values
                
                # Calculate basic correlation
                param_corr, param_pvalue = self._calculate_correlation(humidity, param_values)
                
                # Calculate rolling correlation for different window sizes
                rolling_param_corr = {}
                for window in window_sizes:
                    if window < len(df):
                        rolling_corr = self._calculate_rolling_correlation(humidity, param_values, window)
                        rolling_param_corr[str(window)] = {
                            'mean': float(np.mean(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                            'std': float(np.std(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                            'min': float(np.min(rolling_corr)) if len(rolling_corr) > 0 else 0.0,
                            'max': float(np.max(rolling_corr)) if len(rolling_corr) > 0 else 0.0
                        }
                
                # Calculate lag correlation
                max_lag = min(24, len(df) // 4)  # Use at most 1/4 of the data length
                lag_param_corr = self._calculate_lag_correlation(humidity, param_values, max_lag)
                
                # Store parameter correlation features
                features[param] = {
                    'correlation': param_corr,
                    'p_value': param_pvalue,
                    'rolling_correlation': rolling_param_corr,
                    'lag_correlation': lag_param_corr
                }
        
        # Calculate correlation stability (standard deviation of rolling correlation)
        correlation_stability = {}
        for feature_name, feature_data in features.items():
            if 'rolling_correlation' in feature_data:
                rolling_corr_data = feature_data['rolling_correlation']
                if rolling_corr_data:
                    # Use the largest window size
                    largest_window = max(rolling_corr_data.keys(), key=lambda x: int(x))
                    stability = rolling_corr_data[largest_window].get('std', 1.0)
                    correlation_stability[feature_name] = 1.0 - min(stability, 1.0)  # Higher is more stable
        
        features['correlation_stability'] = correlation_stability
        
        return features