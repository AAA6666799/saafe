"""
Pressure Change Extractor for environmental data analysis.

This module provides a feature extractor that extracts pressure change features.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal

from ...base import EnvironmentalFeatureExtractor


class PressureChangeExtractor(EnvironmentalFeatureExtractor):
    """
    Feature extractor for pressure changes in environmental data.
    
    This class analyzes environmental data to extract features related to
    pressure changes, including rates of change, trends, and patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pressure change extractor.
        
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
        if 'pressure_column' not in self.config:
            self.config['pressure_column'] = 'pressure'
        
        if 'time_windows' not in self.config:
            self.config['time_windows'] = [5, 15, 30, 60]  # Window sizes for rate calculation
        
        if 'significant_change_threshold' not in self.config:
            self.config['significant_change_threshold'] = 1.0  # hPa
        
        if 'rapid_change_threshold' not in self.config:
            self.config['rapid_change_threshold'] = 2.0  # hPa per hour
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 5
        
        if 'detect_pressure_systems' not in self.config:
            self.config['detect_pressure_systems'] = True
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract pressure change features from environmental data.
        
        Args:
            data: Input environmental data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting pressure change features")
        
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
        pressure_column = self.config['pressure_column']
        if pressure_column not in df.columns:
            self.logger.warning(f"Missing pressure column '{pressure_column}' in data")
            return {}
        
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
        
        # Apply smoothing if configured
        if self.config.get('apply_smoothing', True):
            df_processed = self._apply_smoothing(df, [pressure_column])
        else:
            df_processed = df
        
        # Extract pressure change features
        pressure_features = self._extract_pressure_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'pressure_change_features': pressure_features
        }
        
        self.logger.info(f"Extracted pressure change features from {len(df)} samples")
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
        
        # Add pressure change features
        pressure_features = features.get('pressure_change_features', {})
        for feature_name, feature_value in pressure_features.items():
            if isinstance(feature_value, list):
                # Skip time series data in the flattened representation
                if len(feature_value) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_value):
                        flat_features[f"pressure_change_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"pressure_change_{feature_name}_mean"] = np.mean(feature_value)
                    flat_features[f"pressure_change_{feature_name}_max"] = np.max(feature_value)
                    flat_features[f"pressure_change_{feature_name}_min"] = np.min(feature_value)
            elif isinstance(feature_value, dict):
                for sub_name, sub_value in feature_value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 20:
                        # For large lists, just include summary statistics
                        flat_features[f"pressure_change_{feature_name}_{sub_name}_count"] = len(sub_value)
                    else:
                        flat_features[f"pressure_change_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"pressure_change_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved pressure change features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'mean_pressure',
            'std_pressure',
            'min_pressure',
            'max_pressure',
            'pressure_range',
            'mean_change_rate',
            'max_change_rate',
            'min_change_rate',
            'std_change_rate',
            'significant_changes_count',
            'rapid_changes_count',
            'change_direction_switches',
            'pressure_trend',
            'pressure_system_type',
            'pressure_system_strength',
            'pressure_oscillation_frequency',
            'pressure_oscillation_amplitude',
            'pressure_stability'
        ]
    
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
    
    def _calculate_change_rates(self, 
                              pressure_values: np.ndarray, 
                              timestamps: pd.Series) -> Dict[str, np.ndarray]:
        """
        Calculate pressure change rates.
        
        Args:
            pressure_values: Array of pressure values
            timestamps: Series of timestamps
            
        Returns:
            Dictionary containing change rates
        """
        # Check if we have enough data
        if len(pressure_values) < 2:
            return {
                'point_change_rates': np.array([]),
                'hourly_change_rates': np.array([])
            }
        
        # Calculate point-to-point change rates
        pressure_diffs = np.diff(pressure_values)
        
        # Calculate time differences in seconds
        if pd.api.types.is_datetime64_any_dtype(timestamps):
            time_diffs = np.array([(timestamps.iloc[i+1] - timestamps.iloc[i]).total_seconds() 
                                 for i in range(len(timestamps)-1)])
        else:
            # Assume uniform time steps
            time_diffs = np.ones(len(pressure_values) - 1)
        
        # Avoid division by zero
        time_diffs[time_diffs == 0] = 1.0
        
        # Calculate change rates (pressure units per second)
        point_change_rates = pressure_diffs / time_diffs
        
        # Calculate hourly change rates (pressure units per hour)
        hourly_change_rates = point_change_rates * 3600
        
        return {
            'point_change_rates': point_change_rates,
            'hourly_change_rates': hourly_change_rates
        }
    
    def _detect_significant_changes(self, 
                                  pressure_values: np.ndarray, 
                                  change_rates: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect significant pressure changes.
        
        Args:
            pressure_values: Array of pressure values
            change_rates: Array of pressure change rates
            
        Returns:
            List of dictionaries containing significant change information
        """
        # Get thresholds
        significant_threshold = self.config.get('significant_change_threshold', 1.0)
        rapid_threshold = self.config.get('rapid_change_threshold', 2.0)
        
        # Initialize results
        significant_changes = []
        
        # Check if we have enough data
        if len(pressure_values) < 2 or len(change_rates) < 1:
            return significant_changes
        
        # Find significant changes
        in_significant_change = False
        change_start_idx = 0
        change_direction = 0
        accumulated_change = 0.0
        
        for i in range(1, len(pressure_values)):
            # Calculate point change
            point_change = pressure_values[i] - pressure_values[i-1]
            
            # Determine direction
            current_direction = 1 if point_change > 0 else (-1 if point_change < 0 else 0)
            
            # Check if we're in a significant change
            if in_significant_change:
                # Check if direction changed
                if current_direction != 0 and current_direction != change_direction:
                    # End of significant change
                    change_end_idx = i - 1
                    
                    # Calculate change metrics
                    total_change = pressure_values[change_end_idx] - pressure_values[change_start_idx]
                    duration = change_end_idx - change_start_idx + 1
                    avg_rate = total_change / duration if duration > 0 else 0.0
                    
                    # Add to results if significant
                    if abs(total_change) >= significant_threshold:
                        significant_changes.append({
                            'start_idx': change_start_idx,
                            'end_idx': change_end_idx,
                            'duration': duration,
                            'total_change': float(total_change),
                            'avg_rate': float(avg_rate),
                            'direction': 'rising' if total_change > 0 else 'falling',
                            'is_rapid': abs(avg_rate) >= rapid_threshold
                        })
                    
                    # Reset
                    in_significant_change = False
                    accumulated_change = 0.0
                else:
                    # Continue accumulating change
                    accumulated_change += point_change
            
            # Check if we should start a new significant change
            if not in_significant_change and current_direction != 0:
                in_significant_change = True
                change_start_idx = i - 1
                change_direction = current_direction
                accumulated_change = point_change
        
        # Check if we ended in a significant change
        if in_significant_change:
            change_end_idx = len(pressure_values) - 1
            
            # Calculate change metrics
            total_change = pressure_values[change_end_idx] - pressure_values[change_start_idx]
            duration = change_end_idx - change_start_idx + 1
            avg_rate = total_change / duration if duration > 0 else 0.0
            
            # Add to results if significant
            if abs(total_change) >= significant_threshold:
                significant_changes.append({
                    'start_idx': change_start_idx,
                    'end_idx': change_end_idx,
                    'duration': duration,
                    'total_change': float(total_change),
                    'avg_rate': float(avg_rate),
                    'direction': 'rising' if total_change > 0 else 'falling',
                    'is_rapid': abs(avg_rate) >= rapid_threshold
                })
        
        return significant_changes
    
    def _detect_pressure_system(self, 
                              pressure_values: np.ndarray, 
                              significant_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect pressure system type and characteristics.
        
        Args:
            pressure_values: Array of pressure values
            significant_changes: List of significant pressure changes
            
        Returns:
            Dictionary containing pressure system information
        """
        # Check if we have enough data
        if len(pressure_values) < 10:
            return {
                'system_type': 'unknown',
                'system_strength': 0.0,
                'is_stable': True
            }
        
        # Calculate overall trend
        trend_slope, _, _, _, _ = stats.linregress(np.arange(len(pressure_values)), pressure_values)
        
        # Calculate pressure range
        pressure_range = np.max(pressure_values) - np.min(pressure_values)
        
        # Calculate pressure stability (inverse of normalized standard deviation)
        pressure_std = np.std(pressure_values)
        pressure_mean = np.mean(pressure_values)
        if pressure_mean > 0:
            stability = 1.0 - min(1.0, pressure_std / pressure_mean)
        else:
            stability = 0.0
        
        # Count rising and falling changes
        rising_changes = sum(1 for change in significant_changes if change['direction'] == 'rising')
        falling_changes = sum(1 for change in significant_changes if change['direction'] == 'falling')
        
        # Determine system type
        if trend_slope > 0.05:
            system_type = 'rising_pressure'
        elif trend_slope < -0.05:
            system_type = 'falling_pressure'
        elif pressure_range < 1.0:
            system_type = 'stable_high_pressure'
        elif rising_changes > falling_changes * 2:
            system_type = 'building_high_pressure'
        elif falling_changes > rising_changes * 2:
            system_type = 'developing_low_pressure'
        elif stability > 0.9:
            system_type = 'stable_pressure'
        else:
            system_type = 'variable_pressure'
        
        # Determine system strength
        system_strength = min(1.0, pressure_range / 10.0)  # Normalize to [0, 1]
        
        return {
            'system_type': system_type,
            'system_strength': float(system_strength),
            'is_stable': stability > 0.8
        }
    
    def _detect_oscillations(self, pressure_values: np.ndarray) -> Dict[str, float]:
        """
        Detect oscillations in pressure values.
        
        Args:
            pressure_values: Array of pressure values
            
        Returns:
            Dictionary containing oscillation features
        """
        # Check if we have enough data
        if len(pressure_values) < 10:
            return {
                'frequency': 0.0,
                'amplitude': 0.0,
                'regularity': 0.0
            }
        
        try:
            # Detrend the data
            trend_slope, trend_intercept, _, _, _ = stats.linregress(
                np.arange(len(pressure_values)), pressure_values)
            trend = trend_slope * np.arange(len(pressure_values)) + trend_intercept
            detrended = pressure_values - trend
            
            # Find peaks and troughs
            peaks, _ = signal.find_peaks(detrended)
            troughs, _ = signal.find_peaks(-detrended)
            
            # Check if we found oscillations
            if len(peaks) < 2 or len(troughs) < 2:
                return {
                    'frequency': 0.0,
                    'amplitude': 0.0,
                    'regularity': 0.0
                }
            
            # Calculate frequency (cycles per sample)
            peak_intervals = np.diff(peaks)
            frequency = 1.0 / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 0.0
            
            # Calculate amplitude
            peak_values = detrended[peaks]
            trough_values = detrended[troughs]
            amplitude = (np.mean(peak_values) - np.mean(trough_values)) / 2
            
            # Calculate regularity (inverse of normalized standard deviation of intervals)
            if np.mean(peak_intervals) > 0:
                regularity = 1.0 - min(1.0, np.std(peak_intervals) / np.mean(peak_intervals))
            else:
                regularity = 0.0
            
            return {
                'frequency': float(frequency),
                'amplitude': float(amplitude),
                'regularity': float(regularity)
            }
        
        except Exception as e:
            self.logger.warning(f"Error detecting oscillations: {str(e)}")
            return {
                'frequency': 0.0,
                'amplitude': 0.0,
                'regularity': 0.0
            }
    
    def _extract_pressure_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract pressure change features.
        
        Args:
            df: DataFrame containing environmental readings
            
        Returns:
            Dictionary containing pressure change features
        """
        # Get pressure column
        pressure_column = self.config['pressure_column']
        
        # Get pressure values and timestamps
        pressure_values = df[pressure_column].values
        timestamps = df['timestamp']
        
        # Calculate basic statistics
        pressure_mean = float(np.mean(pressure_values))
        pressure_std = float(np.std(pressure_values))
        pressure_min = float(np.min(pressure_values))
        pressure_max = float(np.max(pressure_values))
        pressure_range = float(pressure_max - pressure_min)
        
        # Calculate change rates
        change_rates = self._calculate_change_rates(pressure_values, timestamps)
        point_rates = change_rates['point_change_rates']
        hourly_rates = change_rates['hourly_change_rates']
        
        # Calculate change rate statistics
        if len(hourly_rates) > 0:
            mean_change_rate = float(np.mean(hourly_rates))
            max_change_rate = float(np.max(hourly_rates))
            min_change_rate = float(np.min(hourly_rates))
            std_change_rate = float(np.std(hourly_rates))
        else:
            mean_change_rate = 0.0
            max_change_rate = 0.0
            min_change_rate = 0.0
            std_change_rate = 0.0
        
        # Detect significant changes
        significant_changes = self._detect_significant_changes(pressure_values, hourly_rates)
        
        # Count significant and rapid changes
        significant_count = len(significant_changes)
        rapid_count = sum(1 for change in significant_changes if change['is_rapid'])
        
        # Calculate direction switches
        if len(point_rates) > 1:
            direction_switches = sum(1 for i in range(1, len(point_rates)) 
                                   if (point_rates[i] > 0 and point_rates[i-1] < 0) or 
                                      (point_rates[i] < 0 and point_rates[i-1] > 0))
        else:
            direction_switches = 0
        
        # Determine pressure trend
        if len(pressure_values) > 1:
            trend_slope, _, _, _, _ = stats.linregress(np.arange(len(pressure_values)), pressure_values)
            
            if trend_slope > 0.1:
                trend = 'strongly_rising'
            elif trend_slope > 0.01:
                trend = 'rising'
            elif trend_slope < -0.1:
                trend = 'strongly_falling'
            elif trend_slope < -0.01:
                trend = 'falling'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
            trend_slope = 0.0
        
        # Detect pressure system if configured
        pressure_system = {}
        if self.config.get('detect_pressure_systems', True):
            pressure_system = self._detect_pressure_system(pressure_values, significant_changes)
        
        # Detect oscillations
        oscillations = self._detect_oscillations(pressure_values)
        
        # Calculate pressure stability
        if pressure_mean > 0:
            pressure_stability = 1.0 - min(1.0, pressure_std / pressure_mean)
        else:
            pressure_stability = 0.0
        
        # Return features
        return {
            'mean_pressure': pressure_mean,
            'std_pressure': pressure_std,
            'min_pressure': pressure_min,
            'max_pressure': pressure_max,
            'pressure_range': pressure_range,
            'mean_change_rate': mean_change_rate,
            'max_change_rate': max_change_rate,
            'min_change_rate': min_change_rate,
            'std_change_rate': std_change_rate,
            'significant_changes_count': significant_count,
            'rapid_changes_count': rapid_count,
            'change_direction_switches': direction_switches,
            'pressure_trend': trend,
            'trend_slope': float(trend_slope),
            'pressure_system_type': pressure_system.get('system_type', 'unknown'),
            'pressure_system_strength': pressure_system.get('system_strength', 0.0),
            'pressure_system_stable': pressure_system.get('is_stable', True),
            'pressure_oscillation_frequency': oscillations['frequency'],
            'pressure_oscillation_amplitude': oscillations['amplitude'],
            'pressure_oscillation_regularity': oscillations['regularity'],
            'pressure_stability': float(pressure_stability),
            'hourly_change_rates': hourly_rates.tolist() if len(hourly_rates) > 0 else []
        }