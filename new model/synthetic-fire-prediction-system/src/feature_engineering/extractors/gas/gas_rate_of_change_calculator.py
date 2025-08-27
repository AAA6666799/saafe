"""
Gas Rate of Change Calculator for gas sensor data analysis.

This module provides a feature extractor that calculates rate of change in gas concentrations.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal

from ...base import GasFeatureExtractor


class GasRateOfChangeCalculator(GasFeatureExtractor):
    """
    Feature extractor for calculating rate of change in gas concentrations.
    
    This class analyzes gas concentration data to extract features related to
    the rate of change, including derivatives, acceleration, and change patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gas rate of change calculator.
        
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
        # Check required parameters
        required_params = ['gas_types']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Set default values for optional parameters
        if 'time_windows' not in self.config:
            self.config['time_windows'] = [1, 3, 5, 10]  # Window sizes for rate calculation
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 3
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'significant_change_threshold' not in self.config:
            # Default thresholds for significant changes in ppm/s
            self.config['significant_change_threshold'] = {
                'methane': 5.0,
                'propane': 2.0,
                'hydrogen': 5.0,
                'carbon_monoxide': 1.0,
                'default': 2.0
            }
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract gas rate of change features from gas data.
        
        Args:
            data: Input gas data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting gas rate of change features")
        
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
        gas_types = self.config['gas_types']
        missing_columns = [gas for gas in gas_types if gas not in df.columns]
        if missing_columns:
            self.logger.warning(f"Missing gas columns in data: {missing_columns}")
            return {}
        
        # Check if timestamp column exists
        if 'timestamp' not in df.columns:
            # Create a dummy timestamp column
            self.logger.warning("No timestamp column found, creating dummy timestamps")
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='S')
        
        # Convert timestamps to seconds if they're datetime objects
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            # Convert to seconds since first timestamp
            first_timestamp = df['timestamp'].iloc[0]
            df['time_seconds'] = (df['timestamp'] - first_timestamp).dt.total_seconds()
        else:
            # Assume timestamps are already numeric
            df['time_seconds'] = df['timestamp']
        
        # Apply smoothing if configured
        if self.config.get('apply_smoothing', True):
            df_processed = self._apply_smoothing(df, gas_types)
        else:
            df_processed = df
        
        # Extract rate of change features
        rate_features = self._extract_rate_features(df_processed, gas_types)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'gas_types': gas_types,
            'rate_features': rate_features
        }
        
        self.logger.info(f"Extracted gas rate of change features from {len(df)} samples")
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
        
        # Add rate features
        rate_features = features.get('rate_features', {})
        for gas_type, gas_features in rate_features.items():
            for feature_name, feature_value in gas_features.items():
                if isinstance(feature_value, list):
                    # Skip time series data in the flattened representation
                    flat_features[f"{gas_type}_{feature_name}_mean"] = np.mean(feature_value)
                    flat_features[f"{gas_type}_{feature_name}_max"] = np.max(feature_value)
                    flat_features[f"{gas_type}_{feature_name}_min"] = np.min(feature_value)
                elif isinstance(feature_value, dict):
                    for sub_name, sub_value in feature_value.items():
                        flat_features[f"{gas_type}_{feature_name}_{sub_name}"] = sub_value
                else:
                    flat_features[f"{gas_type}_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved gas rate of change features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        # Base feature names (will be prefixed with gas type)
        base_features = [
            'mean_rate',
            'max_rate',
            'min_rate',
            'std_rate',
            'median_rate',
            'mean_abs_rate',
            'max_abs_rate',
            'positive_rate_percentage',
            'negative_rate_percentage',
            'zero_crossings',
            'significant_changes',
            'acceleration_mean',
            'acceleration_max',
            'rate_pattern'
        ]
        
        # Add window-specific features
        for window in self.config.get('time_windows', [1, 3, 5, 10]):
            base_features.append(f'rate_window_{window}')
        
        # Generate full feature names for each gas type
        feature_names = []
        for gas in self.config.get('gas_types', []):
            for feature in base_features:
                feature_names.append(f"{gas}_{feature}")
        
        return feature_names
    
    def calculate_concentration_slope(self,
                                    gas_readings: pd.DataFrame,
                                    gas_type: str,
                                    window_sizes: List[int]) -> Dict[str, float]:
        """
        Calculate concentration slope over different time windows.
        
        Args:
            gas_readings: DataFrame containing gas concentration readings
            gas_type: Type of gas to analyze
            window_sizes: List of window sizes (in samples) to calculate slopes for
            
        Returns:
            Dictionary mapping window size to calculated slope
        """
        if gas_type not in gas_readings.columns or 'time_seconds' not in gas_readings.columns:
            return {str(window): 0.0 for window in window_sizes}
        
        slopes = {}
        
        for window in window_sizes:
            if window >= len(gas_readings):
                slopes[str(window)] = 0.0
                continue
            
            # Calculate rolling linear regression
            y = gas_readings[gas_type].values
            x = gas_readings['time_seconds'].values
            
            # Use rolling window
            window_slopes = []
            for i in range(len(y) - window + 1):
                x_window = x[i:i+window]
                y_window = y[i:i+window]
                
                # Calculate slope using linear regression
                if len(np.unique(x_window)) > 1:  # Ensure x values are not all the same
                    slope, _, _, _, _ = stats.linregress(x_window, y_window)
                    window_slopes.append(slope)
                else:
                    window_slopes.append(0.0)
            
            # Use the mean slope as the feature
            slopes[str(window)] = float(np.mean(window_slopes)) if window_slopes else 0.0
        
        return slopes
    
    def detect_concentration_peaks(self,
                                 gas_readings: pd.DataFrame,
                                 gas_type: str,
                                 threshold: float) -> List[Dict[str, Any]]:
        """
        Detect peaks in gas concentration.
        
        Args:
            gas_readings: DataFrame containing gas concentration readings
            gas_type: Type of gas to analyze
            threshold: Threshold for peak detection
            
        Returns:
            List of dictionaries containing peak information
        """
        if gas_type not in gas_readings.columns:
            return []
        
        # Get gas concentration values
        concentrations = gas_readings[gas_type].values
        
        # Detect peaks
        peaks, properties = signal.find_peaks(concentrations, height=threshold, distance=3)
        
        # Create peak information
        peak_info = []
        for i, peak_idx in enumerate(peaks):
            peak_info.append({
                'index': int(peak_idx),
                'timestamp': gas_readings['timestamp'].iloc[peak_idx] if 'timestamp' in gas_readings.columns else None,
                'concentration': float(concentrations[peak_idx]),
                'width': float(properties.get('widths', [0])[i]) if 'widths' in properties else 0.0,
                'prominence': float(properties.get('prominences', [0])[i]) if 'prominences' in properties else 0.0
            })
        
        return peak_info
    
    def _apply_smoothing(self, df: pd.DataFrame, gas_types: List[str]) -> pd.DataFrame:
        """
        Apply smoothing to gas concentration data.
        
        Args:
            df: DataFrame containing gas concentration readings
            gas_types: List of gas types to smooth
            
        Returns:
            DataFrame with smoothed gas concentration values
        """
        # Create a copy of the DataFrame
        df_smoothed = df.copy()
        
        # Get smoothing window size
        window = self.config.get('smoothing_window', 3)
        
        # Apply smoothing to each gas type
        for gas in gas_types:
            if gas in df.columns:
                # Apply moving average smoothing
                df_smoothed[gas] = df[gas].rolling(window=window, center=True).mean()
                
                # Fill NaN values at the edges
                df_smoothed[gas] = df_smoothed[gas].fillna(df[gas])
        
        return df_smoothed
    
    def _calculate_derivatives(self, df: pd.DataFrame, gas_types: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate derivatives (rates of change) for gas concentrations.
        
        Args:
            df: DataFrame containing gas concentration readings
            gas_types: List of gas types to analyze
            
        Returns:
            Dictionary mapping gas type to derivative values
        """
        derivatives = {}
        
        for gas in gas_types:
            if gas in df.columns and 'time_seconds' in df.columns:
                # Get concentration and time values
                concentrations = df[gas].values
                times = df['time_seconds'].values
                
                # Calculate time differences
                dt = np.diff(times)
                
                # Avoid division by zero
                dt[dt == 0] = 1e-6
                
                # Calculate derivatives
                dC = np.diff(concentrations)
                derivatives[gas] = dC / dt
                
                # Add a zero at the beginning to maintain the same length
                derivatives[gas] = np.insert(derivatives[gas], 0, 0)
        
        return derivatives
    
    def _calculate_second_derivatives(self, derivatives: Dict[str, np.ndarray], df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate second derivatives (accelerations) for gas concentrations.
        
        Args:
            derivatives: Dictionary mapping gas type to first derivative values
            df: DataFrame containing gas concentration readings
            
        Returns:
            Dictionary mapping gas type to second derivative values
        """
        accelerations = {}
        
        for gas, first_deriv in derivatives.items():
            if 'time_seconds' in df.columns:
                # Get time values
                times = df['time_seconds'].values
                
                # Calculate time differences
                dt = np.diff(times)
                
                # Avoid division by zero
                dt[dt == 0] = 1e-6
                
                # Calculate second derivatives
                d2C = np.diff(first_deriv)
                acceleration = d2C / dt
                
                # Add a zero at the beginning to maintain the same length
                accelerations[gas] = np.insert(acceleration, 0, 0)
            else:
                # If no time information, use simple differencing
                d2C = np.diff(first_deriv)
                accelerations[gas] = np.insert(d2C, 0, 0)
        
        return accelerations
    
    def _detect_significant_changes(self, derivatives: Dict[str, np.ndarray], gas_types: List[str]) -> Dict[str, List[int]]:
        """
        Detect significant changes in gas concentrations.
        
        Args:
            derivatives: Dictionary mapping gas type to derivative values
            gas_types: List of gas types to analyze
            
        Returns:
            Dictionary mapping gas type to list of indices with significant changes
        """
        # Get thresholds for significant changes
        thresholds = self.config.get('significant_change_threshold', {})
        default_threshold = thresholds.get('default', 2.0)
        
        significant_changes = {}
        
        for gas in gas_types:
            if gas in derivatives:
                # Get threshold for this gas
                threshold = thresholds.get(gas, default_threshold)
                
                # Detect significant changes
                changes = np.where(np.abs(derivatives[gas]) > threshold)[0]
                significant_changes[gas] = changes.tolist()
        
        return significant_changes
    
    def _analyze_rate_patterns(self, derivatives: Dict[str, np.ndarray], gas_types: List[str]) -> Dict[str, str]:
        """
        Analyze patterns in the rate of change.
        
        Args:
            derivatives: Dictionary mapping gas type to derivative values
            gas_types: List of gas types to analyze
            
        Returns:
            Dictionary mapping gas type to rate pattern description
        """
        patterns = {}
        
        for gas in gas_types:
            if gas in derivatives:
                # Get derivative values
                rates = derivatives[gas]
                
                # Calculate statistics
                mean_rate = np.mean(rates)
                std_rate = np.std(rates)
                positive_percentage = np.mean(rates > 0) * 100.0
                negative_percentage = np.mean(rates < 0) * 100.0
                zero_crossings = np.sum(np.diff(np.signbit(rates)))
                
                # Determine pattern
                if mean_rate > 0.5 * std_rate and positive_percentage > 70:
                    pattern = "consistently_increasing"
                elif mean_rate < -0.5 * std_rate and negative_percentage > 70:
                    pattern = "consistently_decreasing"
                elif zero_crossings > len(rates) * 0.2:
                    pattern = "fluctuating"
                elif std_rate < 0.1 * np.abs(mean_rate) and np.abs(mean_rate) > 0.1:
                    pattern = "steady_change"
                elif std_rate < 0.1:
                    pattern = "stable"
                else:
                    pattern = "irregular"
                
                patterns[gas] = pattern
            else:
                patterns[gas] = "unknown"
        
        return patterns
    
    def _extract_rate_features(self, df: pd.DataFrame, gas_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract rate of change features for each gas type.
        
        Args:
            df: DataFrame containing gas concentration readings
            gas_types: List of gas types to analyze
            
        Returns:
            Dictionary mapping gas type to rate features
        """
        # Calculate derivatives
        derivatives = self._calculate_derivatives(df, gas_types)
        
        # Calculate second derivatives (accelerations)
        accelerations = self._calculate_second_derivatives(derivatives, df)
        
        # Detect significant changes
        significant_changes = self._detect_significant_changes(derivatives, gas_types)
        
        # Analyze rate patterns
        patterns = self._analyze_rate_patterns(derivatives, gas_types)
        
        # Calculate slopes for different time windows
        time_windows = self.config.get('time_windows', [1, 3, 5, 10])
        
        # Extract features for each gas type
        features = {}
        
        for gas in gas_types:
            if gas in derivatives:
                # Get derivative values
                rates = derivatives[gas]
                
                # Calculate basic statistics
                mean_rate = float(np.mean(rates))
                max_rate = float(np.max(rates))
                min_rate = float(np.min(rates))
                std_rate = float(np.std(rates))
                median_rate = float(np.median(rates))
                
                # Calculate absolute rate statistics
                abs_rates = np.abs(rates)
                mean_abs_rate = float(np.mean(abs_rates))
                max_abs_rate = float(np.max(abs_rates))
                
                # Calculate direction statistics
                positive_rate_percentage = float(np.mean(rates > 0) * 100.0)
                negative_rate_percentage = float(np.mean(rates < 0) * 100.0)
                
                # Calculate zero crossings
                zero_crossings = int(np.sum(np.diff(np.signbit(rates))))
                
                # Get significant changes
                sig_changes = significant_changes.get(gas, [])
                
                # Get acceleration statistics
                if gas in accelerations:
                    accel = accelerations[gas]
                    accel_mean = float(np.mean(accel))
                    accel_max = float(np.max(np.abs(accel)))
                else:
                    accel_mean = 0.0
                    accel_max = 0.0
                
                # Calculate slopes for different time windows
                slopes = self.calculate_concentration_slope(df, gas, time_windows)
                
                # Store features
                features[gas] = {
                    'mean_rate': mean_rate,
                    'max_rate': max_rate,
                    'min_rate': min_rate,
                    'std_rate': std_rate,
                    'median_rate': median_rate,
                    'mean_abs_rate': mean_abs_rate,
                    'max_abs_rate': max_abs_rate,
                    'positive_rate_percentage': positive_rate_percentage,
                    'negative_rate_percentage': negative_rate_percentage,
                    'zero_crossings': zero_crossings,
                    'significant_changes_count': len(sig_changes),
                    'significant_changes_indices': sig_changes,
                    'acceleration_mean': accel_mean,
                    'acceleration_max': accel_max,
                    'rate_pattern': patterns.get(gas, "unknown"),
                    'rate_values': rates.tolist(),
                    'window_slopes': slopes
                }
        
        return features