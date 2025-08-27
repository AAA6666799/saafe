"""
Gas Ratio Calculator for gas sensor data analysis.

This module provides a feature extractor that calculates ratios between different gas types.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats

from ...base import GasFeatureExtractor


class GasRatioCalculator(GasFeatureExtractor):
    """
    Feature extractor for calculating ratios between different gas types.
    
    This class analyzes gas concentration data to calculate ratios between different
    gas types, which can be indicative of specific fire types or combustion processes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gas ratio calculator.
        
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
        
        # Ensure we have at least two gas types for ratio calculation
        if len(self.config['gas_types']) < 2:
            raise ValueError("At least two gas types are required for ratio calculation")
        
        # Set default values for optional parameters
        if 'ratio_pairs' not in self.config:
            # By default, calculate ratios between all pairs of gases
            gas_types = self.config['gas_types']
            self.config['ratio_pairs'] = [(gas1, gas2) for i, gas1 in enumerate(gas_types) 
                                        for gas2 in gas_types[i+1:]]
        
        if 'min_denominator_value' not in self.config:
            self.config['min_denominator_value'] = 1.0  # Minimum value to avoid division by zero
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 5
        
        if 'calculate_inverse_ratios' not in self.config:
            self.config['calculate_inverse_ratios'] = False
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract gas ratio features from gas data.
        
        Args:
            data: Input gas data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting gas ratio features")
        
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
        
        # Apply smoothing if configured
        if self.config.get('apply_smoothing', True):
            df_processed = self._apply_smoothing(df, gas_types)
        else:
            df_processed = df
        
        # Extract ratio features
        ratio_features = self._extract_ratio_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'gas_types': gas_types,
            'ratio_features': ratio_features
        }
        
        self.logger.info(f"Extracted gas ratio features from {len(df)} samples")
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
        
        # Add ratio features
        ratio_features = features.get('ratio_features', {})
        for ratio_name, ratio_values in ratio_features.items():
            if isinstance(ratio_values, dict):
                for stat_name, stat_value in ratio_values.items():
                    if isinstance(stat_value, list):
                        # Skip time series data in the flattened representation
                        flat_features[f"ratio_{ratio_name}_{stat_name}_mean"] = np.mean(stat_value)
                        flat_features[f"ratio_{ratio_name}_{stat_name}_max"] = np.max(stat_value)
                        flat_features[f"ratio_{ratio_name}_{stat_name}_min"] = np.min(stat_value)
                    else:
                        flat_features[f"ratio_{ratio_name}_{stat_name}"] = stat_value
            else:
                flat_features[f"ratio_{ratio_name}"] = ratio_values
        
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
        
        self.logger.info(f"Saved gas ratio features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        # Get ratio pairs
        ratio_pairs = self.config.get('ratio_pairs', [])
        
        # Base feature names for each ratio
        base_features = [
            'mean',
            'median',
            'std',
            'min',
            'max',
            'q25',
            'q75',
            'iqr',
            'skewness',
            'kurtosis',
            'stability',
            'trend'
        ]
        
        # Generate feature names for each ratio pair
        feature_names = []
        for gas1, gas2 in ratio_pairs:
            ratio_name = f"{gas1}_to_{gas2}"
            for feature in base_features:
                feature_names.append(f"ratio_{ratio_name}_{feature}")
            
            # Add inverse ratio features if configured
            if self.config.get('calculate_inverse_ratios', False):
                inverse_ratio_name = f"{gas2}_to_{gas1}"
                for feature in base_features:
                    feature_names.append(f"ratio_{inverse_ratio_name}_{feature}")
        
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
        if gas_type not in gas_readings.columns:
            return {str(window): 0.0 for window in window_sizes}
        
        slopes = {}
        
        for window in window_sizes:
            if window >= len(gas_readings):
                slopes[str(window)] = 0.0
                continue
            
            # Calculate rolling linear regression
            y = gas_readings[gas_type].values
            x = np.arange(len(y))
            
            # Use rolling window
            window_slopes = []
            for i in range(len(y) - window + 1):
                x_window = x[i:i+window]
                y_window = y[i:i+window]
                
                # Calculate slope using linear regression
                slope, _, _, _, _ = stats.linregress(x_window, y_window)
                window_slopes.append(slope)
            
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
        from scipy import signal
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
        window = self.config.get('smoothing_window', 5)
        
        # Apply smoothing to each gas type
        for gas in gas_types:
            if gas in df.columns:
                # Apply moving average smoothing
                df_smoothed[gas] = df[gas].rolling(window=window, center=True).mean()
                
                # Fill NaN values at the edges
                df_smoothed[gas] = df_smoothed[gas].fillna(df[gas])
        
        return df_smoothed
    
    def _calculate_ratio(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """
        Calculate ratio between two gas concentrations with safeguards.
        
        Args:
            numerator: Numerator values
            denominator: Denominator values
            
        Returns:
            Array of ratio values
        """
        # Get minimum denominator value
        min_value = self.config.get('min_denominator_value', 1.0)
        
        # Replace zeros and very small values in denominator
        safe_denominator = np.copy(denominator)
        safe_denominator[safe_denominator < min_value] = min_value
        
        # Calculate ratio
        ratio = numerator / safe_denominator
        
        # Cap extreme values
        ratio = np.clip(ratio, -1e6, 1e6)
        
        return ratio
    
    def _calculate_ratio_statistics(self, ratio_values: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics for a ratio time series.
        
        Args:
            ratio_values: Array of ratio values
            
        Returns:
            Dictionary containing ratio statistics
        """
        # Calculate basic statistics
        mean_ratio = float(np.mean(ratio_values))
        median_ratio = float(np.median(ratio_values))
        std_ratio = float(np.std(ratio_values))
        min_ratio = float(np.min(ratio_values))
        max_ratio = float(np.max(ratio_values))
        q25_ratio = float(np.percentile(ratio_values, 25))
        q75_ratio = float(np.percentile(ratio_values, 75))
        iqr_ratio = float(q75_ratio - q25_ratio)
        
        # Calculate higher-order statistics
        skewness = float(stats.skew(ratio_values))
        kurtosis = float(stats.kurtosis(ratio_values))
        
        # Calculate stability (inverse of coefficient of variation)
        if mean_ratio != 0:
            stability = float(1.0 / (std_ratio / abs(mean_ratio))) if std_ratio > 0 else float('inf')
        else:
            stability = 0.0
        
        # Calculate trend (slope of linear regression)
        x = np.arange(len(ratio_values))
        slope, _, _, _, _ = stats.linregress(x, ratio_values)
        trend = float(slope)
        
        return {
            'mean': mean_ratio,
            'median': median_ratio,
            'std': std_ratio,
            'min': min_ratio,
            'max': max_ratio,
            'q25': q25_ratio,
            'q75': q75_ratio,
            'iqr': iqr_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'stability': stability,
            'trend': trend,
            'values': ratio_values.tolist()
        }
    
    def _extract_ratio_features(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Extract ratio features from gas concentration data.
        
        Args:
            df: DataFrame containing gas concentration readings
            
        Returns:
            Dictionary mapping ratio name to ratio features
        """
        # Get ratio pairs
        ratio_pairs = self.config.get('ratio_pairs', [])
        
        # Calculate ratios and statistics
        ratio_features = {}
        
        for gas1, gas2 in ratio_pairs:
            if gas1 in df.columns and gas2 in df.columns:
                # Get concentration values
                values1 = df[gas1].values
                values2 = df[gas2].values
                
                # Calculate ratio
                ratio_values = self._calculate_ratio(values1, values2)
                
                # Calculate statistics
                ratio_name = f"{gas1}_to_{gas2}"
                ratio_features[ratio_name] = self._calculate_ratio_statistics(ratio_values)
                
                # Calculate inverse ratio if configured
                if self.config.get('calculate_inverse_ratios', False):
                    inverse_ratio_values = self._calculate_ratio(values2, values1)
                    inverse_ratio_name = f"{gas2}_to_{gas1}"
                    ratio_features[inverse_ratio_name] = self._calculate_ratio_statistics(inverse_ratio_values)
        
        # Calculate composite ratios if we have more than two gas types
        gas_types = self.config['gas_types']
        if len(gas_types) >= 3:
            # Calculate sum ratios (e.g., (gas1 + gas2) / gas3)
            for i, gas1 in enumerate(gas_types):
                for j, gas2 in enumerate(gas_types[i+1:], i+1):
                    for k, gas3 in enumerate(gas_types):
                        if k != i and k != j and gas1 in df.columns and gas2 in df.columns and gas3 in df.columns:
                            # Calculate sum ratio
                            sum_values = df[gas1].values + df[gas2].values
                            ratio_values = self._calculate_ratio(sum_values, df[gas3].values)
                            
                            # Calculate statistics
                            ratio_name = f"{gas1}_plus_{gas2}_to_{gas3}"
                            ratio_features[ratio_name] = self._calculate_ratio_statistics(ratio_values)
        
        return ratio_features