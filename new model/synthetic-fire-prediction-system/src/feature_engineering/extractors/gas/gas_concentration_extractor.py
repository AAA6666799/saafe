"""
Gas Concentration Extractor for gas sensor data analysis.

This module provides a feature extractor that extracts gas concentration level features.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal

from ...base import GasFeatureExtractor


class GasConcentrationExtractor(GasFeatureExtractor):
    """
    Feature extractor for gas concentration levels.
    
    This class analyzes gas concentration data to extract features related to
    concentration levels, including absolute values, normalized values, and
    statistical properties.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gas concentration extractor.
        
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
        if 'concentration_thresholds' not in self.config:
            # Default thresholds for common gases in ppm
            self.config['concentration_thresholds'] = {
                'methane': 500.0,  # Lower explosive limit is ~50,000 ppm
                'propane': 200.0,  # Lower explosive limit is ~20,000 ppm
                'hydrogen': 400.0,  # Lower explosive limit is ~40,000 ppm
                'carbon_monoxide': 50.0,  # OSHA PEL is 50 ppm
                'default': 100.0  # Default threshold for other gases
            }
        
        if 'normalization_method' not in self.config:
            self.config['normalization_method'] = 'min_max'  # Options: 'min_max', 'z_score', 'none'
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 5
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract gas concentration features from gas data.
        
        Args:
            data: Input gas data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting gas concentration features")
        
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
        
        # Extract concentration features
        concentration_features = self._extract_concentration_features(df_processed, gas_types)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'gas_types': gas_types,
            'concentration_features': concentration_features
        }
        
        self.logger.info(f"Extracted gas concentration features from {len(df)} samples")
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
        
        # Add concentration features
        concentration_features = features.get('concentration_features', {})
        for gas_type, gas_features in concentration_features.items():
            for feature_name, feature_value in gas_features.items():
                if isinstance(feature_value, list):
                    # Skip time series data in the flattened representation
                    flat_features[f"{gas_type}_{feature_name}_mean"] = np.mean(feature_value)
                    flat_features[f"{gas_type}_{feature_name}_max"] = np.max(feature_value)
                    flat_features[f"{gas_type}_{feature_name}_min"] = np.min(feature_value)
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
        
        self.logger.info(f"Saved gas concentration features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        # Base feature names (will be prefixed with gas type)
        base_features = [
            'mean_concentration',
            'max_concentration',
            'min_concentration',
            'std_concentration',
            'median_concentration',
            'q25_concentration',
            'q75_concentration',
            'iqr_concentration',
            'skewness',
            'kurtosis',
            'above_threshold_percentage',
            'peak_count',
            'peak_mean',
            'peak_max',
            'normalized_concentration'
        ]
        
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
    
    def _normalize_concentrations(self, df: pd.DataFrame, gas_types: List[str]) -> Dict[str, np.ndarray]:
        """
        Normalize gas concentration values.
        
        Args:
            df: DataFrame containing gas concentration readings
            gas_types: List of gas types to normalize
            
        Returns:
            Dictionary mapping gas type to normalized concentration values
        """
        normalized = {}
        method = self.config.get('normalization_method', 'min_max')
        
        for gas in gas_types:
            if gas in df.columns:
                values = df[gas].values
                
                if method == 'min_max':
                    # Min-max normalization
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    if max_val > min_val:
                        normalized[gas] = (values - min_val) / (max_val - min_val)
                    else:
                        normalized[gas] = np.zeros_like(values)
                
                elif method == 'z_score':
                    # Z-score normalization
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    if std_val > 0:
                        normalized[gas] = (values - mean_val) / std_val
                    else:
                        normalized[gas] = np.zeros_like(values)
                
                else:  # 'none' or unknown method
                    normalized[gas] = values
        
        return normalized
    
    def _extract_concentration_features(self, df: pd.DataFrame, gas_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract concentration features for each gas type.
        
        Args:
            df: DataFrame containing gas concentration readings
            gas_types: List of gas types to analyze
            
        Returns:
            Dictionary mapping gas type to concentration features
        """
        # Get concentration thresholds
        thresholds = self.config.get('concentration_thresholds', {})
        default_threshold = thresholds.get('default', 100.0)
        
        # Normalize concentrations
        normalized = self._normalize_concentrations(df, gas_types)
        
        # Extract features for each gas type
        features = {}
        
        for gas in gas_types:
            if gas in df.columns:
                # Get concentration values
                concentrations = df[gas].values
                
                # Get threshold for this gas
                threshold = thresholds.get(gas, default_threshold)
                
                # Calculate basic statistics
                mean_conc = float(np.mean(concentrations))
                max_conc = float(np.max(concentrations))
                min_conc = float(np.min(concentrations))
                std_conc = float(np.std(concentrations))
                median_conc = float(np.median(concentrations))
                q25_conc = float(np.percentile(concentrations, 25))
                q75_conc = float(np.percentile(concentrations, 75))
                iqr_conc = float(q75_conc - q25_conc)
                
                # Calculate higher-order statistics
                skewness = float(stats.skew(concentrations))
                kurtosis = float(stats.kurtosis(concentrations))
                
                # Calculate threshold-based features
                above_threshold = concentrations > threshold
                above_threshold_percentage = float(np.mean(above_threshold) * 100.0)
                
                # Detect peaks
                peaks = self.detect_concentration_peaks(df, gas, threshold)
                peak_count = len(peaks)
                peak_concentrations = [p['concentration'] for p in peaks]
                peak_mean = float(np.mean(peak_concentrations)) if peak_concentrations else 0.0
                peak_max = float(np.max(peak_concentrations)) if peak_concentrations else 0.0
                
                # Store features
                features[gas] = {
                    'mean_concentration': mean_conc,
                    'max_concentration': max_conc,
                    'min_concentration': min_conc,
                    'std_concentration': std_conc,
                    'median_concentration': median_conc,
                    'q25_concentration': q25_conc,
                    'q75_concentration': q75_conc,
                    'iqr_concentration': iqr_conc,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'above_threshold_percentage': above_threshold_percentage,
                    'peak_count': peak_count,
                    'peak_mean': peak_mean,
                    'peak_max': peak_max,
                    'normalized_concentration': normalized.get(gas, []).tolist(),
                    'concentration_values': concentrations.tolist()
                }
        
        return features