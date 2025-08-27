"""
Gas Anomaly Detector for gas sensor data analysis.

This module provides a feature extractor that detects anomalies in gas concentration data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ...base import GasFeatureExtractor


class GasAnomalyDetector(GasFeatureExtractor):
    """
    Feature extractor for detecting anomalies in gas concentration data.
    
    This class analyzes gas concentration data to identify anomalous patterns,
    unusual concentration levels, and outliers in gas behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gas anomaly detector.
        
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
        if 'anomaly_detection_method' not in self.config:
            self.config['anomaly_detection_method'] = 'zscore'  # Options: 'zscore', 'isolation_forest', 'lof'
        
        if 'zscore_threshold' not in self.config:
            self.config['zscore_threshold'] = 3.0
        
        if 'contamination' not in self.config:
            self.config['contamination'] = 0.05  # For isolation forest and LOF
        
        if 'n_estimators' not in self.config:
            self.config['n_estimators'] = 100  # For isolation forest
        
        if 'n_neighbors' not in self.config:
            self.config['n_neighbors'] = 20  # For LOF
        
        if 'window_size' not in self.config:
            self.config['window_size'] = 10  # Window size for contextual anomaly detection
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 3
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract anomaly features from gas data.
        
        Args:
            data: Input gas data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting gas anomaly features")
        
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
        
        # Apply smoothing if configured
        if self.config.get('apply_smoothing', True):
            df_processed = self._apply_smoothing(df, gas_types)
        else:
            df_processed = df
        
        # Extract anomaly features
        anomaly_features = self._extract_anomaly_features(df_processed, gas_types)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'gas_types': gas_types,
            'anomaly_features': anomaly_features
        }
        
        self.logger.info(f"Extracted gas anomaly features from {len(df)} samples")
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
        
        # Add anomaly features
        anomaly_features = features.get('anomaly_features', {})
        for gas_type, gas_features in anomaly_features.items():
            for feature_name, feature_value in gas_features.items():
                if isinstance(feature_value, list):
                    # Skip time series data in the flattened representation
                    if len(feature_value) <= 20:  # Only include reasonably sized lists
                        for i, value in enumerate(feature_value):
                            flat_features[f"{gas_type}_anomaly_{feature_name}_{i}"] = value
                    else:
                        # For large lists, just include summary statistics
                        flat_features[f"{gas_type}_anomaly_{feature_name}_mean"] = np.mean(feature_value)
                        flat_features[f"{gas_type}_anomaly_{feature_name}_max"] = np.max(feature_value)
                        flat_features[f"{gas_type}_anomaly_{feature_name}_min"] = np.min(feature_value)
                elif isinstance(feature_value, dict):
                    for sub_name, sub_value in feature_value.items():
                        flat_features[f"{gas_type}_anomaly_{feature_name}_{sub_name}"] = sub_value
                else:
                    flat_features[f"{gas_type}_anomaly_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved gas anomaly features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        # Base feature names (will be prefixed with gas type)
        base_features = [
            'anomaly_score_mean',
            'anomaly_score_max',
            'anomaly_count',
            'anomaly_percentage',
            'anomaly_severity_mean',
            'anomaly_severity_max',
            'anomaly_duration_mean',
            'anomaly_duration_max',
            'anomaly_frequency',
            'contextual_anomaly_count',
            'collective_anomaly_count',
            'point_anomaly_count'
        ]
        
        # Generate full feature names for each gas type
        feature_names = []
        for gas in self.config.get('gas_types', []):
            for feature in base_features:
                feature_names.append(f"{gas}_anomaly_{feature}")
        
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
        window = self.config.get('smoothing_window', 3)
        
        # Apply smoothing to each gas type
        for gas in gas_types:
            if gas in df.columns:
                # Apply moving average smoothing
                df_smoothed[gas] = df[gas].rolling(window=window, center=True).mean()
                
                # Fill NaN values at the edges
                df_smoothed[gas] = df_smoothed[gas].fillna(df[gas])
        
        return df_smoothed
    
    def _detect_zscore_anomalies(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            values: Time series values
            
        Returns:
            Tuple of (anomaly_scores, anomaly_mask)
        """
        # Calculate mean and standard deviation
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Calculate Z-scores
        if std_val > 0:
            z_scores = np.abs((values - mean_val) / std_val)
        else:
            z_scores = np.zeros_like(values)
        
        # Create anomaly mask
        threshold = self.config.get('zscore_threshold', 3.0)
        anomaly_mask = z_scores > threshold
        
        return z_scores, anomaly_mask
    
    def _detect_isolation_forest_anomalies(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            values: Time series values
            
        Returns:
            Tuple of (anomaly_scores, anomaly_mask)
        """
        # Reshape for scikit-learn
        X = values.reshape(-1, 1)
        
        # Add temporal context
        window_size = self.config.get('window_size', 10)
        if len(values) > window_size:
            # Create rolling windows
            windows = []
            for i in range(len(values) - window_size + 1):
                windows.append(values[i:i+window_size])
            
            # Use the windows as features
            X = np.array(windows)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        contamination = self.config.get('contamination', 0.05)
        n_estimators = self.config.get('n_estimators', 100)
        
        clf = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
        
        try:
            # Fit and predict
            clf.fit(X_scaled)
            y_pred = clf.decision_function(X_scaled)
            
            # Convert to anomaly scores (higher = more anomalous)
            anomaly_scores = -y_pred
            
            # Create anomaly mask
            anomaly_mask = clf.predict(X_scaled) == -1
            
            # If we used windows, we need to map back to the original time series
            if len(values) > window_size:
                # For scores, use the maximum score of windows containing each point
                full_scores = np.zeros_like(values)
                full_mask = np.zeros_like(values, dtype=bool)
                
                for i in range(len(anomaly_scores)):
                    for j in range(window_size):
                        idx = i + j
                        if idx < len(values):
                            full_scores[idx] = max(full_scores[idx], anomaly_scores[i])
                            full_mask[idx] = full_mask[idx] or anomaly_mask[i]
                
                return full_scores, full_mask
            else:
                return anomaly_scores, anomaly_mask
        
        except Exception as e:
            self.logger.warning(f"Isolation Forest failed: {str(e)}")
            return np.zeros_like(values), np.zeros_like(values, dtype=bool)
    
    def _detect_lof_anomalies(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            values: Time series values
            
        Returns:
            Tuple of (anomaly_scores, anomaly_mask)
        """
        # Reshape for scikit-learn
        X = values.reshape(-1, 1)
        
        # Add temporal context
        window_size = self.config.get('window_size', 10)
        if len(values) > window_size:
            # Create rolling windows
            windows = []
            for i in range(len(values) - window_size + 1):
                windows.append(values[i:i+window_size])
            
            # Use the windows as features
            X = np.array(windows)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train LOF
        contamination = self.config.get('contamination', 0.05)
        n_neighbors = self.config.get('n_neighbors', 20)
        
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
        
        try:
            # Fit and predict
            y_pred = clf.fit_predict(X_scaled)
            
            # Get negative outlier factor (higher = more anomalous)
            anomaly_scores = -clf._decision_function(X_scaled)
            
            # Create anomaly mask
            anomaly_mask = y_pred == -1
            
            # If we used windows, we need to map back to the original time series
            if len(values) > window_size:
                # For scores, use the maximum score of windows containing each point
                full_scores = np.zeros_like(values)
                full_mask = np.zeros_like(values, dtype=bool)
                
                for i in range(len(anomaly_scores)):
                    for j in range(window_size):
                        idx = i + j
                        if idx < len(values):
                            full_scores[idx] = max(full_scores[idx], anomaly_scores[i])
                            full_mask[idx] = full_mask[idx] or anomaly_mask[i]
                
                return full_scores, full_mask
            else:
                return anomaly_scores, anomaly_mask
        
        except Exception as e:
            self.logger.warning(f"LOF failed: {str(e)}")
            return np.zeros_like(values), np.zeros_like(values, dtype=bool)
    
    def _detect_contextual_anomalies(self, values: np.ndarray) -> np.ndarray:
        """
        Detect contextual anomalies in a time series.
        
        Args:
            values: Time series values
            
        Returns:
            Binary mask of contextual anomalies
        """
        window_size = self.config.get('window_size', 10)
        if len(values) <= window_size:
            return np.zeros_like(values, dtype=bool)
        
        # Calculate rolling statistics
        contextual_anomalies = np.zeros_like(values, dtype=bool)
        
        for i in range(window_size, len(values)):
            # Get window
            window = values[i-window_size:i]
            
            # Calculate window statistics
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            # Check if current value is anomalous in this context
            if window_std > 0:
                z_score = abs(values[i] - window_mean) / window_std
                if z_score > self.config.get('zscore_threshold', 3.0):
                    contextual_anomalies[i] = True
        
        return contextual_anomalies
    
    def _detect_collective_anomalies(self, values: np.ndarray, anomaly_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect collective anomalies (sequences of anomalous points).
        
        Args:
            values: Time series values
            anomaly_mask: Binary mask of anomalies
            
        Returns:
            List of (start_idx, end_idx) tuples for collective anomalies
        """
        min_length = 3  # Minimum length for a collective anomaly
        
        # Find sequences of anomalies
        collective_anomalies = []
        in_anomaly = False
        start_idx = 0
        
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly and not in_anomaly:
                # Start of a new anomaly sequence
                in_anomaly = True
                start_idx = i
            elif not is_anomaly and in_anomaly:
                # End of an anomaly sequence
                if i - start_idx >= min_length:
                    collective_anomalies.append((start_idx, i - 1))
                in_anomaly = False
        
        # Check if we ended in an anomaly sequence
        if in_anomaly and len(values) - start_idx >= min_length:
            collective_anomalies.append((start_idx, len(values) - 1))
        
        return collective_anomalies
    
    def _extract_anomaly_features(self, df: pd.DataFrame, gas_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract anomaly features for each gas type.
        
        Args:
            df: DataFrame containing gas concentration readings
            gas_types: List of gas types to analyze
            
        Returns:
            Dictionary mapping gas type to anomaly features
        """
        # Get anomaly detection method
        method = self.config.get('anomaly_detection_method', 'zscore')
        
        # Extract features for each gas type
        features = {}
        
        for gas in gas_types:
            if gas in df.columns:
                # Get concentration values
                values = df[gas].values
                
                # Detect anomalies using the configured method
                if method == 'zscore':
                    anomaly_scores, anomaly_mask = self._detect_zscore_anomalies(values)
                elif method == 'isolation_forest':
                    anomaly_scores, anomaly_mask = self._detect_isolation_forest_anomalies(values)
                elif method == 'lof':
                    anomaly_scores, anomaly_mask = self._detect_lof_anomalies(values)
                else:
                    self.logger.warning(f"Unknown anomaly detection method: {method}, using zscore")
                    anomaly_scores, anomaly_mask = self._detect_zscore_anomalies(values)
                
                # Detect contextual anomalies
                contextual_anomalies = self._detect_contextual_anomalies(values)
                
                # Detect collective anomalies
                collective_anomalies = self._detect_collective_anomalies(values, anomaly_mask)
                
                # Calculate anomaly statistics
                anomaly_count = int(np.sum(anomaly_mask))
                anomaly_percentage = float(np.mean(anomaly_mask) * 100.0)
                
                # Calculate anomaly severity
                anomaly_severity = anomaly_scores[anomaly_mask] if anomaly_count > 0 else np.array([0.0])
                anomaly_severity_mean = float(np.mean(anomaly_severity))
                anomaly_severity_max = float(np.max(anomaly_severity))
                
                # Calculate anomaly durations
                anomaly_durations = []
                in_anomaly = False
                current_duration = 0
                
                for is_anomaly in anomaly_mask:
                    if is_anomaly:
                        current_duration += 1
                        in_anomaly = True
                    elif in_anomaly:
                        anomaly_durations.append(current_duration)
                        current_duration = 0
                        in_anomaly = False
                
                # Add last anomaly if we ended in an anomaly
                if in_anomaly and current_duration > 0:
                    anomaly_durations.append(current_duration)
                
                # Calculate anomaly duration statistics
                anomaly_duration_mean = float(np.mean(anomaly_durations)) if anomaly_durations else 0.0
                anomaly_duration_max = float(np.max(anomaly_durations)) if anomaly_durations else 0.0
                
                # Calculate anomaly frequency (anomalies per 100 samples)
                if len(values) > 0:
                    anomaly_frequency = float(anomaly_count / len(values) * 100.0)
                else:
                    anomaly_frequency = 0.0
                
                # Count different types of anomalies
                contextual_anomaly_count = int(np.sum(contextual_anomalies))
                collective_anomaly_count = len(collective_anomalies)
                point_anomaly_count = anomaly_count - contextual_anomaly_count - sum(
                    end - start + 1 for start, end in collective_anomalies)
                point_anomaly_count = max(0, point_anomaly_count)  # Ensure non-negative
                
                # Store features
                features[gas] = {
                    'anomaly_scores': anomaly_scores.tolist(),
                    'anomaly_mask': anomaly_mask.tolist(),
                    'anomaly_score_mean': float(np.mean(anomaly_scores)),
                    'anomaly_score_max': float(np.max(anomaly_scores)),
                    'anomaly_count': anomaly_count,
                    'anomaly_percentage': anomaly_percentage,
                    'anomaly_severity_mean': anomaly_severity_mean,
                    'anomaly_severity_max': anomaly_severity_max,
                    'anomaly_duration_mean': anomaly_duration_mean,
                    'anomaly_duration_max': anomaly_duration_max,
                    'anomaly_frequency': anomaly_frequency,
                    'contextual_anomaly_count': contextual_anomaly_count,
                    'collective_anomaly_count': collective_anomaly_count,
                    'point_anomaly_count': point_anomaly_count,
                    'collective_anomalies': [(int(start), int(end)) for start, end in collective_anomalies]
                }
        
        return features