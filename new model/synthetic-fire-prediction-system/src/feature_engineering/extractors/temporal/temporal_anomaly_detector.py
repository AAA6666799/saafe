"""
Temporal Anomaly Detector for time-series data analysis.

This module provides a feature extractor that detects and characterizes anomalies in time-series data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal
import matplotlib.pyplot as plt

from ...base_temporal import TemporalFeatureExtractor


class TemporalAnomalyDetector(TemporalFeatureExtractor):
    """
    Feature extractor for anomaly detection in time-series data.
    
    This class analyzes time-series data to identify and characterize anomalies,
    including point anomalies, contextual anomalies, and collective anomalies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the temporal anomaly detector.
        
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
        
        if 'anomaly_detection_method' not in self.config:
            self.config['anomaly_detection_method'] = 'z_score'  # Options: 'z_score', 'iqr', 'isolation_forest', 'lof', 'autoencoder'
        
        if 'window_size' not in self.config:
            self.config['window_size'] = 10  # Window size for contextual anomaly detection
        
        if 'threshold' not in self.config:
            self.config['threshold'] = 3.0  # Threshold for anomaly detection
        
        if 'contamination' not in self.config:
            self.config['contamination'] = 0.05  # Contamination parameter for isolation forest and LOF
        
        if 'n_neighbors' not in self.config:
            self.config['n_neighbors'] = 20  # Number of neighbors for LOF
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 3
        
        if 'anomaly_types' not in self.config:
            self.config['anomaly_types'] = ['point', 'contextual', 'collective']  # Types of anomalies to detect
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract anomaly features from time-series data.
        
        Args:
            data: Input time-series data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting temporal anomaly features")
        
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
        
        # Extract anomaly features
        anomaly_features = self._extract_anomaly_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'anomaly_features': anomaly_features
        }
        
        self.logger.info(f"Extracted temporal anomaly features from {len(df)} samples")
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
        for feature_name, feature_value in anomaly_features.items():
            if isinstance(feature_value, list):
                # Skip time series data in the flattened representation
                if len(feature_value) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_value):
                        flat_features[f"anomaly_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"anomaly_{feature_name}_count"] = len(feature_value)
            elif isinstance(feature_value, dict):
                for sub_name, sub_value in feature_value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 20:
                        # For large lists, just include summary statistics
                        flat_features[f"anomaly_{feature_name}_{sub_name}_count"] = len(sub_value)
                    else:
                        flat_features[f"anomaly_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"anomaly_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved temporal anomaly features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'anomalies',
            'anomaly_count',
            'anomaly_indices',
            'anomaly_scores',
            'anomaly_types',
            'anomaly_severities',
            'point_anomalies',
            'contextual_anomalies',
            'collective_anomalies',
            'anomaly_statistics',
            'anomaly_distribution',
            'anomaly_clusters'
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
    
    def _extract_anomaly_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract anomaly features from time-series data.
        
        Args:
            df: DataFrame containing time-series data
            
        Returns:
            Dictionary containing anomaly features
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
            self.logger.warning("Not enough data for anomaly detection (need at least 10 points)")
            return {
                'anomalies': [],
                'anomaly_count': 0
            }
        
        # Get anomaly detection method
        method = self.config.get('anomaly_detection_method', 'z_score')
        
        # Get anomaly types to detect
        anomaly_types = self.config.get('anomaly_types', ['point', 'contextual', 'collective'])
        
        # Initialize anomaly lists
        point_anomalies = []
        contextual_anomalies = []
        collective_anomalies = []
        
        # Detect point anomalies
        if 'point' in anomaly_types:
            point_anomalies = self._detect_point_anomalies(values, method)
        
        # Detect contextual anomalies
        if 'contextual' in anomaly_types:
            contextual_anomalies = self._detect_contextual_anomalies(values)
        
        # Detect collective anomalies
        if 'collective' in anomaly_types:
            collective_anomalies = self._detect_collective_anomalies(values)
        
        # Combine all anomalies
        all_anomalies = point_anomalies + contextual_anomalies + collective_anomalies
        
        # Remove duplicates
        unique_anomalies = []
        unique_indices = set()
        
        for anomaly in all_anomalies:
            if anomaly['index'] not in unique_indices:
                unique_indices.add(anomaly['index'])
                unique_anomalies.append(anomaly)
        
        # Sort anomalies by index
        unique_anomalies = sorted(unique_anomalies, key=lambda x: x['index'])
        
        # Calculate anomaly statistics
        anomaly_statistics = self._calculate_anomaly_statistics(values, unique_anomalies)
        
        # Calculate anomaly distribution
        anomaly_distribution = self._calculate_anomaly_distribution(values, unique_anomalies)
        
        # Cluster anomalies
        anomaly_clusters = self._cluster_anomalies(unique_anomalies)
        
        # Prepare result
        anomaly_features = {
            'anomalies': unique_anomalies,
            'anomaly_count': len(unique_anomalies),
            'anomaly_indices': [anomaly['index'] for anomaly in unique_anomalies],
            'anomaly_scores': [anomaly['score'] for anomaly in unique_anomalies],
            'anomaly_types': [anomaly['type'] for anomaly in unique_anomalies],
            'anomaly_severities': [anomaly['severity'] for anomaly in unique_anomalies],
            'point_anomalies': point_anomalies,
            'contextual_anomalies': contextual_anomalies,
            'collective_anomalies': collective_anomalies,
            'anomaly_statistics': anomaly_statistics,
            'anomaly_distribution': anomaly_distribution,
            'anomaly_clusters': anomaly_clusters
        }
        
        return anomaly_features
    
    def _detect_point_anomalies(self, values: np.ndarray, method: str) -> List[Dict[str, Any]]:
        """
        Detect point anomalies in time-series data.
        
        Args:
            values: Time series values
            method: Anomaly detection method
            
        Returns:
            List of dictionaries containing point anomaly information
        """
        # Get threshold
        threshold = self.config.get('threshold', 3.0)
        
        # Get contamination
        contamination = self.config.get('contamination', 0.05)
        
        # Get number of neighbors
        n_neighbors = self.config.get('n_neighbors', 20)
        
        # Detect anomalies using the specified method
        if method == 'z_score':
            return self._detect_anomalies_z_score(values, threshold)
        elif method == 'iqr':
            return self._detect_anomalies_iqr(values, threshold)
        elif method == 'isolation_forest':
            return self._detect_anomalies_isolation_forest(values, contamination)
        elif method == 'lof':
            return self._detect_anomalies_lof(values, n_neighbors, contamination)
        elif method == 'autoencoder':
            return self._detect_anomalies_autoencoder(values, threshold)
        else:
            self.logger.warning(f"Unknown anomaly detection method: {method}, using z_score")
            return self._detect_anomalies_z_score(values, threshold)
    
    def _detect_anomalies_z_score(self, values: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            values: Time series values
            threshold: Z-score threshold
            
        Returns:
            List of dictionaries containing anomaly information
        """
        # Calculate mean and standard deviation
        mean = np.mean(values)
        std = np.std(values)
        
        # Check if standard deviation is valid
        if std == 0:
            return []
        
        # Calculate Z-scores
        z_scores = np.abs((values - mean) / std)
        
        # Detect anomalies
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        # Create anomaly information
        anomalies = []
        for idx in anomaly_indices:
            # Calculate anomaly score
            score = float(z_scores[idx])
            
            # Calculate anomaly severity
            severity = self._calculate_anomaly_severity(score, threshold)
            
            anomalies.append({
                'index': int(idx),
                'value': float(values[idx]),
                'score': score,
                'type': 'point',
                'method': 'z_score',
                'severity': severity
            })
        
        return anomalies
    
    def _detect_anomalies_iqr(self, values: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            values: Time series values
            threshold: IQR multiplier
            
        Returns:
            List of dictionaries containing anomaly information
        """
        # Calculate quartiles
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        
        # Calculate IQR
        iqr = q3 - q1
        
        # Check if IQR is valid
        if iqr == 0:
            return []
        
        # Calculate lower and upper bounds
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Detect anomalies
        anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
        
        # Create anomaly information
        anomalies = []
        for idx in anomaly_indices:
            # Calculate anomaly score
            if values[idx] < lower_bound:
                score = float(abs(values[idx] - lower_bound) / iqr)
            else:
                score = float(abs(values[idx] - upper_bound) / iqr)
            
            # Calculate anomaly severity
            severity = self._calculate_anomaly_severity(score, threshold)
            
            anomalies.append({
                'index': int(idx),
                'value': float(values[idx]),
                'score': score,
                'type': 'point',
                'method': 'iqr',
                'severity': severity
            })
        
        return anomalies
    
    def _detect_anomalies_isolation_forest(self, values: np.ndarray, contamination: float) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Args:
            values: Time series values
            contamination: Expected proportion of anomalies
            
        Returns:
            List of dictionaries containing anomaly information
        """
        try:
            # Try to import scikit-learn
            from sklearn.ensemble import IsolationForest
            
            # Reshape values for scikit-learn
            X = values.reshape(-1, 1)
            
            # Create and fit model
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(X)
            
            # Predict anomalies
            # -1 for anomalies, 1 for normal points
            predictions = model.predict(X)
            
            # Calculate anomaly scores
            scores = -model.score_samples(X)  # Negated to make higher values more anomalous
            
            # Detect anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            
            # Create anomaly information
            anomalies = []
            for idx in anomaly_indices:
                # Calculate anomaly score
                score = float(scores[idx])
                
                # Calculate anomaly severity
                severity = self._calculate_anomaly_severity(score, np.percentile(scores, 95))
                
                anomalies.append({
                    'index': int(idx),
                    'value': float(values[idx]),
                    'score': score,
                    'type': 'point',
                    'method': 'isolation_forest',
                    'severity': severity
                })
            
            return anomalies
        
        except ImportError:
            self.logger.warning("scikit-learn package not available, using Z-score method")
            return self._detect_anomalies_z_score(values, self.config.get('threshold', 3.0))
    
    def _detect_anomalies_lof(self, values: np.ndarray, n_neighbors: int, contamination: float) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Local Outlier Factor (LOF) algorithm.
        
        Args:
            values: Time series values
            n_neighbors: Number of neighbors
            contamination: Expected proportion of anomalies
            
        Returns:
            List of dictionaries containing anomaly information
        """
        try:
            # Try to import scikit-learn
            from sklearn.neighbors import LocalOutlierFactor
            
            # Reshape values for scikit-learn
            X = values.reshape(-1, 1)
            
            # Create and fit model
            model = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(X) - 1), contamination=contamination)
            
            # Predict anomalies
            # -1 for anomalies, 1 for normal points
            predictions = model.fit_predict(X)
            
            # Calculate anomaly scores
            scores = -model.negative_outlier_factor_  # Negated to make higher values more anomalous
            
            # Detect anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            
            # Create anomaly information
            anomalies = []
            for idx in anomaly_indices:
                # Calculate anomaly score
                score = float(scores[idx])
                
                # Calculate anomaly severity
                severity = self._calculate_anomaly_severity(score, np.percentile(scores, 95))
                
                anomalies.append({
                    'index': int(idx),
                    'value': float(values[idx]),
                    'score': score,
                    'type': 'point',
                    'method': 'lof',
                    'severity': severity
                })
            
            return anomalies
        
        except ImportError:
            self.logger.warning("scikit-learn package not available, using Z-score method")
            return self._detect_anomalies_z_score(values, self.config.get('threshold', 3.0))
    
    def _detect_anomalies_autoencoder(self, values: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Autoencoder.
        
        Args:
            values: Time series values
            threshold: Reconstruction error threshold
            
        Returns:
            List of dictionaries containing anomaly information
        """
        try:
            # Try to import tensorflow
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            
            # Normalize values
            min_val = np.min(values)
            max_val = np.max(values)
            
            if max_val > min_val:
                normalized_values = (values - min_val) / (max_val - min_val)
            else:
                normalized_values = np.zeros_like(values)
            
            # Reshape values for tensorflow
            X = normalized_values.reshape(-1, 1)
            
            # Create autoencoder model
            model = Sequential([
                Dense(8, activation='relu', input_shape=(1,)),
                Dense(4, activation='relu'),
                Dense(4, activation='relu'),
                Dense(8, activation='relu'),
                Dense(1)
            ])
            
            # Compile model
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(X, X, epochs=50, batch_size=32, verbose=0)
            
            # Predict values
            predictions = model.predict(X)
            
            # Calculate reconstruction errors
            errors = np.abs(X - predictions)
            
            # Calculate error threshold
            error_threshold = np.mean(errors) + threshold * np.std(errors)
            
            # Detect anomalies
            anomaly_indices = np.where(errors > error_threshold)[0]
            
            # Create anomaly information
            anomalies = []
            for idx in anomaly_indices:
                # Calculate anomaly score
                score = float(errors[idx])
                
                # Calculate anomaly severity
                severity = self._calculate_anomaly_severity(score, error_threshold)
                
                anomalies.append({
                    'index': int(idx),
                    'value': float(values[idx]),
                    'score': score,
                    'type': 'point',
                    'method': 'autoencoder',
                    'severity': severity
                })
            
            return anomalies
        
        except ImportError:
            self.logger.warning("tensorflow package not available, using Z-score method")
            return self._detect_anomalies_z_score(values, self.config.get('threshold', 3.0))
    
    def _detect_contextual_anomalies(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect contextual anomalies in time-series data.
        
        Args:
            values: Time series values
            
        Returns:
            List of dictionaries containing contextual anomaly information
        """
        # Get window size
        window_size = self.config.get('window_size', 10)
        
        # Get threshold
        threshold = self.config.get('threshold', 3.0)
        
        # Check if we have enough data
        if len(values) <= window_size * 2:
            return []
        
        # Calculate rolling statistics
        anomalies = []
        
        for i in range(window_size, len(values) - window_size):
            # Get windows
            before_window = values[i-window_size:i]
            after_window = values[i+1:i+window_size+1]
            
            # Calculate window statistics
            before_mean = np.mean(before_window)
            before_std = np.std(before_window)
            after_mean = np.mean(after_window)
            after_std = np.std(after_window)
            
            # Calculate combined window statistics
            combined_window = np.concatenate([before_window, after_window])
            combined_mean = np.mean(combined_window)
            combined_std = np.std(combined_window)
            
            # Check if standard deviation is valid
            if combined_std == 0:
                continue
            
            # Calculate Z-score
            z_score = abs(values[i] - combined_mean) / combined_std
            
            # Check if point is anomalous
            if z_score > threshold:
                # Calculate anomaly score
                score = float(z_score)
                
                # Calculate anomaly severity
                severity = self._calculate_anomaly_severity(score, threshold)
                
                # Calculate context difference
                context_diff = abs(before_mean - after_mean) / max(before_std, after_std, 1e-10)
                
                anomalies.append({
                    'index': int(i),
                    'value': float(values[i]),
                    'score': score,
                    'type': 'contextual',
                    'method': 'z_score',
                    'severity': severity,
                    'context_diff': float(context_diff),
                    'before_mean': float(before_mean),
                    'after_mean': float(after_mean)
                })
        
        return anomalies
    
    def _detect_collective_anomalies(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect collective anomalies in time-series data.
        
        Args:
            values: Time series values
            
        Returns:
            List of dictionaries containing collective anomaly information
        """
        # Get window size
        window_size = self.config.get('window_size', 10)
        
        # Get threshold
        threshold = self.config.get('threshold', 3.0)
        
        # Check if we have enough data
        if len(values) <= window_size * 3:
            return []
        
        # Calculate rolling statistics
        anomalies = []
        
        for i in range(window_size, len(values) - window_size * 2):
            # Get windows
            current_window = values[i:i+window_size]
            reference_window = np.concatenate([values[i-window_size:i], values[i+window_size:i+window_size*2]])
            
            # Calculate window statistics
            current_mean = np.mean(current_window)
            current_std = np.std(current_window)
            reference_mean = np.mean(reference_window)
            reference_std = np.std(reference_window)
            
            # Check if standard deviation is valid
            if reference_std == 0:
                continue
            
            # Calculate window Z-score
            window_z_score = abs(current_mean - reference_mean) / reference_std
            
            # Check if window is anomalous
            if window_z_score > threshold:
                # Calculate anomaly score
                score = float(window_z_score)
                
                # Calculate anomaly severity
                severity = self._calculate_anomaly_severity(score, threshold)
                
                # Calculate additional metrics
                trend_diff = abs(np.mean(np.diff(current_window)) - np.mean(np.diff(reference_window)))
                var_ratio = current_std / reference_std if reference_std > 0 else float('inf')
                
                # Add anomaly for each point in the window
                for j in range(window_size):
                    idx = i + j
                    
                    # Calculate point-specific score
                    point_z_score = abs(values[idx] - reference_mean) / reference_std
                    point_score = float(point_z_score)
                    
                    # Calculate point-specific severity
                    point_severity = self._calculate_anomaly_severity(point_score, threshold)
                    
                    anomalies.append({
                        'index': int(idx),
                        'value': float(values[idx]),
                        'score': point_score,
                        'type': 'collective',
                        'method': 'window_z_score',
                        'severity': point_severity,
                        'window_score': score,
                        'window_severity': severity,
                        'window_start': int(i),
                        'window_end': int(i + window_size),
                        'trend_diff': float(trend_diff),
                        'var_ratio': float(var_ratio)
                    })
        
        return anomalies
    
    def _calculate_anomaly_severity(self, score: float, threshold: float) -> str:
        """
        Calculate anomaly severity based on score.
        
        Args:
            score: Anomaly score
            threshold: Anomaly threshold
            
        Returns:
            Anomaly severity ('low', 'medium', 'high', 'critical')
        """
        if score < threshold:
            return 'normal'
        elif score < threshold * 1.5:
            return 'low'
        elif score < threshold * 2.0:
            return 'medium'
        elif score < threshold * 3.0:
            return 'high'
        else:
            return 'critical'
    
    def _calculate_anomaly_statistics(self, values: np.ndarray, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for detected anomalies.
        
        Args:
            values: Time series values
            anomalies: List of anomalies
            
        Returns:
            Dictionary containing anomaly statistics
        """
        # Check if we have anomalies
        if not anomalies:
            return {
                'anomaly_rate': 0.0,
                'mean_score': 0.0,
                'max_score': 0.0,
                'severity_counts': {
                    'low': 0,
                    'medium': 0,
                    'high': 0,
                    'critical': 0
                },
                'type_counts': {
                    'point': 0,
                    'contextual': 0,
                    'collective': 0
                }
            }
        
        # Calculate anomaly rate
        anomaly_rate = float(len(anomalies) / len(values))
        
        # Calculate score statistics
        scores = [anomaly['score'] for anomaly in anomalies]
        mean_score = float(np.mean(