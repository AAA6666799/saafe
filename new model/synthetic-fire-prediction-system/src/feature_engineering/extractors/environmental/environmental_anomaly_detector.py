"""
Environmental Anomaly Detector for environmental data analysis.

This module provides a feature extractor that detects anomalies in environmental data.
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

from ...base import EnvironmentalFeatureExtractor


class EnvironmentalAnomalyDetector(EnvironmentalFeatureExtractor):
    """
    Feature extractor for detecting anomalies in environmental data.
    
    This class analyzes environmental data to identify anomalous patterns,
    unusual values, and outliers in environmental parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environmental anomaly detector.
        
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
        
        if 'multivariate_anomaly_detection' not in self.config:
            self.config['multivariate_anomaly_detection'] = True
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract anomaly features from environmental data.
        
        Args:
            data: Input environmental data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting environmental anomaly features")
        
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
        
        # Extract anomaly features
        anomaly_features = self._extract_anomaly_features(df_processed, available_params)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'parameters_analyzed': available_params,
            'anomaly_features': anomaly_features
        }
        
        self.logger.info(f"Extracted environmental anomaly features from {len(df)} samples")
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
        
        # Add univariate anomaly features
        for param_name, param_features in anomaly_features.get('univariate', {}).items():
            for feature_name, feature_value in param_features.items():
                if isinstance(feature_value, list):
                    # Skip time series data in the flattened representation
                    if len(feature_value) <= 20:  # Only include reasonably sized lists
                        for i, value in enumerate(feature_value):
                            flat_features[f"{param_name}_anomaly_{feature_name}_{i}"] = value
                    else:
                        # For large lists, just include summary statistics
                        flat_features[f"{param_name}_anomaly_{feature_name}_mean"] = np.mean(feature_value)
                        flat_features[f"{param_name}_anomaly_{feature_name}_max"] = np.max(feature_value)
                        flat_features[f"{param_name}_anomaly_{feature_name}_min"] = np.min(feature_value)
                else:
                    flat_features[f"{param_name}_anomaly_{feature_name}"] = feature_value
        
        # Add multivariate anomaly features
        multivariate_features = anomaly_features.get('multivariate', {})
        for feature_name, feature_value in multivariate_features.items():
            if isinstance(feature_value, list):
                # Skip time series data in the flattened representation
                if len(feature_value) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_value):
                        flat_features[f"multivariate_anomaly_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"multivariate_anomaly_{feature_name}_mean"] = np.mean(feature_value)
                    flat_features[f"multivariate_anomaly_{feature_name}_max"] = np.max(feature_value)
                    flat_features[f"multivariate_anomaly_{feature_name}_min"] = np.min(feature_value)
            else:
                flat_features[f"multivariate_anomaly_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved environmental anomaly features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        # Base feature names for univariate anomalies
        univariate_features = [
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
        
        # Base feature names for multivariate anomalies
        multivariate_features = [
            'anomaly_score_mean',
            'anomaly_score_max',
            'anomaly_count',
            'anomaly_percentage',
            'anomaly_severity_mean',
            'anomaly_severity_max',
            'correlation_anomaly_count',
            'joint_anomaly_count'
        ]
        
        # Generate feature names for each parameter
        feature_names = []
        for param in self.config.get('parameters', []):
            for feature in univariate_features:
                feature_names.append(f"{param}_anomaly_{feature}")
        
        # Add multivariate feature names
        for feature in multivariate_features:
            feature_names.append(f"multivariate_anomaly_{feature}")
        
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
        window = self.config.get('smoothing_window', 3)
        
        # Apply smoothing to each column
        for col in columns:
            if col in df.columns:
                # Apply moving average smoothing
                df_smoothed[col] = df[col].rolling(window=window, center=True).mean()
                
                # Fill NaN values at the edges
                df_smoothed[col] = df_smoothed[col].fillna(df[col])
        
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
    
    def _detect_multivariate_anomalies(self, df: pd.DataFrame, parameters: List[str]) -> Dict[str, Any]:
        """
        Detect multivariate anomalies across multiple parameters.
        
        Args:
            df: DataFrame containing environmental readings
            parameters: List of parameters to analyze
            
        Returns:
            Dictionary containing multivariate anomaly features
        """
        # Check if we have enough parameters
        if len(parameters) < 2:
            return {
                'anomaly_scores': [],
                'anomaly_mask': [],
                'anomaly_count': 0,
                'anomaly_percentage': 0.0,
                'correlation_anomaly_count': 0,
                'joint_anomaly_count': 0
            }
        
        # Extract parameter values
        X = df[parameters].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get anomaly detection method
        method = self.config.get('anomaly_detection_method', 'zscore')
        
        try:
            if method == 'isolation_forest':
                # Use Isolation Forest for multivariate anomaly detection
                contamination = self.config.get('contamination', 0.05)
                n_estimators = self.config.get('n_estimators', 100)
                
                clf = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
                clf.fit(X_scaled)
                
                # Get anomaly scores and mask
                anomaly_scores = -clf.decision_function(X_scaled)
                anomaly_mask = clf.predict(X_scaled) == -1
            
            elif method == 'lof':
                # Use LOF for multivariate anomaly detection
                contamination = self.config.get('contamination', 0.05)
                n_neighbors = self.config.get('n_neighbors', 20)
                
                clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
                y_pred = clf.fit_predict(X_scaled)
                
                # Get anomaly scores and mask
                anomaly_scores = -clf._decision_function(X_scaled)
                anomaly_mask = y_pred == -1
            
            else:  # 'zscore' or other
                # Calculate Mahalanobis distance for multivariate Z-score
                mean_vec = np.mean(X_scaled, axis=0)
                cov_matrix = np.cov(X_scaled, rowvar=False)
                
                # Add small regularization to ensure invertibility
                cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
                
                # Calculate Mahalanobis distance
                inv_cov = np.linalg.inv(cov_matrix)
                mahalanobis_dist = []
                
                for i in range(X_scaled.shape[0]):
                    x = X_scaled[i, :]
                    dist = np.sqrt(np.dot(np.dot((x - mean_vec), inv_cov), (x - mean_vec).T))
                    mahalanobis_dist.append(dist)
                
                anomaly_scores = np.array(mahalanobis_dist)
                
                # Create anomaly mask
                threshold = self.config.get('zscore_threshold', 3.0)
                anomaly_mask = anomaly_scores > threshold
        
        except Exception as e:
            self.logger.warning(f"Multivariate anomaly detection failed: {str(e)}")
            anomaly_scores = np.zeros(len(df))
            anomaly_mask = np.zeros(len(df), dtype=bool)
        
        # Calculate anomaly statistics
        anomaly_count = int(np.sum(anomaly_mask))
        anomaly_percentage = float(np.mean(anomaly_mask) * 100.0)
        
        # Calculate anomaly severity
        anomaly_severity = anomaly_scores[anomaly_mask] if anomaly_count > 0 else np.array([0.0])
        anomaly_severity_mean = float(np.mean(anomaly_severity))
        anomaly_severity_max = float(np.max(anomaly_severity))
        
        # Detect correlation anomalies
        correlation_anomalies = self._detect_correlation_anomalies(df, parameters)
        correlation_anomaly_count = len(correlation_anomalies)
        
        # Detect joint anomalies (points that are anomalous in multiple parameters)
        joint_anomaly_count = self._count_joint_anomalies(df, parameters)
        
        return {
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_mask': anomaly_mask.tolist(),
            'anomaly_count': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'anomaly_severity_mean': anomaly_severity_mean,
            'anomaly_severity_max': anomaly_severity_max,
            'correlation_anomaly_count': correlation_anomaly_count,
            'joint_anomaly_count': joint_anomaly_count
        }
    
    def _detect_correlation_anomalies(self, df: pd.DataFrame, parameters: List[str]) -> List[Tuple[int, int, float]]:
        """
        Detect anomalies in the correlation between parameters.
        
        Args:
            df: DataFrame containing environmental readings
            parameters: List of parameters to analyze
            
        Returns:
            List of (start_idx, end_idx, correlation) tuples for correlation anomalies
        """
        # Check if we have enough parameters and data points
        if len(parameters) < 2 or len(df) < 10:
            return []
        
        window_size = self.config.get('window_size', 10)
        if len(df) <= window_size * 2:
            return []
        
        # Calculate rolling correlations between all parameter pairs
        correlation_anomalies = []
        
        for i in range(len(parameters)):
            for j in range(i+1, len(parameters)):
                param1 = parameters[i]
                param2 = parameters[j]
                
                # Calculate baseline correlation
                baseline_corr, _ = stats.pearsonr(df[param1].values, df[param2].values)
                
                # Calculate rolling correlations
                rolling_corrs = []
                for k in range(len(df) - window_size + 1):
                    window_df = df.iloc[k:k+window_size]
                    try:
                        corr, _ = stats.pearsonr(window_df[param1].values, window_df[param2].values)
                        rolling_corrs.append(corr)
                    except:
                        rolling_corrs.append(0.0)
                
                # Detect anomalies in correlation
                for k in range(len(rolling_corrs)):
                    if abs(rolling_corrs[k] - baseline_corr) > 0.5:  # Significant correlation change
                        correlation_anomalies.append((k, k + window_size - 1, rolling_corrs[k]))
        
        return correlation_anomalies
    
    def _count_joint_anomalies(self, df: pd.DataFrame, parameters: List[str]) -> int:
        """
        Count points that are anomalous in multiple parameters.
        
        Args:
            df: DataFrame containing environmental readings
            parameters: List of parameters to analyze
            
        Returns:
            Count of joint anomalies
        """
        # Get anomaly detection method
        method = self.config.get('anomaly_detection_method', 'zscore')
        
        # Detect anomalies in each parameter
        anomaly_masks = {}
        for param in parameters:
            values = df[param].values
            
            if method == 'isolation_forest':
                _, anomaly_mask = self._detect_isolation_forest_anomalies(values)
            elif method == 'lof':
                _, anomaly_mask = self._detect_lof_anomalies(values)
            else:  # 'zscore' or other
                _, anomaly_mask = self._detect_zscore_anomalies(values)
            
            anomaly_masks[param] = anomaly_mask
        
        # Count points that are anomalous in at least 2 parameters
        joint_anomalies = np.zeros(len(df), dtype=bool)
        
        for i in range(len(df)):
            anomaly_count = sum(1 for param in parameters if anomaly_masks[param][i])
            if anomaly_count >= 2:
                joint_anomalies[i] = True
        
        return int(np.sum(joint_anomalies))
    
    def _extract_anomaly_features(self, df: pd.DataFrame, parameters: List[str]) -> Dict[str, Any]:
        """
        Extract anomaly features for environmental parameters.
        
        Args:
            df: DataFrame containing environmental readings
            parameters: List of parameters to analyze
            
        Returns:
            Dictionary containing anomaly features
        """
        # Get anomaly detection method
        method = self.config.get('anomaly_detection_method', 'zscore')
        
        # Initialize results
        univariate_features = {}
        
        # Extract univariate anomaly features for each parameter
        for param in parameters:
            values = df[param].values
            
            # Detect anomalies using the configured method
            if method == 'isolation_forest':
                anomaly_scores, anomaly_mask = self._detect_isolation_forest_anomalies(values)
            elif method == 'lof':
                anomaly_scores, anomaly_mask = self._detect_lof_anomalies(values)
            else:  # 'zscore' or other
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
            univariate_features[param] = {
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
        
        # Extract multivariate anomaly features if configured
        multivariate_features = {}
        if self.config.get('multivariate_anomaly_detection', True) and len(parameters) >= 2:
            multivariate_features = self._detect_multivariate_anomalies(df, parameters)
        
        return {
            'univariate': univariate_features,
            'multivariate': multivariate_features
        }