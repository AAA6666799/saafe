"""
Sequence Pattern Extractor for time-series data analysis.

This module provides a feature extractor that extracts sequence patterns in time-series data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import stumpy

from ...base_temporal import TemporalFeatureExtractor


class SequencePatternExtractor(TemporalFeatureExtractor):
    """
    Feature extractor for sequence patterns in time-series data.
    
    This class analyzes time-series data to identify recurring patterns,
    motifs, and characteristic sequences.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sequence pattern extractor.
        
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
        
        if 'pattern_detection_method' not in self.config:
            self.config['pattern_detection_method'] = 'matrix_profile'  # Options: 'matrix_profile', 'clustering', 'autocorrelation'
        
        if 'window_size' not in self.config:
            self.config['window_size'] = 10  # Window size for pattern detection
        
        if 'num_patterns' not in self.config:
            self.config['num_patterns'] = 3  # Number of patterns to extract
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 3
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract sequence pattern features from time-series data.
        
        Args:
            data: Input time-series data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting sequence pattern features")
        
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
        
        # Extract sequence pattern features
        pattern_features = self._extract_pattern_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'sequence_pattern_features': pattern_features
        }
        
        self.logger.info(f"Extracted sequence pattern features from {len(df)} samples")
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
        
        # Add sequence pattern features
        pattern_features = features.get('sequence_pattern_features', {})
        for feature_name, feature_value in pattern_features.items():
            if isinstance(feature_value, list):
                # Skip time series data in the flattened representation
                if len(feature_value) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_value):
                        flat_features[f"sequence_pattern_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"sequence_pattern_{feature_name}_mean"] = np.mean(feature_value)
                    flat_features[f"sequence_pattern_{feature_name}_max"] = np.max(feature_value)
                    flat_features[f"sequence_pattern_{feature_name}_min"] = np.min(feature_value)
            elif isinstance(feature_value, dict):
                for sub_name, sub_value in feature_value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 20:
                        # For large lists, just include summary statistics
                        flat_features[f"sequence_pattern_{feature_name}_{sub_name}_count"] = len(sub_value)
                    else:
                        flat_features[f"sequence_pattern_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"sequence_pattern_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved sequence pattern features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'pattern_count',
            'pattern_lengths',
            'pattern_similarities',
            'pattern_locations',
            'pattern_frequencies',
            'pattern_periodicity',
            'pattern_complexity',
            'pattern_variability',
            'pattern_significance',
            'pattern_coverage',
            'pattern_recurrence_rate',
            'pattern_entropy'
        ]
    
    def extract_sequence_patterns(self,
                                time_series: pd.DataFrame,
                                column_name: str,
                                window_size: int) -> Dict[str, Any]:
        """
        Extract sequence patterns from time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            window_size: Size of the window for pattern extraction
            
        Returns:
            Dictionary containing extracted sequence patterns
        """
        # Check if DataFrame has the required column
        if column_name not in time_series.columns:
            self.logger.warning(f"Column '{column_name}' not found in time series data")
            return {}
        
        # Get time series values
        values = time_series[column_name].values
        
        # Check if we have enough data
        if len(values) < window_size * 2:
            self.logger.warning(f"Not enough data for pattern extraction (need at least {window_size * 2} points)")
            return {}
        
        # Get pattern detection method
        method = self.config.get('pattern_detection_method', 'matrix_profile')
        
        # Extract patterns using the configured method
        if method == 'matrix_profile':
            patterns = self._extract_patterns_matrix_profile(values, window_size)
        elif method == 'clustering':
            patterns = self._extract_patterns_clustering(values, window_size)
        elif method == 'autocorrelation':
            patterns = self._extract_patterns_autocorrelation(values, window_size)
        else:
            self.logger.warning(f"Unknown pattern detection method: {method}, using matrix_profile")
            patterns = self._extract_patterns_matrix_profile(values, window_size)
        
        return patterns
    
    def analyze_trend(self,
                     time_series: pd.DataFrame,
                     column_name: str,
                     window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze trends in time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            window_size: Optional window size for trend analysis
            
        Returns:
            Dictionary containing trend analysis results
        """
        # Check if DataFrame has the required column
        if column_name not in time_series.columns:
            self.logger.warning(f"Column '{column_name}' not found in time series data")
            return {}
        
        # Get time series values
        values = time_series[column_name].values
        
        # Set default window size if not provided
        if window_size is None:
            window_size = min(len(values) // 4, 10)
        
        # Check if we have enough data
        if len(values) < window_size:
            self.logger.warning(f"Not enough data for trend analysis (need at least {window_size} points)")
            return {}
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if slope > 0.01:
            direction = 'rising'
        elif slope < -0.01:
            direction = 'falling'
        else:
            direction = 'stable'
        
        # Calculate trend strength (R-squared)
        trend_strength = r_value ** 2
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'direction': direction,
            'strength': float(trend_strength)
        }
    
    def detect_seasonality(self,
                          time_series: pd.DataFrame,
                          column_name: str,
                          period: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect seasonality in time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            period: Optional period length for seasonality detection
            
        Returns:
            Dictionary containing seasonality detection results
        """
        # Check if DataFrame has the required column
        if column_name not in time_series.columns:
            self.logger.warning(f"Column '{column_name}' not found in time series data")
            return {}
        
        # Get time series values
        values = time_series[column_name].values
        
        # Check if we have enough data
        if len(values) < 4:
            self.logger.warning("Not enough data for seasonality detection (need at least 4 points)")
            return {}
        
        # Detect period if not provided
        if period is None:
            # Calculate autocorrelation
            autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr, height=0.1)
            
            # Find first significant peak (excluding lag 0)
            if len(peaks) > 0 and peaks[0] > 0:
                period = int(peaks[0])
            else:
                period = 0
        
        # Check if we have enough data for the detected period
        if period <= 1 or len(values) < period * 2:
            return {
                'has_seasonality': False,
                'period': 0,
                'strength': 0.0,
                'significance': 0.0
            }
        
        try:
            # Apply STL decomposition
            from statsmodels.tsa.seasonal import STL
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
            
            # Calculate seasonality significance (F-test)
            from scipy import stats
            
            # Calculate variance explained by seasonality
            var_explained = np.var(seasonal)
            var_total = np.var(values)
            
            # Calculate F-statistic
            if var_resid > 0 and var_explained > 0:
                f_statistic = var_explained / var_resid
                
                # Calculate degrees of freedom
                df1 = period - 1  # Seasonality component
                df2 = len(values) - period  # Residual
                
                # Calculate p-value
                p_value = 1.0 - stats.f.cdf(f_statistic, df1, df2)
                
                # Calculate significance (1 - p_value)
                significance = 1.0 - p_value
            else:
                significance = 0.0
            
            return {
                'has_seasonality': seasonality_strength > 0.1,
                'period': period,
                'strength': float(seasonality_strength),
                'significance': float(significance)
            }
        
        except Exception as e:
            self.logger.warning(f"Error detecting seasonality: {str(e)}")
            return {
                'has_seasonality': False,
                'period': 0,
                'strength': 0.0,
                'significance': 0.0
            }
    
    def detect_change_points(self,
                            time_series: pd.DataFrame,
                            column_name: str,
                            threshold: float) -> List[Dict[str, Any]]:
        """
        Detect change points in time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            threshold: Threshold for change point detection
            
        Returns:
            List of dictionaries containing change point information
        """
        # Check if DataFrame has the required column
        if column_name not in time_series.columns:
            self.logger.warning(f"Column '{column_name}' not found in time series data")
            return []
        
        # Get time series values
        values = time_series[column_name].values
        
        # Check if we have enough data
        if len(values) < 10:
            self.logger.warning("Not enough data for change point detection (need at least 10 points)")
            return []
        
        try:
            # Use ruptures for change point detection
            import ruptures as rpt
            
            # Create change point detection model
            model = "l2"  # L2 norm
            algo = rpt.Pelt(model=model).fit(values.reshape(-1, 1))
            
            # Find change points
            change_points = algo.predict(pen=threshold)
            
            # Remove the last change point if it's the end of the series
            if change_points and change_points[-1] == len(values):
                change_points = change_points[:-1]
            
            # Create change point information
            change_point_info = []
            for cp in change_points:
                # Calculate change metrics
                if cp > 1 and cp < len(values):
                    before_mean = np.mean(values[cp-1:cp])
                    after_mean = np.mean(values[cp:cp+1])
                    change_magnitude = abs(after_mean - before_mean)
                    
                    change_point_info.append({
                        'index': int(cp),
                        'timestamp': time_series['timestamp'].iloc[cp] if 'timestamp' in time_series.columns else None,
                        'value': float(values[cp]),
                        'change_magnitude': float(change_magnitude),
                        'before_mean': float(before_mean),
                        'after_mean': float(after_mean)
                    })
            
            return change_point_info
        
        except ImportError:
            self.logger.warning("ruptures package not available, using simple change point detection")
            
            # Simple change point detection using rolling statistics
            window_size = 5
            change_points = []
            
            if len(values) <= window_size * 2:
                return []
            
            # Calculate rolling mean and standard deviation
            rolling_mean = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            
            # Calculate differences between consecutive means
            mean_diff = np.abs(np.diff(rolling_mean))
            
            # Find points where difference exceeds threshold
            cp_indices = np.where(mean_diff > threshold)[0]
            
            # Create change point information
            change_point_info = []
            for cp in cp_indices:
                # Adjust index to account for rolling window
                adjusted_cp = cp + window_size // 2
                
                # Calculate change metrics
                if adjusted_cp > 0 and adjusted_cp < len(values) - 1:
                    before_mean = np.mean(values[adjusted_cp-window_size:adjusted_cp])
                    after_mean = np.mean(values[adjusted_cp:adjusted_cp+window_size])
                    change_magnitude = abs(after_mean - before_mean)
                    
                    change_point_info.append({
                        'index': int(adjusted_cp),
                        'timestamp': time_series['timestamp'].iloc[adjusted_cp] if 'timestamp' in time_series.columns else None,
                        'value': float(values[adjusted_cp]),
                        'change_magnitude': float(change_magnitude),
                        'before_mean': float(before_mean),
                        'after_mean': float(after_mean)
                    })
            
            return change_point_info
    
    def detect_anomalies(self,
                        time_series: pd.DataFrame,
                        column_name: str,
                        window_size: int,
                        threshold: float) -> List[Dict[str, Any]]:
        """
        Detect anomalies in time-series data.
        
        Args:
            time_series: DataFrame containing time-series data
            column_name: Name of the column to analyze
            window_size: Size of the window for anomaly detection
            threshold: Threshold for anomaly detection
            
        Returns:
            List of dictionaries containing anomaly information
        """
        # Check if DataFrame has the required column
        if column_name not in time_series.columns:
            self.logger.warning(f"Column '{column_name}' not found in time series data")
            return []
        
        # Get time series values
        values = time_series[column_name].values
        
        # Check if we have enough data
        if len(values) < window_size:
            self.logger.warning(f"Not enough data for anomaly detection (need at least {window_size} points)")
            return []
        
        # Calculate Z-scores using rolling window
        z_scores = np.zeros_like(values)
        
        for i in range(window_size, len(values)):
            # Get window
            window = values[i-window_size:i]
            
            # Calculate window statistics
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            # Calculate Z-score
            if window_std > 0:
                z_scores[i] = abs(values[i] - window_mean) / window_std
            else:
                z_scores[i] = 0.0
        
        # Detect anomalies
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        # Create anomaly information
        anomalies = []
        for idx in anomaly_indices:
            anomalies.append({
                'index': int(idx),
                'timestamp': time_series['timestamp'].iloc[idx] if 'timestamp' in time_series.columns else None,
                'value': float(values[idx]),
                'z_score': float(z_scores[idx])
            })
        
        return anomalies
    
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
    
    def _extract_patterns_matrix_profile(self, values: np.ndarray, window_size: int) -> Dict[str, Any]:
        """
        Extract patterns using Matrix Profile.
        
        Args:
            values: Time series values
            window_size: Window size for pattern extraction
            
        Returns:
            Dictionary containing pattern features
        """
        try:
            # Calculate Matrix Profile
            matrix_profile = stumpy.stump(values, window_size)
            
            # Extract motifs (patterns)
            num_patterns = self.config.get('num_patterns', 3)
            motifs = stumpy.motifs(values, matrix_profile[:, 0], window_size, n_motifs=num_patterns)
            
            # Extract pattern information
            pattern_indices = motifs[0]
            pattern_neighbors = motifs[1]
            
            # Calculate pattern features
            patterns = []
            pattern_locations = []
            pattern_similarities = []
            
            for i, idx in enumerate(pattern_indices):
                # Extract pattern
                pattern = values[idx:idx+window_size]
                
                # Get pattern neighbors
                neighbors = pattern_neighbors[i]
                
                # Calculate pattern similarity
                similarities = []
                for neighbor_idx in neighbors:
                    neighbor = values[neighbor_idx:neighbor_idx+window_size]
                    similarity = 1.0 - np.mean(np.abs(pattern - neighbor)) / (np.max(pattern) - np.min(pattern) + 1e-10)
                    similarities.append(similarity)
                
                patterns.append(pattern.tolist())
                pattern_locations.append([int(idx)] + [int(n) for n in neighbors])
                pattern_similarities.append(float(np.mean(similarities)) if similarities else 1.0)
            
            # Calculate pattern complexity (entropy)
            pattern_complexity = []
            for pattern in patterns:
                # Calculate entropy
                hist, _ = np.histogram(pattern, bins=10)
                hist = hist / np.sum(hist)
                entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
                pattern_complexity.append(float(entropy))
            
            # Calculate pattern variability
            pattern_variability = []
            for pattern in patterns:
                variability = float(np.std(pattern) / (np.mean(pattern) + 1e-10))
                pattern_variability.append(variability)
            
            # Calculate pattern significance
            pattern_significance = []
            for i, idx in enumerate(pattern_indices):
                # Calculate Z-score of pattern
                pattern_mean = np.mean(values[idx:idx+window_size])
                overall_mean = np.mean(values)
                overall_std = np.std(values)
                
                if overall_std > 0:
                    z_score = abs(pattern_mean - overall_mean) / overall_std
                else:
                    z_score = 0.0
                
                pattern_significance.append(float(z_score))
            
            # Calculate pattern coverage
            total_covered = set()
            for locs in pattern_locations:
                for loc in locs:
                    for i in range(loc, min(loc + window_size, len(values))):
                        total_covered.add(i)
            
            pattern_coverage = float(len(total_covered) / len(values))
            
            # Calculate pattern recurrence rate
            pattern_recurrence = []
            for locs in pattern_locations:
                recurrence = float(len(locs) / (len(values) - window_size + 1))
                pattern_recurrence.append(recurrence)
            
            # Calculate pattern periodicity
            pattern_periodicity = []
            for locs in pattern_locations:
                if len(locs) > 1:
                    # Calculate intervals between occurrences
                    intervals = np.diff(sorted(locs))
                    
                    # Calculate coefficient of variation (lower means more periodic)
                    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
                    
                    # Convert to periodicity (1 = perfectly periodic, 0 = not periodic)
                    periodicity = float(1.0 / (1.0 + cv))
                else:
                    periodicity = 0.0
                
                pattern_periodicity.append(periodicity)
            
            return {
                'pattern_count': len(patterns),
                'pattern_lengths': [window_size] * len(patterns),
                'pattern_similarities': pattern_similarities,
                'pattern_locations': pattern_locations,
                'pattern_frequencies': [float(len(locs) / len(values)) for locs in pattern_locations],
                'pattern_periodicity': pattern_periodicity,
                'pattern_complexity': pattern_complexity,
                'pattern_variability': pattern_variability,
                'pattern_significance': pattern_significance,
                'pattern_coverage': pattern_coverage,
                'pattern_recurrence_rate': pattern_recurrence,
                'pattern_entropy': float(np.mean(pattern_complexity)) if pattern_complexity else 0.0,
                'patterns': patterns
            }
        
        except Exception as e:
            self.logger.warning(f"Error extracting patterns using Matrix Profile: {str(e)}")
            return {
                'pattern_count': 0,
                'pattern_lengths': [],
                'pattern_similarities': [],
                'pattern_locations': [],
                'pattern_frequencies': [],
                'pattern_periodicity': [],
                'pattern_complexity': [],
                'pattern_variability': [],
                'pattern_significance': [],
                'pattern_coverage': 0.0,
                'pattern_recurrence_rate': [],
                'pattern_entropy': 0.0,
                'patterns': []
            }
    
    def _extract_patterns_clustering(self, values: np.ndarray, window_size: int) -> Dict[str, Any]:
        """
        Extract patterns using clustering of sliding windows.
        
        Args:
            values: Time series values
            window_size: Window size for pattern extraction
            
        Returns:
            Dictionary containing pattern features
        """
        # Check if we have enough data
        if len(values) < window_size * 2:
            return {
                'pattern_count': 0,
                'pattern_lengths': [],
                'pattern_similarities': [],
                'pattern_locations': [],
                'pattern_frequencies': [],
                'pattern_periodicity': [],
                'pattern_complexity': [],
                'pattern_variability': [],
                'pattern_significance': [],
                'pattern_coverage': 0.0,
                'pattern_recurrence_rate': [],
                'pattern_entropy': 0.0,
                'patterns': []
            }
        
        # Create sliding windows
        windows = []
        indices = []
        
        for i in range(len(values) - window_size + 1):
            window = values[i:i+window_size]
            windows.append(window)
            indices.append(i)
        
        # Standardize windows
        scaler = StandardScaler()
        windows_scaled = scaler.fit_transform(windows)
        
        # Apply clustering
        num_patterns = self.config.get('num_patterns', 3)
        num_clusters = min(num_patterns, len(windows))
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(windows_scaled)
        
        # Extract pattern information
        patterns = []
        pattern_locations = []
        pattern_similarities = []
        
        for cluster_id in range(num_clusters):
            # Get indices of windows in this cluster
            cluster_indices = [indices[i] for i, c in enumerate(clusters) if c == cluster_id]
            
            if cluster_indices:
                # Use the window closest to the cluster center as the pattern
                cluster_windows = [windows[i] for i, c in enumerate(clusters) if c == cluster_id]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                # Calculate distances to center
                distances = [np.linalg.norm(window - cluster_center) for window in cluster_windows]
                
                # Find window closest to center
                closest_idx = np.argmin(distances)
                pattern = cluster_windows[closest_idx]
                
                # Calculate pattern similarity
                similarities = []
                for window in cluster_windows:
                    similarity = 1.0 - np.mean(np.abs(pattern - window)) / (np.max(pattern) - np.min(pattern) + 1e-10)
                    similarities.append(similarity)
                
                patterns.append(pattern.tolist())
                pattern_locations.append([int(idx) for idx in cluster_indices])
                pattern_similarities.append(float(np.mean(similarities)) if similarities else 1.0)
        
        # Calculate pattern complexity (entropy)
        pattern_complexity = []
        for pattern in patterns:
            # Calculate entropy
            hist, _ = np.histogram(pattern, bins=10)
            hist = hist / np.sum(hist)
            entropy =