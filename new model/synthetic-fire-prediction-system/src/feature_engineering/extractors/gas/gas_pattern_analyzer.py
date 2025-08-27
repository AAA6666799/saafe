"""
Gas Pattern Analyzer for gas sensor data analysis.

This module provides a feature extractor that analyzes patterns in gas concentration data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal, fft
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ...base import GasFeatureExtractor


class GasPatternAnalyzer(GasFeatureExtractor):
    """
    Feature extractor for analyzing patterns in gas concentration data.
    
    This class analyzes gas concentration data to identify recurring patterns,
    temporal structures, and characteristic signatures in gas behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gas pattern analyzer.
        
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
        if 'pattern_detection_method' not in self.config:
            self.config['pattern_detection_method'] = 'fft'  # Options: 'fft', 'autocorrelation', 'clustering'
        
        if 'window_size' not in self.config:
            self.config['window_size'] = 60  # Window size for pattern detection (in samples)
        
        if 'overlap' not in self.config:
            self.config['overlap'] = 0.5  # Overlap between windows (as a fraction)
        
        if 'num_clusters' not in self.config:
            self.config['num_clusters'] = 3  # Number of clusters for pattern clustering
        
        if 'min_pattern_duration' not in self.config:
            self.config['min_pattern_duration'] = 5  # Minimum duration for a pattern (in samples)
        
        if 'max_pattern_duration' not in self.config:
            self.config['max_pattern_duration'] = 120  # Maximum duration for a pattern (in samples)
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 5
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract pattern features from gas data.
        
        Args:
            data: Input gas data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting gas pattern features")
        
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
        
        # Extract pattern features
        pattern_features = self._extract_pattern_features(df_processed, gas_types)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'gas_types': gas_types,
            'pattern_features': pattern_features
        }
        
        self.logger.info(f"Extracted gas pattern features from {len(df)} samples")
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
        
        # Add pattern features
        pattern_features = features.get('pattern_features', {})
        for gas_type, gas_features in pattern_features.items():
            for feature_name, feature_value in gas_features.items():
                if isinstance(feature_value, list):
                    # Skip time series data in the flattened representation
                    if len(feature_value) <= 20:  # Only include reasonably sized lists
                        for i, value in enumerate(feature_value):
                            flat_features[f"{gas_type}_{feature_name}_{i}"] = value
                    else:
                        # For large lists, just include summary statistics
                        flat_features[f"{gas_type}_{feature_name}_mean"] = np.mean(feature_value)
                        flat_features[f"{gas_type}_{feature_name}_max"] = np.max(feature_value)
                        flat_features[f"{gas_type}_{feature_name}_min"] = np.min(feature_value)
                elif isinstance(feature_value, dict):
                    for sub_name, sub_value in feature_value.items():
                        if isinstance(sub_value, list) and len(sub_value) > 20:
                            # For large lists, just include summary statistics
                            flat_features[f"{gas_type}_{feature_name}_{sub_name}_count"] = len(sub_value)
                        else:
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
        
        self.logger.info(f"Saved gas pattern features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        # Base feature names (will be prefixed with gas type)
        base_features = [
            'dominant_frequency',
            'dominant_period',
            'spectral_entropy',
            'spectral_flatness',
            'spectral_centroid',
            'spectral_bandwidth',
            'autocorrelation_peak',
            'autocorrelation_lag',
            'pattern_count',
            'pattern_duration_mean',
            'pattern_duration_std',
            'pattern_similarity',
            'pattern_regularity',
            'pattern_complexity',
            'trend_strength',
            'seasonality_strength',
            'residual_strength'
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
        window = self.config.get('smoothing_window', 5)
        
        # Apply smoothing to each gas type
        for gas in gas_types:
            if gas in df.columns:
                # Apply moving average smoothing
                df_smoothed[gas] = df[gas].rolling(window=window, center=True).mean()
                
                # Fill NaN values at the edges
                df_smoothed[gas] = df_smoothed[gas].fillna(df[gas])
        
        return df_smoothed
    
    def _calculate_fft_features(self, values: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, Any]:
        """
        Calculate FFT-based features for a time series.
        
        Args:
            values: Time series values
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary containing FFT features
        """
        # Apply FFT
        n = len(values)
        fft_values = fft.fft(values - np.mean(values))
        fft_magnitudes = np.abs(fft_values[:n//2])
        
        # Calculate frequencies
        freqs = fft.fftfreq(n, 1/sampling_rate)[:n//2]
        
        # Find dominant frequency
        if len(freqs) > 0 and len(fft_magnitudes) > 0:
            dominant_idx = np.argmax(fft_magnitudes)
            dominant_freq = freqs[dominant_idx]
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else n
        else:
            dominant_freq = 0.0
            dominant_period = 0.0
        
        # Calculate spectral entropy
        if np.sum(fft_magnitudes) > 0:
            normalized_magnitudes = fft_magnitudes / np.sum(fft_magnitudes)
            spectral_entropy = -np.sum(normalized_magnitudes * np.log2(normalized_magnitudes + 1e-10))
        else:
            spectral_entropy = 0.0
        
        # Calculate spectral flatness
        if np.prod(fft_magnitudes + 1e-10) > 0 and np.mean(fft_magnitudes) > 0:
            spectral_flatness = np.exp(np.mean(np.log(fft_magnitudes + 1e-10))) / np.mean(fft_magnitudes)
        else:
            spectral_flatness = 0.0
        
        # Calculate spectral centroid
        if np.sum(fft_magnitudes) > 0:
            spectral_centroid = np.sum(freqs * fft_magnitudes) / np.sum(fft_magnitudes)
        else:
            spectral_centroid = 0.0
        
        # Calculate spectral bandwidth
        if np.sum(fft_magnitudes) > 0:
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_magnitudes) / np.sum(fft_magnitudes))
        else:
            spectral_bandwidth = 0.0
        
        return {
            'dominant_frequency': float(dominant_freq),
            'dominant_period': float(dominant_period),
            'spectral_entropy': float(spectral_entropy),
            'spectral_flatness': float(spectral_flatness),
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'fft_magnitudes': fft_magnitudes.tolist()[:20]  # Only include first 20 components
        }
    
    def _calculate_autocorrelation_features(self, values: np.ndarray, max_lag: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate autocorrelation-based features for a time series.
        
        Args:
            values: Time series values
            max_lag: Maximum lag to consider
            
        Returns:
            Dictionary containing autocorrelation features
        """
        # Set default max_lag if not provided
        if max_lag is None:
            max_lag = min(len(values) - 1, 100)
        
        # Calculate autocorrelation
        autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        autocorr = autocorr[:max_lag+1]  # Limit to max_lag
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr, height=0.1)
        
        # Find first significant peak (excluding lag 0)
        peak_lag = 0
        peak_value = 0.0
        
        if len(peaks) > 0 and peaks[0] > 0:
            peak_lag = int(peaks[0])
            peak_value = float(autocorr[peak_lag])
        
        return {
            'autocorrelation': autocorr.tolist(),
            'autocorrelation_peak': peak_value,
            'autocorrelation_lag': peak_lag
        }
    
    def _detect_patterns_clustering(self, values: np.ndarray, window_size: int, overlap: float) -> Dict[str, Any]:
        """
        Detect patterns using clustering of sliding windows.
        
        Args:
            values: Time series values
            window_size: Size of sliding window
            overlap: Overlap between windows (as a fraction)
            
        Returns:
            Dictionary containing pattern features
        """
        if len(values) < window_size:
            return {
                'pattern_count': 0,
                'pattern_durations': [],
                'pattern_indices': [],
                'pattern_similarity': 0.0
            }
        
        # Create sliding windows
        step = int(window_size * (1 - overlap))
        if step < 1:
            step = 1
        
        windows = []
        indices = []
        
        for i in range(0, len(values) - window_size + 1, step):
            window = values[i:i+window_size]
            windows.append(window)
            indices.append(i)
        
        if not windows:
            return {
                'pattern_count': 0,
                'pattern_durations': [],
                'pattern_indices': [],
                'pattern_similarity': 0.0
            }
        
        # Standardize windows
        scaler = StandardScaler()
        windows_scaled = scaler.fit_transform(windows)
        
        # Apply dimensionality reduction if window size is large
        if window_size > 10:
            pca = PCA(n_components=min(10, window_size))
            windows_reduced = pca.fit_transform(windows_scaled)
        else:
            windows_reduced = windows_scaled
        
        # Apply clustering
        n_clusters = min(self.config.get('num_clusters', 3), len(windows))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        try:
            clusters = kmeans.fit_predict(windows_reduced)
        except Exception as e:
            self.logger.warning(f"Clustering failed: {str(e)}")
            return {
                'pattern_count': 0,
                'pattern_durations': [],
                'pattern_indices': [],
                'pattern_similarity': 0.0
            }
        
        # Identify patterns
        patterns = []
        for cluster_id in range(n_clusters):
            # Get indices of windows in this cluster
            cluster_indices = [indices[i] for i, c in enumerate(clusters) if c == cluster_id]
            
            # Group consecutive indices into patterns
            if cluster_indices:
                pattern_groups = []
                current_group = [cluster_indices[0]]
                
                for idx in cluster_indices[1:]:
                    if idx - current_group[-1] <= window_size:
                        # Continue current pattern
                        current_group.append(idx)
                    else:
                        # Start new pattern
                        if len(current_group) >= self.config.get('min_pattern_duration', 5) / step:
                            pattern_groups.append(current_group)
                        current_group = [idx]
                
                # Add last group if it's long enough
                if len(current_group) >= self.config.get('min_pattern_duration', 5) / step:
                    pattern_groups.append(current_group)
                
                # Calculate pattern durations
                for group in pattern_groups:
                    start_idx = group[0]
                    end_idx = group[-1] + window_size
                    duration = end_idx - start_idx
                    
                    if duration <= self.config.get('max_pattern_duration', 120):
                        patterns.append({
                            'start': start_idx,
                            'end': end_idx,
                            'duration': duration,
                            'cluster': cluster_id
                        })
        
        # Calculate pattern similarity
        if patterns:
            # Calculate average distance to cluster center for each pattern
            similarities = []
            for pattern in patterns:
                cluster_id = pattern['cluster']
                center = kmeans.cluster_centers_[cluster_id]
                
                # Find windows in this pattern
                pattern_windows = []
                for i, idx in enumerate(indices):
                    if pattern['start'] <= idx < pattern['end']:
                        pattern_windows.append(windows_reduced[i])
                
                if pattern_windows:
                    # Calculate average distance to center
                    distances = [np.linalg.norm(window - center) for window in pattern_windows]
                    avg_distance = np.mean(distances)
                    max_distance = np.max([np.linalg.norm(center) for center in kmeans.cluster_centers_])
                    
                    if max_distance > 0:
                        similarity = 1.0 - (avg_distance / max_distance)
                    else:
                        similarity = 1.0
                    
                    similarities.append(similarity)
            
            pattern_similarity = float(np.mean(similarities)) if similarities else 0.0
        else:
            pattern_similarity = 0.0
        
        return {
            'pattern_count': len(patterns),
            'pattern_durations': [p['duration'] for p in patterns],
            'pattern_indices': [(p['start'], p['end']) for p in patterns],
            'pattern_similarity': pattern_similarity
        }
    
    def _calculate_decomposition_features(self, values: np.ndarray) -> Dict[str, float]:
        """
        Calculate time series decomposition features.
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing decomposition features
        """
        # Check if we have enough data points
        if len(values) < 10:
            return {
                'trend_strength': 0.0,
                'seasonality_strength': 0.0,
                'residual_strength': 0.0
            }
        
        try:
            # Determine period using autocorrelation
            autocorr_features = self._calculate_autocorrelation_features(values)
            period = autocorr_features['autocorrelation_lag']
            
            # Use default period if autocorrelation didn't find one
            if period < 2:
                period = min(len(values) // 2, 10)
            
            # Apply STL decomposition if we have enough data
            if len(values) >= 2 * period:
                # Create a pandas Series
                ts = pd.Series(values)
                
                # Apply decomposition
                decomposition = stats.STL(ts, period=period).fit()
                
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
                
                # Calculate strengths
                var_trend_resid = np.var(trend + residual)
                var_seas_resid = np.var(seasonal + residual)
                var_resid = np.var(residual)
                var_data = np.var(values)
                
                if var_trend_resid > 0:
                    trend_strength = max(0, min(1, 1 - var_resid / var_trend_resid))
                else:
                    trend_strength = 0.0
                
                if var_seas_resid > 0:
                    seasonality_strength = max(0, min(1, 1 - var_resid / var_seas_resid))
                else:
                    seasonality_strength = 0.0
                
                if var_data > 0:
                    residual_strength = max(0, min(1, var_resid / var_data))
                else:
                    residual_strength = 0.0
            else:
                # Not enough data for decomposition
                trend_strength = 0.0
                seasonality_strength = 0.0
                residual_strength = 1.0
            
            return {
                'trend_strength': float(trend_strength),
                'seasonality_strength': float(seasonality_strength),
                'residual_strength': float(residual_strength)
            }
        
        except Exception as e:
            self.logger.warning(f"Decomposition failed: {str(e)}")
            return {
                'trend_strength': 0.0,
                'seasonality_strength': 0.0,
                'residual_strength': 1.0
            }
    
    def _extract_pattern_features(self, df: pd.DataFrame, gas_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract pattern features for each gas type.
        
        Args:
            df: DataFrame containing gas concentration readings
            gas_types: List of gas types to analyze
            
        Returns:
            Dictionary mapping gas type to pattern features
        """
        # Get pattern detection parameters
        method = self.config.get('pattern_detection_method', 'fft')
        window_size = self.config.get('window_size', 60)
        overlap = self.config.get('overlap', 0.5)
        
        # Calculate sampling rate if time information is available
        if 'time_seconds' in df.columns and len(df) > 1:
            time_diff = df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]
            if time_diff > 0:
                sampling_rate = (len(df) - 1) / time_diff
            else:
                sampling_rate = 1.0
        else:
            sampling_rate = 1.0
        
        # Extract features for each gas type
        features = {}
        
        for gas in gas_types:
            if gas in df.columns:
                # Get concentration values
                values = df[gas].values
                
                # Calculate FFT features
                fft_features = self._calculate_fft_features(values, sampling_rate)
                
                # Calculate autocorrelation features
                autocorr_features = self._calculate_autocorrelation_features(values)
                
                # Detect patterns using clustering
                pattern_features = self._detect_patterns_clustering(values, window_size, overlap)
                
                # Calculate decomposition features
                decomp_features = self._calculate_decomposition_features(values)
                
                # Calculate pattern regularity
                if pattern_features['pattern_count'] > 1:
                    durations = pattern_features['pattern_durations']
                    regularity = 1.0 - (np.std(durations) / np.mean(durations)) if np.mean(durations) > 0 else 0.0
                    regularity = max(0.0, min(1.0, regularity))
                else:
                    regularity = 0.0
                
                # Calculate pattern complexity
                if fft_features['spectral_entropy'] > 0:
                    # Normalize entropy to [0, 1] range (assuming max entropy is log2(n/2))
                    max_entropy = np.log2(len(values) / 2) if len(values) > 0 else 1.0
                    complexity = min(1.0, fft_features['spectral_entropy'] / max_entropy) if max_entropy > 0 else 0.0
                else:
                    complexity = 0.0
                
                # Store features
                features[gas] = {
                    # FFT features
                    'dominant_frequency': fft_features['dominant_frequency'],
                    'dominant_period': fft_features['dominant_period'],
                    'spectral_entropy': fft_features['spectral_entropy'],
                    'spectral_flatness': fft_features['spectral_flatness'],
                    'spectral_centroid': fft_features['spectral_centroid'],
                    'spectral_bandwidth': fft_features['spectral_bandwidth'],
                    
                    # Autocorrelation features
                    'autocorrelation_peak': autocorr_features['autocorrelation_peak'],
                    'autocorrelation_lag': autocorr_features['autocorrelation_lag'],
                    
                    # Pattern features
                    'pattern_count': pattern_features['pattern_count'],
                    'pattern_duration_mean': float(np.mean(pattern_features['pattern_durations'])) 
                                          if pattern_features['pattern_durations'] else 0.0,
                    'pattern_duration_std': float(np.std(pattern_features['pattern_durations'])) 
                                         if pattern_features['pattern_durations'] else 0.0,
                    'pattern_similarity': pattern_features['pattern_similarity'],
                    'pattern_regularity': float(regularity),
                    'pattern_complexity': float(complexity),
                    
                    # Decomposition features
                    'trend_strength': decomp_features['trend_strength'],
                    'seasonality_strength': decomp_features['seasonality_strength'],
                    'residual_strength': decomp_features['residual_strength']
                }
        
        return features