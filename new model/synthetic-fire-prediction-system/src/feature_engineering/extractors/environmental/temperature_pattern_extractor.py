"""
Temperature Pattern Extractor for environmental data analysis.

This module provides a feature extractor that extracts temperature pattern features.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal, fft
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ...base import EnvironmentalFeatureExtractor


class TemperaturePatternExtractor(EnvironmentalFeatureExtractor):
    """
    Feature extractor for temperature patterns in environmental data.
    
    This class analyzes temperature data to identify patterns, cycles,
    and characteristic signatures in temperature behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the temperature pattern extractor.
        
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
        if 'pattern_detection_method' not in self.config:
            self.config['pattern_detection_method'] = 'fft'  # Options: 'fft', 'autocorrelation', 'clustering'
        
        if 'window_size' not in self.config:
            self.config['window_size'] = 60  # Window size for pattern detection (in samples)
        
        if 'overlap' not in self.config:
            self.config['overlap'] = 0.5  # Overlap between windows (as a fraction)
        
        if 'num_clusters' not in self.config:
            self.config['num_clusters'] = 3  # Number of clusters for pattern clustering
        
        if 'temperature_column' not in self.config:
            self.config['temperature_column'] = 'temperature'
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 5
        
        if 'daily_cycle_detection' not in self.config:
            self.config['daily_cycle_detection'] = True
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract temperature pattern features from environmental data.
        
        Args:
            data: Input environmental data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting temperature pattern features")
        
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
        temp_column = self.config['temperature_column']
        if temp_column not in df.columns:
            self.logger.warning(f"Missing temperature column '{temp_column}' in data")
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
            df_processed = self._apply_smoothing(df, [temp_column])
        else:
            df_processed = df
        
        # Extract pattern features
        pattern_features = self._extract_pattern_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'temperature_pattern_features': pattern_features
        }
        
        self.logger.info(f"Extracted temperature pattern features from {len(df)} samples")
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
        pattern_features = features.get('temperature_pattern_features', {})
        for feature_name, feature_value in pattern_features.items():
            if isinstance(feature_value, list):
                # Skip time series data in the flattened representation
                if len(feature_value) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_value):
                        flat_features[f"temp_pattern_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"temp_pattern_{feature_name}_mean"] = np.mean(feature_value)
                    flat_features[f"temp_pattern_{feature_name}_max"] = np.max(feature_value)
                    flat_features[f"temp_pattern_{feature_name}_min"] = np.min(feature_value)
            elif isinstance(feature_value, dict):
                for sub_name, sub_value in feature_value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 20:
                        # For large lists, just include summary statistics
                        flat_features[f"temp_pattern_{feature_name}_{sub_name}_count"] = len(sub_value)
                    else:
                        flat_features[f"temp_pattern_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"temp_pattern_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved temperature pattern features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
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
            'daily_cycle_strength',
            'daily_cycle_phase',
            'daily_amplitude',
            'trend_strength',
            'seasonality_strength',
            'residual_strength'
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
        
        # Apply clustering
        n_clusters = min(self.config.get('num_clusters', 3), len(windows))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        try:
            clusters = kmeans.fit_predict(windows_scaled)
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
                        if len(current_group) >= 3:  # Minimum pattern length
                            pattern_groups.append(current_group)
                        current_group = [idx]
                
                # Add last group if it's long enough
                if len(current_group) >= 3:
                    pattern_groups.append(current_group)
                
                # Calculate pattern durations
                for group in pattern_groups:
                    start_idx = group[0]
                    end_idx = group[-1] + window_size
                    duration = end_idx - start_idx
                    
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
                        pattern_windows.append(windows_scaled[i])
                
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
    
    def _detect_daily_cycle(self, values: np.ndarray, timestamps: pd.Series) -> Dict[str, float]:
        """
        Detect and characterize daily temperature cycles.
        
        Args:
            values: Temperature values
            timestamps: Timestamp series
            
        Returns:
            Dictionary containing daily cycle features
        """
        # Check if we have datetime timestamps and enough data
        if not pd.api.types.is_datetime64_any_dtype(timestamps) or len(values) < 24:
            return {
                'daily_cycle_strength': 0.0,
                'daily_cycle_phase': 0.0,
                'daily_amplitude': 0.0
            }
        
        try:
            # Extract hour of day
            hours = timestamps.dt.hour.values
            
            # Calculate average temperature by hour
            hour_temps = {}
            for hour in range(24):
                hour_indices = np.where(hours == hour)[0]
                if len(hour_indices) > 0:
                    hour_temps[hour] = np.mean(values[hour_indices])
            
            # If we don't have data for all hours, return default values
            if len(hour_temps) < 12:  # Need at least half the hours
                return {
                    'daily_cycle_strength': 0.0,
                    'daily_cycle_phase': 0.0,
                    'daily_amplitude': 0.0
                }
            
            # Fill missing hours by interpolation
            all_hours = np.arange(24)
            available_hours = np.array(list(hour_temps.keys()))
            available_temps = np.array(list(hour_temps.values()))
            
            # Interpolate missing hours
            if len(available_hours) < 24:
                missing_hours = np.setdiff1d(all_hours, available_hours)
                interp_temps = np.interp(missing_hours, available_hours, available_temps, period=24)
                
                for hour, temp in zip(missing_hours, interp_temps):
                    hour_temps[hour] = temp
            
            # Get hourly temperatures in order
            hourly_temps = np.array([hour_temps[hour] for hour in range(24)])
            
            # Calculate daily amplitude
            daily_amplitude = np.max(hourly_temps) - np.min(hourly_temps)
            
            # Fit sinusoid to detect phase
            x = np.arange(24)
            y = hourly_temps
            
            # Normalize data for fitting
            y_norm = (y - np.mean(y)) / (np.max(y) - np.min(y)) if np.max(y) > np.min(y) else np.zeros_like(y)
            
            # Generate sinusoid with different phases
            phases = np.linspace(0, 2*np.pi, 24)
            best_phase = 0
            best_correlation = -1
            
            for phase in phases:
                sinusoid = np.sin(2*np.pi*x/24 + phase)
                correlation = np.corrcoef(y_norm, sinusoid)[0, 1]
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_phase = phase
            
            # Convert phase to hour of day (0-24)
            phase_hour = (24 - (best_phase * 24 / (2*np.pi))) % 24
            
            # Calculate cycle strength (correlation with best-fit sinusoid)
            cycle_strength = max(0, best_correlation)  # Ensure non-negative
            
            return {
                'daily_cycle_strength': float(cycle_strength),
                'daily_cycle_phase': float(phase_hour),
                'daily_amplitude': float(daily_amplitude)
            }
        
        except Exception as e:
            self.logger.warning(f"Daily cycle detection failed: {str(e)}")
            return {
                'daily_cycle_strength': 0.0,
                'daily_cycle_phase': 0.0,
                'daily_amplitude': 0.0
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
                period = min(len(values) // 2, 24)  # Default to 24 for daily data
            
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
    
    def _extract_pattern_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract temperature pattern features.
        
        Args:
            df: DataFrame containing environmental readings
            
        Returns:
            Dictionary containing pattern features
        """
        # Get temperature column
        temp_column = self.config['temperature_column']
        
        # Get temperature values
        values = df[temp_column].values
        
        # Calculate sampling rate if timestamps are available
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']) and len(df) > 1:
            # Calculate time difference in seconds
            time_diff = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
            if time_diff > 0:
                sampling_rate = (len(df) - 1) / time_diff
            else:
                sampling_rate = 1.0
        else:
            sampling_rate = 1.0
        
        # Calculate FFT features
        fft_features = self._calculate_fft_features(values, sampling_rate)
        
        # Calculate autocorrelation features
        autocorr_features = self._calculate_autocorrelation_features(values)
        
        # Detect patterns using clustering
        window_size = self.config.get('window_size', 60)
        overlap = self.config.get('overlap', 0.5)
        pattern_features = self._detect_patterns_clustering(values, window_size, overlap)
        
        # Detect daily cycle if configured
        daily_cycle_features = {}
        if self.config.get('daily_cycle_detection', True):
            daily_cycle_features = self._detect_daily_cycle(values, df['timestamp'])
        
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
        
        # Combine all features
        return {
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
            
            # Daily cycle features
            'daily_cycle_strength': daily_cycle_features.get('daily_cycle_strength', 0.0),
            'daily_cycle_phase': daily_cycle_features.get('daily_cycle_phase', 0.0),
            'daily_amplitude': daily_cycle_features.get('daily_amplitude', 0.0),
            
            # Decomposition features
            'trend_strength': decomp_features['trend_strength'],
            'seasonality_strength': decomp_features['seasonality_strength'],
            'residual_strength': decomp_features['residual_strength']
        }