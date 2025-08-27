"""
Seasonality Detector for time-series data analysis.

This module provides a feature extractor that detects and characterizes seasonality in time-series data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import stats, signal, fftpack
import matplotlib.pyplot as plt

from ...base_temporal import TemporalFeatureExtractor


class SeasonalityDetector(TemporalFeatureExtractor):
    """
    Feature extractor for seasonality detection in time-series data.
    
    This class analyzes time-series data to identify and characterize seasonal patterns,
    including periodicity, strength, and significance of seasonality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the seasonality detector.
        
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
        
        if 'seasonality_detection_method' not in self.config:
            self.config['seasonality_detection_method'] = 'auto'  # Options: 'auto', 'fft', 'acf', 'stl'
        
        if 'max_lag' not in self.config:
            self.config['max_lag'] = 50  # Maximum lag for autocorrelation
        
        if 'significance_threshold' not in self.config:
            self.config['significance_threshold'] = 0.05  # Threshold for statistical significance
        
        if 'apply_smoothing' not in self.config:
            self.config['apply_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 3
        
        if 'known_period' not in self.config:
            self.config['known_period'] = None  # Known period for seasonality detection
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract seasonality features from time-series data.
        
        Args:
            data: Input time-series data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting seasonality features")
        
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
        
        # Extract seasonality features
        seasonality_features = self._extract_seasonality_features(df_processed)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'sample_count': len(df),
            'seasonality_features': seasonality_features
        }
        
        self.logger.info(f"Extracted seasonality features from {len(df)} samples")
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
        
        # Add seasonality features
        seasonality_features = features.get('seasonality_features', {})
        for feature_name, feature_value in seasonality_features.items():
            if isinstance(feature_value, list):
                # Skip time series data in the flattened representation
                if len(feature_value) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_value):
                        flat_features[f"seasonality_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"seasonality_{feature_name}_mean"] = np.mean(feature_value)
                    flat_features[f"seasonality_{feature_name}_max"] = np.max(feature_value)
                    flat_features[f"seasonality_{feature_name}_min"] = np.min(feature_value)
            elif isinstance(feature_value, dict):
                for sub_name, sub_value in feature_value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 20:
                        # For large lists, just include summary statistics
                        flat_features[f"seasonality_{feature_name}_{sub_name}_count"] = len(sub_value)
                    else:
                        flat_features[f"seasonality_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"seasonality_{feature_name}"] = feature_value
        
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
        
        self.logger.info(f"Saved seasonality features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'has_seasonality',
            'seasonality_period',
            'seasonality_strength',
            'seasonality_significance',
            'seasonality_amplitude',
            'seasonality_phase',
            'seasonality_consistency',
            'seasonality_components',
            'seasonality_residuals',
            'seasonality_decomposition',
            'seasonality_autocorrelation',
            'seasonality_spectrum'
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
    
    def _extract_seasonality_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract seasonality features from time-series data.
        
        Args:
            df: DataFrame containing time-series data
            
        Returns:
            Dictionary containing seasonality features
        """
        # Get time series column
        time_series_column = self.config['time_series_column']
        
        # Check if DataFrame has the required column
        if time_series_column not in df.columns:
            self.logger.warning(f"Column '{time_series_column}' not found in time series data")
            return {}
        
        # Get time series values
        values = df[time_series_column].values
        
        # Check if we have enough data
        if len(values) < 4:
            self.logger.warning("Not enough data for seasonality detection (need at least 4 points)")
            return {
                'has_seasonality': False,
                'seasonality_period': 0,
                'seasonality_strength': 0.0,
                'seasonality_significance': 0.0
            }
        
        # Get seasonality detection method
        method = self.config.get('seasonality_detection_method', 'auto')
        
        # Get known period if available
        known_period = self.config.get('known_period')
        
        # Extract seasonality features using the configured method
        if method == 'auto':
            # Try different methods and use the one with the highest significance
            fft_features = self._extract_seasonality_fft(values)
            acf_features = self._extract_seasonality_acf(values)
            
            # Choose the method with the highest significance
            if fft_features.get('seasonality_significance', 0) > acf_features.get('seasonality_significance', 0):
                seasonality_features = fft_features
                seasonality_features['detection_method'] = 'fft'
            else:
                seasonality_features = acf_features
                seasonality_features['detection_method'] = 'acf'
            
            # If known period is provided, use STL decomposition
            if known_period is not None:
                stl_features = self._extract_seasonality_stl(values, known_period)
                seasonality_features.update(stl_features)
                seasonality_features['detection_method'] = 'stl'
        
        elif method == 'fft':
            seasonality_features = self._extract_seasonality_fft(values)
            seasonality_features['detection_method'] = 'fft'
        
        elif method == 'acf':
            seasonality_features = self._extract_seasonality_acf(values)
            seasonality_features['detection_method'] = 'acf'
        
        elif method == 'stl':
            # Check if known period is provided
            if known_period is None:
                # Try to detect period using FFT
                fft_features = self._extract_seasonality_fft(values)
                known_period = fft_features.get('seasonality_period', 0)
                
                # If period is still not detected, try ACF
                if known_period <= 1:
                    acf_features = self._extract_seasonality_acf(values)
                    known_period = acf_features.get('seasonality_period', 0)
            
            # Check if period is valid
            if known_period > 1:
                seasonality_features = self._extract_seasonality_stl(values, known_period)
                seasonality_features['detection_method'] = 'stl'
            else:
                self.logger.warning("Could not detect seasonality period for STL decomposition")
                seasonality_features = {
                    'has_seasonality': False,
                    'seasonality_period': 0,
                    'seasonality_strength': 0.0,
                    'seasonality_significance': 0.0,
                    'detection_method': 'stl'
                }
        
        else:
            self.logger.warning(f"Unknown seasonality detection method: {method}, using auto")
            seasonality_features = self._extract_seasonality_fft(values)
            seasonality_features['detection_method'] = 'fft'
        
        return seasonality_features
    
    def _extract_seasonality_fft(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract seasonality features using Fast Fourier Transform (FFT).
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing seasonality features
        """
        # Check if we have enough data
        if len(values) < 4:
            return {
                'has_seasonality': False,
                'seasonality_period': 0,
                'seasonality_strength': 0.0,
                'seasonality_significance': 0.0
            }
        
        try:
            # Detrend the time series
            detrended = self._detrend_time_series(values)
            
            # Apply FFT
            fft_result = fftpack.fft(detrended)
            
            # Get power spectrum
            power = np.abs(fft_result) ** 2
            
            # Get frequencies
            freqs = fftpack.fftfreq(len(detrended))
            
            # Find the peak frequency (excluding DC component)
            positive_freqs = freqs[1:len(freqs)//2]
            positive_power = power[1:len(power)//2]
            
            if len(positive_freqs) == 0:
                return {
                    'has_seasonality': False,
                    'seasonality_period': 0,
                    'seasonality_strength': 0.0,
                    'seasonality_significance': 0.0
                }
            
            # Find peaks in the power spectrum
            peaks, _ = signal.find_peaks(positive_power)
            
            if len(peaks) == 0:
                return {
                    'has_seasonality': False,
                    'seasonality_period': 0,
                    'seasonality_strength': 0.0,
                    'seasonality_significance': 0.0
                }
            
            # Get the peak with the highest power
            peak_idx = peaks[np.argmax(positive_power[peaks])]
            peak_freq = positive_freqs[peak_idx]
            
            # Calculate period
            if peak_freq > 0:
                period = int(round(1.0 / peak_freq))
            else:
                period = 0
            
            # Check if period is valid
            if period <= 1 or period >= len(values) // 2:
                return {
                    'has_seasonality': False,
                    'seasonality_period': 0,
                    'seasonality_strength': 0.0,
                    'seasonality_significance': 0.0
                }
            
            # Calculate seasonality strength
            # (Ratio of power at the peak frequency to total power)
            peak_power = positive_power[peak_idx]
            total_power = np.sum(positive_power)
            
            if total_power > 0:
                seasonality_strength = float(peak_power / total_power)
            else:
                seasonality_strength = 0.0
            
            # Calculate seasonality significance
            # (Based on Fisher's g-test)
            g_statistic = float(peak_power / np.sum(positive_power))
            
            # Calculate p-value using Fisher's g-test approximation
            n = len(values)
            p_value = 1.0 - (1.0 - np.exp(-g_statistic)) ** ((n - 1) // 2)
            
            # Calculate significance (1 - p_value)
            significance = float(1.0 - p_value)
            
            # Calculate seasonality amplitude
            amplitude = float(np.sqrt(peak_power) * 2 / len(values))
            
            # Calculate seasonality phase
            phase = float(np.angle(fft_result[peak_idx + 1]))
            
            # Extract top seasonality components
            num_components = min(5, len(peaks))
            components = []
            
            for i in range(num_components):
                if i < len(peaks):
                    idx = peaks[i]
                    freq = positive_freqs[idx]
                    
                    if freq > 0:
                        comp_period = int(round(1.0 / freq))
                    else:
                        comp_period = 0
                    
                    comp_power = positive_power[idx]
                    comp_amplitude = float(np.sqrt(comp_power) * 2 / len(values))
                    comp_phase = float(np.angle(fft_result[idx + 1]))
                    
                    components.append({
                        'period': comp_period,
                        'frequency': float(freq),
                        'power': float(comp_power),
                        'amplitude': comp_amplitude,
                        'phase': comp_phase,
                        'relative_power': float(comp_power / total_power) if total_power > 0 else 0.0
                    })
            
            # Sort components by power
            components = sorted(components, key=lambda x: x['power'], reverse=True)
            
            # Calculate spectrum features
            spectrum = {
                'frequencies': positive_freqs.tolist(),
                'power': positive_power.tolist(),
                'peak_indices': peaks.tolist()
            }
            
            return {
                'has_seasonality': seasonality_strength > 0.1 and significance > 0.8,
                'seasonality_period': period,
                'seasonality_strength': seasonality_strength,
                'seasonality_significance': significance,
                'seasonality_amplitude': amplitude,
                'seasonality_phase': phase,
                'seasonality_components': components,
                'seasonality_spectrum': spectrum
            }
        
        except Exception as e:
            self.logger.warning(f"Error extracting seasonality using FFT: {str(e)}")
            return {
                'has_seasonality': False,
                'seasonality_period': 0,
                'seasonality_strength': 0.0,
                'seasonality_significance': 0.0
            }
    
    def _extract_seasonality_acf(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Extract seasonality features using Autocorrelation Function (ACF).
        
        Args:
            values: Time series values
            
        Returns:
            Dictionary containing seasonality features
        """
        # Check if we have enough data
        if len(values) < 4:
            return {
                'has_seasonality': False,
                'seasonality_period': 0,
                'seasonality_strength': 0.0,
                'seasonality_significance': 0.0
            }
        
        try:
            # Detrend the time series
            detrended = self._detrend_time_series(values)
            
            # Get maximum lag
            max_lag = min(self.config.get('max_lag', 50), len(detrended) // 2)
            
            # Calculate autocorrelation
            acf = np.correlate(detrended - np.mean(detrended), detrended - np.mean(detrended), mode='full')
            acf = acf[len(acf)//2:]  # Keep only positive lags
            acf = acf[:max_lag+1]  # Limit to max_lag
            
            # Normalize autocorrelation
            acf = acf / acf[0]
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(acf)
            
            if len(peaks) == 0:
                return {
                    'has_seasonality': False,
                    'seasonality_period': 0,
                    'seasonality_strength': 0.0,
                    'seasonality_significance': 0.0
                }
            
            # Get the peak with the highest autocorrelation
            peak_idx = peaks[np.argmax(acf[peaks])]
            
            # Calculate period
            period = int(peak_idx)
            
            # Check if period is valid
            if period <= 1:
                return {
                    'has_seasonality': False,
                    'seasonality_period': 0,
                    'seasonality_strength': 0.0,
                    'seasonality_significance': 0.0
                }
            
            # Calculate seasonality strength
            # (Autocorrelation at the peak lag)
            seasonality_strength = float(acf[period])
            
            # Calculate seasonality significance
            # (Based on Ljung-Box test)
            from scipy import stats
            
            # Calculate Ljung-Box statistic
            n = len(values)
            lb_statistic = n * (n + 2) * np.sum([(acf[i] ** 2) / (n - i) for i in range(1, min(period + 1, len(acf)))])
            
            # Calculate p-value
            p_value = 1.0 - stats.chi2.cdf(lb_statistic, period)
            
            # Calculate significance (1 - p_value)
            significance = float(1.0 - p_value)
            
            # Calculate seasonality consistency
            # (Standard deviation of autocorrelation at multiples of the period)
            period_multiples = [i * period for i in range(1, 5) if i * period < len(acf)]
            
            if period_multiples:
                acf_at_multiples = [acf[i] for i in period_multiples]
                consistency = float(1.0 - np.std(acf_at_multiples) / np.mean(acf_at_multiples)) if np.mean(acf_at_multiples) > 0 else 0.0
            else:
                consistency = 0.0
            
            return {
                'has_seasonality': seasonality_strength > 0.3 and significance > 0.8,
                'seasonality_period': period,
                'seasonality_strength': seasonality_strength,
                'seasonality_significance': significance,
                'seasonality_consistency': consistency,
                'seasonality_autocorrelation': acf.tolist()
            }
        
        except Exception as e:
            self.logger.warning(f"Error extracting seasonality using ACF: {str(e)}")
            return {
                'has_seasonality': False,
                'seasonality_period': 0,
                'seasonality_strength': 0.0,
                'seasonality_significance': 0.0
            }
    
    def _extract_seasonality_stl(self, values: np.ndarray, period: int) -> Dict[str, Any]:
        """
        Extract seasonality features using Seasonal-Trend decomposition using LOESS (STL).
        
        Args:
            values: Time series values
            period: Seasonality period
            
        Returns:
            Dictionary containing seasonality features
        """
        # Check if we have enough data
        if len(values) < period * 2:
            return {
                'has_seasonality': False,
                'seasonality_period': period,
                'seasonality_strength': 0.0,
                'seasonality_significance': 0.0
            }
        
        try:
            # Apply STL decomposition
            from statsmodels.tsa.seasonal import STL
            
            # Create pandas Series
            ts = pd.Series(values)
            
            # Apply STL decomposition
            stl = STL(ts, period=period, robust=True)
            result = stl.fit()
            
            # Extract components
            trend = result.trend.values
            seasonal = result.seasonal.values
            residual = result.resid.values
            
            # Calculate seasonality strength
            var_trend_resid = np.var(trend + residual)
            var_seas_resid = np.var(seasonal + residual)
            var_resid = np.var(residual)
            
            if var_trend_resid > 0:
                seasonality_strength = float(max(0, min(1, 1 - var_resid / var_seas_resid)))
            else:
                seasonality_strength = 0.0
            
            # Calculate seasonality significance
            # (F-test for seasonal component)
            var_seasonal = np.var(seasonal)
            
            if var_resid > 0 and var_seasonal > 0:
                f_statistic = var_seasonal / var_resid
                
                # Calculate degrees of freedom
                df1 = period - 1  # Seasonal component
                df2 = len(values) - period  # Residual
                
                # Calculate p-value
                from scipy import stats
                p_value = 1.0 - stats.f.cdf(f_statistic, df1, df2)
                
                # Calculate significance (1 - p_value)
                significance = float(1.0 - p_value)
            else:
                significance = 0.0
            
            # Calculate seasonality amplitude
            amplitude = float(np.max(seasonal) - np.min(seasonal))
            
            # Calculate seasonality consistency
            # (Correlation between seasonal components in different cycles)
            seasonal_cycles = []
            
            for i in range(len(values) // period):
                start_idx = i * period
                end_idx = min((i + 1) * period, len(values))
                
                if end_idx - start_idx == period:
                    seasonal_cycles.append(seasonal[start_idx:end_idx])
            
            if len(seasonal_cycles) >= 2:
                correlations = []
                
                for i in range(len(seasonal_cycles)):
                    for j in range(i + 1, len(seasonal_cycles)):
                        corr = np.corrcoef(seasonal_cycles[i], seasonal_cycles[j])[0, 1]
                        correlations.append(corr)
                
                consistency = float(np.mean(correlations))
            else:
                consistency = 0.0
            
            # Extract seasonal pattern
            # (Average seasonal component for each position in the period)
            seasonal_pattern = []
            
            for i in range(period):
                positions = [i + j * period for j in range(len(values) // period) if i + j * period < len(values)]
                if positions:
                    pattern_value = float(np.mean([seasonal[pos] for pos in positions]))
                    seasonal_pattern.append(pattern_value)
            
            return {
                'has_seasonality': seasonality_strength > 0.1 and significance > 0.8,
                'seasonality_period': period,
                'seasonality_strength': seasonality_strength,
                'seasonality_significance': significance,
                'seasonality_amplitude': amplitude,
                'seasonality_consistency': consistency,
                'seasonality_pattern': seasonal_pattern,
                'seasonality_decomposition': {
                    'trend': trend.tolist(),
                    'seasonal': seasonal.tolist(),
                    'residual': residual.tolist()
                }
            }
        
        except Exception as e:
            self.logger.warning(f"Error extracting seasonality using STL: {str(e)}")
            return {
                'has_seasonality': False,
                'seasonality_period': period,
                'seasonality_strength': 0.0,
                'seasonality_significance': 0.0
            }
    
    def _detrend_time_series(self, values: np.ndarray) -> np.ndarray:
        """
        Detrend time series by removing linear trend.
        
        Args:
            values: Time series values
            
        Returns:
            Detrended time series
        """
        # Create time index
        x = np.arange(len(values))
        
        # Fit linear regression
        slope, intercept, _, _, _ = stats.linregress(x, values)
        
        # Calculate trend
        trend = intercept + slope * x
        
        # Remove trend
        detrended = values - trend
        
        return detrended
    
    def plot_seasonality(self, features: Dict[str, Any], filepath: Optional[str] = None) -> None:
        """
        Plot seasonality features.
        
        Args:
            features: Extracted features from the extract_features method
            filepath: Optional path to save the plot
        """
        try:
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # Get seasonality features
            seasonality_features = features.get('seasonality_features', {})
            
            # Plot autocorrelation
            if 'seasonality_autocorrelation' in seasonality_features:
                acf = seasonality_features['seasonality_autocorrelation']
                axes[0].plot(acf)
                axes[0].set_title('Autocorrelation Function')
                axes[0].set_xlabel('Lag')
                axes[0].set_ylabel('Autocorrelation')
                
                # Add period marker
                period = seasonality_features.get('seasonality_period', 0)
                if period > 0:
                    axes[0].axvline(x=period, color='r', linestyle='--')
                    axes[0].text(period, 0.8, f'Period = {period}', color='r')
            
            # Plot spectrum
            if 'seasonality_spectrum' in seasonality_features:
                spectrum = seasonality_features['seasonality_spectrum']
                frequencies = spectrum.get('frequencies', [])
                power = spectrum.get('power', [])
                
                if frequencies and power and len(frequencies) == len(power):
                    axes[1].plot(frequencies, power)
                    axes[1].set_title('Power Spectrum')
                    axes[1].set_xlabel('Frequency')
                    axes[1].set_ylabel('Power')
                    
                    # Add peak markers
                    peak_indices = spectrum.get('peak_indices', [])
                    for idx in peak_indices:
                        if idx < len(frequencies) and idx < len(power):
                            freq = frequencies[idx]
                            pow_val = power[idx]
                            
                            if freq > 0:
                                period = int(round(1.0 / freq))
                                axes[1].plot(freq, pow_val, 'ro')
                                axes[1].text(freq, pow_val, f'Period = {period}', color='r')
            
            # Plot decomposition
            if 'seasonality_decomposition' in seasonality_features:
                decomposition = seasonality_features['seasonality_decomposition']
                trend = decomposition.get('trend', [])
                seasonal = decomposition.get('seasonal', [])
                residual = decomposition.get('residual', [])
                
                if trend and seasonal and residual:
                    axes[2].plot(trend, label='Trend')
                    axes[2].plot(seasonal, label='Seasonal')
                    axes[2].plot(residual, label='Residual')
                    axes[2].set_title('STL Decomposition')
                    axes[2].set_xlabel('Time')
                    axes[2].set_ylabel('Value')
                    axes[2].legend()
            
            # Add overall seasonality information
            has_seasonality = seasonality_features.get('has_seasonality', False)
            period =