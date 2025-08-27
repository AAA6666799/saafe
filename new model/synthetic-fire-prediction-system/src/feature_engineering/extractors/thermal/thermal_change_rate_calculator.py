"""
Thermal Change Rate Calculator for thermal image analysis.

This module provides a feature extractor that calculates rate of change in thermal data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import ndimage, stats, signal
import cv2

from ...base import ThermalFeatureExtractor


class ThermalChangeRateCalculator(ThermalFeatureExtractor):
    """
    Feature extractor for calculating rate of change in thermal data.
    
    This class analyzes thermal images to calculate various rates of change,
    including temperature rise rates, spatial change rates, and temporal derivatives.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the thermal change rate calculator.
        
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
        if 'time_windows' not in self.config:
            self.config['time_windows'] = [1, 3, 5, 10]  # Number of frames for different time windows
        
        if 'spatial_derivative_method' not in self.config:
            self.config['spatial_derivative_method'] = 'sobel'  # Options: 'sobel', 'prewitt', 'scharr'
        
        if 'temporal_smoothing' not in self.config:
            self.config['temporal_smoothing'] = True
        
        if 'smoothing_window' not in self.config:
            self.config['smoothing_window'] = 3
        
        if 'regions_of_interest' not in self.config:
            self.config['regions_of_interest'] = None  # Optional regions to analyze
        
        if 'change_threshold' not in self.config:
            self.config['change_threshold'] = 2.0  # Threshold for significant changes
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract change rate features from thermal data.
        
        Args:
            data: Input thermal data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting thermal change rate features")
        
        # Check if data is a dictionary or DataFrame
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to dictionary
            data_dict = {
                'timestamps': data['timestamp'].tolist(),
                'frames': [row['frame'] for _, row in data.iterrows()]
            }
        else:
            data_dict = data
        
        # Extract frames and timestamps
        frames = data_dict.get('frames', [])
        timestamps = data_dict.get('timestamps', [])
        
        if not frames:
            self.logger.warning("No thermal frames found in data")
            return {}
        
        # Convert frames to numpy arrays if they're not already
        frames = [np.array(frame) if not isinstance(frame, np.ndarray) else frame for frame in frames]
        
        # Convert timestamps to seconds if they're strings
        if timestamps and isinstance(timestamps[0], str):
            try:
                timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
                time_diffs = [(timestamps[i] - timestamps[0]).total_seconds() for i in range(len(timestamps))]
            except ValueError:
                # If parsing fails, assume uniform time steps
                time_diffs = list(range(len(timestamps)))
        else:
            # If timestamps are already numeric or not provided
            time_diffs = list(range(len(frames)))
        
        # Extract change rate features
        change_rate_features = self._extract_change_rate_features(frames, time_diffs)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'frame_count': len(frames),
            'frame_shape': frames[0].shape if frames else None,
            'change_rate_features': change_rate_features
        }
        
        self.logger.info(f"Extracted thermal change rate features from {len(frames)} frames")
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
        
        # Add change rate features
        change_rate_features = features.get('change_rate_features', {})
        for feature_name, feature_values in change_rate_features.items():
            if isinstance(feature_values, list):
                if len(feature_values) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_values):
                        flat_features[f"change_rate_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"change_rate_{feature_name}_mean"] = np.mean(feature_values)
                    flat_features[f"change_rate_{feature_name}_max"] = np.max(feature_values)
                    flat_features[f"change_rate_{feature_name}_min"] = np.min(feature_values)
                    flat_features[f"change_rate_{feature_name}_std"] = np.std(feature_values)
            elif isinstance(feature_values, dict):
                for sub_name, sub_value in feature_values.items():
                    flat_features[f"change_rate_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"change_rate_{feature_name}"] = feature_values
        
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
        
        self.logger.info(f"Saved thermal change rate features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'temperature_rise_rate',
            'max_temperature_rise_rate',
            'mean_temperature_rise_rate',
            'temperature_acceleration',
            'max_temperature_acceleration',
            'spatial_gradient_magnitude',
            'max_spatial_gradient',
            'mean_spatial_gradient',
            'change_area_percentage',
            'change_intensity',
            'change_frequency',
            'change_duration',
            'change_propagation_speed'
        ]
    
    def extract_temperature_statistics(self, 
                                     thermal_frame: np.ndarray,
                                     regions: Optional[List[Tuple[int, int, int, int]]] = None) -> Dict[str, float]:
        """
        Extract temperature statistics from a thermal frame.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            regions: Optional list of regions (x, y, width, height) to analyze
            
        Returns:
            Dictionary containing temperature statistics
        """
        if regions:
            # Extract statistics for each region
            region_stats = []
            for x, y, width, height in regions:
                # Extract region
                region = thermal_frame[y:y+height, x:x+width]
                
                # Calculate statistics
                region_stats.append({
                    'min_temperature': float(np.min(region)),
                    'max_temperature': float(np.max(region)),
                    'mean_temperature': float(np.mean(region)),
                    'temperature_range': float(np.max(region) - np.min(region)),
                    'temperature_std': float(np.std(region))
                })
            
            return {
                'regions': region_stats
            }
        else:
            # Calculate statistics for the entire frame
            return {
                'min_temperature': float(np.min(thermal_frame)),
                'max_temperature': float(np.max(thermal_frame)),
                'mean_temperature': float(np.mean(thermal_frame)),
                'temperature_range': float(np.max(thermal_frame) - np.min(thermal_frame)),
                'temperature_std': float(np.std(thermal_frame))
            }
    
    def detect_hotspots(self, 
                      thermal_frame: np.ndarray,
                      threshold_temp: float) -> List[Dict[str, Any]]:
        """
        Detect hotspots in a thermal frame.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            threshold_temp: Temperature threshold for hotspot detection
            
        Returns:
            List of dictionaries containing hotspot information
        """
        # Create binary mask of hotspots
        hotspot_mask = thermal_frame > threshold_temp
        
        # Label connected components
        labeled_mask, num_hotspots = ndimage.label(hotspot_mask)
        
        # Extract properties of each hotspot
        if num_hotspots > 0:
            from skimage import measure
            props = measure.regionprops(labeled_mask, intensity_image=thermal_frame)
            
            hotspots = []
            for prop in props:
                y0, x0, y1, x1 = prop.bbox
                hotspots.append({
                    'area': int(prop.area),
                    'centroid': (int(prop.centroid[1]), int(prop.centroid[0])),
                    'bbox': (x0, y0, x1 - x0, y1 - y0),
                    'max_temperature': float(prop.max_intensity),
                    'mean_temperature': float(prop.mean_intensity)
                })
            
            return hotspots
        else:
            return []
    
    def _calculate_spatial_gradient(self, thermal_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate spatial gradient for a thermal frame.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Tuple of (gradient_magnitude, gradient_x, gradient_y)
        """
        method = self.config.get('spatial_derivative_method', 'sobel')
        
        if method == 'sobel':
            from scipy import ndimage
            gradient_y = ndimage.sobel(thermal_frame, axis=0)
            gradient_x = ndimage.sobel(thermal_frame, axis=1)
        elif method == 'prewitt':
            from scipy import ndimage
            gradient_y = ndimage.prewitt(thermal_frame, axis=0)
            gradient_x = ndimage.prewitt(thermal_frame, axis=1)
        elif method == 'scharr':
            from skimage import filters
            gradient_y = filters.scharr_h(thermal_frame)
            gradient_x = filters.scharr_v(thermal_frame)
        else:
            self.logger.warning(f"Unknown gradient method: {method}, using sobel")
            from scipy import ndimage
            gradient_y = ndimage.sobel(thermal_frame, axis=0)
            gradient_x = ndimage.sobel(thermal_frame, axis=1)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        return gradient_magnitude, gradient_x, gradient_y
    
    def _calculate_temporal_derivative(self, 
                                     frames: List[np.ndarray], 
                                     time_diffs: List[float],
                                     window_size: int = 1) -> List[np.ndarray]:
        """
        Calculate temporal derivative for a sequence of frames.
        
        Args:
            frames: List of thermal frames
            time_diffs: List of time differences in seconds
            window_size: Size of the window for derivative calculation
            
        Returns:
            List of temporal derivative frames
        """
        if len(frames) <= window_size:
            return [np.zeros_like(frames[0])]
        
        derivatives = []
        
        for i in range(window_size, len(frames)):
            # Calculate time difference
            dt = time_diffs[i] - time_diffs[i - window_size]
            
            if dt > 0:
                # Calculate temperature difference
                dT = frames[i] - frames[i - window_size]
                
                # Calculate derivative
                derivative = dT / dt
                
                derivatives.append(derivative)
            else:
                derivatives.append(np.zeros_like(frames[0]))
        
        # Add placeholders for the first window_size frames
        for _ in range(window_size):
            derivatives.insert(0, np.zeros_like(frames[0]))
        
        return derivatives
    
    def _calculate_temporal_acceleration(self, 
                                       derivatives: List[np.ndarray], 
                                       time_diffs: List[float],
                                       window_size: int = 1) -> List[np.ndarray]:
        """
        Calculate temporal acceleration (second derivative) for a sequence of frames.
        
        Args:
            derivatives: List of first derivative frames
            time_diffs: List of time differences in seconds
            window_size: Size of the window for derivative calculation
            
        Returns:
            List of temporal acceleration frames
        """
        if len(derivatives) <= window_size:
            return [np.zeros_like(derivatives[0])]
        
        accelerations = []
        
        for i in range(window_size, len(derivatives)):
            # Calculate time difference
            dt = time_diffs[i] - time_diffs[i - window_size]
            
            if dt > 0:
                # Calculate derivative difference
                dd = derivatives[i] - derivatives[i - window_size]
                
                # Calculate acceleration
                acceleration = dd / dt
                
                accelerations.append(acceleration)
            else:
                accelerations.append(np.zeros_like(derivatives[0]))
        
        # Add placeholders for the first window_size frames
        for _ in range(window_size):
            accelerations.insert(0, np.zeros_like(derivatives[0]))
        
        return accelerations
    
    def _detect_significant_changes(self, 
                                  derivatives: List[np.ndarray], 
                                  threshold: float) -> List[Dict[str, Any]]:
        """
        Detect significant changes in thermal data.
        
        Args:
            derivatives: List of derivative frames
            threshold: Threshold for significant changes
            
        Returns:
            List of dictionaries containing change information
        """
        changes = []
        
        for i, derivative in enumerate(derivatives):
            # Create binary mask of significant changes
            change_mask = np.abs(derivative) > threshold
            
            # Calculate change metrics
            change_area = np.sum(change_mask)
            change_intensity = np.mean(np.abs(derivative[change_mask])) if np.any(change_mask) else 0.0
            
            # Label connected components
            labeled_mask, num_changes = ndimage.label(change_mask)
            
            changes.append({
                'frame_index': i,
                'change_area': int(change_area),
                'change_area_percentage': float(change_area / derivative.size * 100.0),
                'change_intensity': float(change_intensity),
                'change_count': int(num_changes)
            })
        
        return changes
    
    def _calculate_change_propagation(self, 
                                    frames: List[np.ndarray], 
                                    time_diffs: List[float],
                                    threshold: float) -> Dict[str, Any]:
        """
        Calculate change propagation metrics.
        
        Args:
            frames: List of thermal frames
            time_diffs: List of time differences in seconds
            threshold: Threshold for change detection
            
        Returns:
            Dictionary containing change propagation metrics
        """
        if len(frames) < 2:
            return {
                'propagation_speed': 0.0,
                'propagation_direction': (0.0, 0.0),
                'propagation_area_growth': 0.0
            }
        
        # Detect hotspots in each frame
        hotspot_masks = []
        for frame in frames:
            hotspot_mask = frame > threshold
            hotspot_masks.append(hotspot_mask)
        
        # Calculate hotspot areas
        hotspot_areas = [np.sum(mask) for mask in hotspot_masks]
        
        # Calculate area growth rates
        area_growth_rates = []
        for i in range(1, len(hotspot_areas)):
            dt = time_diffs[i] - time_diffs[i-1]
            if dt > 0 and hotspot_areas[i-1] > 0:
                growth_rate = (hotspot_areas[i] - hotspot_areas[i-1]) / (dt * hotspot_areas[i-1])
                area_growth_rates.append(growth_rate)
        
        # Calculate hotspot centroids
        centroids = []
        for mask in hotspot_masks:
            if np.any(mask):
                # Calculate centroid
                y_indices, x_indices = np.where(mask)
                centroid_y = np.mean(y_indices)
                centroid_x = np.mean(x_indices)
                centroids.append((centroid_x, centroid_y))
            else:
                centroids.append(None)
        
        # Calculate propagation speeds
        propagation_speeds = []
        propagation_directions = []
        
        for i in range(1, len(centroids)):
            if centroids[i] is not None and centroids[i-1] is not None:
                dt = time_diffs[i] - time_diffs[i-1]
                if dt > 0:
                    dx = centroids[i][0] - centroids[i-1][0]
                    dy = centroids[i][1] - centroids[i-1][1]
                    distance = np.sqrt(dx**2 + dy**2)
                    speed = distance / dt
                    direction = (dx / distance if distance > 0 else 0.0, 
                               dy / distance if distance > 0 else 0.0)
                    
                    propagation_speeds.append(speed)
                    propagation_directions.append(direction)
        
        # Calculate average metrics
        avg_propagation_speed = np.mean(propagation_speeds) if propagation_speeds else 0.0
        avg_area_growth_rate = np.mean(area_growth_rates) if area_growth_rates else 0.0
        
        # Calculate average direction
        if propagation_directions:
            avg_dx = np.mean([d[0] for d in propagation_directions])
            avg_dy = np.mean([d[1] for d in propagation_directions])
            magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
            avg_direction = (avg_dx / magnitude if magnitude > 0 else 0.0, 
                           avg_dy / magnitude if magnitude > 0 else 0.0)
        else:
            avg_direction = (0.0, 0.0)
        
        return {
            'propagation_speed': float(avg_propagation_speed),
            'propagation_direction': (float(avg_direction[0]), float(avg_direction[1])),
            'propagation_area_growth': float(avg_area_growth_rate)
        }
    
    def _extract_change_rate_features(self, frames: List[np.ndarray], time_diffs: List[float]) -> Dict[str, Any]:
        """
        Extract change rate features from all frames.
        
        Args:
            frames: List of thermal frames
            time_diffs: List of time differences in seconds
            
        Returns:
            Dictionary containing change rate features
        """
        # Apply temporal smoothing if configured
        if self.config.get('temporal_smoothing', True) and len(frames) > 1:
            window_size = self.config.get('smoothing_window', 3)
            smoothed_frames = []
            
            for i in range(len(frames)):
                # Calculate window boundaries
                start = max(0, i - window_size // 2)
                end = min(len(frames), i + window_size // 2 + 1)
                
                # Calculate weighted average
                weights = np.exp(-0.5 * ((np.arange(start, end) - i) / (window_size / 4))**2)
                weighted_sum = np.zeros_like(frames[0], dtype=float)
                
                for j, w in zip(range(start, end), weights):
                    weighted_sum += frames[j] * w
                
                smoothed_frames.append(weighted_sum / np.sum(weights))
            
            processed_frames = smoothed_frames
        else:
            processed_frames = frames
        
        # Calculate temporal derivatives for different time windows
        time_windows = self.config.get('time_windows', [1, 3, 5, 10])
        derivatives_by_window = {}
        
        for window in time_windows:
            if window < len(processed_frames):
                derivatives = self._calculate_temporal_derivative(processed_frames, time_diffs, window)
                derivatives_by_window[window] = derivatives
        
        # Calculate spatial gradients for each frame
        spatial_gradients = []
        for frame in processed_frames:
            gradient_magnitude, _, _ = self._calculate_spatial_gradient(frame)
            spatial_gradients.append(gradient_magnitude)
        
        # Calculate temperature rise rates (using the smallest time window)
        smallest_window = min(time_windows)
        if smallest_window in derivatives_by_window:
            derivatives = derivatives_by_window[smallest_window]
            
            # Extract max and mean temperature rise rates
            max_rise_rates = [float(np.max(np.abs(d))) for d in derivatives]
            mean_rise_rates = [float(np.mean(np.abs(d))) for d in derivatives]
            
            # Calculate acceleration (second derivative)
            accelerations = self._calculate_temporal_acceleration(derivatives, time_diffs, smallest_window)
            max_accelerations = [float(np.max(np.abs(a))) for a in accelerations]
            mean_accelerations = [float(np.mean(np.abs(a))) for a in accelerations]
        else:
            max_rise_rates = [0.0] * len(processed_frames)
            mean_rise_rates = [0.0] * len(processed_frames)
            accelerations = [np.zeros_like(processed_frames[0])] * len(processed_frames)
            max_accelerations = [0.0] * len(processed_frames)
            mean_accelerations = [0.0] * len(processed_frames)
        
        # Calculate spatial gradient statistics
        max_spatial_gradients = [float(np.max(g)) for g in spatial_gradients]
        mean_spatial_gradients = [float(np.mean(g)) for g in spatial_gradients]
        
        # Detect significant changes
        threshold = self.config.get('change_threshold', 2.0)
        changes = self._detect_significant_changes(derivatives_by_window.get(smallest_window, []), threshold)
        
        # Calculate change frequency (changes per second)
        if len(time_diffs) > 1 and time_diffs[-1] > time_diffs[0]:
            total_time = time_diffs[-1] - time_diffs[0]
            change_count = sum(1 for c in changes if c['change_count'] > 0)
            change_frequency = change_count / total_time
        else:
            change_frequency = 0.0
        
        # Calculate change duration (average duration of change events)
        change_durations = []
        current_duration = 0
        in_change = False
        
        for change in changes:
            if change['change_count'] > 0:
                if not in_change:
                    in_change = True
                current_duration += 1
            else:
                if in_change:
                    change_durations.append(current_duration)
                    current_duration = 0
                    in_change = False
        
        if in_change and current_duration > 0:
            change_durations.append(current_duration)
        
        avg_change_duration = np.mean(change_durations) if change_durations else 0.0
        
        # Calculate change propagation metrics
        propagation_metrics = self._calculate_change_propagation(processed_frames, time_diffs, threshold)
        
        # Return change rate features
        return {
            'temperature_rise_rate': max_rise_rates,
            'max_temperature_rise_rate': float(np.max(max_rise_rates)) if max_rise_rates else 0.0,
            'mean_temperature_rise_rate': float(np.mean(mean_rise_rates)) if mean_rise_rates else 0.0,
            'temperature_acceleration': max_accelerations,
            'max_temperature_acceleration': float(np.max(max_accelerations)) if max_accelerations else 0.0,
            'mean_temperature_acceleration': float(np.mean(mean_accelerations)) if mean_accelerations else 0.0,
            'spatial_gradient_magnitude': max_spatial_gradients,
            'max_spatial_gradient': float(np.max(max_spatial_gradients)) if max_spatial_gradients else 0.0,
            'mean_spatial_gradient': float(np.mean(mean_spatial_gradients)) if mean_spatial_gradients else 0.0,
            'change_area_percentage': [c['change_area_percentage'] for c in changes],
            'max_change_area_percentage': float(np.max([c['change_area_percentage'] for c in changes])) if changes else 0.0,
            'change_intensity': [c['change_intensity'] for c in changes],
            'max_change_intensity': float(np.max([c['change_intensity'] for c in changes])) if changes else 0.0,
            'change_frequency': float(change_frequency),
            'change_duration': float(avg_change_duration),
            'change_propagation_speed': float(propagation_metrics['propagation_speed']),
            'change_propagation_direction_x': float(propagation_metrics['propagation_direction'][0]),
            'change_propagation_direction_y': float(propagation_metrics['propagation_direction'][1]),
            'change_area_growth_rate': float(propagation_metrics['propagation_area_growth'])
        }