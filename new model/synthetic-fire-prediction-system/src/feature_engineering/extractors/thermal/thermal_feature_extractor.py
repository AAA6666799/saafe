"""
Thermal feature extractor implementation.

This module provides an implementation of the ThermalFeatureExtractor
for extracting features from thermal image data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
from scipy import ndimage
from skimage import measure

from ...base import ThermalFeatureExtractor


class BasicThermalFeatureExtractor(ThermalFeatureExtractor):
    """
    FLIR Lepton 3.5 thermal feature extractor implementation.
    
    This class extracts the 15 specific features from FLIR Lepton 3.5 thermal data:
    - t_mean: Mean temperature
    - t_std: Temperature standard deviation
    - t_max: Maximum temperature
    - t_p95: 95th percentile temperature
    - t_hot_area_pct: Percentage of hot area
    - t_hot_largest_blob_pct: Largest hot blob percentage
    - t_grad_mean: Mean temperature gradient
    - t_grad_std: Standard deviation of temperature gradient
    - t_diff_mean: Mean temperature difference between frames
    - t_diff_std: Standard deviation of temperature difference
    - flow_mag_mean: Mean optical flow magnitude
    - flow_mag_std: Standard deviation of optical flow magnitude
    - tproxy_val: Temperature proxy value
    - tproxy_delta: Temperature proxy delta
    - tproxy_vel: Temperature proxy velocity
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FLIR Lepton 3.5 thermal feature extractor.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # FLIR Lepton 3.5 specific parameters
        self.flir_features = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel'
        ]
        
        # Previous frame for temporal calculations
        self.previous_frame = None
        self.frame_history = []
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters for FLIR Lepton 3.5.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Set default values for FLIR Lepton 3.5 parameters
        if 'hot_temperature_threshold' not in self.config:
            self.config['hot_temperature_threshold'] = 50.0  # Celsius
        
        if 'gradient_kernel_size' not in self.config:
            self.config['gradient_kernel_size'] = 3
        
        if 'percentile_threshold' not in self.config:
            self.config['percentile_threshold'] = 95
        
        if 'flow_history_length' not in self.config:
            self.config['flow_history_length'] = 3
        
        if 'temperature_proxy_alpha' not in self.config:
            self.config['temperature_proxy_alpha'] = 0.8
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract FLIR Lepton 3.5 thermal features from thermal data.
        
        Args:
            data: Input thermal data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the 15 FLIR-specific thermal features
        """
        self.logger.info("Extracting FLIR Lepton 3.5 thermal features")
        
        # Handle both DataFrame and dictionary input
        if isinstance(data, pd.DataFrame):
            if 'thermal_frame' in data.columns:
                thermal_frame = data['thermal_frame'].iloc[-1]  # Get latest frame
            elif 'frame' in data.columns:
                thermal_frame = data['frame'].iloc[-1]
            else:
                raise ValueError("No thermal frame data found in DataFrame")
        else:
            # Handle dictionary input
            if 'thermal_frame' in data:
                thermal_frame = data['thermal_frame']
            elif 'frames' in data and data['frames']:
                thermal_frame = data['frames'][-1]  # Get latest frame
            else:
                raise ValueError("No thermal frame data found in input")
        
        # Convert to numpy array if needed
        if not isinstance(thermal_frame, np.ndarray):
            thermal_frame = np.array(thermal_frame)
        
        # Ensure frame is 2D
        if thermal_frame.ndim != 2:
            raise ValueError(f"Expected 2D thermal frame, got {thermal_frame.ndim}D")
        
        # Extract all 15 FLIR Lepton 3.5 features
        features = {}
        
        # Basic temperature statistics
        features['t_mean'] = float(np.mean(thermal_frame))
        features['t_std'] = float(np.std(thermal_frame))
        features['t_max'] = float(np.max(thermal_frame))
        features['t_p95'] = float(np.percentile(thermal_frame, self.config['percentile_threshold']))
        
        # Hot area features
        hot_threshold = self.config['hot_temperature_threshold']
        hot_mask = thermal_frame > hot_threshold
        total_pixels = thermal_frame.size
        hot_pixels = np.sum(hot_mask)
        features['t_hot_area_pct'] = float(hot_pixels / total_pixels * 100.0)
        
        # Largest hot blob percentage
        features['t_hot_largest_blob_pct'] = self._calculate_largest_blob_percentage(hot_mask)
        
        # Temperature gradient features
        grad_features = self._calculate_gradient_features(thermal_frame)
        features['t_grad_mean'] = grad_features['mean']
        features['t_grad_std'] = grad_features['std']
        
        # Temporal difference features (requires previous frame)
        diff_features = self._calculate_temporal_difference_features(thermal_frame)
        features['t_diff_mean'] = diff_features['mean']
        features['t_diff_std'] = diff_features['std']
        
        # Optical flow features (requires previous frame)
        flow_features = self._calculate_optical_flow_features(thermal_frame)
        features['flow_mag_mean'] = flow_features['mean']
        features['flow_mag_std'] = flow_features['std']
        
        # Temperature proxy features
        proxy_features = self._calculate_temperature_proxy_features(thermal_frame)
        features['tproxy_val'] = proxy_features['val']
        features['tproxy_delta'] = proxy_features['delta']
        features['tproxy_vel'] = proxy_features['vel']
        
        # Update frame history
        self.previous_frame = thermal_frame.copy()
        self.frame_history.append(thermal_frame.copy())
        
        # Keep only recent frames for efficiency
        max_history = self.config.get('flow_history_length', 3)
        if len(self.frame_history) > max_history:
            self.frame_history = self.frame_history[-max_history:]
        
        # Add metadata
        features['extraction_time'] = datetime.now().isoformat()
        features['frame_shape'] = thermal_frame.shape
        
        self.logger.info(f"Extracted {len(self.flir_features)} FLIR thermal features")
        return features
    
    def to_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert FLIR Lepton 3.5 thermal features to a pandas DataFrame.
        
        Args:
            features: Extracted FLIR features from the extract_features method
            
        Returns:
            DataFrame containing the 15 thermal features in a single row
        """
        # Create DataFrame with FLIR features only
        flir_data = {}
        for feature_name in self.flir_features:
            if feature_name in features:
                flir_data[feature_name] = features[feature_name]
            else:
                flir_data[feature_name] = 0.0  # Default value if missing
        
        # Create DataFrame with a single row
        df = pd.DataFrame([flir_data])
        
        return df
    
    def save(self, features: Dict[str, Any], filepath: str) -> None:
        """
        Save extracted features to a file.
        
        Args:
            features: Extracted features from the extract_features method
            filepath: Path to save the features
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(features, f, indent=2)
        
        self.logger.info(f"Saved thermal features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all FLIR Lepton 3.5 thermal features.
        
        Returns:
            List of the 15 FLIR-specific feature names
        """
        return self.flir_features.copy()
    
    def extract_temperature_statistics(self, 
                                     thermal_frame: np.ndarray,
                                     regions: Optional[List[Tuple[int, int, int, int]]] = None) -> Dict[str, Any]:
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
    
    def _extract_temperature_statistics(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract temperature statistics from all frames.
        
        Args:
            frames: List of thermal frames
            
        Returns:
            Dictionary containing temperature statistics
        """
        # Extract statistics for each frame
        frame_stats = []
        for frame in frames:
            frame_stats.append(self.extract_temperature_statistics(frame, self.config.get('regions_of_interest')))
        
        # Calculate aggregate statistics
        max_temperatures = [stats.get('max_temperature', 0) for stats in frame_stats 
                           if isinstance(stats, dict) and 'max_temperature' in stats]
        
        mean_temperatures = [stats.get('mean_temperature', 0) for stats in frame_stats 
                            if isinstance(stats, dict) and 'mean_temperature' in stats]
        
        min_temperatures = [stats.get('min_temperature', 0) for stats in frame_stats 
                           if isinstance(stats, dict) and 'min_temperature' in stats]
        
        # Return aggregated statistics
        return {
            'max_temperature': max_temperatures,
            'mean_temperature': mean_temperatures,
            'min_temperature': min_temperatures,
            'max_max_temperature': float(np.max(max_temperatures)) if max_temperatures else 0,
            'max_mean_temperature': float(np.max(mean_temperatures)) if mean_temperatures else 0,
            'min_min_temperature': float(np.min(min_temperatures)) if min_temperatures else 0,
            'temperature_range': float(np.max(max_temperatures) - np.min(min_temperatures)) 
                               if max_temperatures and min_temperatures else 0
        }
    
    def _extract_hotspot_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract hotspot features from all frames.
        
        Args:
            frames: List of thermal frames
            
        Returns:
            Dictionary containing hotspot features
        """
        hotspot_threshold = self.config['hotspot_threshold']
        
        # Detect hotspots in each frame
        frame_hotspots = []
        for frame in frames:
            frame_hotspots.append(self.detect_hotspots(frame, hotspot_threshold))
        
        # Calculate hotspot counts
        hotspot_counts = [len(hotspots) for hotspots in frame_hotspots]
        
        # Calculate hotspot areas
        hotspot_areas = []
        for hotspots in frame_hotspots:
            if hotspots:
                total_area = sum(hotspot['area'] for hotspot in hotspots)
                frame_area = frames[0].size  # Assuming all frames have the same size
                hotspot_areas.append(total_area / frame_area * 100.0)  # As percentage
            else:
                hotspot_areas.append(0.0)
        
        # Calculate maximum hotspot temperatures
        max_hotspot_temps = []
        for hotspots in frame_hotspots:
            if hotspots:
                max_temp = max(hotspot['max_temperature'] for hotspot in hotspots)
                max_hotspot_temps.append(max_temp)
            else:
                max_hotspot_temps.append(0.0)
        
        # Return hotspot features
        return {
            'hotspot_count': hotspot_counts,
            'max_hotspot_count': int(np.max(hotspot_counts)) if hotspot_counts else 0,
            'hotspot_area_percentage': hotspot_areas,
            'max_hotspot_area_percentage': float(np.max(hotspot_areas)) if hotspot_areas else 0.0,
            'max_hotspot_temperature': max_hotspot_temps,
            'max_max_hotspot_temperature': float(np.max(max_hotspot_temps)) if max_hotspot_temps else 0.0
        }
    
    def _extract_entropy_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract entropy features from all frames.
        
        Args:
            frames: List of thermal frames
            
        Returns:
            Dictionary containing entropy features
        """
        num_bins = self.config.get('entropy_bins', 32)
        
        # Calculate entropy for each frame
        entropies = []
        for frame in frames:
            # Calculate histogram
            hist, _ = np.histogram(frame, bins=num_bins)
            
            # Normalize histogram to get probability distribution
            hist = hist / hist.sum()
            
            # Calculate entropy (ignoring zero probabilities)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(entropy)
        
        # Calculate entropy changes between consecutive frames
        entropy_changes = [abs(entropies[i] - entropies[i-1]) for i in range(1, len(entropies))]
        
        # Return entropy features
        return {
            'temperature_entropy': entropies,
            'mean_entropy': float(np.mean(entropies)) if entropies else 0.0,
            'max_entropy': float(np.max(entropies)) if entropies else 0.0,
            'entropy_change': entropy_changes,
            'max_entropy_change': float(np.max(entropy_changes)) if entropy_changes else 0.0,
            'mean_entropy_change': float(np.mean(entropy_changes)) if entropy_changes else 0.0
        }
    
    def _extract_motion_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract motion features from all frames.
        
        Args:
            frames: List of thermal frames
            
        Returns:
            Dictionary containing motion features
        """
        if len(frames) < 2:
            return {
                'motion_detected': [False],
                'motion_area_percentage': [0.0],
                'motion_intensity': [0.0]
            }
        
        motion_threshold = self.config.get('motion_threshold', 5.0)
        
        # Calculate frame differences
        motion_detected = []
        motion_areas = []
        motion_intensities = []
        
        for i in range(1, len(frames)):
            # Calculate absolute difference between consecutive frames
            diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
            
            # Apply threshold to identify significant changes
            motion_mask = diff > motion_threshold
            
            # Calculate motion metrics
            motion_area = np.sum(motion_mask) / motion_mask.size * 100.0  # As percentage
            motion_intensity = np.mean(diff[motion_mask]) if np.any(motion_mask) else 0.0
            
            # Determine if motion is detected
            is_motion_detected = motion_area > 1.0  # More than 1% of the frame changed
            
            motion_detected.append(is_motion_detected)
            motion_areas.append(motion_area)
            motion_intensities.append(motion_intensity)
        
        # Add a placeholder for the first frame
        motion_detected.insert(0, False)
        motion_areas.insert(0, 0.0)
        motion_intensities.insert(0, 0.0)
        
        # Return motion features
        return {
            'motion_detected': motion_detected,
            'motion_area_percentage': motion_areas,
            'max_motion_area_percentage': float(np.max(motion_areas)) if motion_areas else 0.0,
            'motion_intensity': motion_intensities,
            'max_motion_intensity': float(np.max(motion_intensities)) if motion_intensities else 0.0,
            'motion_frames_percentage': (sum(motion_detected) / len(motion_detected) * 100.0) 
                                      if motion_detected else 0.0
        }
    
    def _extract_temperature_rise_features(self, 
                                         frames: List[np.ndarray],
                                         timestamps: List[Any]) -> Dict[str, Any]:
        """
        Extract temperature rise features from all frames.
        
        Args:
            frames: List of thermal frames
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary containing temperature rise features
        """
        if len(frames) < 2 or len(timestamps) < 2:
            return {
                'max_temperature_slope': 0.0,
                'mean_temperature_slope': 0.0,
                'max_temperature_acceleration': 0.0
            }
        
        # Convert timestamps to seconds if they're not already
        if isinstance(timestamps[0], str):
            # Parse ISO format timestamps
            try:
                timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
                time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                             for i in range(1, len(timestamps))]
            except ValueError:
                # If parsing fails, assume uniform time steps
                time_diffs = [1.0] * (len(timestamps) - 1)
        else:
            # Assume timestamps are already numeric
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # Extract maximum temperature for each frame
        max_temps = [np.max(frame) for frame in frames]
        
        # Calculate temperature slopes (rate of change)
        temp_slopes = []
        for i in range(1, len(max_temps)):
            if time_diffs[i-1] > 0:
                slope = (max_temps[i] - max_temps[i-1]) / time_diffs[i-1]
                temp_slopes.append(slope)
            else:
                temp_slopes.append(0.0)
        
        # Calculate temperature acceleration (rate of change of slope)
        temp_accelerations = []
        for i in range(1, len(temp_slopes)):
            if time_diffs[i-1] > 0:
                acceleration = (temp_slopes[i] - temp_slopes[i-1]) / time_diffs[i-1]
                temp_accelerations.append(acceleration)
            else:
                temp_accelerations.append(0.0)
        
        # Return temperature rise features
        return {
            'max_temperature_slope': float(np.max(temp_slopes)) if temp_slopes else 0.0,
            'mean_temperature_slope': float(np.mean(temp_slopes)) if temp_slopes else 0.0,
            'temperature_slopes': temp_slopes,
            'max_temperature_acceleration': float(np.max(temp_accelerations)) if temp_accelerations else 0.0,
            'temperature_accelerations': temp_accelerations
        }
    
    def _calculate_largest_blob_percentage(self, binary_mask: np.ndarray) -> float:
        """
        Calculate the percentage of the largest connected blob in a binary mask.
        
        Args:
            binary_mask: Binary mask where True indicates hot pixels
            
        Returns:
            Percentage of the largest blob relative to total frame size
        """
        if not np.any(binary_mask):
            return 0.0
        
        # Label connected components
        labeled_mask, num_labels = ndimage.label(binary_mask)
        
        if num_labels == 0:
            return 0.0
        
        # Find the largest blob
        blob_sizes = [np.sum(labeled_mask == i) for i in range(1, num_labels + 1)]
        largest_blob_size = max(blob_sizes)
        
        # Calculate percentage
        total_pixels = binary_mask.size
        return float(largest_blob_size / total_pixels * 100.0)
    
    def _calculate_gradient_features(self, thermal_frame: np.ndarray) -> Dict[str, float]:
        """
        Calculate temperature gradient features using Sobel operators.
        
        Args:
            thermal_frame: 2D thermal frame
            
        Returns:
            Dictionary with gradient mean and standard deviation
        """
        # Calculate gradients using Sobel operators
        grad_x = ndimage.sobel(thermal_frame, axis=1)
        grad_y = ndimage.sobel(thermal_frame, axis=0)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'mean': float(np.mean(gradient_magnitude)),
            'std': float(np.std(gradient_magnitude))
        }
    
    def _calculate_temporal_difference_features(self, thermal_frame: np.ndarray) -> Dict[str, float]:
        """
        Calculate temporal difference features between current and previous frame.
        
        Args:
            thermal_frame: Current thermal frame
            
        Returns:
            Dictionary with temporal difference mean and standard deviation
        """
        if self.previous_frame is None:
            # No previous frame available
            return {'mean': 0.0, 'std': 0.0}
        
        # Ensure frames have the same shape
        if thermal_frame.shape != self.previous_frame.shape:
            self.logger.warning("Frame shape mismatch for temporal difference calculation")
            return {'mean': 0.0, 'std': 0.0}
        
        # Calculate temporal difference
        temp_diff = thermal_frame.astype(float) - self.previous_frame.astype(float)
        
        return {
            'mean': float(np.mean(np.abs(temp_diff))),
            'std': float(np.std(temp_diff))
        }
    
    def _calculate_optical_flow_features(self, thermal_frame: np.ndarray) -> Dict[str, float]:
        """
        Calculate optical flow features using Farneback dense optical flow.
        
        Args:
            thermal_frame: Current thermal frame
            
        Returns:
            Dictionary with flow magnitude mean and standard deviation
        """
        if self.previous_frame is None or not CV2_AVAILABLE:
            # No previous frame or OpenCV not available
            return {'mean': 0.0, 'std': 0.0}
        
        try:
            # Convert frames to uint8 for OpenCV
            current_gray = ((thermal_frame - thermal_frame.min()) / 
                          (thermal_frame.max() - thermal_frame.min() + 1e-8) * 255).astype(np.uint8)
            previous_gray = ((self.previous_frame - self.previous_frame.min()) / 
                           (self.previous_frame.max() - self.previous_frame.min() + 1e-8) * 255).astype(np.uint8)
            
            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, 
                                          pyr_scale=0.5, levels=3, winsize=15, 
                                          iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            
            if flow is not None and flow.size > 0:
                # Calculate flow magnitude
                flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                
                return {
                    'mean': float(np.mean(flow_magnitude)),
                    'std': float(np.std(flow_magnitude))
                }
            else:
                return {'mean': 0.0, 'std': 0.0}
                
        except Exception as e:
            self.logger.warning(f"Optical flow calculation failed: {e}")
            # Fallback to simple frame difference for flow estimation
            try:
                frame_diff = np.abs(thermal_frame.astype(float) - self.previous_frame.astype(float))
                return {
                    'mean': float(np.mean(frame_diff)),
                    'std': float(np.std(frame_diff))
                }
            except:
                return {'mean': 0.0, 'std': 0.0}
    
    def _calculate_temperature_proxy_features(self, thermal_frame: np.ndarray) -> Dict[str, float]:
        """
        Calculate temperature proxy features for trend analysis.
        
        Args:
            thermal_frame: Current thermal frame
            
        Returns:
            Dictionary with proxy value, delta, and velocity
        """
        # Calculate current proxy value (weighted mean of hottest regions)
        alpha = self.config.get('temperature_proxy_alpha', 0.8)
        
        # Get top percentile temperatures for proxy calculation
        top_percentile = np.percentile(thermal_frame, 90)
        hot_pixels = thermal_frame[thermal_frame >= top_percentile]
        
        if len(hot_pixels) > 0:
            current_proxy = float(np.mean(hot_pixels))
        else:
            current_proxy = float(np.mean(thermal_frame))
        
        # Initialize tracking variables if not exist
        if not hasattr(self, '_proxy_history'):
            self._proxy_history = []
            self._proxy_smoothed = current_proxy
        
        # Update smoothed proxy value (exponential moving average)
        self._proxy_smoothed = alpha * current_proxy + (1 - alpha) * self._proxy_smoothed
        
        # Calculate delta (change from previous)
        if self._proxy_history:
            proxy_delta = current_proxy - self._proxy_history[-1]
        else:
            proxy_delta = 0.0
        
        # Calculate velocity (rate of change)
        if len(self._proxy_history) >= 2:
            # Use last two values for velocity calculation
            proxy_vel = self._proxy_history[-1] - self._proxy_history[-2]
        else:
            proxy_vel = 0.0
        
        # Update history
        self._proxy_history.append(current_proxy)
        
        # Keep only recent history for efficiency
        max_history = 10
        if len(self._proxy_history) > max_history:
            self._proxy_history = self._proxy_history[-max_history:]
        
        return {
            'val': float(self._proxy_smoothed),
            'delta': float(proxy_delta),
            'vel': float(proxy_vel)
        }