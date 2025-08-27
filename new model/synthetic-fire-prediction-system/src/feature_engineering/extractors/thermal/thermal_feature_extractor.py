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
    Basic implementation of a thermal feature extractor.
    
    This class extracts features from thermal image data, including:
    - Maximum and mean temperature
    - Hotspot area percentage
    - Temperature entropy
    - Motion detection between frames
    - Temperature rise slope
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the thermal feature extractor.
        
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
        required_params = ['max_temperature_threshold', 'hotspot_threshold']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Set default values for optional parameters
        if 'regions_of_interest' not in self.config:
            self.config['regions_of_interest'] = None
        
        if 'entropy_bins' not in self.config:
            self.config['entropy_bins'] = 32
        
        if 'motion_threshold' not in self.config:
            self.config['motion_threshold'] = 5.0
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract features from thermal data.
        
        Args:
            data: Input thermal data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting thermal features")
        
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
        
        # Extract features
        features = {
            'extraction_time': datetime.now().isoformat(),
            'frame_count': len(frames),
            'frame_shape': frames[0].shape if frames else None,
            'temperature_statistics': self._extract_temperature_statistics(frames),
            'hotspot_features': self._extract_hotspot_features(frames),
            'entropy_features': self._extract_entropy_features(frames),
            'motion_features': self._extract_motion_features(frames),
            'temperature_rise_features': self._extract_temperature_rise_features(frames, timestamps)
        }
        
        self.logger.info(f"Extracted {len(features)} thermal feature groups")
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
        
        # Add temperature statistics
        temp_stats = features.get('temperature_statistics', {})
        for stat_name, stat_values in temp_stats.items():
            if isinstance(stat_values, list):
                for i, value in enumerate(stat_values):
                    flat_features[f"temp_{stat_name}_{i}"] = value
            else:
                flat_features[f"temp_{stat_name}"] = stat_values
        
        # Add hotspot features
        hotspot_features = features.get('hotspot_features', {})
        for feature_name, feature_values in hotspot_features.items():
            if isinstance(feature_values, list):
                for i, value in enumerate(feature_values):
                    flat_features[f"hotspot_{feature_name}_{i}"] = value
            else:
                flat_features[f"hotspot_{feature_name}"] = feature_values
        
        # Add entropy features
        entropy_features = features.get('entropy_features', {})
        for feature_name, feature_values in entropy_features.items():
            if isinstance(feature_values, list):
                for i, value in enumerate(feature_values):
                    flat_features[f"entropy_{feature_name}_{i}"] = value
            else:
                flat_features[f"entropy_{feature_name}"] = feature_values
        
        # Add motion features
        motion_features = features.get('motion_features', {})
        for feature_name, feature_values in motion_features.items():
            if isinstance(feature_values, list):
                for i, value in enumerate(feature_values):
                    flat_features[f"motion_{feature_name}_{i}"] = value
            else:
                flat_features[f"motion_{feature_name}"] = feature_values
        
        # Add temperature rise features
        rise_features = features.get('temperature_rise_features', {})
        for feature_name, feature_values in rise_features.items():
            if isinstance(feature_values, list):
                for i, value in enumerate(feature_values):
                    flat_features[f"rise_{feature_name}_{i}"] = value
            else:
                flat_features[f"rise_{feature_name}"] = feature_values
        
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
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(features, f, indent=2)
        
        self.logger.info(f"Saved thermal features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'temperature_statistics.max_temperature',
            'temperature_statistics.mean_temperature',
            'temperature_statistics.min_temperature',
            'temperature_statistics.temperature_range',
            'temperature_statistics.temperature_std',
            'hotspot_features.hotspot_count',
            'hotspot_features.hotspot_area_percentage',
            'hotspot_features.max_hotspot_area',
            'hotspot_features.max_hotspot_temperature',
            'entropy_features.temperature_entropy',
            'entropy_features.entropy_change',
            'motion_features.motion_detected',
            'motion_features.motion_area_percentage',
            'motion_features.motion_intensity',
            'temperature_rise_features.max_temperature_slope',
            'temperature_rise_features.mean_temperature_slope',
            'temperature_rise_features.max_temperature_acceleration'
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