"""
Hotspot Detector for thermal image analysis.

This module provides a feature extractor that detects and characterizes hotspots in thermal images.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import ndimage
from skimage import measure, filters

from ...base import ThermalFeatureExtractor


class HotspotDetector(ThermalFeatureExtractor):
    """
    Feature extractor for detecting and characterizing hotspots in thermal images.
    
    This class analyzes thermal images to identify regions with temperatures above
    specified thresholds, characterizes their properties, and tracks their evolution
    over time.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hotspot detector.
        
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
        required_params = ['hotspot_threshold', 'min_hotspot_size']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Set default values for optional parameters
        if 'multi_threshold_levels' not in self.config:
            self.config['multi_threshold_levels'] = [
                self.config['hotspot_threshold'],
                self.config['hotspot_threshold'] + 10,
                self.config['hotspot_threshold'] + 20
            ]
        
        if 'track_hotspots' not in self.config:
            self.config['track_hotspots'] = True
        
        if 'adaptive_threshold' not in self.config:
            self.config['adaptive_threshold'] = False
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract hotspot features from thermal data.
        
        Args:
            data: Input thermal data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting hotspot features")
        
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
        
        # Extract hotspot features
        hotspot_features = self._extract_hotspot_features(frames, timestamps)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'frame_count': len(frames),
            'frame_shape': frames[0].shape if frames else None,
            'hotspot_features': hotspot_features
        }
        
        self.logger.info(f"Extracted hotspot features from {len(frames)} frames")
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
        
        # Add hotspot features
        hotspot_features = features.get('hotspot_features', {})
        for feature_name, feature_values in hotspot_features.items():
            if isinstance(feature_values, list):
                for i, value in enumerate(feature_values):
                    flat_features[f"hotspot_{feature_name}_{i}"] = value
            else:
                flat_features[f"hotspot_{feature_name}"] = feature_values
        
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
        
        self.logger.info(f"Saved hotspot features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'hotspot_count',
            'hotspot_area_percentage',
            'max_hotspot_area',
            'max_hotspot_temperature',
            'hotspot_centroid_x',
            'hotspot_centroid_y',
            'hotspot_intensity_ratio',
            'hotspot_growth_rate',
            'hotspot_temperature_gradient',
            'hotspot_shape_complexity'
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
        # Use adaptive threshold if configured
        if self.config.get('adaptive_threshold', False):
            # Calculate adaptive threshold based on image statistics
            mean_temp = np.mean(thermal_frame)
            std_temp = np.std(thermal_frame)
            threshold_temp = mean_temp + (std_temp * 2)  # 2 standard deviations above mean
        
        # Create binary mask of hotspots
        hotspot_mask = thermal_frame > threshold_temp
        
        # Apply minimum size filter
        min_size = self.config.get('min_hotspot_size', 5)
        if min_size > 1:
            hotspot_mask = ndimage.binary_opening(hotspot_mask, structure=np.ones((3, 3)))
            labeled_mask, _ = ndimage.label(hotspot_mask)
            sizes = np.bincount(labeled_mask.ravel())
            mask_sizes = sizes > min_size
            mask_sizes[0] = 0  # Background should not be removed
            hotspot_mask = mask_sizes[labeled_mask]
        
        # Label connected components
        labeled_mask, num_hotspots = ndimage.label(hotspot_mask)
        
        # Extract properties of each hotspot
        if num_hotspots > 0:
            props = measure.regionprops(labeled_mask, intensity_image=thermal_frame)
            
            hotspots = []
            for prop in props:
                y0, x0, y1, x1 = prop.bbox
                
                # Calculate temperature gradient within hotspot
                if prop.area > 1:
                    # Use Sobel filter to calculate gradient magnitude
                    region = thermal_frame[y0:y1, x0:x1]
                    gradient_y = filters.sobel_h(region)
                    gradient_x = filters.sobel_v(region)
                    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                    mean_gradient = np.mean(gradient_magnitude)
                else:
                    mean_gradient = 0.0
                
                # Calculate shape complexity (perimeter^2 / area)
                shape_complexity = (prop.perimeter**2) / (4 * np.pi * prop.area) if prop.perimeter > 0 else 1.0
                
                hotspots.append({
                    'area': int(prop.area),
                    'centroid': (int(prop.centroid[1]), int(prop.centroid[0])),
                    'bbox': (x0, y0, x1 - x0, y1 - y0),
                    'max_temperature': float(prop.max_intensity),
                    'mean_temperature': float(prop.mean_intensity),
                    'temperature_gradient': float(mean_gradient),
                    'shape_complexity': float(shape_complexity),
                    'eccentricity': float(prop.eccentricity) if hasattr(prop, 'eccentricity') else 0.0,
                    'orientation': float(prop.orientation) if hasattr(prop, 'orientation') else 0.0
                })
            
            return hotspots
        else:
            return []
    
    def _extract_hotspot_features(self, frames: List[np.ndarray], timestamps: List[Any]) -> Dict[str, Any]:
        """
        Extract hotspot features from all frames.
        
        Args:
            frames: List of thermal frames
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary containing hotspot features
        """
        # Get thresholds for multi-level analysis
        thresholds = self.config.get('multi_threshold_levels', [self.config['hotspot_threshold']])
        
        # Detect hotspots in each frame at each threshold level
        frame_hotspots = []
        for frame in frames:
            frame_results = []
            for threshold in thresholds:
                hotspots = self.detect_hotspots(frame, threshold)
                frame_results.append({
                    'threshold': threshold,
                    'hotspots': hotspots
                })
            frame_hotspots.append(frame_results)
        
        # Calculate hotspot counts for each threshold
        hotspot_counts = {}
        for threshold_idx, threshold in enumerate(thresholds):
            counts = [len(frame[threshold_idx]['hotspots']) for frame in frame_hotspots]
            hotspot_counts[f'threshold_{threshold}'] = counts
        
        # Calculate hotspot areas for primary threshold
        primary_threshold_idx = 0
        hotspot_areas = []
        for frame_idx, frame_result in enumerate(frame_hotspots):
            hotspots = frame_result[primary_threshold_idx]['hotspots']
            if hotspots:
                total_area = sum(hotspot['area'] for hotspot in hotspots)
                frame_area = frames[frame_idx].size  # Assuming all frames have the same size
                hotspot_areas.append(total_area / frame_area * 100.0)  # As percentage
            else:
                hotspot_areas.append(0.0)
        
        # Calculate maximum hotspot temperatures
        max_hotspot_temps = []
        for frame_result in frame_hotspots:
            hotspots = frame_result[primary_threshold_idx]['hotspots']
            if hotspots:
                max_temp = max(hotspot['max_temperature'] for hotspot in hotspots)
                max_hotspot_temps.append(max_temp)
            else:
                max_hotspot_temps.append(0.0)
        
        # Track hotspot growth if enabled and we have multiple frames
        hotspot_growth_rates = []
        if self.config.get('track_hotspots', True) and len(frames) > 1:
            for i in range(1, len(frame_hotspots)):
                prev_hotspots = frame_hotspots[i-1][primary_threshold_idx]['hotspots']
                curr_hotspots = frame_hotspots[i][primary_threshold_idx]['hotspots']
                
                if prev_hotspots and curr_hotspots:
                    # Simple tracking based on centroid proximity
                    # In a real implementation, more sophisticated tracking would be used
                    prev_total_area = sum(h['area'] for h in prev_hotspots)
                    curr_total_area = sum(h['area'] for h in curr_hotspots)
                    
                    if prev_total_area > 0:
                        growth_rate = (curr_total_area - prev_total_area) / prev_total_area
                    else:
                        growth_rate = 1.0 if curr_total_area > 0 else 0.0
                    
                    hotspot_growth_rates.append(growth_rate)
                else:
                    hotspot_growth_rates.append(0.0)
            
            # Add a placeholder for the first frame
            hotspot_growth_rates.insert(0, 0.0)
        
        # Calculate hotspot intensity ratios (max temp / mean temp)
        intensity_ratios = []
        for frame_result in frame_hotspots:
            hotspots = frame_result[primary_threshold_idx]['hotspots']
            if hotspots:
                ratios = [hotspot['max_temperature'] / hotspot['mean_temperature'] 
                         if hotspot['mean_temperature'] > 0 else 1.0 
                         for hotspot in hotspots]
                intensity_ratios.append(np.mean(ratios))
            else:
                intensity_ratios.append(1.0)
        
        # Calculate average temperature gradients within hotspots
        temperature_gradients = []
        for frame_result in frame_hotspots:
            hotspots = frame_result[primary_threshold_idx]['hotspots']
            if hotspots:
                gradients = [hotspot['temperature_gradient'] for hotspot in hotspots]
                temperature_gradients.append(np.mean(gradients))
            else:
                temperature_gradients.append(0.0)
        
        # Calculate average shape complexity
        shape_complexities = []
        for frame_result in frame_hotspots:
            hotspots = frame_result[primary_threshold_idx]['hotspots']
            if hotspots:
                complexities = [hotspot['shape_complexity'] for hotspot in hotspots]
                shape_complexities.append(np.mean(complexities))
            else:
                shape_complexities.append(1.0)  # Circle has complexity 1.0
        
        # Return hotspot features
        return {
            'hotspot_count': hotspot_counts[f'threshold_{thresholds[primary_threshold_idx]}'],
            'max_hotspot_count': int(np.max(hotspot_counts[f'threshold_{thresholds[primary_threshold_idx]}'])) 
                               if hotspot_counts else 0,
            'hotspot_area_percentage': hotspot_areas,
            'max_hotspot_area_percentage': float(np.max(hotspot_areas)) if hotspot_areas else 0.0,
            'max_hotspot_temperature': max_hotspot_temps,
            'max_max_hotspot_temperature': float(np.max(max_hotspot_temps)) if max_hotspot_temps else 0.0,
            'hotspot_growth_rate': hotspot_growth_rates,
            'max_growth_rate': float(np.max(hotspot_growth_rates)) if hotspot_growth_rates else 0.0,
            'hotspot_intensity_ratio': intensity_ratios,
            'avg_intensity_ratio': float(np.mean(intensity_ratios)) if intensity_ratios else 1.0,
            'temperature_gradient': temperature_gradients,
            'avg_temperature_gradient': float(np.mean(temperature_gradients)) if temperature_gradients else 0.0,
            'shape_complexity': shape_complexities,
            'avg_shape_complexity': float(np.mean(shape_complexities)) if shape_complexities else 1.0,
            'multi_threshold_counts': hotspot_counts
        }