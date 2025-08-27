"""
Temperature Gradient Extractor for thermal image analysis.

This module provides a feature extractor that extracts temperature gradient features from thermal images.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import ndimage
from skimage import filters, feature

from ...base import ThermalFeatureExtractor


class TemperatureGradientExtractor(ThermalFeatureExtractor):
    """
    Feature extractor for temperature gradients in thermal images.
    
    This class analyzes thermal images to extract gradient-based features,
    including gradient magnitude, direction, and spatial patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the temperature gradient extractor.
        
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
        if 'gradient_method' not in self.config:
            self.config['gradient_method'] = 'sobel'  # Options: 'sobel', 'prewitt', 'scharr'
        
        if 'edge_detection_threshold' not in self.config:
            self.config['edge_detection_threshold'] = 0.1
        
        if 'gradient_regions' not in self.config:
            self.config['gradient_regions'] = None  # Optional regions to analyze
        
        if 'normalize_gradients' not in self.config:
            self.config['normalize_gradients'] = True
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract temperature gradient features from thermal data.
        
        Args:
            data: Input thermal data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting temperature gradient features")
        
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
        
        # Extract gradient features
        gradient_features = self._extract_gradient_features(frames, timestamps)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'frame_count': len(frames),
            'frame_shape': frames[0].shape if frames else None,
            'gradient_features': gradient_features
        }
        
        self.logger.info(f"Extracted temperature gradient features from {len(frames)} frames")
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
        
        # Add gradient features
        gradient_features = features.get('gradient_features', {})
        for feature_name, feature_values in gradient_features.items():
            if isinstance(feature_values, list):
                for i, value in enumerate(feature_values):
                    flat_features[f"gradient_{feature_name}_{i}"] = value
            else:
                flat_features[f"gradient_{feature_name}"] = feature_values
        
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
        
        self.logger.info(f"Saved temperature gradient features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'mean_gradient_magnitude',
            'max_gradient_magnitude',
            'gradient_direction_histogram',
            'edge_pixel_percentage',
            'gradient_magnitude_histogram',
            'gradient_entropy',
            'gradient_uniformity',
            'gradient_contrast',
            'gradient_energy',
            'gradient_homogeneity'
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
            props = feature.regionprops(labeled_mask, intensity_image=thermal_frame)
            
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
    
    def _calculate_gradient(self, thermal_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate gradient magnitude and direction for a thermal frame.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Tuple of (gradient_magnitude, gradient_x, gradient_y)
        """
        method = self.config.get('gradient_method', 'sobel')
        
        if method == 'sobel':
            gradient_y = filters.sobel_h(thermal_frame)
            gradient_x = filters.sobel_v(thermal_frame)
        elif method == 'prewitt':
            gradient_y = filters.prewitt_h(thermal_frame)
            gradient_x = filters.prewitt_v(thermal_frame)
        elif method == 'scharr':
            gradient_y = filters.scharr_h(thermal_frame)
            gradient_x = filters.scharr_v(thermal_frame)
        else:
            self.logger.warning(f"Unknown gradient method: {method}, using sobel")
            gradient_y = filters.sobel_h(thermal_frame)
            gradient_x = filters.sobel_v(thermal_frame)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize if configured
        if self.config.get('normalize_gradients', True):
            max_val = np.max(gradient_magnitude)
            if max_val > 0:
                gradient_magnitude = gradient_magnitude / max_val
        
        return gradient_magnitude, gradient_x, gradient_y
    
    def _calculate_gradient_direction(self, gradient_x: np.ndarray, gradient_y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient direction from x and y gradients.
        
        Args:
            gradient_x: X component of gradient
            gradient_y: Y component of gradient
            
        Returns:
            Gradient direction in radians
        """
        return np.arctan2(gradient_y, gradient_x)
    
    def _calculate_gradient_histogram(self, gradient_magnitude: np.ndarray, bins: int = 10) -> List[float]:
        """
        Calculate histogram of gradient magnitudes.
        
        Args:
            gradient_magnitude: Gradient magnitude array
            bins: Number of histogram bins
            
        Returns:
            List of histogram values
        """
        hist, _ = np.histogram(gradient_magnitude, bins=bins, range=(0, 1))
        return hist.tolist()
    
    def _calculate_direction_histogram(self, gradient_direction: np.ndarray, bins: int = 8) -> List[float]:
        """
        Calculate histogram of gradient directions.
        
        Args:
            gradient_direction: Gradient direction array
            bins: Number of histogram bins
            
        Returns:
            List of histogram values
        """
        hist, _ = np.histogram(gradient_direction, bins=bins, range=(-np.pi, np.pi))
        return hist.tolist()
    
    def _detect_edges(self, gradient_magnitude: np.ndarray) -> np.ndarray:
        """
        Detect edges based on gradient magnitude.
        
        Args:
            gradient_magnitude: Gradient magnitude array
            
        Returns:
            Binary edge mask
        """
        threshold = self.config.get('edge_detection_threshold', 0.1)
        return gradient_magnitude > threshold
    
    def _calculate_texture_features(self, gradient_magnitude: np.ndarray) -> Dict[str, float]:
        """
        Calculate texture features from gradient magnitude.
        
        Args:
            gradient_magnitude: Gradient magnitude array
            
        Returns:
            Dictionary of texture features
        """
        # Normalize gradient magnitude to [0, 1] range
        if np.max(gradient_magnitude) > 0:
            normalized = gradient_magnitude / np.max(gradient_magnitude)
        else:
            normalized = gradient_magnitude
        
        # Calculate texture features
        entropy = -np.sum(normalized[normalized > 0] * np.log2(normalized[normalized > 0]))
        uniformity = np.sum(normalized**2)
        contrast = np.sum((normalized - np.mean(normalized))**2)
        energy = np.sqrt(uniformity)
        homogeneity = np.sum(1 / (1 + (normalized - np.mean(normalized))**2))
        
        return {
            'entropy': float(entropy),
            'uniformity': float(uniformity),
            'contrast': float(contrast),
            'energy': float(energy),
            'homogeneity': float(homogeneity)
        }
    
    def _extract_gradient_features(self, frames: List[np.ndarray], timestamps: List[Any]) -> Dict[str, Any]:
        """
        Extract gradient features from all frames.
        
        Args:
            frames: List of thermal frames
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary containing gradient features
        """
        # Extract gradient features for each frame
        mean_gradient_magnitudes = []
        max_gradient_magnitudes = []
        edge_percentages = []
        texture_features_list = []
        gradient_histograms = []
        direction_histograms = []
        
        for frame in frames:
            # Calculate gradient
            gradient_magnitude, gradient_x, gradient_y = self._calculate_gradient(frame)
            gradient_direction = self._calculate_gradient_direction(gradient_x, gradient_y)
            
            # Calculate basic statistics
            mean_gradient_magnitudes.append(float(np.mean(gradient_magnitude)))
            max_gradient_magnitudes.append(float(np.max(gradient_magnitude)))
            
            # Detect edges
            edges = self._detect_edges(gradient_magnitude)
            edge_percentages.append(float(np.sum(edges) / edges.size * 100.0))
            
            # Calculate histograms
            gradient_hist = self._calculate_gradient_histogram(gradient_magnitude)
            direction_hist = self._calculate_direction_histogram(gradient_direction)
            gradient_histograms.append(gradient_hist)
            direction_histograms.append(direction_hist)
            
            # Calculate texture features
            texture_features = self._calculate_texture_features(gradient_magnitude)
            texture_features_list.append(texture_features)
        
        # Calculate temporal features if we have multiple frames
        temporal_features = {}
        if len(frames) > 1:
            # Calculate changes in mean gradient magnitude
            mean_gradient_changes = [abs(mean_gradient_magnitudes[i] - mean_gradient_magnitudes[i-1]) 
                                   for i in range(1, len(mean_gradient_magnitudes))]
            
            temporal_features = {
                'mean_gradient_changes': mean_gradient_changes,
                'max_gradient_change': float(np.max(mean_gradient_changes)) if mean_gradient_changes else 0.0,
                'mean_gradient_change': float(np.mean(mean_gradient_changes)) if mean_gradient_changes else 0.0
            }
        
        # Extract texture features across all frames
        avg_texture_features = {}
        for key in ['entropy', 'uniformity', 'contrast', 'energy', 'homogeneity']:
            values = [features[key] for features in texture_features_list]
            avg_texture_features[key] = float(np.mean(values)) if values else 0.0
        
        # Return gradient features
        return {
            'mean_gradient_magnitude': mean_gradient_magnitudes,
            'avg_mean_gradient_magnitude': float(np.mean(mean_gradient_magnitudes)) if mean_gradient_magnitudes else 0.0,
            'max_gradient_magnitude': max_gradient_magnitudes,
            'avg_max_gradient_magnitude': float(np.mean(max_gradient_magnitudes)) if max_gradient_magnitudes else 0.0,
            'edge_pixel_percentage': edge_percentages,
            'avg_edge_pixel_percentage': float(np.mean(edge_percentages)) if edge_percentages else 0.0,
            'gradient_magnitude_histogram': gradient_histograms[-1] if gradient_histograms else [],
            'gradient_direction_histogram': direction_histograms[-1] if direction_histograms else [],
            'gradient_entropy': avg_texture_features.get('entropy', 0.0),
            'gradient_uniformity': avg_texture_features.get('uniformity', 0.0),
            'gradient_contrast': avg_texture_features.get('contrast', 0.0),
            'gradient_energy': avg_texture_features.get('energy', 0.0),
            'gradient_homogeneity': avg_texture_features.get('homogeneity', 0.0),
            **temporal_features
        }