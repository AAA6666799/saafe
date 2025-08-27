"""
Thermal Pattern Recognizer for thermal image analysis.

This module provides a feature extractor that identifies patterns in thermal data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import ndimage, stats, signal
from skimage import feature, measure, segmentation
import cv2

from ...base import ThermalFeatureExtractor


class ThermalPatternRecognizer(ThermalFeatureExtractor):
    """
    Feature extractor for identifying patterns in thermal data.
    
    This class analyzes thermal images to identify recurring patterns,
    spatial structures, and characteristic thermal signatures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the thermal pattern recognizer.
        
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
            self.config['pattern_detection_method'] = 'glcm'  # Options: 'glcm', 'lbp', 'hog'
        
        if 'glcm_distances' not in self.config:
            self.config['glcm_distances'] = [1, 3, 5]
        
        if 'glcm_angles' not in self.config:
            self.config['glcm_angles'] = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        if 'lbp_radius' not in self.config:
            self.config['lbp_radius'] = 3
        
        if 'lbp_n_points' not in self.config:
            self.config['lbp_n_points'] = 24
        
        if 'hog_orientations' not in self.config:
            self.config['hog_orientations'] = 9
        
        if 'hog_pixels_per_cell' not in self.config:
            self.config['hog_pixels_per_cell'] = (8, 8)
        
        if 'hog_cells_per_block' not in self.config:
            self.config['hog_cells_per_block'] = (3, 3)
        
        if 'pattern_similarity_threshold' not in self.config:
            self.config['pattern_similarity_threshold'] = 0.8
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract pattern features from thermal data.
        
        Args:
            data: Input thermal data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting thermal pattern features")
        
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
        
        # Extract pattern features
        pattern_features = self._extract_pattern_features(frames, timestamps)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'frame_count': len(frames),
            'frame_shape': frames[0].shape if frames else None,
            'pattern_features': pattern_features
        }
        
        self.logger.info(f"Extracted thermal pattern features from {len(frames)} frames")
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
        for feature_name, feature_values in pattern_features.items():
            if isinstance(feature_values, list):
                if all(isinstance(item, (int, float)) for item in feature_values):
                    for i, value in enumerate(feature_values):
                        flat_features[f"pattern_{feature_name}_{i}"] = value
                else:
                    # Skip complex nested lists
                    flat_features[f"pattern_{feature_name}_count"] = len(feature_values)
            elif isinstance(feature_values, dict):
                for sub_name, sub_value in feature_values.items():
                    flat_features[f"pattern_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"pattern_{feature_name}"] = feature_values
        
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
        
        self.logger.info(f"Saved thermal pattern features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'texture_features',
            'pattern_count',
            'pattern_sizes',
            'pattern_temperatures',
            'pattern_shapes',
            'pattern_orientations',
            'pattern_symmetry',
            'pattern_regularity',
            'pattern_complexity',
            'pattern_similarity'
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
    
    def _calculate_glcm_features(self, thermal_frame: np.ndarray) -> Dict[str, float]:
        """
        Calculate GLCM (Gray-Level Co-occurrence Matrix) texture features.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Dictionary of GLCM texture features
        """
        from skimage.feature import greycomatrix, greycoprops
        
        # Normalize and quantize the image to 256 levels
        min_val = np.min(thermal_frame)
        max_val = np.max(thermal_frame)
        if max_val > min_val:
            normalized = ((thermal_frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(thermal_frame, dtype=np.uint8)
        
        # Calculate GLCM
        distances = self.config.get('glcm_distances', [1, 3, 5])
        angles = self.config.get('glcm_angles', [0, np.pi/4, np.pi/2, 3*np.pi/4])
        
        try:
            glcm = greycomatrix(normalized, distances=distances, angles=angles, 
                              levels=256, symmetric=True, normed=True)
            
            # Calculate GLCM properties
            contrast = greycoprops(glcm, 'contrast').mean()
            dissimilarity = greycoprops(glcm, 'dissimilarity').mean()
            homogeneity = greycoprops(glcm, 'homogeneity').mean()
            energy = greycoprops(glcm, 'energy').mean()
            correlation = greycoprops(glcm, 'correlation').mean()
            ASM = greycoprops(glcm, 'ASM').mean()
            
            return {
                'contrast': float(contrast),
                'dissimilarity': float(dissimilarity),
                'homogeneity': float(homogeneity),
                'energy': float(energy),
                'correlation': float(correlation),
                'ASM': float(ASM)
            }
        except Exception as e:
            self.logger.warning(f"Error calculating GLCM features: {str(e)}")
            return {
                'contrast': 0.0,
                'dissimilarity': 0.0,
                'homogeneity': 0.0,
                'energy': 0.0,
                'correlation': 0.0,
                'ASM': 0.0
            }
    
    def _calculate_lbp_features(self, thermal_frame: np.ndarray) -> Dict[str, Any]:
        """
        Calculate LBP (Local Binary Pattern) texture features.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Dictionary of LBP texture features
        """
        from skimage.feature import local_binary_pattern
        
        # Normalize the image
        min_val = np.min(thermal_frame)
        max_val = np.max(thermal_frame)
        if max_val > min_val:
            normalized = (thermal_frame - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(thermal_frame, dtype=float)
        
        # Calculate LBP
        radius = self.config.get('lbp_radius', 3)
        n_points = self.config.get('lbp_n_points', 24)
        
        try:
            lbp = local_binary_pattern(normalized, n_points, radius, method='uniform')
            
            # Calculate LBP histogram
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
            
            # Calculate statistics from histogram
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            uniformity = np.sum(hist**2)
            
            return {
                'histogram': hist.tolist(),
                'entropy': float(entropy),
                'uniformity': float(uniformity),
                'mean': float(np.mean(lbp)),
                'std': float(np.std(lbp))
            }
        except Exception as e:
            self.logger.warning(f"Error calculating LBP features: {str(e)}")
            return {
                'histogram': [],
                'entropy': 0.0,
                'uniformity': 0.0,
                'mean': 0.0,
                'std': 0.0
            }
    
    def _calculate_hog_features(self, thermal_frame: np.ndarray) -> Dict[str, Any]:
        """
        Calculate HOG (Histogram of Oriented Gradients) features.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Dictionary of HOG features
        """
        from skimage.feature import hog
        
        # Normalize the image
        min_val = np.min(thermal_frame)
        max_val = np.max(thermal_frame)
        if max_val > min_val:
            normalized = (thermal_frame - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(thermal_frame, dtype=float)
        
        # Calculate HOG
        orientations = self.config.get('hog_orientations', 9)
        pixels_per_cell = self.config.get('hog_pixels_per_cell', (8, 8))
        cells_per_block = self.config.get('hog_cells_per_block', (3, 3))
        
        try:
            hog_features = hog(normalized, orientations=orientations,
                             pixels_per_cell=pixels_per_cell,
                             cells_per_block=cells_per_block,
                             visualize=False, feature_vector=True)
            
            # Calculate statistics from HOG features
            return {
                'mean': float(np.mean(hog_features)),
                'std': float(np.std(hog_features)),
                'max': float(np.max(hog_features)),
                'min': float(np.min(hog_features)),
                'feature_count': len(hog_features)
            }
        except Exception as e:
            self.logger.warning(f"Error calculating HOG features: {str(e)}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0,
                'feature_count': 0
            }
    
    def _detect_patterns(self, thermal_frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect patterns in a thermal frame using segmentation.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            List of dictionaries containing pattern information
        """
        # Normalize the image
        min_val = np.min(thermal_frame)
        max_val = np.max(thermal_frame)
        if max_val > min_val:
            normalized = ((thermal_frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(thermal_frame, dtype=np.uint8)
        
        try:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Label connected components
            labeled, num_patterns = ndimage.label(binary)
            
            if num_patterns == 0:
                return []
            
            # Extract properties of each pattern
            props = measure.regionprops(labeled, intensity_image=thermal_frame)
            
            patterns = []
            for prop in props:
                # Skip very small regions
                if prop.area < 10:
                    continue
                
                y0, x0, y1, x1 = prop.bbox
                
                # Calculate shape features
                eccentricity = prop.eccentricity if hasattr(prop, 'eccentricity') else 0.0
                orientation = prop.orientation if hasattr(prop, 'orientation') else 0.0
                extent = prop.extent if hasattr(prop, 'extent') else 0.0
                solidity = prop.solidity if hasattr(prop, 'solidity') else 0.0
                
                # Calculate symmetry
                region = binary[y0:y1, x0:x1]
                flipped_h = np.fliplr(region)
                flipped_v = np.flipud(region)
                symmetry_h = np.sum(region == flipped_h) / region.size
                symmetry_v = np.sum(region == flipped_v) / region.size
                
                patterns.append({
                    'area': int(prop.area),
                    'centroid': (int(prop.centroid[1]), int(prop.centroid[0])),
                    'bbox': (x0, y0, x1 - x0, y1 - y0),
                    'max_temperature': float(prop.max_intensity),
                    'mean_temperature': float(prop.mean_intensity),
                    'eccentricity': float(eccentricity),
                    'orientation': float(orientation),
                    'extent': float(extent),
                    'solidity': float(solidity),
                    'symmetry_h': float(symmetry_h),
                    'symmetry_v': float(symmetry_v)
                })
            
            return patterns
        except Exception as e:
            self.logger.warning(f"Error detecting patterns: {str(e)}")
            return []
    
    def _calculate_pattern_similarity(self, patterns1: List[Dict[str, Any]], patterns2: List[Dict[str, Any]]) -> float:
        """
        Calculate similarity between two sets of patterns.
        
        Args:
            patterns1: First set of patterns
            patterns2: Second set of patterns
            
        Returns:
            Similarity score between 0 and 1
        """
        if not patterns1 or not patterns2:
            return 0.0
        
        # Extract feature vectors for each pattern
        features1 = np.array([[p['area'], p['max_temperature'], p['mean_temperature'], 
                             p['eccentricity'], p['extent'], p['solidity']] 
                            for p in patterns1])
        
        features2 = np.array([[p['area'], p['max_temperature'], p['mean_temperature'], 
                             p['eccentricity'], p['extent'], p['solidity']] 
                            for p in patterns2])
        
        # Normalize features
        if features1.shape[0] > 0 and features2.shape[0] > 0:
            # Combine features for normalization
            all_features = np.vstack([features1, features2])
            
            # Calculate min and max for each feature
            min_vals = np.min(all_features, axis=0)
            max_vals = np.max(all_features, axis=0)
            
            # Avoid division by zero
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0
            
            # Normalize
            features1_norm = (features1 - min_vals) / range_vals
            features2_norm = (features2 - min_vals) / range_vals
            
            # Calculate pairwise distances
            from scipy.spatial.distance import cdist
            distances = cdist(features1_norm, features2_norm, 'euclidean')
            
            # Calculate similarity as inverse of average minimum distance
            min_distances = np.min(distances, axis=1)
            similarity = 1.0 / (1.0 + np.mean(min_distances))
            
            return float(similarity)
        else:
            return 0.0
    
    def _extract_pattern_features(self, frames: List[np.ndarray], timestamps: List[Any]) -> Dict[str, Any]:
        """
        Extract pattern features from all frames.
        
        Args:
            frames: List of thermal frames
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary containing pattern features
        """
        # Extract texture features for each frame
        texture_features = []
        pattern_counts = []
        pattern_lists = []
        
        method = self.config.get('pattern_detection_method', 'glcm')
        
        for frame in frames:
            # Calculate texture features based on selected method
            if method == 'glcm':
                texture = self._calculate_glcm_features(frame)
            elif method == 'lbp':
                texture = self._calculate_lbp_features(frame)
            elif method == 'hog':
                texture = self._calculate_hog_features(frame)
            else:
                texture = self._calculate_glcm_features(frame)
            
            texture_features.append(texture)
            
            # Detect patterns
            patterns = self._detect_patterns(frame)
            pattern_lists.append(patterns)
            pattern_counts.append(len(patterns))
        
        # Calculate pattern statistics
        pattern_sizes = []
        pattern_temperatures = []
        pattern_shapes = []
        pattern_orientations = []
        pattern_symmetries = []
        
        for patterns in pattern_lists:
            if patterns:
                sizes = [p['area'] for p in patterns]
                temps = [p['mean_temperature'] for p in patterns]
                eccentricities = [p['eccentricity'] for p in patterns]
                orientations = [p['orientation'] for p in patterns]
                symmetries = [(p['symmetry_h'] + p['symmetry_v']) / 2 for p in patterns]
                
                pattern_sizes.append(np.mean(sizes))
                pattern_temperatures.append(np.mean(temps))
                pattern_shapes.append(np.mean(eccentricities))
                pattern_orientations.append(np.mean(orientations))
                pattern_symmetries.append(np.mean(symmetries))
            else:
                pattern_sizes.append(0.0)
                pattern_temperatures.append(0.0)
                pattern_shapes.append(0.0)
                pattern_orientations.append(0.0)
                pattern_symmetries.append(0.0)
        
        # Calculate pattern regularity (consistency of pattern count)
        pattern_regularity = 1.0 - (np.std(pattern_counts) / np.mean(pattern_counts)) if np.mean(pattern_counts) > 0 else 0.0
        
        # Calculate pattern complexity (average of pattern count and shape complexity)
        pattern_complexity = np.mean(pattern_counts) * np.mean(pattern_shapes) if pattern_shapes else 0.0
        
        # Calculate pattern similarity between consecutive frames
        pattern_similarities = []
        if len(pattern_lists) > 1:
            for i in range(1, len(pattern_lists)):
                similarity = self._calculate_pattern_similarity(pattern_lists[i-1], pattern_lists[i])
                pattern_similarities.append(similarity)
            
            # Add a placeholder for the first frame
            pattern_similarities.insert(0, 1.0)
        else:
            pattern_similarities = [1.0] * len(pattern_lists)
        
        # Aggregate texture features
        aggregated_texture = {}
        if texture_features:
            for key in texture_features[0].keys():
                if key == 'histogram':
                    continue  # Skip histograms for aggregation
                
                values = [tf.get(key, 0.0) for tf in texture_features]
                aggregated_texture[key] = float(np.mean(values))
        
        # Return pattern features
        return {
            'texture_features': aggregated_texture,
            'pattern_count': pattern_counts,
            'avg_pattern_count': float(np.mean(pattern_counts)) if pattern_counts else 0.0,
            'pattern_sizes': pattern_sizes,
            'avg_pattern_size': float(np.mean(pattern_sizes)) if pattern_sizes else 0.0,
            'pattern_temperatures': pattern_temperatures,
            'avg_pattern_temperature': float(np.mean(pattern_temperatures)) if pattern_temperatures else 0.0,
            'pattern_shapes': pattern_shapes,
            'avg_pattern_shape': float(np.mean(pattern_shapes)) if pattern_shapes else 0.0,
            'pattern_orientations': pattern_orientations,
            'avg_pattern_orientation': float(np.mean(pattern_orientations)) if pattern_orientations else 0.0,
            'pattern_symmetry': pattern_symmetries,
            'avg_pattern_symmetry': float(np.mean(pattern_symmetries)) if pattern_symmetries else 0.0,
            'pattern_regularity': float(pattern_regularity),
            'pattern_complexity': float(pattern_complexity),
            'pattern_similarity': pattern_similarities,
            'avg_pattern_similarity': float(np.mean(pattern_similarities)) if pattern_similarities else 1.0
        }