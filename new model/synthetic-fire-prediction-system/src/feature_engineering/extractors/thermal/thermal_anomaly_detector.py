"""
Thermal Anomaly Detector for thermal image analysis.

This module provides a feature extractor that detects anomalies in thermal data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy import ndimage, stats, signal
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ...base import ThermalFeatureExtractor


class ThermalAnomalyDetector(ThermalFeatureExtractor):
    """
    Feature extractor for detecting anomalies in thermal data.
    
    This class analyzes thermal images to identify anomalous patterns,
    unusual temperature distributions, and outliers in thermal behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the thermal anomaly detector.
        
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
        if 'anomaly_detection_method' not in self.config:
            self.config['anomaly_detection_method'] = 'zscore'  # Options: 'zscore', 'isolation_forest', 'lof'
        
        if 'zscore_threshold' not in self.config:
            self.config['zscore_threshold'] = 3.0
        
        if 'contamination' not in self.config:
            self.config['contamination'] = 0.05  # For isolation forest and LOF
        
        if 'n_estimators' not in self.config:
            self.config['n_estimators'] = 100  # For isolation forest
        
        if 'n_neighbors' not in self.config:
            self.config['n_neighbors'] = 20  # For LOF
        
        if 'temporal_window' not in self.config:
            self.config['temporal_window'] = 5  # Number of frames for temporal analysis
        
        if 'spatial_window' not in self.config:
            self.config['spatial_window'] = 5  # Size of spatial window for local anomaly detection
        
        if 'use_spatial_context' not in self.config:
            self.config['use_spatial_context'] = True
        
        if 'use_temporal_context' not in self.config:
            self.config['use_temporal_context'] = True
    
    def extract_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract anomaly features from thermal data.
        
        Args:
            data: Input thermal data, either as a dictionary or DataFrame
            
        Returns:
            Dictionary containing the extracted features
        """
        self.logger.info("Extracting thermal anomaly features")
        
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
        
        # Extract anomaly features
        anomaly_features = self._extract_anomaly_features(frames, timestamps)
        
        # Prepare result
        features = {
            'extraction_time': datetime.now().isoformat(),
            'frame_count': len(frames),
            'frame_shape': frames[0].shape if frames else None,
            'anomaly_features': anomaly_features
        }
        
        self.logger.info(f"Extracted thermal anomaly features from {len(frames)} frames")
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
        
        # Add anomaly features
        anomaly_features = features.get('anomaly_features', {})
        for feature_name, feature_values in anomaly_features.items():
            if isinstance(feature_values, list):
                if len(feature_values) <= 20:  # Only include reasonably sized lists
                    for i, value in enumerate(feature_values):
                        flat_features[f"anomaly_{feature_name}_{i}"] = value
                else:
                    # For large lists, just include summary statistics
                    flat_features[f"anomaly_{feature_name}_mean"] = np.mean(feature_values)
                    flat_features[f"anomaly_{feature_name}_max"] = np.max(feature_values)
                    flat_features[f"anomaly_{feature_name}_min"] = np.min(feature_values)
                    flat_features[f"anomaly_{feature_name}_std"] = np.std(feature_values)
            elif isinstance(feature_values, dict):
                for sub_name, sub_value in feature_values.items():
                    flat_features[f"anomaly_{feature_name}_{sub_name}"] = sub_value
            else:
                flat_features[f"anomaly_{feature_name}"] = feature_values
        
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
        
        self.logger.info(f"Saved thermal anomaly features to {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        return [
            'anomaly_scores',
            'anomaly_mask',
            'anomaly_count',
            'anomaly_area_percentage',
            'anomaly_intensity',
            'anomaly_locations',
            'anomaly_clusters',
            'anomaly_persistence',
            'anomaly_growth_rate',
            'anomaly_temperature_stats',
            'anomaly_spatial_distribution',
            'anomaly_temporal_pattern'
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
    
    def _detect_zscore_anomalies(self, thermal_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Tuple of (anomaly_scores, anomaly_mask)
        """
        # Calculate mean and standard deviation
        mean_temp = np.mean(thermal_frame)
        std_temp = np.std(thermal_frame)
        
        # Calculate Z-scores
        z_scores = np.abs((thermal_frame - mean_temp) / std_temp)
        
        # Create anomaly mask
        threshold = self.config.get('zscore_threshold', 3.0)
        anomaly_mask = z_scores > threshold
        
        return z_scores, anomaly_mask
    
    def _detect_isolation_forest_anomalies(self, thermal_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Tuple of (anomaly_scores, anomaly_mask)
        """
        # Reshape the frame for scikit-learn
        height, width = thermal_frame.shape
        X = thermal_frame.reshape(-1, 1)
        
        # Add spatial coordinates if configured
        if self.config.get('use_spatial_context', True):
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))
            
            # Normalize coordinates
            coords = coords / np.array([height, width])
            
            # Combine temperature and coordinates
            X = np.column_stack((X, coords))
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        contamination = self.config.get('contamination', 0.05)
        n_estimators = self.config.get('n_estimators', 100)
        
        clf = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
        
        try:
            # Fit and predict
            clf.fit(X_scaled)
            y_pred = clf.decision_function(X_scaled)
            
            # Convert to anomaly scores (higher = more anomalous)
            anomaly_scores = -y_pred
            
            # Reshape back to image dimensions
            anomaly_scores = anomaly_scores.reshape(height, width)
            
            # Create anomaly mask
            anomaly_mask = clf.predict(X_scaled) == -1
            anomaly_mask = anomaly_mask.reshape(height, width)
            
            return anomaly_scores, anomaly_mask
        
        except Exception as e:
            self.logger.warning(f"Error in Isolation Forest: {str(e)}")
            return np.zeros_like(thermal_frame), np.zeros_like(thermal_frame, dtype=bool)
    
    def _detect_lof_anomalies(self, thermal_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Tuple of (anomaly_scores, anomaly_mask)
        """
        # Reshape the frame for scikit-learn
        height, width = thermal_frame.shape
        X = thermal_frame.reshape(-1, 1)
        
        # Add spatial coordinates if configured
        if self.config.get('use_spatial_context', True):
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))
            
            # Normalize coordinates
            coords = coords / np.array([height, width])
            
            # Combine temperature and coordinates
            X = np.column_stack((X, coords))
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train LOF
        contamination = self.config.get('contamination', 0.05)
        n_neighbors = self.config.get('n_neighbors', 20)
        
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
        
        try:
            # Fit and predict
            y_pred = clf.fit_predict(X_scaled)
            
            # Get negative outlier factor (higher = more anomalous)
            anomaly_scores = -clf._decision_function(X_scaled)
            
            # Reshape back to image dimensions
            anomaly_scores = anomaly_scores.reshape(height, width)
            
            # Create anomaly mask
            anomaly_mask = y_pred == -1
            anomaly_mask = anomaly_mask.reshape(height, width)
            
            return anomaly_scores, anomaly_mask
        
        except Exception as e:
            self.logger.warning(f"Error in LOF: {str(e)}")
            return np.zeros_like(thermal_frame), np.zeros_like(thermal_frame, dtype=bool)
    
    def _detect_anomalies(self, thermal_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in a thermal frame using the configured method.
        
        Args:
            thermal_frame: 2D numpy array representing the thermal image
            
        Returns:
            Tuple of (anomaly_scores, anomaly_mask)
        """
        method = self.config.get('anomaly_detection_method', 'zscore')
        
        if method == 'zscore':
            return self._detect_zscore_anomalies(thermal_frame)
        elif method == 'isolation_forest':
            return self._detect_isolation_forest_anomalies(thermal_frame)
        elif method == 'lof':
            return self._detect_lof_anomalies(thermal_frame)
        else:
            self.logger.warning(f"Unknown anomaly detection method: {method}, using zscore")
            return self._detect_zscore_anomalies(thermal_frame)
    
    def _analyze_anomaly_clusters(self, anomaly_mask: np.ndarray) -> Dict[str, Any]:
        """
        Analyze clusters of anomalies.
        
        Args:
            anomaly_mask: Binary mask of anomalies
            
        Returns:
            Dictionary containing cluster analysis results
        """
        # Label connected components
        labeled_mask, num_clusters = ndimage.label(anomaly_mask)
        
        if num_clusters == 0:
            return {
                'cluster_count': 0,
                'cluster_sizes': [],
                'cluster_centroids': [],
                'max_cluster_size': 0,
                'mean_cluster_size': 0
            }
        
        # Extract properties of each cluster
        from skimage import measure
        props = measure.regionprops(labeled_mask)
        
        # Calculate cluster metrics
        cluster_sizes = [prop.area for prop in props]
        cluster_centroids = [(int(prop.centroid[1]), int(prop.centroid[0])) for prop in props]
        
        return {
            'cluster_count': num_clusters,
            'cluster_sizes': cluster_sizes,
            'cluster_centroids': cluster_centroids,
            'max_cluster_size': int(np.max(cluster_sizes)) if cluster_sizes else 0,
            'mean_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0
        }
    
    def _analyze_temporal_anomalies(self, 
                                  anomaly_masks: List[np.ndarray], 
                                  anomaly_scores: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze temporal patterns in anomalies.
        
        Args:
            anomaly_masks: List of binary anomaly masks
            anomaly_scores: List of anomaly scores
            
        Returns:
            Dictionary containing temporal analysis results
        """
        if not anomaly_masks or len(anomaly_masks) < 2:
            return {
                'persistence': 0.0,
                'growth_rate': 0.0,
                'max_persistence': 0,
                'temporal_pattern': 'unknown'
            }
        
        # Calculate anomaly areas
        anomaly_areas = [np.sum(mask) for mask in anomaly_masks]
        
        # Calculate persistence (how long anomalies stay in the same location)
        persistence_map = np.zeros_like(anomaly_masks[0], dtype=int)
        for mask in anomaly_masks:
            persistence_map[mask] += 1
        
        max_persistence = int(np.max(persistence_map))
        mean_persistence = float(np.mean(persistence_map[persistence_map > 0])) if np.any(persistence_map > 0) else 0.0
        
        # Calculate growth rates
        growth_rates = []
        for i in range(1, len(anomaly_areas)):
            if anomaly_areas[i-1] > 0:
                growth_rate = (anomaly_areas[i] - anomaly_areas[i-1]) / anomaly_areas[i-1]
                growth_rates.append(growth_rate)
        
        mean_growth_rate = float(np.mean(growth_rates)) if growth_rates else 0.0
        
        # Determine temporal pattern
        if mean_growth_rate > 0.1:
            pattern = 'growing'
        elif mean_growth_rate < -0.1:
            pattern = 'shrinking'
        elif max_persistence > len(anomaly_masks) * 0.7:
            pattern = 'persistent'
        elif np.std(anomaly_areas) / (np.mean(anomaly_areas) + 1e-10) > 0.5:
            pattern = 'fluctuating'
        else:
            pattern = 'stable'
        
        return {
            'persistence': mean_persistence,
            'growth_rate': mean_growth_rate,
            'max_persistence': max_persistence,
            'temporal_pattern': pattern
        }
    
    def _analyze_spatial_distribution(self, anomaly_masks: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze spatial distribution of anomalies.
        
        Args:
            anomaly_masks: List of binary anomaly masks
            
        Returns:
            Dictionary containing spatial distribution analysis results
        """
        if not anomaly_masks:
            return {
                'distribution_type': 'unknown',
                'spatial_entropy': 0.0,
                'spatial_concentration': 0.0
            }
        
        # Combine all anomaly masks
        combined_mask = np.any(anomaly_masks, axis=0)
        
        if not np.any(combined_mask):
            return {
                'distribution_type': 'none',
                'spatial_entropy': 0.0,
                'spatial_concentration': 0.0
            }
        
        # Calculate spatial entropy
        y_indices, x_indices = np.where(combined_mask)
        
        # Create 2D histogram
        height, width = combined_mask.shape
        hist, _, _ = np.histogram2d(y_indices, x_indices, bins=[height//10, width//10])
        
        # Normalize histogram
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        # Calculate spatial concentration (inverse of normalized entropy)
        max_entropy = np.log2(hist.size)
        concentration = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Determine distribution type
        if concentration > 0.8:
            distribution_type = 'highly_concentrated'
        elif concentration > 0.5:
            distribution_type = 'concentrated'
        elif concentration > 0.3:
            distribution_type = 'moderately_dispersed'
        else:
            distribution_type = 'widely_dispersed'
        
        return {
            'distribution_type': distribution_type,
            'spatial_entropy': float(entropy),
            'spatial_concentration': float(concentration)
        }
    
    def _extract_anomaly_features(self, frames: List[np.ndarray], timestamps: List[Any]) -> Dict[str, Any]:
        """
        Extract anomaly features from all frames.
        
        Args:
            frames: List of thermal frames
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary containing anomaly features
        """
        # Detect anomalies in each frame
        anomaly_scores_list = []
        anomaly_masks_list = []
        
        for frame in frames:
            anomaly_scores, anomaly_mask = self._detect_anomalies(frame)
            anomaly_scores_list.append(anomaly_scores)
            anomaly_masks_list.append(anomaly_mask)
        
        # Calculate anomaly statistics for each frame
        anomaly_counts = [int(np.sum(mask)) for mask in anomaly_masks_list]
        anomaly_areas = [float(np.sum(mask) / mask.size * 100.0) for mask in anomaly_masks_list]
        anomaly_intensities = [float(np.mean(scores[mask])) if np.any(mask) else 0.0 
                             for scores, mask in zip(anomaly_scores_list, anomaly_masks_list)]
        
        # Analyze anomaly clusters
        cluster_analyses = [self._analyze_anomaly_clusters(mask) for mask in anomaly_masks_list]
        cluster_counts = [analysis['cluster_count'] for analysis in cluster_analyses]
        
        # Analyze temporal patterns if configured
        temporal_analysis = {}
        if self.config.get('use_temporal_context', True) and len(frames) > 1:
            temporal_analysis = self._analyze_temporal_anomalies(anomaly_masks_list, anomaly_scores_list)
        
        # Analyze spatial distribution
        spatial_analysis = self._analyze_spatial_distribution(anomaly_masks_list)
        
        # Calculate temperature statistics of anomalous regions
        anomaly_temp_stats = []
        for frame, mask in zip(frames, anomaly_masks_list):
            if np.any(mask):
                anomaly_temps = frame[mask]
                stats = {
                    'min': float(np.min(anomaly_temps)),
                    'max': float(np.max(anomaly_temps)),
                    'mean': float(np.mean(anomaly_temps)),
                    'std': float(np.std(anomaly_temps))
                }
                anomaly_temp_stats.append(stats)
            else:
                anomaly_temp_stats.append({
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0,
                    'std': 0.0
                })
        
        # Return anomaly features
        return {
            'anomaly_scores': [float(np.mean(scores)) for scores in anomaly_scores_list],
            'max_anomaly_score': float(np.max([np.max(scores) for scores in anomaly_scores_list])) 
                               if anomaly_scores_list else 0.0,
            'anomaly_count': anomaly_counts,
            'max_anomaly_count': int(np.max(anomaly_counts)) if anomaly_counts else 0,
            'anomaly_area_percentage': anomaly_areas,
            'max_anomaly_area_percentage': float(np.max(anomaly_areas)) if anomaly_areas else 0.0,
            'anomaly_intensity': anomaly_intensities,
            'max_anomaly_intensity': float(np.max(anomaly_intensities)) if anomaly_intensities else 0.0,
            'anomaly_clusters': cluster_counts,
            'max_anomaly_clusters': int(np.max(cluster_counts)) if cluster_counts else 0,
            'anomaly_persistence': float(temporal_analysis.get('persistence', 0.0)),
            'anomaly_growth_rate': float(temporal_analysis.get('growth_rate', 0.0)),
            'anomaly_temporal_pattern': temporal_analysis.get('temporal_pattern', 'unknown'),
            'anomaly_spatial_distribution': spatial_analysis.get('distribution_type', 'unknown'),
            'anomaly_spatial_concentration': float(spatial_analysis.get('spatial_concentration', 0.0)),
            'anomaly_temperature_stats': {
                'min': float(np.mean([stats['min'] for stats in anomaly_temp_stats])) if anomaly_temp_stats else 0.0,
                'max': float(np.mean([stats['max'] for stats in anomaly_temp_stats])) if anomaly_temp_stats else 0.0,
                'mean': float(np.mean([stats['mean'] for stats in anomaly_temp_stats])) if anomaly_temp_stats else 0.0,
                'std': float(np.mean([stats['std'] for stats in anomaly_temp_stats])) if anomaly_temp_stats else 0.0
            }
        }