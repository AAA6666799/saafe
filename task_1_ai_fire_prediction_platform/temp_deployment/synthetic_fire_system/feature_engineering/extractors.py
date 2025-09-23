"""
Feature extraction for synthetic fire prediction system
"""

import numpy as np
from typing import List, Dict, Any
from synthetic_fire_system.core.interfaces import SensorData, FeatureVector, FeatureExtractor


class ThermalFeatureExtractor(FeatureExtractor):
    """Extract features from thermal image data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_names = [
            "mean_temp", "max_temp", "min_temp", "temp_std",
            "temp_gradient_x", "temp_gradient_y", "hotspot_count", "hotspot_intensity"
        ]
    
    def extract_features(self, sensor_data: SensorData) -> np.ndarray:
        """Extract thermal features from sensor data"""
        if sensor_data.thermal_frame is None:
            return np.zeros(len(self.feature_names))
        
        thermal_frame = sensor_data.thermal_frame
        
        # Basic statistical features
        mean_temp = np.mean(thermal_frame)
        max_temp = np.max(thermal_frame)
        min_temp = np.min(thermal_frame)
        temp_std = np.std(thermal_frame)
        
        # Gradient features
        grad_x = np.gradient(thermal_frame, axis=1)
        grad_y = np.gradient(thermal_frame, axis=0)
        temp_gradient_x = np.mean(np.abs(grad_x))
        temp_gradient_y = np.mean(np.abs(grad_y))
        
        # Hotspot detection (areas significantly hotter than average)
        threshold = mean_temp + 2 * temp_std
        hotspot_mask = thermal_frame > threshold
        hotspot_count = np.sum(hotspot_mask)
        
        # Average intensity of hotspots
        if hotspot_count > 0:
            hotspot_intensity = np.mean(thermal_frame[hotspot_mask])
        else:
            hotspot_intensity = 0.0
        
        features = np.array([
            mean_temp, max_temp, min_temp, temp_std,
            temp_gradient_x, temp_gradient_y, hotspot_count, hotspot_intensity
        ])
        
        return features
    
    def validate_features(self, features: np.ndarray) -> bool:
        """Validate extracted thermal features"""
        if len(features) != len(self.feature_names):
            return False
        
        # Check for valid ranges (basic validation)
        if not np.isfinite(features).all():
            return False
            
        return True
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        return self.feature_names


class GasFeatureExtractor(FeatureExtractor):
    """Extract features from gas sensor data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Updated for real S3 gas data format
        self.gas_types = ["co", "no2", "voc"]
        self.feature_names = [f"{gas}_concentration" for gas in self.gas_types] + \
                           [f"{gas}_anomaly" for gas in self.gas_types]
    
    def extract_features(self, sensor_data: SensorData) -> np.ndarray:
        """Extract gas features from sensor data"""
        if sensor_data.gas_readings is None:
            return np.zeros(len(self.feature_names))
        
        gas_readings = sensor_data.gas_readings
        
        # Concentration features
        concentrations = []
        anomalies = []
        
        # Normal concentration ranges (for anomaly detection)
        normal_ranges = {
            "co": (0, 10),      # CO in ppm
            "no2": (0, 5),      # NO2 in ppm
            "voc": (0, 500)     # VOC in ppb
        }
        
        for gas_type in self.gas_types:
            concentration = gas_readings.get(gas_type, 0.0)
            concentrations.append(concentration)
            
            # Calculate anomaly score (z-score based on normal range)
            if gas_type in normal_ranges:
                normal_min, normal_max = normal_ranges[gas_type]
                normal_mean = (normal_min + normal_max) / 2
                normal_std = (normal_max - normal_min) / 4  # Assuming 95% within range
                if normal_std > 0:
                    anomaly_score = abs(concentration - normal_mean) / normal_std
                else:
                    anomaly_score = 0.0
            else:
                anomaly_score = 0.0
                
            anomalies.append(anomaly_score)
        
        features = np.array(concentrations + anomalies)
        return features
    
    def validate_features(self, features: np.ndarray) -> bool:
        """Validate extracted gas features"""
        if len(features) != len(self.feature_names):
            return False
        
        # Check for valid ranges (basic validation)
        if not np.isfinite(features).all():
            return False
            
        return True
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        return self.feature_names


class EnvironmentalFeatureExtractor(FeatureExtractor):
    """Extract features from environmental sensor data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_names = ["temperature", "humidity", "pressure", "voc"]
    
    def extract_features(self, sensor_data: SensorData) -> np.ndarray:
        """Extract environmental features from sensor data"""
        if sensor_data.environmental_data is None:
            return np.zeros(len(self.feature_names))
        
        env_data = sensor_data.environmental_data
        
        features = np.array([
            env_data.get("temperature", 0.0),
            env_data.get("humidity", 0.0),
            env_data.get("pressure", 0.0),
            env_data.get("voc", 0.0)
        ])
        
        return features
    
    def validate_features(self, features: np.ndarray) -> bool:
        """Validate extracted environmental features"""
        if len(features) != len(self.feature_names):
            return False
        
        # Check for valid ranges (basic validation)
        if not np.isfinite(features).all():
            return False
            
        return True
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        return self.feature_names