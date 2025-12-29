"""
Feature fusion for synthetic fire prediction system
"""

import numpy as np
from typing import List, Dict, Any
from ai_fire_prediction_platform.core.interfaces import SensorData, FeatureVector
from ai_fire_prediction_platform.feature_engineering.extractors import (
    ThermalFeatureExtractor, GasFeatureExtractor, EnvironmentalFeatureExtractor
)


class FeatureFusionEngine:
    """Fuse features from different sensor types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thermal_extractor = ThermalFeatureExtractor(config)
        self.gas_extractor = GasFeatureExtractor(config)
        self.env_extractor = EnvironmentalFeatureExtractor(config)
        
        # Fusion feature names
        self.fusion_feature_names = [
            "temp_gas_correlation",
            "temp_env_correlation",
            "gas_env_correlation",
            "anomaly_score"
        ]
    
    def extract_features(self, sensor_data: SensorData) -> FeatureVector:
        """Extract and fuse features from all sensor data"""
        # Extract individual features
        thermal_features = self.thermal_extractor.extract_features(sensor_data)
        gas_features = self.gas_extractor.extract_features(sensor_data)
        env_features = self.env_extractor.extract_features(sensor_data)
        
        # Create fusion features
        fusion_features = self._create_fusion_features(
            thermal_features, gas_features, env_features
        )
        
        # Calculate feature quality (basic implementation)
        feature_quality = self._calculate_feature_quality(
            thermal_features, gas_features, env_features
        )
        
        return FeatureVector(
            timestamp=sensor_data.timestamp,
            thermal_features=thermal_features,
            gas_features=gas_features,
            environmental_features=env_features,
            fusion_features=fusion_features,
            feature_quality=feature_quality
        )
    
    def _create_fusion_features(self, thermal_features: np.ndarray, 
                               gas_features: np.ndarray, 
                               env_features: np.ndarray) -> np.ndarray:
        """Create fusion features from individual features"""
        # Correlation between thermal and gas features
        if len(thermal_features) > 0 and len(gas_features) > 0:
            # Use mean temperature and mean gas concentration for correlation
            temp_mean = thermal_features[0] if len(thermal_features) > 0 else 0
            gas_mean = np.mean(gas_features[:len(gas_features)//2]) if len(gas_features) > 0 else 0
            temp_gas_corr = self._safe_correlation(temp_mean, gas_mean)
        else:
            temp_gas_corr = 0.0
        
        # Correlation between thermal and environmental features
        if len(thermal_features) > 0 and len(env_features) > 0:
            temp_mean = thermal_features[0] if len(thermal_features) > 0 else 0
            env_temp = env_features[0] if len(env_features) > 0 else 0
            temp_env_corr = self._safe_correlation(temp_mean, env_temp)
        else:
            temp_env_corr = 0.0
        
        # Correlation between gas and environmental features
        if len(gas_features) > 0 and len(env_features) > 0:
            gas_mean = np.mean(gas_features[:len(gas_features)//2]) if len(gas_features) > 0 else 0
            env_voc = env_features[3] if len(env_features) > 3 else 0
            gas_env_corr = self._safe_correlation(gas_mean, env_voc)
        else:
            gas_env_corr = 0.0
        
        # Overall anomaly score (combination of individual anomalies)
        gas_anomalies = gas_features[len(gas_features)//2:] if len(gas_features) > 0 else np.array([])
        if len(gas_anomalies) > 0:
            anomaly_score = np.mean(gas_anomalies)
        else:
            anomaly_score = 0.0
        
        return np.array([temp_gas_corr, temp_env_corr, gas_env_corr, anomaly_score])
    
    def _safe_correlation(self, x: float, y: float) -> float:
        """Calculate a safe correlation-like measure"""
        # Simple implementation - in practice, this would be more sophisticated
        if x == 0 and y == 0:
            return 0.0
        # Use a simple similarity measure to avoid correlation calculation issues
        return np.clip(1.0 - abs(x - y) / (abs(x) + abs(y) + 1e-8), -1.0, 1.0)
    
    def _calculate_feature_quality(self, thermal_features: np.ndarray,
                                  gas_features: np.ndarray,
                                  env_features: np.ndarray) -> float:
        """Calculate overall feature quality score"""
        # Count how many features have valid (non-zero) values
        total_features = len(thermal_features) + len(gas_features) + len(env_features)
        if total_features == 0:
            return 0.0
        
        valid_features = 0
        valid_features += np.sum(np.abs(thermal_features) > 1e-6)
        valid_features += np.sum(np.abs(gas_features) > 1e-6)
        valid_features += np.sum(np.abs(env_features) > 1e-6)
        
        return valid_features / total_features if total_features > 0 else 0.0
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get names of all features"""
        return {
            "thermal": self.thermal_extractor.get_feature_names(),
            "gas": self.gas_extractor.get_feature_names(),
            "environmental": self.env_extractor.get_feature_names(),
            "fusion": self.fusion_feature_names
        }