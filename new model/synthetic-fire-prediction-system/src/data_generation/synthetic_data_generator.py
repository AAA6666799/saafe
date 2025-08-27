#!/usr/bin/env python3
"""
Synthetic Fire Detection Data Generator
Generates realistic sensor data for training fire detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic fire detection training data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the synthetic data generator."""
        self.config = config
        self.thermal_config = config.get('thermal', {})
        self.gas_config = config.get('gas', {})
        self.env_config = config.get('environmental', {})
        
    def generate_training_dataset(self, samples: int = 10000) -> pd.DataFrame:
        """Generate a complete training dataset."""
        logger.info(f"Generating {samples} synthetic training samples...")
        
        # Generate features
        thermal_features = self._generate_thermal_features(samples)
        gas_features = self._generate_gas_features(samples)
        env_features = self._generate_environmental_features(samples)
        
        # Combine all features
        all_features = np.concatenate([thermal_features, gas_features, env_features], axis=1)
        
        # Generate labels based on feature patterns
        labels = self._generate_labels(thermal_features, gas_features, env_features)
        
        # Create column names
        n_thermal = thermal_features.shape[1]
        n_gas = gas_features.shape[1]
        n_env = env_features.shape[1]
        
        column_names = (
            [f'thermal_{i}' for i in range(n_thermal)] +
            [f'gas_{i}' for i in range(n_gas)] +
            [f'env_{i}' for i in range(n_env)]
        )
        
        # Create DataFrame
        data = pd.DataFrame(all_features, columns=column_names)
        data['fire_detected'] = labels
        
        logger.info(f"Generated dataset with {len(data)} samples")
        logger.info(f"Fire samples: {sum(labels)}, Normal samples: {len(labels) - sum(labels)}")
        
        return data
    
    def _generate_thermal_features(self, samples: int) -> np.ndarray:
        """Generate thermal sensor features."""
        n_features = 8  # Number of thermal features
        
        # Base thermal readings (room temperature)
        base_temp = 22.0
        thermal_data = np.random.normal(base_temp, 2.0, (samples, n_features))
        
        # Add some spatial correlation between thermal sensors
        for i in range(1, n_features):
            thermal_data[:, i] += 0.3 * thermal_data[:, i-1] + np.random.normal(0, 0.5, samples)
        
        # Ensure non-negative temperatures
        thermal_data = np.maximum(thermal_data, 0.0)
        
        return thermal_data
    
    def _generate_gas_features(self, samples: int) -> np.ndarray:
        """Generate gas sensor features."""
        n_features = 6  # Number of gas features
        
        # Base gas concentrations (normal levels)
        gas_data = np.random.exponential(0.1, (samples, n_features))
        
        # Add some correlation between different gas types
        gas_data[:, 1] += 0.2 * gas_data[:, 0]  # CO and CO2 correlation
        gas_data[:, 3] += 0.15 * gas_data[:, 2]  # VOC correlation
        
        return gas_data
    
    def _generate_environmental_features(self, samples: int) -> np.ndarray:
        """Generate environmental sensor features."""
        n_features = 6  # Number of environmental features
        
        # Temperature, humidity, pressure, and VOCs
        env_data = np.zeros((samples, n_features))
        
        # Temperature (correlated with thermal but different range)
        env_data[:, 0] = np.random.normal(23.0, 3.0, samples)
        
        # Humidity (anti-correlated with temperature)
        env_data[:, 1] = np.maximum(0, np.minimum(100, 
            60 - 0.5 * (env_data[:, 0] - 23) + np.random.normal(0, 10, samples)))
        
        # Pressure
        env_data[:, 2] = np.random.normal(1013.25, 5.0, samples)
        
        # VOCs (various types)
        env_data[:, 3:] = np.random.lognormal(-2, 1, (samples, 3))
        
        return env_data
    
    def _generate_labels(self, thermal: np.ndarray, gas: np.ndarray, env: np.ndarray) -> np.ndarray:
        """Generate fire detection labels based on sensor patterns."""
        samples = thermal.shape[0]
        labels = np.zeros(samples, dtype=int)
        
        # Create fire scenarios (approximately 15% positive samples)
        fire_probability = 0.15
        
        for i in range(samples):
            fire_score = 0.0
            
            # Thermal indicators
            max_temp = np.max(thermal[i])
            avg_temp = np.mean(thermal[i])
            temp_variance = np.var(thermal[i])
            
            if max_temp > 35:  # High temperature
                fire_score += 0.3
            if avg_temp > 28:  # Elevated average temperature
                fire_score += 0.2
            if temp_variance > 20:  # High temperature variance (hotspots)
                fire_score += 0.2
            
            # Gas indicators
            max_gas = np.max(gas[i])
            gas_sum = np.sum(gas[i])
            
            if max_gas > 0.5:  # High gas concentration
                fire_score += 0.2
            if gas_sum > 1.0:  # Multiple gas types detected
                fire_score += 0.15
            
            # Environmental indicators
            humidity = env[i, 1]
            voc_levels = np.sum(env[i, 3:])
            
            if humidity < 30:  # Low humidity (dry conditions)
                fire_score += 0.1
            if voc_levels > 0.8:  # High VOC levels
                fire_score += 0.15
            
            # Add some randomness to prevent overfitting
            fire_score += np.random.normal(0, 0.1)
            
            # Determine label based on fire score
            if fire_score > 0.5:
                labels[i] = 1
            elif fire_score > 0.3 and np.random.random() < 0.3:
                labels[i] = 1  # Some borderline cases
        
        # Ensure we have roughly the desired proportion of fire samples
        current_fire_rate = np.mean(labels)
        if current_fire_rate < fire_probability * 0.8:
            # Add more fire samples
            normal_indices = np.where(labels == 0)[0]
            num_to_flip = int((fire_probability - current_fire_rate) * samples)
            flip_indices = np.random.choice(normal_indices, min(num_to_flip, len(normal_indices)), replace=False)
            labels[flip_indices] = 1
            
            # Modify features to be more fire-like
            for idx in flip_indices:
                thermal[idx] *= np.random.uniform(1.2, 2.0, thermal.shape[1])
                gas[idx] *= np.random.uniform(1.5, 3.0, gas.shape[1])
        
        elif current_fire_rate > fire_probability * 1.2:
            # Remove some fire samples
            fire_indices = np.where(labels == 1)[0]
            num_to_flip = int((current_fire_rate - fire_probability) * samples)
            flip_indices = np.random.choice(fire_indices, min(num_to_flip, len(fire_indices)), replace=False)
            labels[flip_indices] = 0
        
        return labels