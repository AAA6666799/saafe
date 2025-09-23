#!/usr/bin/env python3
"""
FLIR+SCD41 Synthetic Data Generator
Generates realistic FLIR Lepton 3.5 + SCD41 CO₂ sensor data for training fire detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FlirScd41DataGenerator:
    """Generate synthetic FLIR+SCD41 fire detection training data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the FLIR+SCD41 synthetic data generator."""
        self.config = config
        self.thermal_config = config.get('thermal', {
            'resolution': [160, 120],
            'frame_rate': 9.0,
            'temperature_range': [-10, 150]
        })
        self.gas_config = config.get('gas', {
            'sample_rate': 0.2,
            'co2_range': [400, 40000]
        })
        
        # FLIR Lepton 3.5 thermal features (15 features)
        self.thermal_features = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel'
        ]
        
        # SCD41 CO₂ gas features (3 features)
        self.gas_features = ['gas_val', 'gas_delta', 'gas_vel']
        
        logger.info("Initialized FLIR+SCD41 Data Generator")
        logger.info(f"Thermal features: {len(self.thermal_features)}")
        logger.info(f"Gas features: {len(self.gas_features)}")
    
    def generate_training_dataset(self, samples: int = 10000, fire_ratio: float = 0.15) -> pd.DataFrame:
        """
        Generate a complete training dataset for FLIR+SCD41 system.
        
        Args:
            samples: Number of samples to generate
            fire_ratio: Proportion of fire samples (0.0 to 1.0)
            
        Returns:
            DataFrame with 18 features and fire_detected label
        """
        logger.info(f"Generating {samples} FLIR+SCD41 training samples...")
        logger.info(f"Target fire ratio: {fire_ratio:.2%}")
        
        # Generate features
        thermal_features = self._generate_thermal_features(samples)
        gas_features = self._generate_gas_features(samples)
        
        # Combine all features
        all_features = np.concatenate([thermal_features, gas_features], axis=1)
        
        # Generate labels based on feature patterns
        labels = self._generate_labels(thermal_features, gas_features, fire_ratio)
        
        # Create column names (15 thermal + 3 gas = 18 total)
        column_names = self.thermal_features + self.gas_features
        
        # Create DataFrame
        data = pd.DataFrame(all_features, columns=column_names)
        data['fire_detected'] = labels
        
        # Add timestamp for reference
        data['timestamp'] = [datetime.now() + timedelta(seconds=i) for i in range(len(data))]
        
        logger.info(f"Generated dataset with {len(data)} samples")
        logger.info(f"Fire samples: {sum(labels)}, Normal samples: {len(labels) - sum(labels)}")
        logger.info(f"Actual fire ratio: {sum(labels)/len(labels):.2%}")
        
        return data
    
    def _generate_thermal_features(self, samples: int) -> np.ndarray:
        """Generate FLIR Lepton 3.5 thermal features (15 features)."""
        # Initialize array for 15 thermal features
        thermal_data = np.zeros((samples, len(self.thermal_features)))
        
        # Generate normal operation data (room temperature conditions)
        base_temp = np.random.normal(22.0, 2.0, samples)  # Base room temperature
        
        # t_mean: Mean temperature
        thermal_data[:, 0] = base_temp + np.random.normal(0, 1.0, samples)
        
        # t_std: Temperature standard deviation
        thermal_data[:, 1] = np.random.lognormal(0.5, 0.3, samples)  # Log-normal distribution
        
        # t_max: Maximum temperature
        thermal_data[:, 2] = thermal_data[:, 0] + np.random.exponential(5.0, samples)
        
        # t_p95: 95th percentile temperature
        thermal_data[:, 3] = thermal_data[:, 0] + np.random.exponential(3.0, samples)
        
        # t_hot_area_pct: Percentage of hot area (>30°C)
        thermal_data[:, 4] = np.random.exponential(2.0, samples)  # Most of the time small hot areas
        
        # t_hot_largest_blob_pct: Largest hot blob percentage
        thermal_data[:, 5] = thermal_data[:, 4] * np.random.uniform(0.1, 0.8, samples)
        
        # t_grad_mean: Mean temperature gradient
        thermal_data[:, 6] = np.random.exponential(1.0, samples)
        
        # t_grad_std: Standard deviation of temperature gradient
        thermal_data[:, 7] = np.random.lognormal(0.2, 0.2, samples)
        
        # t_diff_mean: Mean temperature difference between frames
        thermal_data[:, 8] = np.random.normal(0, 0.5, samples)
        
        # t_diff_std: Standard deviation of temperature difference
        thermal_data[:, 9] = np.random.lognormal(0.1, 0.2, samples)
        
        # flow_mag_mean: Mean optical flow magnitude
        thermal_data[:, 10] = np.random.exponential(0.3, samples)
        
        # flow_mag_std: Standard deviation of optical flow magnitude
        thermal_data[:, 11] = np.random.lognormal(0.1, 0.2, samples)
        
        # tproxy_val: Temperature proxy value (similar to max temp)
        thermal_data[:, 12] = thermal_data[:, 2] + np.random.normal(0, 2.0, samples)
        
        # tproxy_delta: Temperature proxy delta (change from previous)
        thermal_data[:, 13] = np.random.normal(0, 1.0, samples)
        
        # tproxy_vel: Temperature proxy velocity (rate of change)
        thermal_data[:, 14] = thermal_data[:, 13] + np.random.normal(0, 0.5, samples)
        
        # Ensure realistic ranges
        thermal_data[:, 0] = np.clip(thermal_data[:, 0], -10, 150)  # t_mean
        thermal_data[:, 1] = np.clip(thermal_data[:, 1], 0, 50)     # t_std
        thermal_data[:, 2] = np.clip(thermal_data[:, 2], -10, 150)  # t_max
        thermal_data[:, 3] = np.clip(thermal_data[:, 3], -10, 150)  # t_p95
        thermal_data[:, 4] = np.clip(thermal_data[:, 4], 0, 100)    # t_hot_area_pct
        thermal_data[:, 5] = np.clip(thermal_data[:, 5], 0, 100)    # t_hot_largest_blob_pct
        
        return thermal_data
    
    def _generate_gas_features(self, samples: int) -> np.ndarray:
        """Generate SCD41 CO₂ gas features (3 features)."""
        # Initialize array for 3 gas features
        gas_data = np.zeros((samples, len(self.gas_features)))
        
        # Generate normal indoor CO₂ levels (400-1000 ppm typically)
        base_co2 = np.random.normal(600, 100, samples)
        
        # gas_val: Current CO₂ concentration (ppm)
        gas_data[:, 0] = np.clip(base_co2, 400, 40000)
        
        # gas_delta: Change from previous reading (ppm)
        gas_data[:, 1] = np.random.normal(0, 20, samples)
        
        # gas_vel: Rate of change (same as delta for SCD41)
        gas_data[:, 2] = gas_data[:, 1] + np.random.normal(0, 5, samples)
        
        # Ensure realistic ranges
        gas_data[:, 0] = np.clip(gas_data[:, 0], 400, 40000)  # gas_val
        gas_data[:, 1] = np.clip(gas_data[:, 1], -1000, 1000)  # gas_delta
        gas_data[:, 2] = np.clip(gas_data[:, 2], -1000, 1000)  # gas_vel
        
        return gas_data
    
    def _generate_labels(self, thermal: np.ndarray, gas: np.ndarray, target_ratio: float = 0.15) -> np.ndarray:
        """Generate fire detection labels based on FLIR+SCD41 sensor patterns."""
        samples = thermal.shape[0]
        labels = np.zeros(samples, dtype=int)
        
        for i in range(samples):
            fire_score = 0.0
            
            # FLIR Lepton 3.5 thermal indicators
            t_mean = thermal[i, 0]
            t_max = thermal[i, 2]
            t_hot_area_pct = thermal[i, 4]
            t_grad_mean = thermal[i, 6]
            tproxy_delta = thermal[i, 13]
            tproxy_vel = thermal[i, 14]
            
            # High temperature indicators
            if t_max > 60:  # High maximum temperature
                fire_score += 0.3
            elif t_max > 40:  # Moderately high
                fire_score += 0.15
                
            if t_mean > 35:  # Elevated mean temperature
                fire_score += 0.2
            elif t_mean > 28:  # Slightly elevated
                fire_score += 0.1
            
            # Hot area indicators
            if t_hot_area_pct > 10:  # Significant hot area
                fire_score += 0.25
            elif t_hot_area_pct > 5:  # Moderate hot area
                fire_score += 0.1
            
            # Gradient indicators (sharp temperature changes)
            if t_grad_mean > 5:  # Sharp gradients
                fire_score += 0.2
            elif t_grad_mean > 3:  # Moderate gradients
                fire_score += 0.1
            
            # Temporal change indicators
            temp_change_magnitude = abs(tproxy_delta) + abs(tproxy_vel)
            if temp_change_magnitude > 15:  # Rapid temperature change
                fire_score += 0.2
            elif temp_change_magnitude > 8:  # Moderate temperature change
                fire_score += 0.1
            
            # SCD41 CO₂ gas indicators
            gas_val = gas[i, 0]
            gas_delta = gas[i, 1]
            gas_vel = gas[i, 2]
            
            # Elevated CO₂ levels
            if gas_val > 2000:  # Significantly elevated CO₂
                fire_score += 0.3
            elif gas_val > 1200:  # Moderately elevated
                fire_score += 0.15
            
            # Rapid CO₂ increase
            co2_change_magnitude = abs(gas_delta) + abs(gas_vel)
            if co2_change_magnitude > 200:  # Rapid CO₂ change
                fire_score += 0.25
            elif co2_change_magnitude > 100:  # Moderate CO₂ change
                fire_score += 0.1
            
            # Combined sensor correlation (both sensors indicate fire)
            if t_max > 50 and gas_val > 1500:
                fire_score += 0.1  # Multi-sensor correlation boost
            
            # Add some randomness to prevent overfitting
            fire_score += np.random.normal(0, 0.1)
            
            # Determine label based on fire score
            if fire_score > 0.6:  # High confidence fire
                labels[i] = 1
            elif fire_score > 0.4 and np.random.random() < 0.3:  # Borderline cases
                labels[i] = 1
        
        # Adjust to achieve target fire ratio
        current_fire_rate = np.mean(labels)
        if current_fire_rate < target_ratio * 0.8:
            # Add more fire samples
            normal_indices = np.where(labels == 0)[0]
            num_to_flip = int((target_ratio - current_fire_rate) * samples)
            if len(normal_indices) > 0 and num_to_flip > 0:
                flip_indices = np.random.choice(normal_indices, min(num_to_flip, len(normal_indices)), replace=False)
                labels[flip_indices] = 1
                
                # Modify features to be more fire-like
                for idx in flip_indices:
                    # Increase thermal features
                    thermal[idx, 0] *= np.random.uniform(1.2, 1.8)  # t_mean
                    thermal[idx, 2] *= np.random.uniform(1.3, 2.0)  # t_max
                    thermal[idx, 4] *= np.random.uniform(1.5, 3.0)  # t_hot_area_pct
                    thermal[idx, 6] *= np.random.uniform(1.2, 2.5)  # t_grad_mean
                    
                    # Increase gas features
                    gas[idx, 0] *= np.random.uniform(1.5, 3.0)     # gas_val
                    gas[idx, 1] *= np.random.uniform(2.0, 5.0)     # gas_delta
                    gas[idx, 2] *= np.random.uniform(2.0, 5.0)     # gas_vel
        
        elif current_fire_rate > target_ratio * 1.2 and current_fire_rate > 0:
            # Remove some fire samples
            fire_indices = np.where(labels == 1)[0]
            num_to_flip = int((current_fire_rate - target_ratio) * samples)
            if len(fire_indices) > 0 and num_to_flip > 0:
                flip_indices = np.random.choice(fire_indices, min(num_to_flip, len(fire_indices)), replace=False)
                labels[flip_indices] = 0
        
        return labels
    
    def generate_scenario_dataset(self, 
                                normal_samples: int = 8000, 
                                fire_samples: int = 2000) -> pd.DataFrame:
        """
        Generate a dataset with specific numbers of normal and fire samples.
        
        Args:
            normal_samples: Number of normal (non-fire) samples
            fire_samples: Number of fire samples
            
        Returns:
            DataFrame with specified sample distribution
        """
        total_samples = normal_samples + fire_samples
        logger.info(f"Generating scenario dataset: {normal_samples} normal, {fire_samples} fire samples")
        
        # Generate all data
        all_data = self.generate_training_dataset(total_samples, fire_samples / total_samples)
        
        return all_data
    
    def save_dataset(self, data: pd.DataFrame, filepath: str) -> None:
        """
        Save dataset to CSV file.
        
        Args:
            data: DataFrame to save
            filepath: Path to save file
        """
        data.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")

# Convenience function for creating generator
def create_flir_scd41_generator(config: Dict[str, Any] = None) -> FlirScd41DataGenerator:
    """
    Create a FLIR+SCD41 data generator with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FlirScd41DataGenerator instance
    """
    if config is None:
        config = {
            'thermal': {
                'resolution': [160, 120],
                'frame_rate': 9.0,
                'temperature_range': [-10, 150]
            },
            'gas': {
                'sample_rate': 0.2,
                'co2_range': [400, 40000]
            }
        }
    
    return FlirScd41DataGenerator(config)