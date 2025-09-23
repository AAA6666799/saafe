"""
Synthetic FLIR Lepton 3.5 Interface Implementation.

This module provides a synthetic implementation of the FLIRLepton35Interface
for testing and simulation purposes.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

from .flir_lepton35_interface import FLIRLepton35Interface


class SyntheticFLIRInterface(FLIRLepton35Interface):
    """
    Synthetic FLIR Lepton 3.5 thermal camera interface implementation.
    
    This class provides a synthetic implementation for testing and simulation
    of the FLIR Lepton 3.5 thermal camera.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the synthetic FLIR Lepton 3.5 interface.
        
        Args:
            config: Dictionary containing FLIR Lepton 3.5 configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Override the connected status for synthetic implementation
        self.connected = True
        
        self.logger.info("Initialized synthetic FLIR Lepton 3.5 interface")
    
    def connect(self) -> bool:
        """
        Connect to the synthetic FLIR Lepton 3.5 thermal camera.
        
        Returns:
            True if connection is successful, False otherwise
        """
        self.connected = True
        self.logger.info("Connected to synthetic FLIR Lepton 3.5")
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the synthetic FLIR Lepton 3.5 thermal camera.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        self.connected = False
        self.logger.info("Disconnected from synthetic FLIR Lepton 3.5")
        return True
    
    def read(self) -> Dict[str, Any]:
        """
        Read synthetic data from the FLIR Lepton 3.5 thermal camera.
        
        Returns:
            Dictionary containing synthetic FLIR Lepton 3.5 thermal data in the format expected by
            the FLIR thermal extractor (15 features: t_mean, t_std, t_max, t_p95, 
            t_hot_area_pct, t_hot_largest_blob_pct, t_grad_mean, t_grad_std, 
            t_diff_mean, t_diff_std, flow_mag_mean, flow_mag_std, tproxy_val, 
            tproxy_delta, tproxy_vel)
        """
        if not self.connected:
            raise RuntimeError("Synthetic FLIR Lepton 3.5 not connected")
        
        # Generate synthetic thermal data
        thermal_image = self.get_thermal_image()
        
        # Calculate the 15 specific FLIR features
        features = {}
        
        # Basic temperature statistics
        features['t_mean'] = float(np.mean(thermal_image))
        features['t_std'] = float(np.std(thermal_image))
        features['t_max'] = float(np.max(thermal_image))
        features['t_p95'] = float(np.percentile(thermal_image, 95))
        
        # Hot area features (using 40°C as hot threshold)
        hot_mask = thermal_image > 40.0
        total_pixels = thermal_image.size
        hot_pixels = np.sum(hot_mask)
        features['t_hot_area_pct'] = float(hot_pixels / total_pixels * 100.0)
        
        # Largest hot blob percentage (simplified implementation)
        features['t_hot_largest_blob_pct'] = float(hot_pixels / total_pixels * 50.0)  # Simplified
        
        # Gradient features (simplified implementation)
        grad_x = np.gradient(thermal_image, axis=1)
        grad_y = np.gradient(thermal_image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['t_grad_mean'] = float(np.mean(gradient_magnitude))
        features['t_grad_std'] = float(np.std(gradient_magnitude))
        
        # Temporal difference features (simplified - using random values)
        features['t_diff_mean'] = float(np.random.normal(0.1, 0.05))
        features['t_diff_std'] = float(np.random.normal(0.05, 0.02))
        
        # Optical flow features (simplified - using random values)
        features['flow_mag_mean'] = float(np.random.normal(0.2, 0.1))
        features['flow_mag_std'] = float(np.random.normal(0.1, 0.05))
        
        # Temperature proxy features (simplified - using max temperature)
        features['tproxy_val'] = features['t_max']
        features['tproxy_delta'] = float(np.random.normal(1.0, 0.5))
        features['tproxy_vel'] = float(np.random.normal(0.5, 0.2))
        
        # Add metadata
        features['timestamp'] = datetime.now().isoformat()
        features['device_type'] = 'flir_lepton_3_5'
        features['thermal_frame'] = thermal_image
        
        self.last_frame_time = datetime.now()
        self.logger.debug("Successfully read synthetic data from FLIR Lepton 3.5")
        
        return features
    
    def calibrate(self) -> bool:
        """
        Calibrate the synthetic FLIR Lepton 3.5 thermal camera.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        if not self.connected:
            return False
        
        self.logger.info("Calibrated synthetic FLIR Lepton 3.5")
        return True
    
    def get_thermal_image(self) -> np.ndarray:
        """
        Get a synthetic thermal image from the FLIR Lepton 3.5.
        
        Returns:
            2D numpy array representing the thermal image (160x120)
        """
        if not self.connected:
            raise RuntimeError("Synthetic FLIR Lepton 3.5 not connected")
        
        # Generate a synthetic thermal image for demonstration
        width, height = self.resolution
        thermal_image = np.random.normal(22.0, 5.0, (height, width))
        
        # Occasionally add a hotspot for testing
        if np.random.random() < 0.1:  # 10% chance of hotspot
            center_y = np.random.randint(10, height - 10)
            center_x = np.random.randint(10, width - 10)
            radius = np.random.randint(5, 15)
            hotspot_temp = np.random.uniform(40, 80)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            thermal_image[mask] = hotspot_temp
        
        return thermal_image


# Convenience function for creating synthetic FLIR interface
def create_synthetic_flir_interface(config: Dict[str, Any]) -> SyntheticFLIRInterface:
    """
    Create a synthetic FLIR interface with default configuration.
    
    Args:
        config: Configuration dictionary for synthetic FLIR
        
    Returns:
        Configured SyntheticFLIRInterface instance
    """
    default_config = {
        'device_path': '/dev/thermal0',
        'resolution': (160, 120),
        'temperature_range': (-10.0, 150.0),
        'frame_rate': 9.0,
        'sensor_type': 'flir_lepton_3_5'
    }
    
    default_config.update(config)
    return SyntheticFLIRInterface(default_config)