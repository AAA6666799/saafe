"""
FLIR Lepton 3.5 Interface Implementation.

This module provides a concrete implementation of the ThermalSensorInterface
specifically for the FLIR Lepton 3.5 thermal camera.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

from ..base import ThermalSensorInterface


class FLIRLepton35Interface(ThermalSensorInterface):
    """
    FLIR Lepton 3.5 thermal camera interface implementation.
    
    This class provides a concrete implementation for interfacing with
    the FLIR Lepton 3.5 thermal camera, supporting its specific features
    and data formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FLIR Lepton 3.5 interface.
        
        Args:
            config: Dictionary containing FLIR Lepton 3.5 configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # FLIR Lepton 3.5 specific parameters
        self.device_path = config.get('device_path', '/dev/thermal0')
        self.resolution = config.get('resolution', (160, 120))  # 160x120 pixels
        self.temperature_range = config.get('temperature_range', (-10.0, 150.0))  # Celsius
        self.frame_rate = config.get('frame_rate', 9.0)  # Hz
        self.connected = False
        self.last_frame_time = None
        
        self.logger.info(f"Initialized FLIR Lepton 3.5 interface for device: {self.device_path}")
    
    def validate_config(self) -> None:
        """
        Validate the FLIR Lepton 3.5 configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.device_path, str):
            raise ValueError("Device path must be a string")
        
        if not isinstance(self.resolution, (tuple, list)) or len(self.resolution) != 2:
            raise ValueError("Resolution must be a tuple of (width, height)")
        
        if not isinstance(self.temperature_range, (tuple, list)) or len(self.temperature_range) != 2:
            raise ValueError("Temperature range must be a tuple of (min, max)")
        
        if self.temperature_range[0] >= self.temperature_range[1]:
            raise ValueError("Invalid temperature range: min must be less than max")
        
        if not isinstance(self.frame_rate, (int, float)) or self.frame_rate <= 0:
            raise ValueError("Frame rate must be a positive number")
    
    def connect(self) -> bool:
        """
        Connect to the FLIR Lepton 3.5 thermal camera.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # In a real implementation, this would establish a connection to the device
            # For now, we'll simulate a successful connection
            self.connected = True
            self.logger.info(f"Connected to FLIR Lepton 3.5 at {self.device_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to FLIR Lepton 3.5: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the FLIR Lepton 3.5 thermal camera.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        try:
            # In a real implementation, this would close the connection to the device
            self.connected = False
            self.logger.info("Disconnected from FLIR Lepton 3.5")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from FLIR Lepton 3.5: {str(e)}")
            return False
    
    def read(self) -> Dict[str, Any]:
        """
        Read data from the FLIR Lepton 3.5 thermal camera.
        
        Returns:
            Dictionary containing FLIR Lepton 3.5 thermal data in the format expected by
            the FLIR thermal extractor (15 features: t_mean, t_std, t_max, t_p95, 
            t_hot_area_pct, t_hot_largest_blob_pct, t_grad_mean, t_grad_std, 
            t_diff_mean, t_diff_std, flow_mag_mean, flow_mag_std, tproxy_val, 
            tproxy_delta, tproxy_vel)
        """
        if not self.connected:
            raise RuntimeError("FLIR Lepton 3.5 not connected")
        
        try:
            # Get thermal image data
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
            self.logger.debug("Successfully read data from FLIR Lepton 3.5")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error reading from FLIR Lepton 3.5: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the FLIR Lepton 3.5 thermal camera.
        
        Returns:
            Dictionary containing FLIR Lepton 3.5 status information
        """
        return {
            'connected': self.connected,
            'device_path': self.device_path,
            'resolution': self.resolution,
            'temperature_range': self.temperature_range,
            'frame_rate': self.frame_rate,
            'last_frame_time': self.last_frame_time.isoformat() if self.last_frame_time else None,
            'sensor_type': 'flir_lepton_3_5'
        }
    
    def calibrate(self) -> bool:
        """
        Calibrate the FLIR Lepton 3.5 thermal camera.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        try:
            if not self.connected:
                return False
            
            # In a real implementation, this would perform actual calibration
            self.logger.info("Calibrated FLIR Lepton 3.5")
            return True
        except Exception as e:
            self.logger.error(f"Error calibrating FLIR Lepton 3.5: {str(e)}")
            return False
    
    def get_thermal_image(self) -> np.ndarray:
        """
        Get a thermal image from the FLIR Lepton 3.5.
        
        Returns:
            2D numpy array representing the thermal image (160x120)
        """
        if not self.connected:
            raise RuntimeError("FLIR Lepton 3.5 not connected")
        
        # Generate a synthetic thermal image for demonstration
        # In a real implementation, this would read from the actual device
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
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the resolution of the FLIR Lepton 3.5 thermal camera.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        return self.resolution
    
    def get_temperature_range(self) -> Tuple[float, float]:
        """
        Get the temperature range of the FLIR Lepton 3.5 thermal camera.
        
        Returns:
            Tuple of (min_temperature, max_temperature) in degrees Celsius
        """
        return self.temperature_range


# Convenience function for creating FLIR Lepton 3.5 interface
def create_flir_lepton35_interface(config: Dict[str, Any]) -> FLIRLepton35Interface:
    """
    Create a FLIR Lepton 3.5 interface with default configuration.
    
    Args:
        config: Configuration dictionary for FLIR Lepton 3.5
        
    Returns:
        Configured FLIRLepton35Interface instance
    """
    default_config = {
        'device_path': '/dev/thermal0',
        'resolution': (160, 120),
        'temperature_range': (-10.0, 150.0),
        'frame_rate': 9.0,
        'sensor_type': 'flir_lepton_3_5'
    }
    
    default_config.update(config)
    return FLIRLepton35Interface(default_config)