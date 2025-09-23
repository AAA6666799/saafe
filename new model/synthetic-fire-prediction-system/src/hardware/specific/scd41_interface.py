"""
SCD41 CO₂ Sensor Interface Implementation.

This module provides a concrete implementation of the GasSensorInterface
specifically for the Sensirion SCD41 CO₂ sensor.
"""

import numpy as np
import logging
from typing import Dict, Any, List
from datetime import datetime

from ..base import GasSensorInterface


class SCD41Interface(GasSensorInterface):
    """
    Sensirion SCD41 CO₂ sensor interface implementation.
    
    This class provides a concrete implementation for interfacing with
    the Sensirion SCD41 CO₂ sensor, supporting its specific features
    and data formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SCD41 interface.
        
        Args:
            config: Dictionary containing SCD41 configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # SCD41 specific parameters
        self.device_address = config.get('device_address', '/dev/ttyUSB0')
        self.measurement_range = config.get('measurement_range', (400, 40000))  # ppm
        self.sampling_rate = config.get('sampling_rate', 0.2)  # Every 5 seconds
        self.connected = False
        self.last_reading_time = None
        self.previous_co2 = 400.0  # Initial CO2 level
        
        # Supported gases for SCD41
        self.supported_gases = ['co2']
        
        self.logger.info(f"Initialized SCD41 interface for device: {self.device_address}")
    
    def validate_config(self) -> None:
        """
        Validate the SCD41 configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.device_address, str):
            raise ValueError("Device address must be a string")
        
        if not isinstance(self.measurement_range, (tuple, list)) or len(self.measurement_range) != 2:
            raise ValueError("Measurement range must be a tuple of (min, max)")
        
        if self.measurement_range[0] >= self.measurement_range[1]:
            raise ValueError("Invalid measurement range: min must be less than max")
        
        if not isinstance(self.sampling_rate, (int, float)) or self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive number")
    
    def connect(self) -> bool:
        """
        Connect to the SCD41 CO₂ sensor.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # In a real implementation, this would establish a connection to the device
            # For now, we'll simulate a successful connection
            self.connected = True
            self.logger.info(f"Connected to SCD41 at {self.device_address}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to SCD41: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the SCD41 CO₂ sensor.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        try:
            # In a real implementation, this would close the connection to the device
            self.connected = False
            self.logger.info("Disconnected from SCD41")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from SCD41: {str(e)}")
            return False
    
    def read(self) -> Dict[str, Any]:
        """
        Read data from the SCD41 CO₂ sensor.
        
        Returns:
            Dictionary containing SCD41 CO₂ data in the format expected by
            the SCD41 gas extractor (3 features: gas_val, gas_delta, gas_vel)
        """
        if not self.connected:
            raise RuntimeError("SCD41 not connected")
        
        try:
            # Get CO2 concentration data
            co2_data = self.get_gas_concentration('co2')
            current_co2 = co2_data.get('co2', 400.0)
            
            # Calculate the 3 specific SCD41 features
            features = {}
            
            # Current CO2 value
            features['gas_val'] = current_co2
            
            # Change from previous reading
            features['gas_delta'] = current_co2 - self.previous_co2
            
            # Rate of change (same as delta for SCD41)
            features['gas_vel'] = features['gas_delta']
            
            # Update previous CO2 level
            self.previous_co2 = current_co2
            
            # Add metadata
            features['timestamp'] = datetime.now().isoformat()
            features['device_type'] = 'sensirion_scd41'
            features['co2_concentration'] = current_co2
            features['sensor_temp'] = float(np.random.normal(25.0, 2.0))  # Simulated sensor temperature
            features['sensor_humidity'] = float(np.random.normal(45.0, 5.0))  # Simulated sensor humidity
            
            self.last_reading_time = datetime.now()
            self.logger.debug("Successfully read data from SCD41")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error reading from SCD41: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the SCD41 CO₂ sensor.
        
        Returns:
            Dictionary containing SCD41 status information
        """
        return {
            'connected': self.connected,
            'device_address': self.device_address,
            'measurement_range': self.measurement_range,
            'sampling_rate': self.sampling_rate,
            'last_reading_time': self.last_reading_time.isoformat() if self.last_reading_time else None,
            'sensor_type': 'sensirion_scd41'
        }
    
    def calibrate(self) -> bool:
        """
        Calibrate the SCD41 CO₂ sensor.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        try:
            if not self.connected:
                return False
            
            # In a real implementation, this would perform actual calibration
            self.logger.info("Calibrated SCD41")
            return True
        except Exception as e:
            self.logger.error(f"Error calibrating SCD41: {str(e)}")
            return False
    
    def get_gas_concentration(self, gas_type: str = None) -> Dict[str, float]:
        """
        Get gas concentration readings from the SCD41.
        
        Args:
            gas_type: Specific gas type to read (defaults to 'co2' for SCD41)
            
        Returns:
            Dictionary mapping gas types to concentration values in PPM
        """
        if not self.connected:
            raise RuntimeError("SCD41 not connected")
        
        if gas_type and gas_type != 'co2':
            raise ValueError(f"SCD41 only supports CO2 measurements, not {gas_type}")
        
        # Generate a synthetic CO2 reading for demonstration
        # In a real implementation, this would read from the actual device
        base_co2 = 400.0  # Outdoor baseline
        variation = np.random.normal(0, 50)  # Normal indoor variation
        
        # Occasionally simulate elevated CO2 (e.g., from human presence or fire)
        if np.random.random() < 0.05:  # 5% chance of elevated CO2
            variation += np.random.uniform(200, 1000)
        
        co2_concentration = max(400.0, base_co2 + variation)  # Minimum 400 ppm
        
        return {
            'co2': co2_concentration
        }
    
    def get_supported_gases(self) -> List[str]:
        """
        Get the list of gas types supported by the SCD41 sensor.
        
        Returns:
            List of supported gas types (only 'co2' for SCD41)
        """
        return self.supported_gases.copy()
    
    def set_alarm_threshold(self, gas_type: str, threshold: float) -> bool:
        """
        Set the alarm threshold for CO2.
        
        Args:
            gas_type: Type of gas ('co2')
            threshold: Threshold value in PPM
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            return False
        
        if gas_type != 'co2':
            return False
        
        if threshold < self.measurement_range[0] or threshold > self.measurement_range[1]:
            return False
        
        # In a real implementation, this would set the actual alarm threshold
        self.logger.info(f"Set CO2 alarm threshold to {threshold} ppm")
        return True


# Convenience function for creating SCD41 interface
def create_scd41_interface(config: Dict[str, Any]) -> SCD41Interface:
    """
    Create a SCD41 interface with default configuration.
    
    Args:
        config: Configuration dictionary for SCD41
        
    Returns:
        Configured SCD41Interface instance
    """
    default_config = {
        'device_address': '/dev/ttyUSB0',
        'measurement_range': (400, 40000),
        'sampling_rate': 0.2,
        'sensor_type': 'sensirion_scd41'
    }
    
    default_config.update(config)
    return SCD41Interface(default_config)