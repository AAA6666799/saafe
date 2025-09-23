"""
Synthetic SCD41 CO₂ Sensor Interface Implementation.

This module provides a synthetic implementation of the SCD41Interface
for testing and simulation purposes.
"""

import numpy as np
import logging
from typing import Dict, Any, List
from datetime import datetime

from .scd41_interface import SCD41Interface


class SyntheticSCD41Interface(SCD41Interface):
    """
    Synthetic Sensirion SCD41 CO₂ sensor interface implementation.
    
    This class provides a synthetic implementation for testing and simulation
    of the Sensirion SCD41 CO₂ sensor.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the synthetic SCD41 interface.
        
        Args:
            config: Dictionary containing SCD41 configuration parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Override the connected status for synthetic implementation
        self.connected = True
        
        self.logger.info("Initialized synthetic SCD41 interface")
    
    def connect(self) -> bool:
        """
        Connect to the synthetic SCD41 CO₂ sensor.
        
        Returns:
            True if connection is successful, False otherwise
        """
        self.connected = True
        self.logger.info("Connected to synthetic SCD41")
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the synthetic SCD41 CO₂ sensor.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        self.connected = False
        self.logger.info("Disconnected from synthetic SCD41")
        return True
    
    def read(self) -> Dict[str, Any]:
        """
        Read synthetic data from the SCD41 CO₂ sensor.
        
        Returns:
            Dictionary containing synthetic SCD41 CO₂ data in the format expected by
            the SCD41 gas extractor (3 features: gas_val, gas_delta, gas_vel)
        """
        if not self.connected:
            raise RuntimeError("Synthetic SCD41 not connected")
        
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
        self.logger.debug("Successfully read synthetic data from SCD41")
        
        return features
    
    def calibrate(self) -> bool:
        """
        Calibrate the synthetic SCD41 CO₂ sensor.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        if not self.connected:
            return False
        
        self.logger.info("Calibrated synthetic SCD41")
        return True
    
    def get_gas_concentration(self, gas_type: str = None) -> Dict[str, float]:
        """
        Get synthetic gas concentration readings from the SCD41.
        
        Args:
            gas_type: Specific gas type to read (defaults to 'co2' for SCD41)
            
        Returns:
            Dictionary mapping gas types to concentration values in PPM
        """
        if not self.connected:
            raise RuntimeError("Synthetic SCD41 not connected")
        
        if gas_type and gas_type != 'co2':
            raise ValueError(f"SCD41 only supports CO2 measurements, not {gas_type}")
        
        # Generate a synthetic CO2 reading for demonstration
        base_co2 = 400.0  # Outdoor baseline
        variation = np.random.normal(0, 50)  # Normal indoor variation
        
        # Occasionally simulate elevated CO2 (e.g., from human presence or fire)
        if np.random.random() < 0.05:  # 5% chance of elevated CO2
            variation += np.random.uniform(200, 1000)
        
        co2_concentration = max(400.0, base_co2 + variation)  # Minimum 400 ppm
        
        return {
            'co2': co2_concentration
        }


# Convenience function for creating synthetic SCD41 interface
def create_synthetic_scd41_interface(config: Dict[str, Any]) -> SyntheticSCD41Interface:
    """
    Create a synthetic SCD41 interface with default configuration.
    
    Args:
        config: Configuration dictionary for synthetic SCD41
        
    Returns:
        Configured SyntheticSCD41Interface instance
    """
    default_config = {
        'device_address': '/dev/ttyUSB0',
        'measurement_range': (400, 40000),
        'sampling_rate': 0.2,
        'sensor_type': 'sensirion_scd41'
    }
    
    default_config.update(config)
    return SyntheticSCD41Interface(default_config)