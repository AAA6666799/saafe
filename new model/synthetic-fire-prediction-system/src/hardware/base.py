"""
Hardware abstraction layer interfaces.

This module defines the core interfaces and abstract classes for hardware abstraction
in the synthetic fire prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime


class SensorInterface(ABC):
    """
    Base abstract class for sensor interfaces.
    
    This class defines the common interface that all sensor interfaces must implement,
    regardless of the sensor type (thermal, gas, environmental).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sensor interface.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.sensor_id = config.get('sensor_id', 'unknown')
        self.sensor_type = config.get('sensor_type', 'unknown')
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the sensor.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the sensor.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def read(self) -> Dict[str, Any]:
        """
        Read data from the sensor.
        
        Returns:
            Dictionary containing sensor readings
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the sensor.
        
        Returns:
            Dictionary containing sensor status information
        """
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """
        Calibrate the sensor.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        pass


class ThermalSensorInterface(SensorInterface):
    """
    Interface for thermal sensors.
    
    This class extends the base SensorInterface with thermal-specific methods.
    """
    
    @abstractmethod
    def get_thermal_image(self) -> np.ndarray:
        """
        Get a thermal image from the sensor.
        
        Returns:
            2D numpy array representing the thermal image
        """
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the resolution of the thermal sensor.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        pass
    
    @abstractmethod
    def get_temperature_range(self) -> Tuple[float, float]:
        """
        Get the temperature range of the thermal sensor.
        
        Returns:
            Tuple of (min_temperature, max_temperature) in degrees Celsius
        """
        pass


class GasSensorInterface(SensorInterface):
    """
    Interface for gas sensors.
    
    This class extends the base SensorInterface with gas-specific methods.
    """
    
    @abstractmethod
    def get_gas_concentration(self, gas_type: Optional[str] = None) -> Dict[str, float]:
        """
        Get gas concentration readings.
        
        Args:
            gas_type: Optional specific gas type to read
            
        Returns:
            Dictionary mapping gas types to concentration values in PPM
        """
        pass
    
    @abstractmethod
    def get_supported_gases(self) -> List[str]:
        """
        Get the list of gas types supported by this sensor.
        
        Returns:
            List of supported gas types
        """
        pass
    
    @abstractmethod
    def set_alarm_threshold(self, gas_type: str, threshold: float) -> bool:
        """
        Set the alarm threshold for a specific gas type.
        
        Args:
            gas_type: Type of gas
            threshold: Threshold value in PPM
            
        Returns:
            True if successful, False otherwise
        """
        pass


class EnvironmentalSensorInterface(SensorInterface):
    """
    Interface for environmental sensors.
    
    This class extends the base SensorInterface with environmental-specific methods.
    """
    
    @abstractmethod
    def get_temperature(self) -> float:
        """
        Get temperature reading.
        
        Returns:
            Temperature in degrees Celsius
        """
        pass
    
    @abstractmethod
    def get_humidity(self) -> float:
        """
        Get humidity reading.
        
        Returns:
            Relative humidity as a percentage
        """
        pass
    
    @abstractmethod
    def get_pressure(self) -> float:
        """
        Get pressure reading.
        
        Returns:
            Pressure in hPa
        """
        pass
    
    @abstractmethod
    def get_voc(self) -> Dict[str, float]:
        """
        Get volatile organic compound readings.
        
        Returns:
            Dictionary mapping VOC types to concentration values
        """
        pass


class HardwareAbstractionLayer:
    """
    Hardware abstraction layer for the system.
    
    This class provides a unified interface for interacting with hardware sensors,
    abstracting away the details of specific sensor implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hardware abstraction layer.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.thermal_sensors = {}
        self.gas_sensors = {}
        self.environmental_sensors = {}
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize all configured sensors.
        
        Returns:
            True if initialization is successful, False otherwise
        """
        success = True
        
        # Initialize thermal sensors
        for sensor_id, sensor_config in self.config.get('thermal_sensors', {}).items():
            sensor_config['sensor_id'] = sensor_id
            sensor_config['sensor_type'] = 'thermal'
            if not self._initialize_thermal_sensor(sensor_id, sensor_config):
                success = False
        
        # Initialize gas sensors
        for sensor_id, sensor_config in self.config.get('gas_sensors', {}).items():
            sensor_config['sensor_id'] = sensor_id
            sensor_config['sensor_type'] = 'gas'
            if not self._initialize_gas_sensor(sensor_id, sensor_config):
                success = False
        
        # Initialize environmental sensors
        for sensor_id, sensor_config in self.config.get('environmental_sensors', {}).items():
            sensor_config['sensor_id'] = sensor_id
            sensor_config['sensor_type'] = 'environmental'
            if not self._initialize_environmental_sensor(sensor_id, sensor_config):
                success = False
        
        self.is_initialized = success
        return success
    
    def _initialize_thermal_sensor(self, sensor_id: str, sensor_config: Dict[str, Any]) -> bool:
        """
        Initialize a thermal sensor.
        
        Args:
            sensor_id: Sensor identifier
            sensor_config: Sensor configuration
            
        Returns:
            True if initialization is successful, False otherwise
        """
        # This would be implemented with actual sensor initialization code
        # For now, we'll use a mock implementation
        from src.hardware.mock import MockThermalSensor
        try:
            sensor = MockThermalSensor(sensor_config)
            if sensor.connect():
                self.thermal_sensors[sensor_id] = sensor
                return True
            return False
        except Exception as e:
            print(f"Failed to initialize thermal sensor {sensor_id}: {str(e)}")
            return False
    
    def _initialize_gas_sensor(self, sensor_id: str, sensor_config: Dict[str, Any]) -> bool:
        """
        Initialize a gas sensor.
        
        Args:
            sensor_id: Sensor identifier
            sensor_config: Sensor configuration
            
        Returns:
            True if initialization is successful, False otherwise
        """
        # This would be implemented with actual sensor initialization code
        # For now, we'll use a mock implementation
        from src.hardware.mock import MockGasSensor
        try:
            sensor = MockGasSensor(sensor_config)
            if sensor.connect():
                self.gas_sensors[sensor_id] = sensor
                return True
            return False
        except Exception as e:
            print(f"Failed to initialize gas sensor {sensor_id}: {str(e)}")
            return False
    
    def _initialize_environmental_sensor(self, sensor_id: str, sensor_config: Dict[str, Any]) -> bool:
        """
        Initialize an environmental sensor.
        
        Args:
            sensor_id: Sensor identifier
            sensor_config: Sensor configuration
            
        Returns:
            True if initialization is successful, False otherwise
        """
        # This would be implemented with actual sensor initialization code
        # For now, we'll use a mock implementation
        from src.hardware.mock import MockEnvironmentalSensor
        try:
            sensor = MockEnvironmentalSensor(sensor_config)
            if sensor.connect():
                self.environmental_sensors[sensor_id] = sensor
                return True
            return False
        except Exception as e:
            print(f"Failed to initialize environmental sensor {sensor_id}: {str(e)}")
            return False
    
    def shutdown(self) -> bool:
        """
        Shut down all sensors.
        
        Returns:
            True if shutdown is successful, False otherwise
        """
        success = True
        
        # Disconnect thermal sensors
        for sensor_id, sensor in self.thermal_sensors.items():
            if not sensor.disconnect():
                print(f"Failed to disconnect thermal sensor {sensor_id}")
                success = False
        
        # Disconnect gas sensors
        for sensor_id, sensor in self.gas_sensors.items():
            if not sensor.disconnect():
                print(f"Failed to disconnect gas sensor {sensor_id}")
                success = False
        
        # Disconnect environmental sensors
        for sensor_id, sensor in self.environmental_sensors.items():
            if not sensor.disconnect():
                print(f"Failed to disconnect environmental sensor {sensor_id}")
                success = False
        
        self.is_initialized = False
        return success
    
    def get_all_sensor_readings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get readings from all sensors.
        
        Returns:
            Dictionary mapping sensor IDs to their readings
        """
        readings = {}
        
        # Get thermal sensor readings
        for sensor_id, sensor in self.thermal_sensors.items():
            readings[sensor_id] = sensor.read()
        
        # Get gas sensor readings
        for sensor_id, sensor in self.gas_sensors.items():
            readings[sensor_id] = sensor.read()
        
        # Get environmental sensor readings
        for sensor_id, sensor in self.environmental_sensors.items():
            readings[sensor_id] = sensor.read()
        
        return readings
    
    def get_all_sensor_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information from all sensors.
        
        Returns:
            Dictionary mapping sensor IDs to their status information
        """
        statuses = {}
        
        # Get thermal sensor statuses
        for sensor_id, sensor in self.thermal_sensors.items():
            statuses[sensor_id] = sensor.get_status()
        
        # Get gas sensor statuses
        for sensor_id, sensor in self.gas_sensors.items():
            statuses[sensor_id] = sensor.get_status()
        
        # Get environmental sensor statuses
        for sensor_id, sensor in self.environmental_sensors.items():
            statuses[sensor_id] = sensor.get_status()
        
        return statuses
    
    def calibrate_all_sensors(self) -> Dict[str, bool]:
        """
        Calibrate all sensors.
        
        Returns:
            Dictionary mapping sensor IDs to calibration success status
        """
        results = {}
        
        # Calibrate thermal sensors
        for sensor_id, sensor in self.thermal_sensors.items():
            results[sensor_id] = sensor.calibrate()
        
        # Calibrate gas sensors
        for sensor_id, sensor in self.gas_sensors.items():
            results[sensor_id] = sensor.calibrate()
        
        # Calibrate environmental sensors
        for sensor_id, sensor in self.environmental_sensors.items():
            results[sensor_id] = sensor.calibrate()
        
        return results