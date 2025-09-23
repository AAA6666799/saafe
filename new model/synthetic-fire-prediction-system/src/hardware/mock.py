"""
Mock implementations of hardware interfaces.

This module provides mock implementations of the hardware interfaces for testing
and development purposes.
"""

import numpy as np
import random
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from src.hardware.base import (
    SensorInterface,
    ThermalSensorInterface,
    GasSensorInterface,
    EnvironmentalSensorInterface
)


class MockSensorBase:
    """
    Base class for mock sensors with common functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mock sensor base.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.sensor_id = config.get('sensor_id', 'unknown')
        self.sensor_type = config.get('sensor_type', 'unknown')
        self.connected = False
        self.last_read_time = None
        self.error_rate = 0.01  # 1% chance of read error
        self.battery_level = 100.0
        self.firmware_version = "1.0.0"
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Base implementation does nothing
        pass
    
    def connect(self) -> bool:
        """
        Connect to the mock sensor.
        
        Returns:
            True if connection is successful, False otherwise
        """
        # Simulate occasional connection failures
        if random.random() < 0.05:  # 5% chance of connection failure
            return False
        
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the mock sensor.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        if not self.connected:
            return False
        
        self.connected = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the mock sensor.
        
        Returns:
            Dictionary containing sensor status information
        """
        # Simulate battery drain
        if self.connected and self.last_read_time is not None:
            time_since_last_read = (datetime.now() - self.last_read_time).total_seconds()
            self.battery_level = max(0.0, self.battery_level - (time_since_last_read * 0.001))
        
        return {
            "connected": self.connected,
            "battery_level": self.battery_level,
            "firmware_version": self.firmware_version,
            "last_read_time": self.last_read_time.isoformat() if self.last_read_time else None,
            "errors": [],
            "warnings": []
        }
    
    def calibrate(self) -> bool:
        """
        Calibrate the mock sensor.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        if not self.connected:
            return False
        
        # Simulate occasional calibration failures
        if random.random() < 0.1:  # 10% chance of calibration failure
            return False
        
        return True


class MockThermalSensor(ThermalSensorInterface):
    """
    Mock implementation of a thermal sensor.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mock thermal sensor.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Initialize the base class first
        super().__init__(config)
        
        # Set attributes after calling super().__init__
        self.width = config.get('width', 384)
        self.height = config.get('height', 288)
        self.min_temp = config.get('min_temp', 0.0)
        self.max_temp = config.get('max_temp', 500.0)
        self.ambient_temp = config.get('ambient_temp', 25.0)
        self.noise_level = config.get('noise_level', 1.0)
        self.connected = False
        self.last_read_time = None
        
        # Generate a base thermal image with ambient temperature
        self.base_image = np.ones((self.height, self.width)) * self.ambient_temp
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid thermal sensor resolution")
        
        if self.min_temp >= self.max_temp:
            raise ValueError("Invalid temperature range")
    
    def connect(self) -> bool:
        """
        Connect to the mock sensor.
        
        Returns:
            True if connection is successful, False otherwise
        """
        # Simulate occasional connection failures
        if random.random() < 0.05:  # 5% chance of connection failure
            return False
        
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the mock sensor.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        if not self.connected:
            return False
        
        self.connected = False
        return True
    
    def read(self) -> Dict[str, Any]:
        """
        Read data from the mock thermal sensor.
        
        Returns:
            Dictionary containing sensor readings
        """
        if not self.connected:
            raise RuntimeError("Sensor not connected")
        
        # Simulate occasional read errors
        if random.random() < 0.01:  # 1% chance of read error
            raise RuntimeError("Sensor read error")
        
        self.last_read_time = datetime.now()
        
        # Generate a thermal image
        thermal_image = self.get_thermal_image()
        
        # Calculate some basic statistics
        mean_temp = np.mean(thermal_image)
        max_temp = np.max(thermal_image)
        min_temp = np.min(thermal_image)
        
        return {
            "timestamp": self.last_read_time.isoformat(),
            "thermal_image": thermal_image,
            "mean_temperature": mean_temp,
            "max_temperature": max_temp,
            "min_temperature": min_temp,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the mock sensor.
        
        Returns:
            Dictionary containing sensor status information
        """
        return {
            "connected": self.connected,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "width": self.width,
            "height": self.height,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "last_read_time": self.last_read_time.isoformat() if self.last_read_time else None
        }
    
    def calibrate(self) -> bool:
        """
        Calibrate the mock sensor.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        if not self.connected:
            return False
        
        # Simulate occasional calibration failures
        if random.random() < 0.1:  # 10% chance of calibration failure
            return False
        
        return True
    
    def get_thermal_image(self) -> np.ndarray:
        """
        Get a thermal image from the mock sensor.
        
        Returns:
            2D numpy array representing the thermal image
        """
        # Start with the base image
        image = self.base_image.copy()
        
        # Add random noise
        noise = np.random.normal(0, self.noise_level, (self.height, self.width))
        image += noise
        
        # Add some random hotspots (1-3)
        num_hotspots = random.randint(1, 3)
        for _ in range(num_hotspots):
            # Random hotspot position
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            # Random hotspot size
            size = random.randint(5, 20)
            
            # Random hotspot temperature
            temp = random.uniform(self.ambient_temp + 10, self.max_temp * 0.8)
            
            # Create hotspot with Gaussian falloff
            for i in range(max(0, y - size), min(self.height, y + size + 1)):
                for j in range(max(0, x - size), min(self.width, x + size + 1)):
                    distance = np.sqrt((i - y) ** 2 + (j - x) ** 2)
                    if distance <= size:
                        falloff = np.exp(-(distance / size) ** 2)
                        image[i, j] += temp * falloff
        
        # Clip to valid temperature range
        image = np.clip(image, self.min_temp, self.max_temp)
        
        return image
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the resolution of the mock thermal sensor.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        return (self.width, self.height)
    
    def get_temperature_range(self) -> Tuple[float, float]:
        """
        Get the temperature range of the mock thermal sensor.
        
        Returns:
            Tuple of (min_temperature, max_temperature) in degrees Celsius
        """
        return (self.min_temp, self.max_temp)


class MockGasSensor(GasSensorInterface):
    """
    Mock implementation of a gas sensor.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mock gas sensor.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Initialize the base class first
        super().__init__(config)
        
        # Set attributes after calling super().__init__
        self.supported_gases = config.get('supported_gases', ['methane', 'propane', 'hydrogen'])
        self.baseline_concentrations = {}
        self.alarm_thresholds = {}
        self.connected = False
        self.last_read_time = None
        
        # Set baseline concentrations and alarm thresholds
        for gas in self.supported_gases:
            self.baseline_concentrations[gas] = config.get(f'{gas}_baseline', 0.0)
            self.alarm_thresholds[gas] = config.get(f'{gas}_threshold', 100.0)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.supported_gases:
            raise ValueError("No supported gases specified")
    
    def connect(self) -> bool:
        """
        Connect to the mock sensor.
        
        Returns:
            True if connection is successful, False otherwise
        """
        # Simulate occasional connection failures
        if random.random() < 0.05:  # 5% chance of connection failure
            return False
        
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the mock sensor.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        if not self.connected:
            return False
        
        self.connected = False
        return True
    
    def read(self) -> Dict[str, Any]:
        """
        Read data from the mock gas sensor.
        
        Returns:
            Dictionary containing sensor readings
        """
        if not self.connected:
            raise RuntimeError("Sensor not connected")
        
        # Simulate occasional read errors
        if random.random() < 0.01:  # 1% chance of read error
            raise RuntimeError("Sensor read error")
        
        self.last_read_time = datetime.now()
        
        # Generate gas concentration readings
        concentrations = self.get_gas_concentration()
        
        # Check for alarms
        alarms = {}
        for gas, concentration in concentrations.items():
            alarms[gas] = concentration > self.alarm_thresholds.get(gas, float('inf'))
        
        return {
            "timestamp": self.last_read_time.isoformat(),
            "concentrations": concentrations,
            "alarms": alarms,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the mock sensor.
        
        Returns:
            Dictionary containing sensor status information
        """
        return {
            "connected": self.connected,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "supported_gases": self.supported_gases,
            "alarm_thresholds": self.alarm_thresholds,
            "last_read_time": self.last_read_time.isoformat() if self.last_read_time else None
        }
    
    def calibrate(self) -> bool:
        """
        Calibrate the mock sensor.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        if not self.connected:
            return False
        
        # Simulate occasional calibration failures
        if random.random() < 0.1:  # 10% chance of calibration failure
            return False
        
        return True
    
    def get_gas_concentration(self, gas_type: Optional[str] = None) -> Dict[str, float]:
        """
        Get gas concentration readings from the mock sensor.
        
        Args:
            gas_type: Optional specific gas type to read
            
        Returns:
            Dictionary mapping gas types to concentration values in PPM
        """
        concentrations = {}
        
        gases_to_read = [gas_type] if gas_type else self.supported_gases
        
        for gas in gases_to_read:
            if gas not in self.supported_gases:
                continue
            
            # Base concentration plus random variation
            base = self.baseline_concentrations.get(gas, 0.0)
            variation = base * 0.1  # 10% variation
            concentration = max(0.0, random.gauss(base, variation))
            
            # Occasionally simulate a spike
            if random.random() < 0.05:  # 5% chance of spike
                concentration += random.uniform(0, self.alarm_thresholds.get(gas, 100.0) * 1.5)
            
            concentrations[gas] = concentration
        
        return concentrations
    
    def get_supported_gases(self) -> List[str]:
        """
        Get the list of gas types supported by this mock sensor.
        
        Returns:
            List of supported gas types
        """
        return self.supported_gases.copy()
    
    def set_alarm_threshold(self, gas_type: str, threshold: float) -> bool:
        """
        Set the alarm threshold for a specific gas type.
        
        Args:
            gas_type: Type of gas
            threshold: Threshold value in PPM
            
        Returns:
            True if successful, False otherwise
        """
        if gas_type not in self.supported_gases:
            return False
        
        if threshold < 0:
            return False
        
        self.alarm_thresholds[gas_type] = threshold
        return True


class MockEnvironmentalSensor(EnvironmentalSensorInterface):
    """
    Mock implementation of an environmental sensor.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mock environmental sensor.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Initialize the base class first
        super().__init__(config)
        
        # Set attributes after calling super().__init__
        self.base_temperature = config.get('base_temperature', 25.0)
        self.base_humidity = config.get('base_humidity', 50.0)
        self.base_pressure = config.get('base_pressure', 1013.25)
        self.supported_vocs = config.get('supported_vocs', ['benzene', 'formaldehyde', 'toluene'])
        self.base_voc_levels = {}
        self.connected = False
        self.last_read_time = None
        
        # Set base VOC levels
        for voc in self.supported_vocs:
            self.base_voc_levels[voc] = config.get(f'{voc}_base', 0.1)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.base_temperature < -273.15:
            raise ValueError("Invalid base temperature")
        
        if self.base_humidity < 0 or self.base_humidity > 100:
            raise ValueError("Invalid base humidity")
        
        if self.base_pressure <= 0:
            raise ValueError("Invalid base pressure")
    
    def connect(self) -> bool:
        """
        Connect to the mock sensor.
        
        Returns:
            True if connection is successful, False otherwise
        """
        # Simulate occasional connection failures
        if random.random() < 0.05:  # 5% chance of connection failure
            return False
        
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the mock sensor.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        if not self.connected:
            return False
        
        self.connected = False
        return True
    
    def read(self) -> Dict[str, Any]:
        """
        Read data from the mock environmental sensor.
        
        Returns:
            Dictionary containing sensor readings
        """
        if not self.connected:
            raise RuntimeError("Sensor not connected")
        
        # Simulate occasional read errors
        if random.random() < 0.01:  # 1% chance of read error
            raise RuntimeError("Sensor read error")
        
        self.last_read_time = datetime.now()
        
        # Get environmental readings
        temperature = self.get_temperature()
        humidity = self.get_humidity()
        pressure = self.get_pressure()
        voc = self.get_voc()
        
        # Calculate dew point
        a = 17.27
        b = 237.7
        alpha = ((a * temperature) / (b + temperature)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return {
            "timestamp": self.last_read_time.isoformat(),
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "voc": voc,
            "dew_point": dew_point,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the mock sensor.
        
        Returns:
            Dictionary containing sensor status information
        """
        return {
            "connected": self.connected,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "base_temperature": self.base_temperature,
            "base_humidity": self.base_humidity,
            "base_pressure": self.base_pressure,
            "supported_vocs": self.supported_vocs,
            "last_read_time": self.last_read_time.isoformat() if self.last_read_time else None
        }
    
    def calibrate(self) -> bool:
        """
        Calibrate the mock sensor.
        
        Returns:
            True if calibration is successful, False otherwise
        """
        if not self.connected:
            return False
        
        # Simulate occasional calibration failures
        if random.random() < 0.1:  # 10% chance of calibration failure
            return False
        
        return True
    
    def get_temperature(self) -> float:
        """
        Get temperature reading from the mock sensor.
        
        Returns:
            Temperature in degrees Celsius
        """
        # Base temperature plus random variation
        variation = 2.0  # +/- 2°C
        return random.gauss(self.base_temperature, variation / 3)
    
    def get_humidity(self) -> float:
        """
        Get humidity reading from the mock sensor.
        
        Returns:
            Relative humidity as a percentage
        """
        # Base humidity plus random variation
        variation = 10.0  # +/- 10%
        humidity = random.gauss(self.base_humidity, variation / 3)
        return max(0.0, min(100.0, humidity))
    
    def get_pressure(self) -> float:
        """
        Get pressure reading from the mock sensor.
        
        Returns:
            Pressure in hPa
        """
        # Base pressure plus random variation
        variation = 5.0  # +/- 5 hPa
        return max(0.0, random.gauss(self.base_pressure, variation / 3))
    
    def get_voc(self) -> Dict[str, float]:
        """
        Get volatile organic compound readings from the mock sensor.
        
        Returns:
            Dictionary mapping VOC types to concentration values
        """
        voc_readings = {}
        
        for voc in self.supported_vocs:
            # Base VOC level plus random variation
            base = self.base_voc_levels.get(voc, 0.1)
            variation = base * 0.2  # 20% variation
            level = max(0.0, random.gauss(base, variation))
            
            # Occasionally simulate a spike
            if random.random() < 0.05:  # 5% chance of spike
                level += random.uniform(0, base * 5)
            
            voc_readings[voc] = level
        
        return voc_readings
