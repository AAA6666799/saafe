"""
Hardware Abstraction Layer for the synthetic fire prediction system.

This package provides unified interfaces for both synthetic and real sensors,
enabling seamless integration between simulation and production environments.
"""

from .sensor_manager import SensorManager, SensorMode, create_sensor_manager
from .real_sensors import RealThermalSensor, RealGasSensor, RealEnvironmentalSensor
from .base import SensorInterface, ThermalSensorInterface, GasSensorInterface, EnvironmentalSensorInterface

__all__ = [
    'SensorManager', 'SensorMode', 'create_sensor_manager',
    'RealThermalSensor', 'RealGasSensor', 'RealEnvironmentalSensor',
    'SensorInterface', 'ThermalSensorInterface', 'GasSensorInterface', 'EnvironmentalSensorInterface'
]