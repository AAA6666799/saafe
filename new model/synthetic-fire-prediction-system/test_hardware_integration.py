#!/usr/bin/env python3
"""
Test script for hardware integration with the new MLX90640 sensor.
"""

import sys
import os
import json

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hardware.base import HardwareAbstractionLayer


def test_hardware_integration():
    """Test the hardware integration with the new MLX90640 sensor."""
    print("Testing hardware integration with MLX90640...")
    
    # Load the configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'hardware_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Hardware configuration: {config}")
    
    # Create the hardware abstraction layer
    try:
        hal = HardwareAbstractionLayer(config)
        print("Successfully created HardwareAbstractionLayer")
    except Exception as e:
        print(f"Failed to create HardwareAbstractionLayer: {e}")
        return False
    
    # Initialize all sensors
    try:
        initialized = hal.initialize()
        print(f"Hardware initialization status: {initialized}")
        if not initialized:
            print("Failed to initialize hardware")
            return False
    except Exception as e:
        print(f"Error initializing hardware: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check if thermal sensors were initialized
    print(f"Thermal sensors: {list(hal.thermal_sensors.keys())}")
    print(f"Gas sensors: {list(hal.gas_sensors.keys())}")
    print(f"Environmental sensors: {list(hal.environmental_sensors.keys())}")
    
    # Test reading from all sensors
    try:
        readings = hal.get_all_sensor_readings()
        print(f"Sensor readings: {readings}")
    except Exception as e:
        print(f"Error reading from sensors: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test getting status from all sensors
    try:
        statuses = hal.get_all_sensor_statuses()
        print(f"Sensor statuses: {statuses}")
    except Exception as e:
        print(f"Error getting sensor statuses: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test calibrating all sensors
    try:
        calibration_results = hal.calibrate_all_sensors()
        print(f"Calibration results: {calibration_results}")
    except Exception as e:
        print(f"Error calibrating sensors: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Shutdown all sensors
    try:
        shutdown = hal.shutdown()
        print(f"Hardware shutdown status: {shutdown}")
    except Exception as e:
        print(f"Error shutting down hardware: {e}")
        return False
    
    print("All hardware integration tests passed!")
    return True


if __name__ == "__main__":
    print("Running hardware integration tests...")
    
    success = test_hardware_integration()
    
    if success:
        print("\nAll hardware integration tests completed successfully!")
        sys.exit(0)
    else:
        print("\nSome hardware integration tests failed!")
        sys.exit(1)