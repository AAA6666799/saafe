#!/usr/bin/env python3
"""
Test script for hardware integration with the new MLX90640 sensor.
This test directly tests our implementation without relying on the mock sensors.
"""

import sys
import os
import json

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hardware.specific.mlx90640_interface import MLX90640Interface
from hardware.specific.grove_multichannel_v2_interface import GroveMultichannelV2Interface
from hardware.specific.scd41_interface import SCD41Interface


def test_mlx90640_directly():
    """Test the MLX90640 interface directly."""
    print("Testing MLX90640 interface directly...")
    
    # Load the configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'hardware_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get the MLX90640 configuration
    mlx90640_config = config['thermal_sensors']['mlx90640_thermal']
    mlx90640_config['sensor_id'] = 'mlx90640_thermal'
    
    print(f"MLX90640 configuration: {mlx90640_config}")
    
    # Create the interface
    try:
        mlx90640 = MLX90640Interface(mlx90640_config)
        print("Successfully created MLX90640 interface")
    except Exception as e:
        print(f"Failed to create MLX90640 interface: {e}")
        return False
    
    # Test connection
    try:
        connected = mlx90640.connect()
        print(f"Connection status: {connected}")
        if not connected:
            print("Failed to connect to MLX90640")
            return False
    except Exception as e:
        print(f"Error connecting to MLX90640: {e}")
        return False
    
    # Test reading data
    try:
        data = mlx90640.read()
        print(f"Successfully read data from MLX90640:")
        print(f"  Device type: {data.get('device_type', 'N/A')}")
        print(f"  Temperature mean: {data.get('t_mean', 'N/A'):.2f}°C")
        print(f"  Temperature max: {data.get('t_max', 'N/A'):.2f}°C")
        print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
    except Exception as e:
        print(f"Error reading from MLX90640: {e}")
        return False
    
    # Test status
    try:
        status = mlx90640.get_status()
        print(f"MLX90640 status: {status}")
    except Exception as e:
        print(f"Error getting MLX90640 status: {e}")
        return False
    
    print("MLX90640 direct test passed!")
    return True


def test_gas_sensor_directly():
    """Test the gas sensor interface directly."""
    print("\nTesting gas sensor interface directly...")
    
    # Load the configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'hardware_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get the gas sensor configuration
    gas_config = config['gas_sensors']['multichannel_gas_v2']
    gas_config['sensor_id'] = 'multichannel_gas_v2'
    
    print(f"Gas sensor configuration: {gas_config}")
    
    # Create the interface
    try:
        gas_sensor = GroveMultichannelV2Interface(gas_config)
        print("Successfully created gas sensor interface")
    except Exception as e:
        print(f"Failed to create gas sensor interface: {e}")
        return False
    
    # Test connection
    try:
        connected = gas_sensor.connect()
        print(f"Connection status: {connected}")
        if not connected:
            print("Failed to connect to gas sensor")
            return False
    except Exception as e:
        print(f"Error connecting to gas sensor: {e}")
        return False
    
    # Test reading data
    try:
        data = gas_sensor.read()
        print(f"Successfully read data from gas sensor:")
        print(f"  CO concentration: {data.get('co_concentration', 'N/A'):.2f} ppm")
        print(f"  NO2 concentration: {data.get('no2_concentration', 'N/A'):.2f} ppm")
        print(f"  VOC total: {data.get('voc_total', 'N/A'):.2f} ppb")
        print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
    except Exception as e:
        print(f"Error reading from gas sensor: {e}")
        return False
    
    print("Gas sensor direct test passed!")
    return True


def test_scd41_directly():
    """Test the SCD41 interface directly."""
    print("\nTesting SCD41 interface directly...")
    
    # Load the configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'hardware_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get the SCD41 configuration
    scd41_config = config['environmental_sensors']['scd41_co2']
    scd41_config['sensor_id'] = 'scd41_co2'
    
    print(f"SCD41 configuration: {scd41_config}")
    
    # Create the interface
    try:
        scd41 = SCD41Interface(scd41_config)
        print("Successfully created SCD41 interface")
    except Exception as e:
        print(f"Failed to create SCD41 interface: {e}")
        return False
    
    # Test connection
    try:
        connected = scd41.connect()
        print(f"Connection status: {connected}")
        if not connected:
            print("Failed to connect to SCD41")
            return False
    except Exception as e:
        print(f"Error connecting to SCD41: {e}")
        return False
    
    # Test reading data
    try:
        data = scd41.read()
        print(f"Successfully read data from SCD41:")
        print(f"  CO2 concentration: {data.get('co2', 'N/A'):.2f} ppm")
        print(f"  Temperature: {data.get('temperature', 'N/A'):.2f}°C")
        print(f"  Humidity: {data.get('humidity', 'N/A'):.2f}%")
        print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
    except Exception as e:
        print(f"Error reading from SCD41: {e}")
        return False
    
    print("SCD41 direct test passed!")
    return True


if __name__ == "__main__":
    print("Running hardware specific tests...")
    
    success = True
    success &= test_mlx90640_directly()
    success &= test_gas_sensor_directly()
    success &= test_scd41_directly()
    
    if success:
        print("\nAll hardware specific tests completed successfully!")
        sys.exit(0)
    else:
        print("\nSome hardware specific tests failed!")
        sys.exit(1)