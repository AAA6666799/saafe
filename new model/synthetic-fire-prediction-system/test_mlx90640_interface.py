#!/usr/bin/env python3
"""
Test script for the MLX90640 interface implementation.
"""

import sys
import os
import json

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hardware.specific.mlx90640_interface import MLX90640Interface, create_mlx90640_interface


def test_mlx90640_interface():
    """Test the MLX90640 interface implementation."""
    print("Testing MLX90640 interface...")
    
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
    
    # Test resolution
    try:
        resolution = mlx90640.get_resolution()
        print(f"MLX90640 resolution: {resolution}")
    except Exception as e:
        print(f"Error getting MLX90640 resolution: {e}")
        return False
    
    # Test temperature range
    try:
        temp_range = mlx90640.get_temperature_range()
        print(f"MLX90640 temperature range: {temp_range}")
    except Exception as e:
        print(f"Error getting MLX90640 temperature range: {e}")
        return False
    
    # Test calibration
    try:
        calibrated = mlx90640.calibrate()
        print(f"Calibration status: {calibrated}")
    except Exception as e:
        print(f"Error calibrating MLX90640: {e}")
        return False
    
    print("All tests passed!")
    return True


def test_create_mlx90640_interface():
    """Test the convenience function for creating MLX90640 interface."""
    print("\nTesting create_mlx90640_interface function...")
    
    try:
        # Test with minimal configuration
        config = {
            'device_address': 0x33,
            'sensor_id': 'test_mlx90640'
        }
        
        mlx90640 = create_mlx90640_interface(config)
        print("Successfully created MLX90640 interface using convenience function")
        
        # Test connection
        connected = mlx90640.connect()
        print(f"Connection status: {connected}")
        
        return True
    except Exception as e:
        print(f"Error testing create_mlx90640_interface: {e}")
        return False


if __name__ == "__main__":
    print("Running MLX90640 interface tests...")
    
    success = True
    success &= test_mlx90640_interface()
    success &= test_create_mlx90640_interface()
    
    if success:
        print("\nAll tests completed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)