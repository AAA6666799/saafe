#!/usr/bin/env python3
"""
Test script for Grove device configuration.

This script tests the configuration and initialization of:
1. Grove Multichannel Gas Sensor v2
2. Grove Thermal Imaging Camera IR-Array MLX90641
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.hardware.base import HardwareAbstractionLayer


def load_hardware_config():
    """Load hardware configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'hardware_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Loaded hardware configuration successfully")
        return config
    except FileNotFoundError:
        print(f"❌ Hardware configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON configuration: {e}")
        return None


def test_hardware_initialization(config):
    """Test hardware initialization with the provided configuration."""
    print("\n🔧 Testing hardware initialization...")
    
    try:
        # Create hardware abstraction layer
        hal = HardwareAbstractionLayer(config)
        
        # Initialize all sensors
        if hal.initialize():
            print("✅ Hardware initialization successful")
            
            # Get sensor statuses
            statuses = hal.get_all_sensor_statuses()
            print("\n📊 Sensor Statuses:")
            for sensor_id, status in statuses.items():
                print(f"  {sensor_id}:")
                for key, value in status.items():
                    print(f"    {key}: {value}")
            
            # Get sensor readings
            print("\n📡 Sensor Readings:")
            readings = hal.get_all_sensor_readings()
            for sensor_id, reading in readings.items():
                print(f"  {sensor_id}:")
                for key, value in reading.items():
                    print(f"    {key}: {value}")
            
            # Calibrate sensors
            print("\n⚙️  Calibrating sensors...")
            calibration_results = hal.calibrate_all_sensors()
            for sensor_id, success in calibration_results.items():
                status = "✅ Success" if success else "❌ Failed"
                print(f"  {sensor_id}: {status}")
            
            # Shutdown sensors
            print("\n🔌 Shutting down sensors...")
            if hal.shutdown():
                print("✅ Sensors shut down successfully")
            else:
                print("❌ Error shutting down sensors")
                
            return True
        else:
            print("❌ Hardware initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during hardware initialization: {e}")
        return False


def main():
    """Main function to run the device configuration test."""
    print("🔍 Testing Grove Device Configuration")
    print("=" * 40)
    
    # Load configuration
    config = load_hardware_config()
    if not config:
        return False
    
    # Print configuration summary
    print("\n📋 Configuration Summary:")
    print(f"  Sensor Mode: {config.get('sensor_mode', 'unknown')}")
    print(f"  Thermal Sensors: {len(config.get('thermal_sensors', {}))}")
    print(f"  Gas Sensors: {len(config.get('gas_sensors', {}))}")
    print(f"  Environmental Sensors: {len(config.get('environmental_sensors', {}))}")
    
    # Test hardware initialization
    success = test_hardware_initialization(config)
    
    if success:
        print("\n🎉 All tests passed! Devices are properly configured.")
    else:
        print("\n💥 Some tests failed. Please check the configuration.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)