#!/usr/bin/env python3
"""
Debug script for sensor initialization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_mock_thermal_sensor():
    """Test mock thermal sensor initialization."""
    print("Testing MockThermalSensor...")
    try:
        from src.hardware.mock import MockThermalSensor
        config = {
            'sensor_id': 'test_thermal',
            'sensor_type': 'thermal',
            'width': 24,
            'height': 32
        }
        sensor = MockThermalSensor(config)
        print("✅ MockThermalSensor created successfully")
        print(f"  Sensor ID: {sensor.sensor_id}")
        print(f"  Sensor Type: {sensor.sensor_type}")
        return True
    except Exception as e:
        print(f"❌ Error creating MockThermalSensor: {e}")
        return False

def test_mlx90641_sensor():
    """Test MLX90641 sensor initialization."""
    print("\nTesting MLX90641Interface...")
    try:
        from src.hardware.specific.mlx90641_interface import MLX90641Interface
        config = {
            'sensor_id': 'test_mlx90641',
            'sensor_type': 'grove_mlx90641',
            'device_address': 0x33,
            'resolution': [24, 32],
            'temperature_range': [-40.0, 300.0],
            'frame_rate': 8.0
        }
        sensor = MLX90641Interface(config)
        print("✅ MLX90641Interface created successfully")
        print(f"  Sensor ID: {sensor.sensor_id}")
        print(f"  Sensor Type: {sensor.sensor_type}")
        print(f"  Device Address: {sensor.device_address}")
        print(f"  Resolution: {sensor.resolution}")
        return True
    except Exception as e:
        print(f"❌ Error creating MLX90641Interface: {e}")
        return False

def test_mock_gas_sensor():
    """Test mock gas sensor initialization."""
    print("\nTesting MockGasSensor...")
    try:
        from src.hardware.mock import MockGasSensor
        config = {
            'sensor_id': 'test_gas',
            'sensor_type': 'gas',
            'supported_gases': ['co', 'no2', 'voc']
        }
        sensor = MockGasSensor(config)
        print("✅ MockGasSensor created successfully")
        print(f"  Sensor ID: {sensor.sensor_id}")
        print(f"  Sensor Type: {sensor.sensor_type}")
        print(f"  Supported Gases: {sensor.supported_gases}")
        return True
    except Exception as e:
        print(f"❌ Error creating MockGasSensor: {e}")
        return False

def test_grove_multichannel_v2_sensor():
    """Test Grove Multichannel Gas Sensor v2 initialization."""
    print("\nTesting GroveMultichannelV2Interface...")
    try:
        from src.hardware.specific.grove_multichannel_v2_interface import GroveMultichannelV2Interface
        config = {
            'sensor_id': 'test_grove_v2',
            'sensor_type': 'grove_multichannel_v2',
            'device_address': '/dev/ttyUSB0',
            'supported_gases': ['co', 'no2', 'voc']
        }
        sensor = GroveMultichannelV2Interface(config)
        print("✅ GroveMultichannelV2Interface created successfully")
        print(f"  Sensor ID: {sensor.sensor_id}")
        print(f"  Sensor Type: {sensor.sensor_type}")
        print(f"  Device Address: {sensor.device_address}")
        print(f"  Supported Gases: {sensor.supported_gases}")
        return True
    except Exception as e:
        print(f"❌ Error creating GroveMultichannelV2Interface: {e}")
        return False

def main():
    """Main function to run sensor tests."""
    print("🔍 Debugging Sensor Initialization")
    print("=" * 35)
    
    tests = [
        test_mock_thermal_sensor,
        test_mlx90641_sensor,
        test_mock_gas_sensor,
        test_grove_multichannel_v2_sensor
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)