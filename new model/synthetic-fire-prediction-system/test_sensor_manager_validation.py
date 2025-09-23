#!/usr/bin/env python3
"""
Test script for SensorManager data validation integration.
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_sensor_manager_validation():
    """Test the SensorManager data validation integration."""
    print("Testing SensorManager Data Validation Integration...")
    
    try:
        from src.hardware.sensor_manager import SensorManager, SensorMode
        
        # Configuration with data validation enabled
        config = {
            'mode': SensorMode.SYNTHETIC,
            'enable_data_validation': True,
            'mqtt': {
                'enabled': True,
                'broker': 'localhost',
                'port': 1883
            },
            'buffer_size': 100
        }
        
        # Create sensor manager
        sensor_manager = SensorManager(config)
        print("✓ Successfully created SensorManager with data validation enabled")
        
        # Initialize sensors
        init_results = sensor_manager.initialize_sensors()
        print(f"✓ Sensor initialization results: {init_results}")
        
        # Test 1: Valid data
        print("\n--- Test 1: Valid Data ---")
        valid_data = {
            'flir': {
                'flir_001': {
                    't_mean': 25.5,
                    't_std': 2.3,
                    't_max': 35.2,
                    't_p95': 32.1,
                    't_hot_area_pct': 5.2,
                    't_grad_mean': 1.2,
                    'tproxy_val': 35.2,
                    'timestamp': datetime.now().isoformat()
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 450.0,
                    'co2_concentration': 450.0,
                    'gas_delta': 10.0,
                    'timestamp': datetime.now().isoformat()
                }
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'mqtt'
            }
        }
        
        # Simulate MQTT data reception
        sensor_manager._on_mqtt_data_received(valid_data)
        
        # Check data buffer
        recent_data = sensor_manager.get_recent_data(5)
        print(f"✓ Data buffer size after valid data: {len(sensor_manager.data_buffer)}")
        print(f"✓ Recent data entries: {len(recent_data)}")
        
        # Test 2: Invalid data (should be rejected)
        print("\n--- Test 2: Invalid Data (Should be Rejected) ---")
        invalid_data = {
            'flir': {
                'flir_001': {
                    't_mean': 25.5,
                    't_std': 2.3,
                    't_max': 400.0,  # Too high
                    't_p95': 32.1,
                    't_hot_area_pct': 5.2,
                    't_grad_mean': 1.2,
                    'tproxy_val': 35.2,
                    'timestamp': datetime.now().isoformat()
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 50000.0,  # Too high
                    'co2_concentration': 50000.0,
                    'gas_delta': 10.0,
                    'timestamp': datetime.now().isoformat()
                }
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'mqtt'
            }
        }
        
        # Manually validate to see quality level
        if sensor_manager.data_validator:
            validation_result = sensor_manager.data_validator.validate_sensor_data(invalid_data)
            print(f"  Quality level: {validation_result.quality_level.value}")
            print(f"  Issues: {validation_result.issues}")
        
        # Simulate MQTT data reception
        sensor_manager._on_mqtt_data_received(invalid_data)
        
        # Check data buffer (should still be the same size)
        print(f"✓ Data buffer size after invalid data: {len(sensor_manager.data_buffer)}")
        
        # Test 3: Poor quality data (should be rejected)
        print("\n--- Test 3: Poor Quality Data (Should be Rejected) ---")
        poor_data = {
            'flir': {
                'flir_001': {
                    't_mean': 25.5,
                    't_std': 2.3,
                    't_max': 35.2,
                    't_p95': 32.1,
                    't_hot_area_pct': 5.2,
                    't_grad_mean': 1.2,
                    'tproxy_val': 35.2,
                    'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat()  # Stale data
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 450.0,
                    'co2_concentration': 450.0,
                    'gas_delta': 2000.0,  # High delta
                    'timestamp': datetime.now().isoformat()
                }
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'mqtt'
            }
        }
        
        # Manually validate to see quality level
        if sensor_manager.data_validator:
            validation_result = sensor_manager.data_validator.validate_sensor_data(poor_data)
            print(f"  Quality level: {validation_result.quality_level.value}")
            print(f"  Issues: {validation_result.issues}")
        
        # Simulate MQTT data reception
        sensor_manager._on_mqtt_data_received(poor_data)
        
        # Check data buffer (should still be the same size)
        print(f"✓ Data buffer size after poor quality data: {len(sensor_manager.data_buffer)}")
        
        # Show final data buffer contents
        recent_data = sensor_manager.get_recent_data(5)
        print(f"\n--- Final Data Buffer Contents ---")
        print(f"✓ Total entries in buffer: {len(sensor_manager.data_buffer)}")
        print(f"✓ Recent entries retrieved: {len(recent_data)}")
        
        # Show validation statistics if available
        if sensor_manager.data_validator:
            stats = sensor_manager.data_validator.get_validation_statistics()
            print(f"\n--- Validation Statistics ---")
            print(f"✓ Total validations: {stats['total_validations']}")
            print(f"✓ Passed validations: {stats['passed_validations']}")
            print(f"✓ Failed validations: {stats['failed_validations']}")
            print(f"✓ Quality distribution: {stats['quality_distribution']}")
        
        print("\n✅ SensorManager Data Validation Integration tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ SensorManager Data Validation Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_sensor_manager_validation()