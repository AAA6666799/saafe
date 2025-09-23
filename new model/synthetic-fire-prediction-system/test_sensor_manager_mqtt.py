#!/usr/bin/env python3
"""
Test script to demonstrate MQTT integration with the SensorManager.
This shows how MQTT data is processed by the sensor manager.
"""

import sys
import os
import json
import time
import random
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_sensor_manager_mqtt_integration():
    """Test the integration between MQTT handler and SensorManager."""
    print("=== SensorManager MQTT Integration Test ===")
    
    try:
        # Import the sensor manager
        from src.hardware.sensor_manager import SensorManager, SensorMode
        print("✓ Successfully imported SensorManager")
        
        # Configuration for sensor manager with MQTT enabled
        config = {
            'mode': SensorMode.SYNTHETIC,
            'mqtt': {
                'enabled': True,
                'broker': 'localhost',
                'port': 1883,
                'topics': {
                    'flir': 'sensors/flir/+/data',
                    'scd41': 'sensors/scd41/+/data'
                }
            },
            'buffer_size': 100,
            'collection_interval': 1.0
        }
        
        # Create sensor manager
        sensor_manager = SensorManager(config)
        print("✓ Created SensorManager with MQTT enabled")
        
        # Initialize sensors (this will also initialize MQTT if enabled)
        print("\n--- Initializing Sensors ---")
        init_results = sensor_manager.initialize_sensors()
        print(f"✓ Sensor initialization results: {init_results}")
        
        # Check if MQTT client was created
        if sensor_manager.mqtt_client:
            print("✓ MQTT client successfully created")
        else:
            print("⚠ MQTT client not created (expected in mock environment)")
        
        # Simulate receiving MQTT data by directly adding to the MQTT client's data store
        print("\n--- Simulating MQTT Data Reception ---")
        if sensor_manager.mqtt_client:
            # Add some FLIR data
            flir_data = {
                'flir_001': generate_flir_data(),
                'flir_002': generate_flir_data()
            }
            
            # Add some SCD41 data
            scd41_data = {
                'scd41_001': generate_scd41_data(),
                'scd41_002': generate_scd41_data()
            }
            
            # Update the MQTT client's data stores
            sensor_manager.mqtt_client.latest_flir_data.update(flir_data)
            sensor_manager.mqtt_client.latest_scd41_data.update(scd41_data)
            
            print(f"✓ Added {len(flir_data)} FLIR and {len(scd41_data)} SCD41 sensors to MQTT client")
            
            # Simulate the MQTT data callback
            mqtt_data = {
                'flir': flir_data,
                'scd41': scd41_data,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'mqtt',
                    'sensor_count': {
                        'flir': len(flir_data),
                        'scd41': len(scd41_data)
                    }
                }
            }
            
            # Trigger the MQTT data received callback
            sensor_manager._on_mqtt_data_received(mqtt_data)
            print("✓ Triggered MQTT data received callback")
        
        # Check the data buffer
        recent_data = sensor_manager.get_recent_data(5)
        print(f"✓ Data buffer contains {len(recent_data)} entries")
        
        if recent_data:
            latest_entry = recent_data[-1]
            flir_count = len(latest_entry.get('flir', {}))
            scd41_count = len(latest_entry.get('scd41', {}))
            print(f"✓ Latest buffer entry contains {flir_count} FLIR and {scd41_count} SCD41 sensors")
            
            # Show sample data
            if latest_entry.get('flir'):
                sensor_id = list(latest_entry['flir'].keys())[0]
                sample_flir = latest_entry['flir'][sensor_id]
                print(f"✓ Sample FLIR data from {sensor_id}:")
                print(f"  - Temperature mean: {sample_flir.get('t_mean', 'N/A')}")
                print(f"  - Temperature max: {sample_flir.get('t_max', 'N/A')}")
            
            if latest_entry.get('scd41'):
                sensor_id = list(latest_entry['scd41'].keys())[0]
                sample_scd41 = latest_entry['scd41'][sensor_id]
                print(f"✓ Sample SCD41 data from {sensor_id}:")
                print(f"  - CO2 concentration: {sample_scd41.get('co2_concentration', 'N/A')} ppm")
        
        # Test reading all sensors (this would normally include MQTT data)
        print("\n--- Reading All Sensors ---")
        all_sensor_data = sensor_manager.read_all_sensors()
        
        flir_sensors = len(all_sensor_data.get('flir', {}))
        scd41_sensors = len(all_sensor_data.get('scd41', {}))
        print(f"✓ read_all_sensors() returned {flir_sensors} FLIR and {scd41_sensors} SCD41 sensors")
        
        # Show sensor health
        health = sensor_manager.get_sensor_health()
        print(f"✓ Overall system health: {health.get('overall_status', 'unknown')}")
        print(f"✓ Number of failed sensors: {len(health.get('failed_sensors', []))}")
        
        print("\n=== SensorManager MQTT Integration Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"✗ SensorManager MQTT Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_flir_data():
    """Generate sample FLIR sensor data."""
    return {
        "t_mean": round(random.uniform(20.0, 30.0), 2),
        "t_std": round(random.uniform(1.0, 5.0), 2),
        "t_max": round(random.uniform(25.0, 45.0), 2),
        "t_p95": round(random.uniform(24.0, 40.0), 2),
        "t_hot_area_pct": round(random.uniform(0.0, 15.0), 2),
        "t_hot_largest_blob_pct": round(random.uniform(0.0, 8.0), 2),
        "t_grad_mean": round(random.uniform(0.5, 2.0), 2),
        "t_grad_std": round(random.uniform(0.2, 1.0), 2),
        "t_diff_mean": round(random.uniform(0.0, 0.5), 2),
        "t_diff_std": round(random.uniform(0.0, 0.2), 2),
        "flow_mag_mean": round(random.uniform(0.1, 1.0), 2),
        "flow_mag_std": round(random.uniform(0.05, 0.5), 2),
        "tproxy_val": round(random.uniform(25.0, 45.0), 2),
        "tproxy_delta": round(random.uniform(0.0, 5.0), 2),
        "tproxy_vel": round(random.uniform(0.0, 2.0), 2),
        "timestamp": datetime.now().isoformat(),
        "device_type": "flir_lepton_3_5"
    }


def generate_scd41_data():
    """Generate sample SCD41 sensor data."""
    # Normal CO2 levels with occasional elevation
    base_co2 = 400.0
    variation = random.uniform(-50, 100)
    
    if random.random() < 0.3:  # 30% chance of elevated CO2
        variation += random.uniform(100, 1500)
    
    co2_concentration = max(400.0, base_co2 + variation)
    
    return {
        "gas_val": round(co2_concentration, 2),
        "gas_delta": round(random.uniform(-20, 50), 2),
        "gas_vel": round(random.uniform(-10, 25), 2),
        "co2_concentration": round(co2_concentration, 2),
        "sensor_temp": round(random.uniform(20.0, 30.0), 2),
        "sensor_humidity": round(random.uniform(30.0, 60.0), 2),
        "timestamp": datetime.now().isoformat(),
        "device_type": "sensirion_scd41"
    }


if __name__ == "__main__":
    test_sensor_manager_mqtt_integration()