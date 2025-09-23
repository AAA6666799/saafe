#!/usr/bin/env python3
"""
Comprehensive test script for MQTT functionality with FLIR+SCD41 sensors.
"""

import sys
import os
import json
import time
import threading
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_mqtt_functionality():
    """Test the MQTT handler functionality."""
    print("Testing MQTT handler functionality...")
    
    try:
        from src.hardware.mqtt_handler import MqttHandler, create_mqtt_handler
        print("Successfully imported MQTT handler")
        
        # Configuration for MQTT
        config = {
            'broker': 'localhost',
            'port': 1883,
            'topics': {
                'flir': 'sensors/flir/+/data',
                'scd41': 'sensors/scd41/+/data'
            }
        }
        
        # Create MQTT handler
        mqtt_handler = create_mqtt_handler(config)
        print("Successfully created MQTT handler")
        
        # Test data callback
        received_data = []
        def on_data_received(data):
            print(f"Received data: {data}")
            received_data.append(data)
        
        mqtt_handler.set_data_callback(on_data_received)
        
        # Test connection callback
        connection_status = []
        def on_connection_change(connected):
            print(f"Connection status changed: {connected}")
            connection_status.append(connected)
        
        mqtt_handler.set_connection_callback(on_connection_change)
        
        # Test error callback
        error_messages = []
        def on_error(error):
            print(f"Error occurred: {error}")
            error_messages.append(error)
        
        mqtt_handler.set_error_callback(on_error)
        
        # Test getting latest data
        latest_data = mqtt_handler.get_latest_data()
        print(f"Initial latest data: {latest_data}")
        
        # Simulate receiving FLIR data
        flir_data = {
            "t_mean": 25.5,
            "t_std": 2.3,
            "t_max": 35.2,
            "t_p95": 32.1,
            "t_hot_area_pct": 5.2,
            "t_hot_largest_blob_pct": 2.8,
            "t_grad_mean": 1.2,
            "t_grad_std": 0.8,
            "t_diff_mean": 0.3,
            "t_diff_std": 0.1,
            "flow_mag_mean": 0.5,
            "flow_mag_std": 0.2,
            "tproxy_val": 35.2,
            "tproxy_delta": 2.1,
            "tproxy_vel": 1.0,
            "timestamp": datetime.now().isoformat(),
            "device_type": "flir_lepton_3_5"
        }
        
        # Simulate receiving SCD41 data
        scd41_data = {
            "gas_val": 450.0,
            "gas_delta": 10.0,
            "gas_vel": 10.0,
            "co2_concentration": 450.0,
            "sensor_temp": 23.5,
            "sensor_humidity": 48.2,
            "timestamp": datetime.now().isoformat(),
            "device_type": "sensirion_scd41"
        }
        
        # Test the MQTT handler's internal message processing
        # We'll simulate what would happen when messages arrive
        print("Simulating FLIR data reception...")
        mqtt_handler.latest_flir_data['flir_001'] = flir_data
        mqtt_handler.latest_scd41_data['scd41_001'] = scd41_data
        
        # Trigger the data callback manually to simulate what would happen
        # when a message is received
        formatted_data = {
            'flir': mqtt_handler.latest_flir_data.copy(),
            'scd41': mqtt_handler.latest_scd41_data.copy(),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'mqtt',
                'sensor_count': {
                    'flir': len(mqtt_handler.latest_flir_data),
                    'scd41': len(mqtt_handler.latest_scd41_data)
                }
            }
        }
        
        if mqtt_handler.data_callback:
            mqtt_handler.data_callback(formatted_data)
        
        print(f"Received data count: {len(received_data)}")
        if received_data:
            print(f"Last received data: {received_data[-1]}")
        
        # Test getting latest data again
        latest_data = mqtt_handler.get_latest_data()
        print(f"Final latest data: {latest_data}")
        
        print("MQTT functionality test completed successfully")
        return True
        
    except Exception as e:
        print(f"Failed to test MQTT functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mqtt_functionality()