#!/usr/bin/env python3
"""
Simple test script for MQTT functionality with FLIR+SCD41 sensors.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_mqtt_import():
    """Test importing the MQTT handler."""
    print("Testing MQTT handler import...")
    
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
        
        # Check if MQTT is available
        try:
            import paho.mqtt.client as mqtt
            print("MQTT library is available")
        except ImportError:
            print("MQTT library is not available")
        
        return True
    except Exception as e:
        print(f"Failed to import MQTT handler: {e}")
        return False

if __name__ == "__main__":
    test_mqtt_import()