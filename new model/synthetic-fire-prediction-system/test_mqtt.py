#!/usr/bin/env python3
"""
Test script for MQTT functionality with FLIR+SCD41 sensors.
"""

import sys
import os
import json
import time
import threading
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

from src.hardware.mqtt_handler import MqttHandler, create_mqtt_handler


def test_mqtt_handler():
    """Test the MQTT handler functionality."""
    print("Testing MQTT handler...")
    
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
    
    # Check if MQTT is available
    try:
        import paho.mqtt.client as mqtt
        print("MQTT library is available")
    except ImportError:
        print("MQTT library is not available")
        return
    
    # Test connection
    print("Testing MQTT connection...")
    if mqtt_handler.connect():
        print("Successfully connected to MQTT broker")
    else:
        print("Failed to connect to MQTT broker")
    
    # Test data callback
    def on_data_received(data):
        print(f"Received data: {data}")
    
    mqtt_handler.set_data_callback(on_data_received)
    
    # Test getting latest data
    latest_data = mqtt_handler.get_latest_data()
    print(f"Latest data: {latest_data}")
    
    # Disconnect
    mqtt_handler.disconnect()
    print("Disconnected from MQTT broker")


if __name__ == "__main__":
    test_mqtt_handler()