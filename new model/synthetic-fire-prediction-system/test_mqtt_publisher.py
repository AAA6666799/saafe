#!/usr/bin/env python3
"""
Test script to simulate MQTT publisher for FLIR+SCD41 sensors.
This script simulates sending data to an MQTT broker that our system would subscribe to.
"""

import sys
import os
import json
import time
import random
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def simulate_mqtt_publisher():
    """Simulate an MQTT publisher sending FLIR+SCD41 data."""
    print("Simulating MQTT publisher for FLIR+SCD41 sensors...")
    
    try:
        # Try to import the real MQTT client
        import paho.mqtt.client as mqtt
        print("Using real MQTT client")
        
        # Create MQTT client
        client = mqtt.Client()
        
        # Connect to broker (this would normally connect to a real broker)
        # For simulation, we'll just show what would be sent
        print("Would connect to MQTT broker at localhost:1883")
        
    except ImportError:
        # Use mock implementation
        print("MQTT library not available, using mock implementation")
        
        class MockMQTTClient:
            def publish(self, topic, payload):
                print(f"Mock publishing to {topic}: {payload}")
        
        client = MockMQTTClient()
    
    # Simulate sending data periodically
    for i in range(5):
        print(f"\n--- Sending data batch {i+1} ---")
        
        # Generate FLIR data
        flir_sensor_id = f"flir_{random.randint(1, 10):03d}"
        flir_topic = f"sensors/flir/{flir_sensor_id}/data"
        
        flir_data = {
            "t_mean": round(random.uniform(20.0, 30.0), 2),
            "t_std": round(random.uniform(1.0, 5.0), 2),
            "t_max": round(random.uniform(25.0, 40.0), 2),
            "t_p95": round(random.uniform(24.0, 38.0), 2),
            "t_hot_area_pct": round(random.uniform(0.0, 10.0), 2),
            "t_hot_largest_blob_pct": round(random.uniform(0.0, 5.0), 2),
            "t_grad_mean": round(random.uniform(0.5, 2.0), 2),
            "t_grad_std": round(random.uniform(0.2, 1.0), 2),
            "t_diff_mean": round(random.uniform(0.0, 0.5), 2),
            "t_diff_std": round(random.uniform(0.0, 0.2), 2),
            "flow_mag_mean": round(random.uniform(0.1, 1.0), 2),
            "flow_mag_std": round(random.uniform(0.05, 0.5), 2),
            "tproxy_val": round(random.uniform(25.0, 40.0), 2),
            "tproxy_delta": round(random.uniform(0.0, 5.0), 2),
            "tproxy_vel": round(random.uniform(0.0, 2.0), 2),
            "timestamp": datetime.now().isoformat(),
            "device_type": "flir_lepton_3_5"
        }
        
        # Send FLIR data
        flir_payload = json.dumps(flir_data)
        client.publish(flir_topic, flir_payload)
        print(f"Sent FLIR data from {flir_sensor_id}")
        
        # Generate SCD41 data
        scd41_sensor_id = f"scd41_{random.randint(1, 10):03d}"
        scd41_topic = f"sensors/scd41/{scd41_sensor_id}/data"
        
        # Normal CO2 levels
        base_co2 = 400.0
        variation = random.uniform(-50, 100)
        
        # Occasionally simulate elevated CO2 (e.g., from human presence or fire)
        if random.random() < 0.2:  # 20% chance of elevated CO2
            variation += random.uniform(100, 1000)
        
        co2_concentration = max(400.0, base_co2 + variation)
        
        scd41_data = {
            "gas_val": round(co2_concentration, 2),
            "gas_delta": round(random.uniform(-20, 50), 2),
            "gas_vel": round(random.uniform(-10, 25), 2),
            "co2_concentration": round(co2_concentration, 2),
            "sensor_temp": round(random.uniform(20.0, 30.0), 2),
            "sensor_humidity": round(random.uniform(30.0, 60.0), 2),
            "timestamp": datetime.now().isoformat(),
            "device_type": "sensirion_scd41"
        }
        
        # Send SCD41 data
        scd41_payload = json.dumps(scd41_data)
        client.publish(scd41_topic, scd41_payload)
        print(f"Sent SCD41 data from {scd41_sensor_id}")
        
        # Wait before sending next batch
        time.sleep(1)
    
    print("\nMQTT publisher simulation completed")

if __name__ == "__main__":
    simulate_mqtt_publisher()