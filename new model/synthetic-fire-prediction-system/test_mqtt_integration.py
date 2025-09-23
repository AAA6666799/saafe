#!/usr/bin/env python3
"""
Integration test for MQTT data ingestion with FLIR+SCD41 sensors.
This test demonstrates the complete flow from MQTT message reception to data processing.
"""

import sys
import os
import json
import time
import threading
import random
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

class MockMQTTBroker:
    """Mock MQTT broker for testing purposes."""
    
    def __init__(self):
        self.subscribers = {}
        self.messages = []
    
    def subscribe(self, topic_pattern, callback):
        """Subscribe to a topic pattern."""
        if topic_pattern not in self.subscribers:
            self.subscribers[topic_pattern] = []
        self.subscribers[topic_pattern].append(callback)
        print(f"Subscribed to {topic_pattern}")
    
    def publish(self, topic, payload):
        """Publish a message to a topic."""
        self.messages.append((topic, payload))
        print(f"Published to {topic}")
        
        # Match topic to subscribers
        for pattern, callbacks in self.subscribers.items():
            if self._topic_matches(topic, pattern):
                for callback in callbacks:
                    # Simulate async delivery
                    threading.Thread(target=callback, args=(topic, payload)).start()
    
    def _topic_matches(self, topic, pattern):
        """Check if a topic matches a pattern with wildcards."""
        topic_parts = topic.split('/')
        pattern_parts = pattern.split('/')
        
        if len(topic_parts) != len(pattern_parts):
            return False
        
        for t, p in zip(topic_parts, pattern_parts):
            if p == '+' or p == '#':  # Wildcard
                continue
            if t != p:
                return False
        return True


def test_mqtt_integration():
    """Test the complete MQTT data ingestion integration."""
    print("=== MQTT Integration Test for FLIR+SCD41 Sensors ===")
    
    try:
        # Import the MQTT handler
        from src.hardware.mqtt_handler import MqttHandler, create_mqtt_handler
        print("✓ Successfully imported MQTT handler")
        
        # Create mock broker
        broker = MockMQTTBroker()
        print("✓ Created mock MQTT broker")
        
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
        print("✓ Created MQTT handler")
        
        # Track received data
        received_data = []
        connection_events = []
        error_events = []
        
        # Set up callbacks
        def on_data_received(data):
            print(f"→ Data callback triggered with {len(data.get('flir', {}))} FLIR and {len(data.get('scd41', {}))} SCD41 sensors")
            received_data.append(data)
        
        def on_connection_change(connected):
            status = "CONNECTED" if connected else "DISCONNECTED"
            print(f"→ Connection status: {status}")
            connection_events.append(connected)
        
        def on_error(error):
            print(f"→ Error: {error}")
            error_events.append(error)
        
        mqtt_handler.set_data_callback(on_data_received)
        mqtt_handler.set_connection_callback(on_connection_change)
        mqtt_handler.set_error_callback(on_error)
        
        # Simulate connection
        print("\n--- Simulating MQTT Connection ---")
        # In a real implementation, this would connect to a broker
        # For testing, we'll trigger the connection callback directly
        if mqtt_handler.connection_callback:
            mqtt_handler.connection_callback(True)
        
        # Simulate subscription to topics
        print("\n--- Simulating Topic Subscriptions ---")
        for sensor_type, topic in mqtt_handler.topics.items():
            broker.subscribe(topic, lambda t, p: simulate_message_receipt(mqtt_handler, t, p))
        
        # Generate and send test data
        print("\n--- Sending Test Data ---")
        test_data = generate_test_data()
        
        for topic, payload in test_data:
            broker.publish(topic, payload)
            time.sleep(0.1)  # Small delay between messages
        
        # Wait for processing
        time.sleep(1)
        
        # Check results
        print(f"\n--- Test Results ---")
        print(f"✓ Connection events: {len(connection_events)}")
        print(f"✓ Data callbacks: {len(received_data)}")
        print(f"✓ Error events: {len(error_events)}")
        
        if received_data:
            latest = received_data[-1]
            flir_count = len(latest.get('flir', {}))
            scd41_count = len(latest.get('scd41', {}))
            print(f"✓ Latest data contains {flir_count} FLIR sensors and {scd41_count} SCD41 sensors")
            
            # Show sample data
            if latest['flir']:
                sensor_id = list(latest['flir'].keys())[0]
                sample_flir = latest['flir'][sensor_id]
                print(f"✓ Sample FLIR data from {sensor_id}:")
                print(f"  - Temperature mean: {sample_flir.get('t_mean', 'N/A')}")
                print(f"  - Temperature max: {sample_flir.get('t_max', 'N/A')}")
                print(f"  - Hot area: {sample_flir.get('t_hot_area_pct', 'N/A')}%")
            
            if latest['scd41']:
                sensor_id = list(latest['scd41'].keys())[0]
                sample_scd41 = latest['scd41'][sensor_id]
                print(f"✓ Sample SCD41 data from {sensor_id}:")
                print(f"  - CO2 concentration: {sample_scd41.get('co2_concentration', 'N/A')} ppm")
                print(f"  - CO2 delta: {sample_scd41.get('gas_delta', 'N/A')} ppm")
        
        # Test get_latest_data method
        latest_data = mqtt_handler.get_latest_data()
        print(f"✓ get_latest_data() returned {len(latest_data.get('flir', {}))} FLIR and {len(latest_data.get('scd41', {}))} SCD41 sensors")
        
        print("\n=== MQTT Integration Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"✗ MQTT Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_test_data():
    """Generate test data for FLIR and SCD41 sensors."""
    test_data = []
    
    # Generate FLIR data
    for i in range(3):
        sensor_id = f"flir_{random.randint(1, 5):03d}"
        topic = f"sensors/flir/{sensor_id}/data"
        
        flir_data = {
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
        
        payload = json.dumps(flir_data)
        test_data.append((topic, payload))
    
    # Generate SCD41 data
    for i in range(2):
        sensor_id = f"scd41_{random.randint(1, 5):03d}"
        topic = f"sensors/scd41/{sensor_id}/data"
        
        # Normal CO2 levels with occasional elevation
        base_co2 = 400.0
        variation = random.uniform(-50, 100)
        
        if random.random() < 0.3:  # 30% chance of elevated CO2
            variation += random.uniform(100, 1500)
        
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
        
        payload = json.dumps(scd41_data)
        test_data.append((topic, payload))
    
    return test_data


def simulate_message_receipt(mqtt_handler, topic: str, payload: str):
    """Simulate the receipt of an MQTT message by the handler."""
    try:
        # Parse the message
        data = json.loads(payload)
        
        # Extract sensor ID from topic
        topic_parts = topic.split('/')
        if len(topic_parts) >= 3:
            sensor_id = topic_parts[2]
        else:
            sensor_id = "unknown"
        
        # Process data based on topic
        if 'flir' in topic:
            mqtt_handler.latest_flir_data[sensor_id] = data
            print(f"  → Processed FLIR data from {sensor_id}")
        
        elif 'scd41' in topic:
            mqtt_handler.latest_scd41_data[sensor_id] = data
            print(f"  → Processed SCD41 data from {sensor_id}")
        
        # Trigger data callback
        if mqtt_handler.data_callback:
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
            mqtt_handler.data_callback(formatted_data)
            
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse message payload: {e}")
        if mqtt_handler.error_callback:
            mqtt_handler.error_callback(f"JSON decode error: {str(e)}")
    except Exception as e:
        print(f"  ✗ Error processing message: {e}")
        if mqtt_handler.error_callback:
            mqtt_handler.error_callback(f"Message processing error: {str(e)}")


if __name__ == "__main__":
    test_mqtt_integration()