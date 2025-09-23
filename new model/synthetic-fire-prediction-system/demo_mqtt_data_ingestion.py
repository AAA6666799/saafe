#!/usr/bin/env python3
"""
Demonstration of MQTT Data Ingestion for FLIR+SCD41 Sensors.
This script shows how the system would work in a real deployment scenario.
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

def demo_mqtt_data_ingestion():
    """Demonstrate the complete MQTT data ingestion workflow."""
    print("🔥 Saafe Fire Detection System - MQTT Data Ingestion Demo 🔥")
    print("=" * 60)
    
    try:
        # Import required modules
        from src.hardware.sensor_manager import SensorManager, SensorMode
        from src.hardware.mqtt_handler import MqttHandler, create_mqtt_handler
        print("✅ System modules loaded successfully")
        
        # Configuration for a real deployment
        config = {
            'mode': SensorMode.REAL,  # In real deployment, we'd use REAL mode
            'mqtt': {
                'enabled': True,
                'broker': 'mqtt.saafe.ai',  # Real MQTT broker
                'port': 1883,
                'topics': {
                    'flir': 'sensors/flir/+/data',
                    'scd41': 'sensors/scd41/+/data'
                },
                'username': 'saafe_user',
                'password': 'saafe_password'
            },
            'buffer_size': 1000,
            'collection_interval': 0.5,  # Collect data every 0.5 seconds
            'flir_sensors': {
                'flir_001': {'device_path': '/dev/flir0'},
                'flir_002': {'device_path': '/dev/flir1'}
            },
            'scd41_sensors': {
                'scd41_001': {'device_address': '/dev/ttyUSB0'},
                'scd41_002': {'device_address': '/dev/ttyUSB1'}
            }
        }
        
        print("\n🔧 System Configuration:")
        print(f"   • Operating Mode: {config['mode']}")
        print(f"   • MQTT Enabled: {config['mqtt']['enabled']}")
        print(f"   • MQTT Broker: {config['mqtt']['broker']}:{config['mqtt']['port']}")
        print(f"   • FLIR Sensors: {len(config['flir_sensors'])}")
        print(f"   • SCD41 Sensors: {len(config['scd41_sensors'])}")
        
        # Create sensor manager
        print("\n🚀 Initializing Sensor Manager...")
        sensor_manager = SensorManager(config)
        
        # Initialize all sensors including MQTT
        print("🔌 Connecting to sensors...")
        init_results = sensor_manager.initialize_sensors()
        
        success_count = sum(1 for success in init_results.values() if success)
        total_count = len(init_results)
        print(f"✅ Sensor initialization: {success_count}/{total_count} successful")
        
        # Check MQTT connection
        if sensor_manager.mqtt_client and sensor_manager.mqtt_client.is_connected():
            print("📡 MQTT client connected successfully")
        else:
            print("⚠️  MQTT client not connected (using mock in this demo)")
        
        # Start data collection
        print("\n📈 Starting continuous data collection...")
        if sensor_manager.start_data_collection():
            print("✅ Data collection started")
        else:
            print("❌ Failed to start data collection")
            return False
        
        # Simulate receiving data for 10 seconds
        print("\n📡 Simulating MQTT data reception for 10 seconds...")
        print("📊 Monitoring incoming sensor data:")
        print("-" * 50)
        
        start_time = time.time()
        data_points_received = 0
        
        # In a real system, data would come from actual MQTT messages
        # For this demo, we'll simulate data arrival
        while time.time() - start_time < 10:
            # Simulate receiving new data every 1-2 seconds
            if random.random() < 0.3:  # 30% chance of new data
                # Generate simulated sensor data
                sensor_data = generate_simulated_sensor_data()
                
                # In a real system, this data would arrive via MQTT
                # For demo purposes, we'll add it directly to the data buffer
                sensor_manager.data_buffer.append(sensor_data)
                data_points_received += 1
                
                # Show what we received
                flir_count = len(sensor_data.get('flir', {}))
                scd41_count = len(sensor_data.get('scd41', {}))
                timestamp = sensor_data.get('metadata', {}).get('timestamp', 'N/A')
                
                print(f"📥 [{timestamp}] Received data from {flir_count} FLIR + {scd41_count} SCD41 sensors")
                
                # Show sample readings
                if sensor_data.get('flir'):
                    for sensor_id, data in list(sensor_data['flir'].items())[:1]:  # Show first FLIR
                        temp_max = data.get('t_max', 'N/A')
                        hot_area = data.get('t_hot_area_pct', 'N/A')
                        print(f"   🔥 FLIR {sensor_id}: Max temp {temp_max}°C, Hot area {hot_area}%")
                
                if sensor_data.get('scd41'):
                    for sensor_id, data in list(sensor_data['scd41'].items())[:1]:  # Show first SCD41
                        co2 = data.get('co2_concentration', 'N/A')
                        print(f"   💨 SCD41 {sensor_id}: CO2 {co2} ppm")
            
            # Small delay to simulate real-time processing
            time.sleep(0.5)
        
        # Get system status
        print(f"\n📈 Data Collection Summary:")
        print(f"   • Duration: 10 seconds")
        print(f"   • Data points received: {data_points_received}")
        print(f"   • Current data buffer size: {len(sensor_manager.data_buffer)}")
        
        # Show recent data
        recent_data = sensor_manager.get_recent_data(3)
        print(f"   • Recent data points retrieved: {len(recent_data)}")
        
        # Show sensor health
        health = sensor_manager.get_sensor_health()
        print(f"   • System health status: {health.get('overall_status', 'unknown')}")
        print(f"   • Total collections: {health.get('performance_metrics', {}).get('collection_count', 0)}")
        
        # Stop data collection
        print("\n🛑 Stopping data collection...")
        sensor_manager.stop_data_collection()
        print("✅ Data collection stopped")
        
        # Shutdown
        print("\n🔌 Shutting down system...")
        sensor_manager.shutdown()
        print("✅ System shutdown complete")
        
        print("\n" + "=" * 60)
        print("🎉 MQTT Data Ingestion Demo Completed Successfully!")
        print("💡 In a real deployment:")
        print("   • Data arrives via actual MQTT messages from IoT sensors")
        print("   • Real FLIR Lepton 3.5 and SCD41 sensors provide data")
        print("   • System runs continuously monitoring for fire conditions")
        print("   • Data is processed by AI models for fire detection")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_simulated_sensor_data():
    """Generate realistic simulated sensor data for demonstration."""
    # Timestamp for this data point
    timestamp = datetime.now().isoformat()
    
    # Simulate 1-3 FLIR sensors
    flir_data = {}
    flir_count = random.randint(1, 3)
    
    for i in range(flir_count):
        sensor_id = f"flir_{random.randint(1, 10):03d}"
        
        # Base temperature (normal room temperature)
        base_temp = random.uniform(20.0, 25.0)
        
        # Occasionally simulate elevated temperatures (fire conditions)
        if random.random() < 0.1:  # 10% chance of fire condition
            base_temp += random.uniform(10.0, 30.0)
        
        flir_data[sensor_id] = {
            "t_mean": round(base_temp + random.uniform(-2, 2), 2),
            "t_std": round(random.uniform(1.0, 5.0), 2),
            "t_max": round(base_temp + random.uniform(5, 15), 2),
            "t_p95": round(base_temp + random.uniform(3, 12), 2),
            "t_hot_area_pct": round(random.uniform(0.0, 20.0), 2),
            "t_hot_largest_blob_pct": round(random.uniform(0.0, 10.0), 2),
            "t_grad_mean": round(random.uniform(0.5, 2.0), 2),
            "t_grad_std": round(random.uniform(0.2, 1.0), 2),
            "t_diff_mean": round(random.uniform(0.0, 0.5), 2),
            "t_diff_std": round(random.uniform(0.0, 0.2), 2),
            "flow_mag_mean": round(random.uniform(0.1, 1.0), 2),
            "flow_mag_std": round(random.uniform(0.05, 0.5), 2),
            "tproxy_val": round(base_temp + random.uniform(5, 15), 2),
            "tproxy_delta": round(random.uniform(0.0, 5.0), 2),
            "tproxy_vel": round(random.uniform(0.0, 2.0), 2),
            "timestamp": timestamp,
            "device_type": "flir_lepton_3_5"
        }
    
    # Simulate 1-2 SCD41 sensors
    scd41_data = {}
    scd41_count = random.randint(1, 2)
    
    for i in range(scd41_count):
        sensor_id = f"scd41_{random.randint(1, 10):03d}"
        
        # Base CO2 level (normal indoor)
        base_co2 = 400.0
        
        # Occasionally simulate elevated CO2 (fire or human presence)
        if random.random() < 0.15:  # 15% chance of elevated CO2
            base_co2 += random.uniform(200, 2000)
        
        co2_level = max(400.0, base_co2 + random.uniform(-50, 100))
        
        scd41_data[sensor_id] = {
            "gas_val": round(co2_level, 2),
            "gas_delta": round(random.uniform(-20, 50), 2),
            "gas_vel": round(random.uniform(-10, 25), 2),
            "co2_concentration": round(co2_level, 2),
            "sensor_temp": round(random.uniform(20.0, 30.0), 2),
            "sensor_humidity": round(random.uniform(30.0, 60.0), 2),
            "timestamp": timestamp,
            "device_type": "sensirion_scd41"
        }
    
    return {
        'flir': flir_data,
        'scd41': scd41_data,
        'metadata': {
            'timestamp': timestamp,
            'source': 'mqtt_simulation',
            'sensor_count': {
                'flir': len(flir_data),
                'scd41': len(scd41_data)
            }
        }
    }


if __name__ == "__main__":
    demo_mqtt_data_ingestion()