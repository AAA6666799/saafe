#!/usr/bin/env python3
"""
Test script for Device Health Monitor.
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_device_health_monitor():
    """Test the device health monitor functionality."""
    print("Testing Device Health Monitor...")
    
    try:
        from src.hardware.device_health_monitor import DeviceHealthMonitor, create_device_health_monitor
        
        # Create health monitor
        monitor = create_device_health_monitor()
        print("✓ Successfully created device health monitor")
        
        # Register devices
        monitor.register_device("flir_001", "flir", {"location": "kitchen"})
        monitor.register_device("flir_002", "flir", {"location": "living_room"})
        monitor.register_device("scd41_001", "scd41", {"location": "kitchen"})
        print("✓ Successfully registered devices")
        
        # Test 1: Record normal readings
        print("\n--- Test 1: Normal Readings ---")
        flir_reading_1 = {
            't_mean': 22.5,
            't_max': 35.2,
            't_hot_area_pct': 2.1,
            'timestamp': datetime.now().isoformat()
        }
        
        scd41_reading_1 = {
            'co2_concentration': 450.0,
            'gas_val': 450.0,
            'timestamp': datetime.now().isoformat()
        }
        
        monitor.record_device_reading("flir_001", flir_reading_1)
        monitor.record_device_reading("scd41_001", scd41_reading_1)
        print("✓ Recorded normal device readings")
        
        # Test 2: Record battery levels
        print("\n--- Test 2: Battery Levels ---")
        monitor.record_battery_level("flir_001", 85.5)
        monitor.record_battery_level("scd41_001", 92.0)
        print("✓ Recorded battery levels")
        
        # Test 3: Record response times
        print("\n--- Test 3: Response Times ---")
        monitor.record_response_time("flir_001", 0.15)
        monitor.record_response_time("scd41_001", 0.08)
        print("✓ Recorded response times")
        
        # Test 4: Record data quality scores
        print("\n--- Test 4: Data Quality Scores ---")
        monitor.record_data_quality_score("flir_001", 95.0, {"features_missing": 0})
        monitor.record_data_quality_score("scd41_001", 98.0, {"features_missing": 0})
        print("✓ Recorded data quality scores")
        
        # Test 5: Record connection status
        print("\n--- Test 5: Connection Status ---")
        monitor.record_connection_status("flir_001", True)
        monitor.record_connection_status("scd41_001", True)
        print("✓ Recorded connection status")
        
        # Test 6: Record errors
        print("\n--- Test 6: Device Errors ---")
        monitor.record_device_error("flir_002", "Connection timeout", "connection_error")
        print("✓ Recorded device errors")
        
        # Test 7: Get device health
        print("\n--- Test 7: Device Health ---")
        flir_health = monitor.get_device_health("flir_001")
        scd41_health = monitor.get_device_health("scd41_001")
        flir2_health = monitor.get_device_health("flir_002")
        
        print(f"✓ FLIR 001 health status: {flir_health['status']}")
        print(f"✓ SCD41 001 health status: {scd41_health['status']}")
        print(f"✓ FLIR 002 health status: {flir2_health['status']}")
        
        # Test 8: Get overall health
        print("\n--- Test 8: Overall Health ---")
        overall_health = monitor.get_overall_health()
        print(f"✓ Overall system status: {overall_health['overall_status']}")
        print(f"✓ Device count: {overall_health['device_count']}")
        print(f"✓ Status distribution: {overall_health['status_distribution']}")
        
        # Test 9: Health with timespan
        print("\n--- Test 9: Health with Timespan ---")
        hour_health = monitor.get_overall_health(timedelta(hours=1))
        print(f"✓ Hourly health status: {hour_health['overall_status']}")
        
        # Test 10: Multiple readings to test trends
        print("\n--- Test 10: Multiple Readings for Trends ---")
        for i in range(5):
            # Simulate degrading battery
            monitor.record_battery_level("flir_001", 85.5 - i * 2)
            time.sleep(0.1)
        
        # Record some errors
        for i in range(3):
            monitor.record_device_error("flir_001", f"Error {i+1}", "test_error")
            time.sleep(0.1)
        
        # Get updated health
        updated_health = monitor.get_device_health("flir_001")
        print(f"✓ Updated FLIR 001 health status: {updated_health['status']}")
        if 'metrics' in updated_health and 'battery_trend' in updated_health['metrics']:
            print(f"✓ Battery trend: {updated_health['metrics']['battery_trend']}")
        
        print("\n✅ Device Health Monitor tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Device Health Monitor tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_device_health_monitor()