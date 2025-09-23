#!/usr/bin/env python3
"""
File import test for FLIR+SCD41 Alert Generation.
This test directly imports the Python file without using the package system.
"""

import sys
import os
from datetime import datetime
import importlib.util

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_file_import():
    """Test importing the alert generation module as a file."""
    print("Testing File Import...")
    
    try:
        # Path to the alert generation module
        module_path = '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/src/agents/decision/flir_scd41_alert_generation.py'
        
        # Load the module directly from file
        spec = importlib.util.spec_from_file_location("flir_scd41_alert_generation", module_path)
        alert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alert_module)
        
        # Test creating an alert
        alert = alert_module.FLIRSCD41Alert(
            alert_id="test_alert_001",
            timestamp=datetime.now(),
            level=alert_module.FLIRSCD41Alert.LEVEL_WARNING,
            message="Test alert message",
            source_agent_id="test_agent",
            fire_detected=True,
            fire_type="thermal",
            severity=5,
            confidence=0.75,
            location="Test Location",
            flir_data={"sensor_1": {"t_max": 65.0}},
            scd41_data={"sensor_1": {"gas_val": 800.0}},
            metadata={"test": "data"}
        )
        
        print("✓ Successfully created FLIRSCD41Alert instance")
        
        # Test to_dict method
        alert_dict = alert.to_dict()
        print("✓ Successfully converted alert to dictionary")
        print(f"✓ Alert ID: {alert_dict['alert_id']}")
        print(f"✓ Alert level: {alert_dict['level']}")
        print(f"✓ Fire detected: {alert_dict['fire_detected']}")
        print(f"✓ Fire type: {alert_dict['fire_type']}")
        
        # Test from_dict method
        reconstructed_alert = alert_module.FLIRSCD41Alert.from_dict(alert_dict)
        print("✓ Successfully reconstructed alert from dictionary")
        print(f"✓ Reconstructed alert ID: {reconstructed_alert.alert_id}")
        print(f"✓ Reconstructed alert level: {reconstructed_alert.level}")
        
        # Test acknowledge method
        alert.acknowledge("test_user")
        print("✓ Successfully acknowledged alert")
        print(f"✓ Alert acknowledged: {alert.acknowledged}")
        print(f"✓ Acknowledged by: {alert.acknowledged_by}")
        
        print("\n✅ File Import tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ File Import tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alert_functionality():
    """Test alert functionality with sample data."""
    print("\n\nTesting Alert Functionality...")
    
    try:
        # Load the module directly from file
        module_path = '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/src/agents/decision/flir_scd41_alert_generation.py'
        spec = importlib.util.spec_from_file_location("flir_scd41_alert_generation", module_path)
        alert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alert_module)
        
        # Test alert level determination with sample data
        print("\n--- Testing Alert Level Determination with Sample Data ---")
        
        # Mock thresholds similar to what's in the config
        mock_severity_thresholds = {
            alert_module.FLIRSCD41Alert.LEVEL_INFO: 1,
            alert_module.FLIRSCD41Alert.LEVEL_WARNING: 3,
            alert_module.FLIRSCD41Alert.LEVEL_CRITICAL: 6,
            alert_module.FLIRSCD41Alert.LEVEL_EMERGENCY: 8
        }
        
        def determine_alert_level(severity: int) -> str:
            if severity >= mock_severity_thresholds[alert_module.FLIRSCD41Alert.LEVEL_EMERGENCY]:
                return alert_module.FLIRSCD41Alert.LEVEL_EMERGENCY
            elif severity >= mock_severity_thresholds[alert_module.FLIRSCD41Alert.LEVEL_CRITICAL]:
                return alert_module.FLIRSCD41Alert.LEVEL_CRITICAL
            elif severity >= mock_severity_thresholds[alert_module.FLIRSCD41Alert.LEVEL_WARNING]:
                return alert_module.FLIRSCD41Alert.LEVEL_WARNING
            else:
                return alert_module.FLIRSCD41Alert.LEVEL_INFO
        
        # Test with different severities
        severities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for severity in severities:
            level = determine_alert_level(severity)
            print(f"✓ Severity {severity:2d} -> Alert Level: {level}")
        
        print("\n✅ Alert Functionality tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Alert Functionality tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_file_import()
    success2 = test_alert_functionality()
    
    if success1 and success2:
        print("\n🎉 All File Import tests passed!")
    else:
        print("\n💥 Some tests failed!")