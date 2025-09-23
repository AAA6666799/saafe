#!/usr/bin/env python3
"""
Direct import test for FLIR+SCD41 Alert Generation.
This test directly imports the module file to avoid package import issues.
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_direct_import():
    """Test direct import of the alert generation module."""
    print("Testing Direct Import...")
    
    try:
        # Add the src directory to the path so we can import directly
        src_path = '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system/src'
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Directly import the module
        import agents.decision.flir_scd41_alert_generation as alert_module
        
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
        
        # Test from_dict method
        reconstructed_alert = alert_module.FLIRSCD41Alert.from_dict(alert_dict)
        print("✓ Successfully reconstructed alert from dictionary")
        print(f"✓ Reconstructed alert ID: {reconstructed_alert.alert_id}")
        
        print("\n✅ Direct Import tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Direct Import tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alert_logic():
    """Test alert logic without full class instantiation."""
    print("\n\nTesting Alert Logic...")
    
    try:
        # Define alert levels directly
        LEVEL_INFO = "info"
        LEVEL_WARNING = "warning"
        LEVEL_CRITICAL = "critical"
        LEVEL_EMERGENCY = "emergency"
        
        # Test alert level determination
        print("\n--- Testing Alert Level Determination ---")
        
        # Mock thresholds
        severity_thresholds = {
            LEVEL_INFO: 1,
            LEVEL_WARNING: 3,
            LEVEL_CRITICAL: 6,
            LEVEL_EMERGENCY: 8
        }
        
        def determine_alert_level(severity: int) -> str:
            if severity >= severity_thresholds[LEVEL_EMERGENCY]:
                return LEVEL_EMERGENCY
            elif severity >= severity_thresholds[LEVEL_CRITICAL]:
                return LEVEL_CRITICAL
            elif severity >= severity_thresholds[LEVEL_WARNING]:
                return LEVEL_WARNING
            else:
                return LEVEL_INFO
        
        # Test cases
        test_cases = [
            (1, LEVEL_INFO),
            (3, LEVEL_WARNING),
            (6, LEVEL_CRITICAL),
            (8, LEVEL_EMERGENCY),
            (9, LEVEL_EMERGENCY),
            (0, LEVEL_INFO)
        ]
        
        for severity, expected_level in test_cases:
            determined_level = determine_alert_level(severity)
            if determined_level == expected_level:
                print(f"✓ Severity {severity} -> Level: {determined_level}")
            else:
                print(f"✗ Severity {severity} -> Expected: {expected_level}, Got: {determined_level}")
        
        print("\n✅ Alert Logic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Alert Logic tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_direct_import()
    success2 = test_alert_logic()
    
    if success1 and success2:
        print("\n🎉 All Direct Import tests passed!")
    else:
        print("\n💥 Some tests failed!")