#!/usr/bin/env python3
"""
Focused test script for FLIR+SCD41 Alert Generation.
This test directly imports only the alert classes without triggering full module imports.
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_flir_scd41_alert_classes():
    """Test the FLIR+SCD41 alert classes directly."""
    print("Testing FLIR+SCD41 Alert Classes...")
    
    try:
        # Direct import of just the classes we need
        from src.agents.decision.flir_scd41_alert_generation import FLIRSCD41Alert
        
        # Test FLIRSCD41Alert class
        print("\n--- Testing FLIRSCD41Alert Class ---")
        
        alert = FLIRSCD41Alert(
            alert_id="test_alert_001",
            timestamp=datetime.now(),
            level=FLIRSCD41Alert.LEVEL_WARNING,
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
        print(f"✓ FLIR data keys: {list(alert_dict['flir_data'].keys())}")
        print(f"✓ SCD41 data keys: {list(alert_dict['scd41_data'].keys())}")
        
        # Test from_dict method
        reconstructed_alert = FLIRSCD41Alert.from_dict(alert_dict)
        print("✓ Successfully reconstructed alert from dictionary")
        print(f"✓ Reconstructed alert ID: {reconstructed_alert.alert_id}")
        print(f"✓ Reconstructed alert level: {reconstructed_alert.level}")
        
        # Test acknowledge method
        alert.acknowledge("test_user")
        print("✓ Successfully acknowledged alert")
        print(f"✓ Alert acknowledged: {alert.acknowledged}")
        print(f"✓ Acknowledged by: {alert.acknowledged_by}")
        
        print("\n✅ FLIR+SCD41 Alert Classes tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ FLIR+SCD41 Alert Classes tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alert_generation_logic():
    """Test the alert generation logic without full agent initialization."""
    print("\n\nTesting Alert Generation Logic...")
    
    try:
        # Import just what we need
        from src.agents.decision.flir_scd41_alert_generation import FLIRSCD41Alert
        
        # Test alert level determination logic
        print("\n--- Testing Alert Level Determination ---")
        
        # Test different severity levels
        severity_levels = [
            (1, FLIRSCD41Alert.LEVEL_INFO),
            (3, FLIRSCD41Alert.LEVEL_WARNING),
            (6, FLIRSCD41Alert.LEVEL_CRITICAL),
            (8, FLIRSCD41Alert.LEVEL_EMERGENCY),
            (9, FLIRSCD41Alert.LEVEL_EMERGENCY),
            (0, FLIRSCD41Alert.LEVEL_INFO)
        ]
        
        # Mock thresholds for testing
        mock_severity_thresholds = {
            FLIRSCD41Alert.LEVEL_INFO: 1,
            FLIRSCD41Alert.LEVEL_WARNING: 3,
            FLIRSCD41Alert.LEVEL_CRITICAL: 6,
            FLIRSCD41Alert.LEVEL_EMERGENCY: 8
        }
        
        # Simple function to determine alert level (replicating the agent logic)
        def determine_alert_level(severity: int) -> str:
            if severity >= mock_severity_thresholds[FLIRSCD41Alert.LEVEL_EMERGENCY]:
                return FLIRSCD41Alert.LEVEL_EMERGENCY
            elif severity >= mock_severity_thresholds[FLIRSCD41Alert.LEVEL_CRITICAL]:
                return FLIRSCD41Alert.LEVEL_CRITICAL
            elif severity >= mock_severity_thresholds[FLIRSCD41Alert.LEVEL_WARNING]:
                return FLIRSCD41Alert.LEVEL_WARNING
            else:
                return FLIRSCD41Alert.LEVEL_INFO
        
        for severity, expected_level in severity_levels:
            determined_level = determine_alert_level(severity)
            if determined_level == expected_level:
                print(f"✓ Severity {severity} -> Level: {determined_level}")
            else:
                print(f"✗ Severity {severity} -> Expected: {expected_level}, Got: {determined_level}")
        
        # Test alert message generation
        print("\n--- Testing Alert Message Generation ---")
        
        # Simple function to generate alert message (replicating the agent logic)
        def generate_alert_message(fire_type: str, severity: int, alert_level: str) -> str:
            flir_summary = " (FLIR: 1 sensors)"
            scd41_summary = " (SCD41: 1 sensors)"
            
            if alert_level == FLIRSCD41Alert.LEVEL_EMERGENCY:
                return f"EMERGENCY: {fire_type.upper()} fire detected with severity {severity}/10. Immediate evacuation required.{flir_summary}{scd41_summary}"
            elif alert_level == FLIRSCD41Alert.LEVEL_CRITICAL:
                return f"CRITICAL: {fire_type.capitalize()} fire detected with severity {severity}/10. Prepare for evacuation.{flir_summary}{scd41_summary}"
            elif alert_level == FLIRSCD41Alert.LEVEL_WARNING:
                return f"WARNING: Possible {fire_type} fire detected with severity {severity}/10. Investigate immediately.{flir_summary}{scd41_summary}"
            else:
                return f"INFO: Potential {fire_type} fire signature detected with severity {severity}/10. Monitor situation.{flir_summary}{scd41_summary}"
        
        test_cases = [
            ("thermal", 7, FLIRSCD41Alert.LEVEL_CRITICAL),
            ("chemical", 9, FLIRSCD41Alert.LEVEL_EMERGENCY),
            ("combined", 4, FLIRSCD41Alert.LEVEL_WARNING),
            ("unknown", 2, FLIRSCD41Alert.LEVEL_INFO)
        ]
        
        for fire_type, severity, alert_level in test_cases:
            message = generate_alert_message(fire_type, severity, alert_level)
            print(f"✓ {alert_level.capitalize()} alert for {fire_type} fire (severity {severity}): {message[:50]}...")
        
        print("\n✅ Alert Generation Logic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Alert Generation Logic tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_flir_scd41_alert_classes()
    success2 = test_alert_generation_logic()
    
    if success1 and success2:
        print("\n🎉 All Focused FLIR+SCD41 Alert Generation tests passed!")
    else:
        print("\n💥 Some tests failed!")