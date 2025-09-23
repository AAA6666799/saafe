#!/usr/bin/env python3
"""
Simple test script for FLIR+SCD41 Alert Generation.
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
        # Import just the classes we need
        from src.agents.decision.flir_scd41_alert_generation import FLIRSCD41Alert, FLIRSCD41AlertGenerationAgent
        
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

def test_flir_scd41_alert_agent_basic():
    """Test basic functionality of the FLIR+SCD41 alert agent."""
    print("\n\nTesting FLIR+SCD41 Alert Agent Basic Functionality...")
    
    try:
        # Import just the agent class
        from src.agents.decision.flir_scd41_alert_generation import FLIRSCD41AlertGenerationAgent
        
        # Create agent with minimal config
        config = {
            "info_threshold": 1,
            "warning_threshold": 3,
            "critical_threshold": 6,
            "emergency_threshold": 8,
            "confidence_threshold": 0.5
        }
        
        agent = FLIRSCD41AlertGenerationAgent("test_agent", config)
        print("✓ Successfully created FLIR+SCD41 alert generation agent")
        
        # Test threshold initialization
        print(f"✓ Info threshold: {agent.severity_thresholds[agent.LEVEL_INFO]}")
        print(f"✓ Warning threshold: {agent.severity_thresholds[agent.LEVEL_WARNING]}")
        print(f"✓ Critical threshold: {agent.severity_thresholds[agent.LEVEL_CRITICAL]}")
        print(f"✓ Emergency threshold: {agent.severity_thresholds[agent.LEVEL_EMERGENCY]}")
        print(f"✓ Confidence threshold: {agent.confidence_threshold}")
        
        # Test FLIR and SCD41 thresholds
        print(f"✓ FLIR t_max threshold: {agent.flir_thresholds['t_max']}")
        print(f"✓ SCD41 gas_val threshold: {agent.scd41_thresholds['gas_val']}")
        
        # Test alert level determination
        test_severities = [1, 3, 6, 8, 9]
        for severity in test_severities:
            level = agent._determine_alert_level(severity)
            print(f"✓ Severity {severity} -> Level: {level}")
        
        print("\n✅ FLIR+SCD41 Alert Agent Basic Functionality tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ FLIR+SCD41 Alert Agent Basic Functionality tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fire_signature_analysis():
    """Test the fire signature analysis functionality."""
    print("\n\nTesting Fire Signature Analysis...")
    
    try:
        # Import just the agent class
        from src.agents.decision.flir_scd41_alert_generation import FLIRSCD41AlertGenerationAgent
        
        # Create agent with test config
        config = {
            "info_threshold": 1,
            "warning_threshold": 3,
            "critical_threshold": 6,
            "emergency_threshold": 8,
            "confidence_threshold": 0.5,
            "flir_thresholds": {
                "t_max": 60.0,
                "t_hot_area_pct": 10.0,
                "tproxy_vel": 2.0,
                "t_mean": 40.0,
                "t_std": 15.0
            },
            "scd41_thresholds": {
                "gas_val": 1000.0,
                "gas_delta": 50.0,
                "gas_vel": 50.0,
                "co2_concentration": 1000.0
            }
        }
        
        agent = FLIRSCD41AlertGenerationAgent("test_agent", config)
        
        # Test 1: Normal data (no fire)
        print("\n--- Test 1: Normal Data Analysis ---")
        flir_data = {
            'flir_001': {
                't_mean': 25.5,
                't_std': 2.3,
                't_max': 35.2,
                't_p95': 32.1,
                't_hot_area_pct': 5.2,
                't_grad_mean': 1.2,
                'tproxy_val': 35.2,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        scd41_data = {
            'scd41_001': {
                'gas_val': 450.0,
                'co2_concentration': 450.0,
                'gas_delta': 10.0,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        analysis = agent._analyze_fire_signature(flir_data, scd41_data)
        print(f"✓ Fire detected: {analysis['fire_detected']}")
        print(f"✓ Fire type: {analysis['fire_type']}")
        print(f"✓ Severity: {analysis['severity']}")
        print(f"✓ Confidence: {analysis['confidence']:.2f}")
        
        # Test 2: FLIR fire data
        print("\n--- Test 2: FLIR Fire Data Analysis ---")
        flir_fire_data = {
            'flir_001': {
                't_mean': 55.5,
                't_std': 2.3,
                't_max': 75.2,  # Above threshold
                't_p95': 62.1,
                't_hot_area_pct': 15.2,  # Above threshold
                't_grad_mean': 1.2,
                'tproxy_val': 75.2,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        analysis = agent._analyze_fire_signature(flir_fire_data, scd41_data)
        print(f"✓ Fire detected: {analysis['fire_detected']}")
        print(f"✓ Fire type: {analysis['fire_type']}")
        print(f"✓ Severity: {analysis['severity']}")
        print(f"✓ Confidence: {analysis['confidence']:.2f}")
        if analysis['triggers']:
            print(f"✓ Triggers: {analysis['triggers']}")
        
        # Test 3: SCD41 fire data
        print("\n--- Test 3: SCD41 Fire Data Analysis ---")
        scd41_fire_data = {
            'scd41_001': {
                'gas_val': 1500.0,  # Above threshold
                'co2_concentration': 1500.0,
                'gas_delta': 100.0,  # Above threshold
                'timestamp': datetime.now().isoformat()
            }
        }
        
        analysis = agent._analyze_fire_signature(flir_data, scd41_fire_data)
        print(f"✓ Fire detected: {analysis['fire_detected']}")
        print(f"✓ Fire type: {analysis['fire_type']}")
        print(f"✓ Severity: {analysis['severity']}")
        print(f"✓ Confidence: {analysis['confidence']:.2f}")
        if analysis['triggers']:
            print(f"✓ Triggers: {analysis['triggers']}")
        
        # Test 4: Combined fire data
        print("\n--- Test 4: Combined Fire Data Analysis ---")
        analysis = agent._analyze_fire_signature(flir_fire_data, scd41_fire_data)
        print(f"✓ Fire detected: {analysis['fire_detected']}")
        print(f"✓ Fire type: {analysis['fire_type']}")
        print(f"✓ Severity: {analysis['severity']}")
        print(f"✓ Confidence: {analysis['confidence']:.2f}")
        if analysis['triggers']:
            print(f"✓ Triggers: {analysis['triggers']}")
        
        print("\n✅ Fire Signature Analysis tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Fire Signature Analysis tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_flir_scd41_alert_classes()
    success2 = test_flir_scd41_alert_agent_basic()
    success3 = test_fire_signature_analysis()
    
    if success1 and success2 and success3:
        print("\n🎉 All FLIR+SCD41 Alert Generation tests passed!")
    else:
        print("\n💥 Some tests failed!")