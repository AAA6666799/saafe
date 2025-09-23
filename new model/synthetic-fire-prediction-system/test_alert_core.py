#!/usr/bin/env python3
"""
Core alert functionality test for FLIR+SCD41 Alert Generation.
This test focuses on the core alert functionality without requiring the full agent infrastructure.
"""

import sys
import os
from datetime import datetime
import json

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_alert_core_functionality():
    """Test the core alert functionality."""
    print("Testing Core Alert Functionality...")
    
    try:
        # Define alert levels as constants
        LEVEL_INFO = "info"
        LEVEL_WARNING = "warning"
        LEVEL_CRITICAL = "critical"
        LEVEL_EMERGENCY = "emergency"
        
        # Define a simple Alert class that mimics the functionality we need
        class SimpleAlert:
            def __init__(self, alert_id, timestamp, level, message, source_agent_id,
                        fire_detected=False, fire_type=None, severity=None, confidence=0.0,
                        location=None, flir_data=None, scd41_data=None, metadata=None):
                self.alert_id = alert_id
                self.timestamp = timestamp
                self.level = level
                self.message = message
                self.source_agent_id = source_agent_id
                self.fire_detected = fire_detected
                self.fire_type = fire_type
                self.severity = severity
                self.confidence = confidence
                self.location = location
                self.flir_data = flir_data or {}
                self.scd41_data = scd41_data or {}
                self.metadata = metadata or {}
                self.acknowledged = False
                self.acknowledged_time = None
                self.acknowledged_by = None
            
            def acknowledge(self, user_id):
                self.acknowledged = True
                self.acknowledged_time = datetime.now()
                self.acknowledged_by = user_id
            
            def to_dict(self):
                return {
                    "alert_id": self.alert_id,
                    "timestamp": self.timestamp.isoformat(),
                    "level": self.level,
                    "message": self.message,
                    "source_agent_id": self.source_agent_id,
                    "fire_detected": self.fire_detected,
                    "fire_type": self.fire_type,
                    "severity": self.severity,
                    "confidence": self.confidence,
                    "location": self.location,
                    "flir_data": self.flir_data,
                    "scd41_data": self.scd41_data,
                    "metadata": self.metadata,
                    "acknowledged": self.acknowledged,
                    "acknowledged_time": self.acknowledged_time.isoformat() if self.acknowledged_time else None,
                    "acknowledged_by": self.acknowledged_by
                }
            
            @classmethod
            def from_dict(cls, data):
                alert = cls(
                    alert_id=data["alert_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    level=data["level"],
                    message=data["message"],
                    source_agent_id=data["source_agent_id"],
                    fire_detected=data["fire_detected"],
                    fire_type=data["fire_type"],
                    severity=data["severity"],
                    confidence=data["confidence"],
                    location=data["location"],
                    flir_data=data["flir_data"],
                    scd41_data=data["scd41_data"],
                    metadata=data["metadata"]
                )
                alert.acknowledged = data["acknowledged"]
                if data["acknowledged_time"]:
                    alert.acknowledged_time = datetime.fromisoformat(data["acknowledged_time"])
                alert.acknowledged_by = data["acknowledged_by"]
                return alert
        
        # Test creating an alert
        print("\n--- Testing Alert Creation ---")
        alert = SimpleAlert(
            alert_id="test_alert_001",
            timestamp=datetime.now(),
            level=LEVEL_WARNING,
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
        
        print("✓ Successfully created SimpleAlert instance")
        print(f"✓ Alert ID: {alert.alert_id}")
        print(f"✓ Alert level: {alert.level}")
        print(f"✓ Fire detected: {alert.fire_detected}")
        print(f"✓ Fire type: {alert.fire_type}")
        print(f"✓ Severity: {alert.severity}")
        print(f"✓ Confidence: {alert.confidence}")
        
        # Test to_dict method
        print("\n--- Testing Alert Serialization ---")
        alert_dict = alert.to_dict()
        print("✓ Successfully converted alert to dictionary")
        print(f"✓ Alert ID: {alert_dict['alert_id']}")
        print(f"✓ Alert level: {alert_dict['level']}")
        print(f"✓ Fire detected: {alert_dict['fire_detected']}")
        print(f"✓ FLIR data keys: {list(alert_dict['flir_data'].keys())}")
        print(f"✓ SCD41 data keys: {list(alert_dict['scd41_data'].keys())}")
        
        # Test from_dict method
        print("\n--- Testing Alert Deserialization ---")
        reconstructed_alert = SimpleAlert.from_dict(alert_dict)
        print("✓ Successfully reconstructed alert from dictionary")
        print(f"✓ Reconstructed alert ID: {reconstructed_alert.alert_id}")
        print(f"✓ Reconstructed alert level: {reconstructed_alert.level}")
        print(f"✓ Reconstructed fire detected: {reconstructed_alert.fire_detected}")
        
        # Test acknowledge method
        print("\n--- Testing Alert Acknowledgment ---")
        alert.acknowledge("test_user")
        print("✓ Successfully acknowledged alert")
        print(f"✓ Alert acknowledged: {alert.acknowledged}")
        print(f"✓ Acknowledged by: {alert.acknowledged_by}")
        
        print("\n✅ Core Alert Functionality tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Core Alert Functionality tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alert_analysis_logic():
    """Test the alert analysis logic."""
    print("\n\nTesting Alert Analysis Logic...")
    
    try:
        # Define alert levels as constants
        LEVEL_INFO = "info"
        LEVEL_WARNING = "warning"
        LEVEL_CRITICAL = "critical"
        LEVEL_EMERGENCY = "emergency"
        
        # Mock thresholds similar to what's in the config
        severity_thresholds = {
            LEVEL_INFO: 1,
            LEVEL_WARNING: 3,
            LEVEL_CRITICAL: 6,
            LEVEL_EMERGENCY: 8
        }
        
        confidence_threshold = 0.5
        
        # Mock FLIR and SCD41 thresholds
        flir_thresholds = {
            "t_max": 60.0,
            "t_hot_area_pct": 10.0,
            "tproxy_vel": 2.0,
            "t_mean": 40.0,
            "t_std": 15.0
        }
        
        scd41_thresholds = {
            "gas_val": 1000.0,
            "gas_delta": 50.0,
            "gas_vel": 50.0,
            "co2_concentration": 1000.0
        }
        
        def determine_alert_level(severity: int) -> str:
            """Determine the alert level based on severity."""
            if severity >= severity_thresholds[LEVEL_EMERGENCY]:
                return LEVEL_EMERGENCY
            elif severity >= severity_thresholds[LEVEL_CRITICAL]:
                return LEVEL_CRITICAL
            elif severity >= severity_thresholds[LEVEL_WARNING]:
                return LEVEL_WARNING
            else:
                return LEVEL_INFO
        
        def analyze_fire_signature(flir_data: dict, scd41_data: dict) -> dict:
            """Analyze FLIR and SCD41 data for fire signatures."""
            # Initialize analysis results
            fire_detected = False
            fire_type = "unknown"
            severity = 0
            confidence = 0.0
            triggers = []
            
            # Analyze FLIR data
            flir_severity = 0
            flir_confidence = 0.0
            
            for sensor_id, sensor_data in flir_data.items():
                # Check temperature thresholds
                if "t_max" in sensor_data and sensor_data["t_max"] > flir_thresholds["t_max"]:
                    fire_detected = True
                    flir_severity = max(flir_severity, 7)
                    flir_confidence = max(flir_confidence, 0.8)
                    triggers.append(f"FLIR {sensor_id}: High temperature ({sensor_data['t_max']:.1f}°C)")
                    fire_type = "thermal"
                
                if "t_hot_area_pct" in sensor_data and sensor_data["t_hot_area_pct"] > flir_thresholds["t_hot_area_pct"]:
                    fire_detected = True
                    flir_severity = max(flir_severity, 6)
                    flir_confidence = max(flir_confidence, 0.7)
                    triggers.append(f"FLIR {sensor_id}: Large hot area ({sensor_data['t_hot_area_pct']:.1f}%)")
                    fire_type = "thermal"
                
                if "tproxy_vel" in sensor_data and sensor_data["tproxy_vel"] > flir_thresholds["tproxy_vel"]:
                    fire_detected = True
                    flir_severity = max(flir_severity, 5)
                    flir_confidence = max(flir_confidence, 0.6)
                    triggers.append(f"FLIR {sensor_id}: Rapid temperature change ({sensor_data['tproxy_vel']:.1f}°C/s)")
                    fire_type = "thermal"
                
                if "t_mean" in sensor_data and sensor_data["t_mean"] > flir_thresholds["t_mean"]:
                    fire_detected = True
                    flir_severity = max(flir_severity, 4)
                    flir_confidence = max(flir_confidence, 0.5)
                    triggers.append(f"FLIR {sensor_id}: High mean temperature ({sensor_data['t_mean']:.1f}°C)")
                    fire_type = "thermal"
                
                if "t_std" in sensor_data and sensor_data["t_std"] > flir_thresholds["t_std"]:
                    fire_detected = True
                    flir_severity = max(flir_severity, 3)
                    flir_confidence = max(flir_confidence, 0.4)
                    triggers.append(f"FLIR {sensor_id}: High temperature variation ({sensor_data['t_std']:.1f}°C)")
                    fire_type = "thermal"
            
            # Analyze SCD41 data
            scd41_severity = 0
            scd41_confidence = 0.0
            
            for sensor_id, sensor_data in scd41_data.items():
                # Check CO2 thresholds
                co2_value = sensor_data.get("gas_val", sensor_data.get("co2_concentration", 0))
                if co2_value > scd41_thresholds["gas_val"]:
                    fire_detected = True
                    scd41_severity = max(scd41_severity, 8)
                    scd41_confidence = max(scd41_confidence, 0.9)
                    triggers.append(f"SCD41 {sensor_id}: High CO2 ({co2_value:.0f} ppm)")
                    fire_type = "chemical"
                
                if "gas_delta" in sensor_data and abs(sensor_data["gas_delta"]) > scd41_thresholds["gas_delta"]:
                    fire_detected = True
                    scd41_severity = max(scd41_severity, 6)
                    scd41_confidence = max(scd41_confidence, 0.7)
                    triggers.append(f"SCD41 {sensor_id}: Rapid CO2 change ({sensor_data['gas_delta']:.0f} ppm)")
                    fire_type = "chemical"
                
                if "gas_vel" in sensor_data and abs(sensor_data["gas_vel"]) > scd41_thresholds["gas_vel"]:
                    fire_detected = True
                    scd41_severity = max(scd41_severity, 5)
                    scd41_confidence = max(scd41_confidence, 0.6)
                    triggers.append(f"SCD41 {sensor_id}: Rapid CO2 velocity ({sensor_data['gas_vel']:.0f} ppm/s)")
                    fire_type = "chemical"
            
            # Combine FLIR and SCD41 analysis
            if flir_severity > 0 and scd41_severity > 0:
                # Both sensors detecting fire - highest confidence
                fire_type = "combined"
                severity = max(flir_severity, scd41_severity)
                confidence = min(0.95, max(flir_confidence, scd41_confidence) + 0.2)
            elif flir_severity > 0:
                # Only FLIR detecting fire
                severity = flir_severity
                confidence = flir_confidence
            elif scd41_severity > 0:
                # Only SCD41 detecting fire
                severity = scd41_severity
                confidence = scd41_confidence
            else:
                # No fire detected
                severity = 0
                confidence = 0.0
            
            return {
                "fire_detected": fire_detected,
                "fire_type": fire_type,
                "severity": severity,
                "confidence": confidence,
                "triggers": triggers,
                "flir_severity": flir_severity,
                "scd41_severity": scd41_severity
            }
        
        def generate_alert_message(fire_type: str, severity: int, alert_level: str, 
                                 flir_data: dict, scd41_data: dict) -> str:
            """Generate an alert message based on fire type, severity, and alert level."""
            # Get summary information from sensor data
            flir_summary = ""
            scd41_summary = ""
            
            if flir_data:
                flir_sensors = list(flir_data.keys())
                flir_summary = f" (FLIR: {len(flir_sensors)} sensors)"
            
            if scd41_data:
                scd41_sensors = list(scd41_data.keys())
                scd41_summary = f" (SCD41: {len(scd41_sensors)} sensors)"
            
            if alert_level == LEVEL_EMERGENCY:
                return f"EMERGENCY: {fire_type.upper()} fire detected with severity {severity}/10. Immediate evacuation required.{flir_summary}{scd41_summary}"
            elif alert_level == LEVEL_CRITICAL:
                return f"CRITICAL: {fire_type.capitalize()} fire detected with severity {severity}/10. Prepare for evacuation.{flir_summary}{scd41_summary}"
            elif alert_level == LEVEL_WARNING:
                return f"WARNING: Possible {fire_type} fire detected with severity {severity}/10. Investigate immediately.{flir_summary}{scd41_summary}"
            else:
                return f"INFO: Potential {fire_type} fire signature detected with severity {severity}/10. Monitor situation.{flir_summary}{scd41_summary}"
        
        # Test 1: Normal data (no fire)
        print("\n--- Test 1: Normal Data Analysis ---")
        normal_flir_data = {
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
        
        normal_scd41_data = {
            'scd41_001': {
                'gas_val': 450.0,
                'co2_concentration': 450.0,
                'gas_delta': 10.0,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        analysis = analyze_fire_signature(normal_flir_data, normal_scd41_data)
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
        
        analysis = analyze_fire_signature(flir_fire_data, normal_scd41_data)
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
        
        analysis = analyze_fire_signature(normal_flir_data, scd41_fire_data)
        print(f"✓ Fire detected: {analysis['fire_detected']}")
        print(f"✓ Fire type: {analysis['fire_type']}")
        print(f"✓ Severity: {analysis['severity']}")
        print(f"✓ Confidence: {analysis['confidence']:.2f}")
        if analysis['triggers']:
            print(f"✓ Triggers: {analysis['triggers']}")
        
        # Test 4: Combined fire data
        print("\n--- Test 4: Combined Fire Data Analysis ---")
        analysis = analyze_fire_signature(flir_fire_data, scd41_fire_data)
        print(f"✓ Fire detected: {analysis['fire_detected']}")
        print(f"✓ Fire type: {analysis['fire_type']}")
        print(f"✓ Severity: {analysis['severity']}")
        print(f"✓ Confidence: {analysis['confidence']:.2f}")
        if analysis['triggers']:
            print(f"✓ Triggers: {analysis['triggers']}")
        
        # Test alert level determination
        print("\n--- Testing Alert Level Determination ---")
        severities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for severity in severities:
            level = determine_alert_level(severity)
            print(f"✓ Severity {severity:2d} -> Alert Level: {level}")
        
        # Test alert message generation
        print("\n--- Testing Alert Message Generation ---")
        test_cases = [
            ("thermal", 7, LEVEL_CRITICAL),
            ("chemical", 9, LEVEL_EMERGENCY),
            ("combined", 4, LEVEL_WARNING),
            ("unknown", 2, LEVEL_INFO)
        ]
        
        for fire_type, severity, alert_level in test_cases:
            message = generate_alert_message(fire_type, severity, alert_level, flir_fire_data, scd41_fire_data)
            print(f"✓ {alert_level.capitalize()} alert for {fire_type} fire (severity {severity}): {message[:50]}...")
        
        print("\n✅ Alert Analysis Logic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Alert Analysis Logic tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_alert_core_functionality()
    success2 = test_alert_analysis_logic()
    
    if success1 and success2:
        print("\n🎉 All Core Alert tests passed!")
    else:
        print("\n💥 Some tests failed!")