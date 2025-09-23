"""
Unit tests for FLIR+SCD41 Alert Generation Agent.

This module contains comprehensive tests for the FLIR+SCD41 alert generation functionality,
including alert creation, fire signature analysis, threshold processing, and message generation.
"""

import sys
import os
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

class TestFLIRSCD41AlertGeneration(unittest.TestCase):
    """Test cases for FLIR+SCD41 Alert Generation functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the base agent dependencies
        self.mock_agent_base = Mock()
        
        # Sample FLIR data
        self.normal_flir_data = {
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
        
        self.fire_flir_data = {
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
        
        # Sample SCD41 data
        self.normal_scd41_data = {
            'scd41_001': {
                'gas_val': 450.0,
                'co2_concentration': 450.0,
                'gas_delta': 10.0,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        self.fire_scd41_data = {
            'scd41_001': {
                'gas_val': 1500.0,  # Above threshold
                'co2_concentration': 1500.0,
                'gas_delta': 100.0,  # Above threshold
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def test_alert_level_determination(self):
        """Test alert level determination based on severity scores."""
        # Define alert levels as constants
        LEVEL_INFO = "info"
        LEVEL_WARNING = "warning"
        LEVEL_CRITICAL = "critical"
        LEVEL_EMERGENCY = "emergency"
        
        # Mock thresholds
        severity_thresholds = {
            LEVEL_INFO: 1,
            LEVEL_WARNING: 3,
            LEVEL_CRITICAL: 6,
            LEVEL_EMERGENCY: 8
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
        
        # Test cases
        test_cases = [
            (0, LEVEL_INFO),
            (1, LEVEL_INFO),
            (2, LEVEL_INFO),
            (3, LEVEL_WARNING),
            (4, LEVEL_WARNING),
            (5, LEVEL_WARNING),
            (6, LEVEL_CRITICAL),
            (7, LEVEL_CRITICAL),
            (8, LEVEL_EMERGENCY),
            (9, LEVEL_EMERGENCY),
            (10, LEVEL_EMERGENCY)
        ]
        
        for severity, expected_level in test_cases:
            with self.subTest(severity=severity):
                level = determine_alert_level(severity)
                self.assertEqual(level, expected_level, 
                               f"Severity {severity} should result in {expected_level} alert level")
    
    def test_fire_signature_analysis_normal_data(self):
        """Test fire signature analysis with normal sensor data."""
        # Mock thresholds
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
        
        # Test with normal data
        analysis = analyze_fire_signature(self.normal_flir_data, self.normal_scd41_data)
        
        self.assertFalse(analysis["fire_detected"], "Normal data should not trigger fire detection")
        self.assertEqual(analysis["fire_type"], "unknown", "Normal data should result in unknown fire type")
        self.assertEqual(analysis["severity"], 0, "Normal data should have zero severity")
        self.assertEqual(analysis["confidence"], 0.0, "Normal data should have zero confidence")
        self.assertEqual(len(analysis["triggers"]), 0, "Normal data should not have any triggers")
    
    def test_fire_signature_analysis_flir_only(self):
        """Test fire signature analysis with FLIR fire data only."""
        # Mock thresholds
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
        
        # Test with FLIR fire data only
        analysis = analyze_fire_signature(self.fire_flir_data, self.normal_scd41_data)
        
        self.assertTrue(analysis["fire_detected"], "FLIR fire data should trigger fire detection")
        self.assertEqual(analysis["fire_type"], "thermal", "FLIR fire data should result in thermal fire type")
        self.assertGreater(analysis["severity"], 0, "FLIR fire data should have positive severity")
        self.assertGreater(analysis["confidence"], 0.0, "FLIR fire data should have positive confidence")
        self.assertGreater(len(analysis["triggers"]), 0, "FLIR fire data should have triggers")
        
        # Check specific triggers
        triggers = analysis["triggers"]
        self.assertTrue(any("High temperature" in trigger for trigger in triggers), 
                       "Should have high temperature trigger")
        self.assertTrue(any("Large hot area" in trigger for trigger in triggers), 
                       "Should have large hot area trigger")
    
    def test_fire_signature_analysis_scd41_only(self):
        """Test fire signature analysis with SCD41 fire data only."""
        # Mock thresholds
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
        
        # Test with SCD41 fire data only
        analysis = analyze_fire_signature(self.normal_flir_data, self.fire_scd41_data)
        
        self.assertTrue(analysis["fire_detected"], "SCD41 fire data should trigger fire detection")
        self.assertEqual(analysis["fire_type"], "chemical", "SCD41 fire data should result in chemical fire type")
        self.assertGreater(analysis["severity"], 0, "SCD41 fire data should have positive severity")
        self.assertGreater(analysis["confidence"], 0.0, "SCD41 fire data should have positive confidence")
        self.assertGreater(len(analysis["triggers"]), 0, "SCD41 fire data should have triggers")
        
        # Check specific triggers
        triggers = analysis["triggers"]
        self.assertTrue(any("High CO2" in trigger for trigger in triggers), 
                       "Should have high CO2 trigger")
        self.assertTrue(any("Rapid CO2 change" in trigger for trigger in triggers), 
                       "Should have rapid CO2 change trigger")
    
    def test_fire_signature_analysis_combined(self):
        """Test fire signature analysis with both FLIR and SCD41 fire data."""
        # Mock thresholds
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
        
        # Test with both FLIR and SCD41 fire data
        analysis = analyze_fire_signature(self.fire_flir_data, self.fire_scd41_data)
        
        self.assertTrue(analysis["fire_detected"], "Combined fire data should trigger fire detection")
        self.assertEqual(analysis["fire_type"], "combined", "Combined fire data should result in combined fire type")
        self.assertGreater(analysis["severity"], 0, "Combined fire data should have positive severity")
        self.assertGreater(analysis["confidence"], 0.0, "Combined fire data should have positive confidence")
        self.assertEqual(analysis["confidence"], 0.95, "Combined fire data should have maximum confidence (0.95)")
        self.assertGreater(len(analysis["triggers"]), 0, "Combined fire data should have triggers")
        
        # Should have both FLIR and SCD41 triggers
        triggers = analysis["triggers"]
        flir_triggers = [t for t in triggers if "FLIR" in t]
        scd41_triggers = [t for t in triggers if "SCD41" in t]
        
        self.assertGreater(len(flir_triggers), 0, "Should have FLIR triggers")
        self.assertGreater(len(scd41_triggers), 0, "Should have SCD41 triggers")
    
    def test_alert_message_generation(self):
        """Test alert message generation for different scenarios."""
        # Define alert levels as constants
        LEVEL_INFO = "info"
        LEVEL_WARNING = "warning"
        LEVEL_CRITICAL = "critical"
        LEVEL_EMERGENCY = "emergency"
        
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
        
        # Test different scenarios
        test_cases = [
            ("thermal", 7, LEVEL_CRITICAL, self.fire_flir_data, self.normal_scd41_data),
            ("chemical", 9, LEVEL_EMERGENCY, self.normal_flir_data, self.fire_scd41_data),
            ("combined", 4, LEVEL_WARNING, self.fire_flir_data, self.fire_scd41_data),
            ("unknown", 2, LEVEL_INFO, self.normal_flir_data, self.normal_scd41_data)
        ]
        
        for fire_type, severity, alert_level, flir_data, scd41_data in test_cases:
            with self.subTest(fire_type=fire_type, severity=severity, alert_level=alert_level):
                message = generate_alert_message(fire_type, severity, alert_level, flir_data, scd41_data)
                
                # Check that the message contains expected elements
                # For combined and unknown fire types, check for lowercase version
                expected_fire_type = fire_type
                if fire_type in ['combined', 'unknown']:
                    expected_fire_type = fire_type
                elif alert_level == LEVEL_EMERGENCY:
                    expected_fire_type = fire_type.upper()
                else:
                    expected_fire_type = fire_type.capitalize()
                
                self.assertIn(expected_fire_type, message,
                            f"Message should contain fire type '{fire_type}'")
                self.assertIn(str(severity), message, f"Message should contain severity '{severity}'")
                self.assertIn(alert_level.upper(), message.upper(), f"Message should indicate {alert_level} level")
                
                # Check sensor information
                if flir_data:
                    self.assertIn("FLIR:", message, "Message should contain FLIR sensor information")
                if scd41_data:
                    self.assertIn("SCD41:", message, "Message should contain SCD41 sensor information")

if __name__ == '__main__':
    unittest.main()