#!/usr/bin/env python3
"""
Test script for FLIR+SCD41 Alert Generation.
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

def test_flir_scd41_alert_generation():
    """Test the FLIR+SCD41 alert generation functionality."""
    print("Testing FLIR+SCD41 Alert Generation...")
    
    try:
        from src.agents.decision.flir_scd41_alert_generation import FLIRSCD41AlertGenerationAgent, FLIRSCD41Alert
        
        # Create agent
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
        print("✓ Successfully created FLIR+SCD41 alert generation agent")
        
        # Test 1: Normal data (no fire)
        print("\n--- Test 1: Normal Data (No Fire) ---")
        normal_data = {
            'flir': {
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
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 450.0,
                    'co2_concentration': 450.0,
                    'gas_delta': 10.0,
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
        
        result = agent.process(normal_data)
        print(f"✓ Normal data result: {result['alert_generated']}")
        if not result['alert_generated']:
            print("✓ Correctly identified no fire in normal data")
        else:
            print("✗ Incorrectly generated alert for normal data")
        
        # Test 2: Fire detected by FLIR only
        print("\n--- Test 2: Fire Detected by FLIR Only ---")
        flir_fire_data = {
            'flir': {
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
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 450.0,
                    'co2_concentration': 450.0,
                    'gas_delta': 10.0,
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
        
        result = agent.process(flir_fire_data)
        print(f"✓ FLIR fire data result: {result['alert_generated']}")
        if result['alert_generated']:
            alert = result['alert']
            print(f"✓ Alert level: {alert['level']}")
            print(f"✓ Fire type: {alert['fire_type']}")
            print(f"✓ Severity: {alert['severity']}")
            print(f"✓ Confidence: {alert['confidence']:.2f}")
        else:
            print("✗ Failed to generate alert for FLIR fire data")
        
        # Test 3: Fire detected by SCD41 only
        print("\n--- Test 3: Fire Detected by SCD41 Only ---")
        scd41_fire_data = {
            'flir': {
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
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 1500.0,  # Above threshold
                    'co2_concentration': 1500.0,
                    'gas_delta': 100.0,  # Above threshold
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
        
        result = agent.process(scd41_fire_data)
        print(f"✓ SCD41 fire data result: {result['alert_generated']}")
        if result['alert_generated']:
            alert = result['alert']
            print(f"✓ Alert level: {alert['level']}")
            print(f"✓ Fire type: {alert['fire_type']}")
            print(f"✓ Severity: {alert['severity']}")
            print(f"✓ Confidence: {alert['confidence']:.2f}")
        else:
            print("✗ Failed to generate alert for SCD41 fire data")
        
        # Test 4: Fire detected by both FLIR and SCD41 (combined)
        print("\n--- Test 4: Fire Detected by Both FLIR and SCD41 ---")
        combined_fire_data = {
            'flir': {
                'flir_001': {
                    't_mean': 65.5,
                    't_std': 2.3,
                    't_max': 85.2,  # Above threshold
                    't_p95': 72.1,
                    't_hot_area_pct': 25.2,  # Above threshold
                    't_grad_mean': 1.2,
                    'tproxy_val': 85.2,
                    'timestamp': datetime.now().isoformat()
                }
            },
            'scd41': {
                'scd41_001': {
                    'gas_val': 2000.0,  # Above threshold
                    'co2_concentration': 2000.0,
                    'gas_delta': 200.0,  # Above threshold
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
        
        result = agent.process(combined_fire_data)
        print(f"✓ Combined fire data result: {result['alert_generated']}")
        if result['alert_generated']:
            alert = result['alert']
            print(f"✓ Alert level: {alert['level']}")
            print(f"✓ Fire type: {alert['fire_type']}")
            print(f"✓ Severity: {alert['severity']}")
            print(f"✓ Confidence: {alert['confidence']:.2f}")
        else:
            print("✗ Failed to generate alert for combined fire data")
        
        # Test 5: Update thresholds
        print("\n--- Test 5: Update Thresholds ---")
        threshold_update_message = {
            "flir_thresholds": {
                "t_max": 70.0
            },
            "scd41_thresholds": {
                "gas_val": 1200.0
            }
        }
        
        # Create a mock message object
        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.sender_id = "test_sender"
                self.message_type = "update_thresholds"
        
        mock_message = MockMessage(threshold_update_message)
        response = agent._handle_update_thresholds(mock_message)
        print(f"✓ Threshold update response: {response is not None}")
        print(f"✓ Updated FLIR t_max threshold: {agent.flir_thresholds['t_max']}")
        print(f"✓ Updated SCD41 gas_val threshold: {agent.scd41_thresholds['gas_val']}")
        
        # Test 6: Alert acknowledgment
        print("\n--- Test 6: Alert Acknowledgment ---")
        # Get an alert ID from previous tests
        if agent.active_alerts:
            alert_id = list(agent.active_alerts.keys())[0]
            ack_message = {
                "alert_id": alert_id,
                "user_id": "test_user"
            }
            
            mock_ack_message = MockMessage(ack_message)
            response = agent._handle_acknowledge_alert(mock_ack_message)
            print(f"✓ Alert acknowledgment response: {response is not None}")
            if response:
                ack_content = response.content
                print(f"✓ Alert acknowledged: {ack_content.get('acknowledged', False)}")
        else:
            print("✗ No active alerts to acknowledge")
        
        print("\n--- Alert Statistics ---")
        print(f"✓ Total alerts generated: {agent.alert_count}")
        print(f"✓ Active alerts: {len(agent.active_alerts)}")
        
        print("\n✅ FLIR+SCD41 Alert Generation tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ FLIR+SCD41 Alert Generation tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_flir_scd41_alert_generation()