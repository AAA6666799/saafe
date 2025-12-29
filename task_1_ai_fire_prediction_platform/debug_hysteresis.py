"""
Debug hysteresis logic
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertEngine, AlertLevel, AlertThresholds


def debug_hysteresis():
    """Debug the hysteresis logic"""
    print("Debugging hysteresis logic...")
    
    # Create thresholds
    thresholds = AlertThresholds()
    print(f"Thresholds: normal_max={thresholds.normal_max}, mild_max={thresholds.mild_max}, elevated_max={thresholds.elevated_max}")
    print(f"Hysteresis margin: {thresholds.hysteresis_margin}")
    
    # Create alert engine
    alert_engine = AlertEngine(thresholds=thresholds)
    
    # Test cases
    test_cases = [
        (15.0, AlertLevel.NORMAL, "Should be Normal"),
        (45.0, AlertLevel.MILD, "Should be Mild"),
        (75.0, AlertLevel.ELEVATED, "Should be Elevated"),
        (95.0, AlertLevel.CRITICAL, "Should be Critical")
    ]
    
    for risk_score, expected_level, description in test_cases:
        print(f"\n--- Testing risk score: {risk_score} ({description}) ---")
        
        # Get current state
        current_level = alert_engine.current_level
        time_at_current = time.time() - alert_engine.level_start_time
        print(f"Current level: {current_level.description}")
        print(f"Time at current level: {time_at_current:.2f}s")
        
        # Test direct mapping
        direct_level = AlertLevel.from_risk_score(risk_score)
        print(f"Direct mapping: {direct_level.description}")
        
        # Test hysteresis logic
        new_level = thresholds.get_level_with_hysteresis(
            risk_score, current_level, time_at_current
        )
        print(f"Hysteresis result: {new_level.description}")
        print(f"Expected: {expected_level.description}")
        
        # Update engine state for next iteration
        if new_level != current_level:
            alert_engine.current_level = new_level
            alert_engine.level_start_time = time.time()
            print(f"Updated engine state to: {new_level.description}")


if __name__ == "__main__":
    debug_hysteresis()