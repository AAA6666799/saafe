"""
Debug the hysteresis method directly
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertLevel, AlertThresholds, AlertEngine


def debug_hysteresis_method():
    """Debug the hysteresis method directly"""
    print("Debugging hysteresis method directly...")
    
    # Create thresholds
    thresholds = AlertThresholds(hysteresis_margin=5.0, min_level_duration=0.0)
    
    # Test cases
    test_cases = [
        (AlertLevel.NORMAL, 45.0, "Normal -> Mild"),
        (AlertLevel.MILD, 48.0, "Mild -> ?"),
        (AlertLevel.MILD, 55.0, "Mild -> Elevated"),
        (AlertLevel.ELEVATED, 52.0, "Elevated -> ?"),
        (AlertLevel.ELEVATED, 90.0, "Elevated -> Critical")
    ]
    
    for current_level, risk_score, description in test_cases:
        time_at_current = 0.0
        
        print(f"\n--- {description} ---")
        print(f"Current level: {current_level.description}")
        print(f"Risk score: {risk_score}")
        
        result = thresholds.get_level_with_hysteresis(risk_score, current_level, time_at_current)
        print(f"Result: {result.description}")


if __name__ == "__main__":
    debug_hysteresis_method()