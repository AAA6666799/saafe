"""
Debug the hysteresis method directly
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertLevel, AlertThresholds


def debug_hysteresis_method():
    """Debug the hysteresis method directly"""
    print("Debugging hysteresis method directly...")
    
    # Create thresholds
    thresholds = AlertThresholds(min_level_duration=0.0, hysteresis_margin=5.0)
    
    # Test case: Current level Normal, risk score 55.0
    current_level = AlertLevel.NORMAL
    risk_score = 55.0
    time_at_current = 0.0
    
    print(f"Current level: {current_level.description}")
    print(f"Risk score: {risk_score}")
    print(f"Time at current: {time_at_current}")
    
    # Manually trace through get_level_with_hysteresis
    print("\n--- Manual trace of get_level_with_hysteresis ---")
    
    # Apply hysteresis margin to prevent rapid oscillation
    if current_level == AlertLevel.NORMAL and risk_score > thresholds.normal_max - thresholds.hysteresis_margin:
        target_level = AlertLevel.MILD
        print(f"Condition met: current_level == NORMAL and risk_score > {thresholds.normal_max - thresholds.hysteresis_margin}")
        print(f"Setting target_level to: {target_level.description}")
    elif current_level == AlertLevel.MILD:
        print("Current level is MILD")
    elif current_level == AlertLevel.ELEVATED:
        print("Current level is ELEVATED")
    elif current_level == AlertLevel.CRITICAL:
        print("Current level is CRITICAL")
    else:
        # Default case - use direct mapping
        target_level = AlertLevel.from_risk_score(risk_score)
        print(f"Default case, using direct mapping: {target_level.description}")
    
    # Apply minimum duration constraint
    duration_constraint = time_at_current < thresholds.min_level_duration and target_level != current_level
    print(f"\nDuration constraint check:")
    print(f"  time_at_current < min_level_duration: {time_at_current} < {thresholds.min_level_duration} = {time_at_current < thresholds.min_level_duration}")
    print(f"  target_level != current_level: {target_level.description} != {current_level.description} = {target_level != current_level}")
    print(f"  Duration constraint prevents change: {duration_constraint}")
    
    if duration_constraint:
        final_level = current_level
        print(f"Final level (duration constraint): {final_level.description}")
    else:
        final_level = target_level
        print(f"Final level (no constraint): {final_level.description}")
    
    # Now call the actual method
    print("\n--- Actual method call ---")
    result = thresholds.get_level_with_hysteresis(risk_score, current_level, time_at_current)
    print(f"Method result: {result.description}")


if __name__ == "__main__":
    debug_hysteresis_method()