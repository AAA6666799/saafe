"""
Detailed debug of hysteresis logic
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertLevel, AlertThresholds


def debug_detailed_hysteresis():
    """Debug the hysteresis logic in detail"""
    print("Detailed debugging of hysteresis logic...")
    
    # Create thresholds
    thresholds = AlertThresholds()
    print(f"Thresholds:")
    print(f"  normal_max: {thresholds.normal_max}")
    print(f"  mild_max: {thresholds.mild_max}")
    print(f"  elevated_max: {thresholds.elevated_max}")
    print(f"  hysteresis_margin: {thresholds.hysteresis_margin}")
    print(f"  min_level_duration: {thresholds.min_level_duration}")
    
    # Test case: current level NORMAL, risk score 45.0
    current_level = AlertLevel.NORMAL
    risk_score = 45.0
    time_at_current = 0.0
    
    print(f"\n--- Test Case ---")
    print(f"Current level: {current_level.description}")
    print(f"Risk score: {risk_score}")
    print(f"Time at current level: {time_at_current}")
    
    # Step through the logic
    print(f"\n--- Logic Evaluation ---")
    
    # First condition
    condition1 = current_level == AlertLevel.NORMAL
    condition2 = risk_score > thresholds.normal_max - thresholds.hysteresis_margin
    
    print(f"current_level == AlertLevel.NORMAL: {condition1}")
    print(f"risk_score > normal_max - hysteresis_margin: {risk_score} > {thresholds.normal_max} - {thresholds.hysteresis_margin} = {thresholds.normal_max - thresholds.hysteresis_margin}")
    print(f"condition2: {condition2}")
    
    if condition1 and condition2:
        target_level = AlertLevel.MILD
        print(f"Setting target_level to: {target_level.description}")
    else:
        print("First condition not met, checking others...")
        
        # Check other conditions
        if current_level == AlertLevel.MILD:
            print("Current level is MILD")
        elif current_level == AlertLevel.ELEVATED:
            print("Current level is ELEVATED")
        elif current_level == AlertLevel.CRITICAL:
            print("Current level is CRITICAL")
        else:
            # Default case
            target_level = AlertLevel.from_risk_score(risk_score)
            print(f"Using default case, target_level from risk score: {target_level.description}")
    
    # Apply minimum duration constraint
    print(f"\n--- Duration Constraint Check ---")
    print(f"time_at_current < min_level_duration: {time_at_current} < {thresholds.min_level_duration} = {time_at_current < thresholds.min_level_duration}")
    print(f"target_level != current_level: {target_level.description} != {current_level.description} = {target_level != current_level}")
    
    if time_at_current < thresholds.min_level_duration and target_level != current_level:
        print("Duration constraint prevents level change, returning current level")
        final_level = current_level
    else:
        print("No duration constraint, returning target level")
        final_level = target_level
    
    print(f"\nFinal result: {final_level.description}")


if __name__ == "__main__":
    debug_detailed_hysteresis()