"""
Final debug of hysteresis logic
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertLevel, AlertThresholds


def debug_final_hysteresis():
    """Debug the hysteresis logic with exact values"""
    print("Final debugging of hysteresis logic...")
    
    # Create thresholds
    thresholds = AlertThresholds(hysteresis_margin=5.0, min_level_duration=0.0)
    print(f"Thresholds:")
    print(f"  normal_max: {thresholds.normal_max}")
    print(f"  mild_max: {thresholds.mild_max}")
    print(f"  elevated_max: {thresholds.elevated_max}")
    print(f"  hysteresis_margin: {thresholds.hysteresis_margin}")
    print(f"  min_level_duration: {thresholds.min_level_duration}")
    
    # Test specific case: score 48, current level Mild
    current_level = AlertLevel.MILD
    risk_score = 48.0
    time_at_current = 0.0
    
    print(f"\n--- Test Case ---")
    print(f"Current level: {current_level.description}")
    print(f"Risk score: {risk_score}")
    print(f"Time at current level: {time_at_current}")
    
    # Evaluate conditions
    print(f"\n--- Condition Evaluation ---")
    
    if current_level == AlertLevel.MILD:
        print("Current level is MILD")
        cond1 = risk_score < thresholds.normal_max + thresholds.hysteresis_margin
        cond2 = risk_score > thresholds.mild_max - thresholds.hysteresis_margin
        
        print(f"  risk_score < normal_max + hysteresis_margin: {risk_score} < {thresholds.normal_max} + {thresholds.hysteresis_margin} = {thresholds.normal_max + thresholds.hysteresis_margin} -> {cond1}")
        print(f"  risk_score > mild_max - hysteresis_margin: {risk_score} > {thresholds.mild_max} - {thresholds.hysteresis_margin} = {thresholds.mild_max - thresholds.hysteresis_margin} -> {cond2}")
        
        if cond1:
            target_level = AlertLevel.NORMAL
            print(f"  Target level: NORMAL")
        elif cond2:
            target_level = AlertLevel.ELEVATED
            print(f"  Target level: ELEVATED")
        else:
            target_level = current_level
            print(f"  Target level: MILD (no change)")
    else:
        # This shouldn't happen in our test
        target_level = AlertLevel.from_risk_score(risk_score)
        print(f"Using direct mapping: {target_level.description}")
    
    # Apply duration constraint
    duration_check = time_at_current < thresholds.min_level_duration and target_level != current_level
    print(f"\nDuration constraint check:")
    print(f"  time_at_current < min_level_duration: {time_at_current} < {thresholds.min_level_duration} = {time_at_current < thresholds.min_level_duration}")
    print(f"  target_level != current_level: {target_level.description} != {current_level.description} = {target_level != current_level}")
    print(f"  Duration constraint prevents change: {duration_check}")
    
    if duration_check:
        final_level = current_level
        print(f"Final level (duration constraint): {final_level.description}")
    else:
        final_level = target_level
        print(f"Final level (no constraint): {final_level.description}")


if __name__ == "__main__":
    debug_final_hysteresis()