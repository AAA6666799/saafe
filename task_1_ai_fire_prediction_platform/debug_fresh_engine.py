"""
Debug fresh engine behavior
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertEngine, AlertLevel, AlertThresholds
from synthetic_fire_system.core.interfaces import PredictionResult


def debug_fresh_engine():
    """Debug fresh engine behavior"""
    print("Debugging fresh engine behavior...")
    
    # Test case: Score 55.0 with fresh engine
    print("\n--- Test Case: Score 55.0 with Fresh Engine ---")
    test_thresholds = AlertThresholds(min_level_duration=0.0)
    alert_engine = AlertEngine(thresholds=test_thresholds)
    
    print(f"Initial level: {alert_engine.current_level.description}")
    
    prediction_result = PredictionResult(
        timestamp=time.time(),
        fire_probability=55.0 / 100.0,
        confidence_score=0.7,
        lead_time_estimate=0.0,
        contributing_factors={},
        model_ensemble_votes={}
    )
    
    alert = alert_engine.process_prediction(prediction_result)
    print(f"Final level: {alert.alert_level.description}")
    
    # Let's trace what happens in process_prediction
    print("\n--- Tracing process_prediction ---")
    
    # Extract risk score and confidence
    risk_score = getattr(prediction_result, 'fire_probability', 50.0) * 100
    confidence = getattr(prediction_result, 'confidence_score', 0.5)
    print(f"Extracted risk_score: {risk_score}")
    print(f"Extracted confidence: {confidence}")
    
    # Apply confidence adjustments
    adjusted_risk_score = alert_engine._apply_confidence_adjustments(risk_score, confidence)
    print(f"Adjusted risk_score: {adjusted_risk_score}")
    
    # Get current time and duration at current level
    current_time = time.time()
    time_at_current = current_time - alert_engine.level_start_time
    previous_level = alert_engine.current_level
    print(f"Current level: {previous_level.description}")
    print(f"Time at current level: {time_at_current:.2f}s")
    
    # Determine new alert level with hysteresis
    new_level = alert_engine.thresholds.get_level_with_hysteresis(
        adjusted_risk_score, alert_engine.current_level, time_at_current
    )
    print(f"New level from hysteresis: {new_level.description}")
    
    # Check if level changed
    level_changed = new_level != alert_engine.current_level
    print(f"Level changed: {level_changed}")
    
    # This is where the current level gets updated in the real method
    if level_changed:
        print(f"Updating current level from {alert_engine.current_level.description} to {new_level.description}")
        alert_engine.current_level = new_level
        alert_engine.level_start_time = current_time
        print(f"Level start time updated to: {alert_engine.level_start_time:.2f}")
    
    # Now let's call hysteresis again with the new current level
    print("\n--- Calling hysteresis again with new current level ---")
    time_at_current = time.time() - alert_engine.level_start_time
    new_level_2 = alert_engine.thresholds.get_level_with_hysteresis(
        adjusted_risk_score, alert_engine.current_level, time_at_current
    )
    print(f"Second hysteresis result: {new_level_2.description}")


if __name__ == "__main__":
    debug_fresh_engine()