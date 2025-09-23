"""
Final debug of alert engine behavior
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertEngine, AlertLevel, AlertThresholds
from synthetic_fire_system.core.interfaces import PredictionResult


def debug_alert_engine():
    """Debug alert engine behavior"""
    print("Debugging alert engine behavior...")
    
    # Create thresholds with no minimum duration
    test_thresholds = AlertThresholds(min_level_duration=0.0)
    
    # Test case: Score 55.0 with fresh engine
    print("\n--- Test Case: Score 55.0 ---")
    alert_engine = AlertEngine(thresholds=test_thresholds)
    
    # Check initial state
    print(f"Initial level: {alert_engine.current_level.description}")
    print(f"Time at current level: {time.time() - alert_engine.level_start_time:.2f}s")
    
    # Create prediction
    prediction_result = PredictionResult(
        timestamp=time.time(),
        fire_probability=55.0 / 100.0,
        confidence_score=0.7,
        lead_time_estimate=0.0,
        contributing_factors={},
        model_ensemble_votes={}
    )
    
    # Process prediction
    alert = alert_engine.process_prediction(prediction_result)
    print(f"Result level: {alert.alert_level.description}")
    print(f"Expected level: Elevated Risk")
    print(f"Match: {'✓' if alert.alert_level == AlertLevel.ELEVATED else '✗'}")
    
    # Test case: Score 95.0 with fresh engine
    print("\n--- Test Case: Score 95.0 ---")
    alert_engine = AlertEngine(thresholds=test_thresholds)
    
    # Check initial state
    print(f"Initial level: {alert_engine.current_level.description}")
    print(f"Time at current level: {time.time() - alert_engine.level_start_time:.2f}s")
    
    # Create prediction
    prediction_result = PredictionResult(
        timestamp=time.time(),
        fire_probability=95.0 / 100.0,
        confidence_score=0.95,
        lead_time_estimate=0.0,
        contributing_factors={},
        model_ensemble_votes={}
    )
    
    # Process prediction
    alert = alert_engine.process_prediction(prediction_result)
    print(f"Result level: {alert.alert_level.description}")
    print(f"Expected level: CRITICAL FIRE ALERT")
    print(f"Match: {'✓' if alert.alert_level == AlertLevel.CRITICAL else '✗'}")
    
    # Debug the hysteresis method directly for score 95.0
    print("\n--- Direct Hysteresis Method Debug for Score 95.0 ---")
    current_level = alert_engine.current_level  # Should be Normal
    risk_score = 95.0
    time_at_current = time.time() - alert_engine.level_start_time
    
    print(f"Current level: {current_level.description}")
    print(f"Risk score: {risk_score}")
    print(f"Time at current: {time_at_current:.2f}s")
    
    result = alert_engine.thresholds.get_level_with_hysteresis(
        risk_score, current_level, time_at_current
    )
    print(f"Hysteresis result: {result.description}")
    print(f"Expected: CRITICAL FIRE ALERT")
    print(f"Match: {'✓' if result == AlertLevel.CRITICAL else '✗'}")


if __name__ == "__main__":
    debug_alert_engine()