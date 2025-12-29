"""
Accurate debug of alert engine behavior
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertEngine, AlertLevel, AlertThresholds
from synthetic_fire_system.core.interfaces import PredictionResult


def debug_accurate():
    """Accurately debug alert engine behavior"""
    print("Accurately debugging alert engine behavior...")
    
    # Create fresh engine
    test_thresholds = AlertThresholds(min_level_duration=0.0)
    alert_engine = AlertEngine(thresholds=test_thresholds)
    
    print(f"Initial level: {alert_engine.current_level.description}")
    
    # Create prediction with score 55.0
    prediction_result = PredictionResult(
        timestamp=time.time(),
        fire_probability=55.0 / 100.0,
        confidence_score=0.7,
        lead_time_estimate=0.0,
        contributing_factors={},
        model_ensemble_votes={}
    )
    
    # Manually trace through process_prediction
    print("\n--- Manual Trace of process_prediction ---")
    
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
    print(f"Previous level: {previous_level.description}")
    print(f"Time at current level: {time_at_current:.2f}s")
    
    # Determine new alert level with hysteresis
    new_level = alert_engine.thresholds.get_level_with_hysteresis(
        adjusted_risk_score, alert_engine.current_level, time_at_current
    )
    print(f"New level from hysteresis: {new_level.description}")
    
    # Check if level changed
    level_changed = new_level != alert_engine.current_level
    print(f"Level changed: {level_changed}")
    
    # Actually call process_prediction
    print("\n--- Actual process_prediction result ---")
    alert = alert_engine.process_prediction(prediction_result)
    print(f"Actual result level: {alert.alert_level.description}")
    
    # Test with a higher score to see multiple transitions
    print("\n--- Testing with score 95.0 ---")
    alert_engine2 = AlertEngine(thresholds=test_thresholds)
    print(f"Initial level: {alert_engine2.current_level.description}")
    
    prediction_result2 = PredictionResult(
        timestamp=time.time(),
        fire_probability=95.0 / 100.0,
        confidence_score=0.95,
        lead_time_estimate=0.0,
        contributing_factors={},
        model_ensemble_votes={}
    )
    
    alert2 = alert_engine2.process_prediction(prediction_result2)
    print(f"Result level: {alert2.alert_level.description}")


if __name__ == "__main__":
    debug_accurate()