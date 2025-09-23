"""
Debug the exact test sequence
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertEngine, AlertLevel, AlertThresholds
from synthetic_fire_system.core.interfaces import PredictionResult


def debug_test_sequence():
    """Debug the exact sequence from our test"""
    print("Debugging the exact test sequence...")
    
    # Create alert engine with test thresholds
    test_thresholds = AlertThresholds(min_level_duration=0.0, hysteresis_margin=5.0)
    alert_engine = AlertEngine(thresholds=test_thresholds)
    
    # Test sequence from our test
    test_sequence = [
        (20, "Should be Normal"),
        (45, "Should transition to Mild"),
        (48, "Should stay Mild (within hysteresis)"),
        (55, "Should transition to Elevated"),
        (52, "Should stay Elevated (within hysteresis)"),
        (45, "Should transition back to Mild"),
        (90, "Should transition to Critical"),
        (80, "Should stay Critical (within hysteresis)")
    ]
    
    print("Processing sequence:")
    for i, (score, description) in enumerate(test_sequence):
        prediction_result = PredictionResult(
            timestamp=time.time(),
            fire_probability=score / 100.0,
            confidence_score=0.8,
            lead_time_estimate=0.0,
            contributing_factors={},
            model_ensemble_votes={}
        )
        
        # Get current state before processing
        current_level = alert_engine.current_level
        time_at_current = time.time() - alert_engine.level_start_time
        
        print(f"\nStep {i+1}: Score {score}")
        print(f"  Before: {current_level.description}")
        print(f"  Time at current level: {time_at_current:.2f}s")
        
        # Process prediction
        alert = alert_engine.process_prediction(prediction_result)
        
        print(f"  After: {alert.alert_level.description} ({description})")
        print(f"  Risk score: {alert.risk_score:.1f}")


if __name__ == "__main__":
    debug_test_sequence()