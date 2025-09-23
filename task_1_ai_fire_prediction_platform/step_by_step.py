"""
Step by step alert level transitions
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertEngine, AlertLevel, AlertThresholds
from synthetic_fire_system.core.interfaces import PredictionResult


def test_step_by_step():
    """Test step by step transitions"""
    print("Testing step by step transitions...")
    
    # Create thresholds with no minimum duration
    test_thresholds = AlertThresholds(min_level_duration=0.0)
    alert_engine = AlertEngine(thresholds=test_thresholds)
    
    # Sequence of scores that should show step by step transitions
    scores = [15, 45, 55, 75, 95]
    
    print("Step by step transitions:")
    for score in scores:
        prediction_result = PredictionResult(
            timestamp=time.time(),
            fire_probability=score / 100.0,
            confidence_score=0.8,
            lead_time_estimate=0.0,
            contributing_factors={},
            model_ensemble_votes={}
        )
        
        # Get state before
        before_level = alert_engine.current_level
        
        # Process prediction
        alert = alert_engine.process_prediction(prediction_result)
        
        # Get state after
        after_level = alert.alert_level
        
        print(f"  Score {score:2d}: {before_level.description} -> {after_level.description}")
    
    print(f"\nFinal level: {alert_engine.current_level.description}")


if __name__ == "__main__":
    test_step_by_step()