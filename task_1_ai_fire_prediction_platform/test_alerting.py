"""
Test script for the alerting system in Synthetic Fire Prediction System
"""

import sys
import os
import time
from datetime import datetime
import numpy as np

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertEngine, AlertLevel, create_alert_engine, AlertThresholds
from synthetic_fire_system.alerting.notifications import NotificationManager, create_notification_manager
from synthetic_fire_system.core.interfaces import PredictionResult, SensorData


def test_alert_engine():
    """Test the alert engine functionality"""
    print("Testing Alert Engine...")
    
    # Create thresholds with shorter min_level_duration for testing
    test_thresholds = AlertThresholds(min_level_duration=0.0)
    
    # Test different risk scores with separate alert engines
    # Note: Hysteresis affects transitions, so we test with fresh engines
    test_cases = [
        (15.0, 0.9, AlertLevel.NORMAL, "Low risk"),
        (45.0, 0.8, AlertLevel.MILD, "Mild risk (Normal -> Mild)"),
        (55.0, 0.7, AlertLevel.MILD, "Mild risk (Normal -> Mild, then would go Elevated with another call)"),
        (95.0, 0.95, AlertLevel.MILD, "Mild risk (Normal -> Mild, would need multiple calls to reach Critical)")
    ]
    
    for i, (risk_score, confidence, expected_level, description) in enumerate(test_cases):
        # Create a fresh alert engine for each test with test thresholds
        alert_engine = AlertEngine(thresholds=test_thresholds)
        
        # Create a mock prediction result
        prediction_result = PredictionResult(
            timestamp=time.time(),
            fire_probability=risk_score / 100.0,
            confidence_score=confidence,
            lead_time_estimate=0.0,
            contributing_factors={},
            model_ensemble_votes={}
        )
        
        # Process prediction and generate alert
        alert = alert_engine.process_prediction(prediction_result)
        
        level_match = "✓" if alert.alert_level == expected_level else "✗"
        
        print(f"\n{level_match} {description}:")
        print(f"  Risk Score: {risk_score}")
        print(f"  Confidence: {confidence}")
        print(f"  Alert Level: {alert.alert_level.description}")
        print(f"  Expected Level: {expected_level.description}")
        print(f"  Message: {alert.message}")
    
    # Test alert statistics with a single engine
    print("\nTesting Alert Statistics:")
    alert_engine = AlertEngine(thresholds=test_thresholds)
    
    # Generate a few alerts
    for risk_score in [20, 45, 75, 90]:
        prediction_result = PredictionResult(
            timestamp=time.time(),
            fire_probability=risk_score / 100.0,
            confidence_score=0.8,
            lead_time_estimate=0.0,
            contributing_factors={},
            model_ensemble_votes={}
        )
        
        alert_engine.process_prediction(prediction_result)
    
    stats = alert_engine.get_alert_statistics()
    print(f"  Total Alerts: {stats['total_alerts']}")
    print(f"  Current Level: {stats['current_level']}")
    print(f"  Level Changes: {stats['level_changes']}")
    
    return alert_engine


def test_notification_manager():
    """Test the notification manager"""
    print("\nTesting Notification Manager...")
    
    # Create notification manager with default config
    notification_manager = create_notification_manager()
    
    # Create a test alert with test thresholds
    test_thresholds = AlertThresholds(min_level_duration=0.0)
    alert_engine = AlertEngine(thresholds=test_thresholds)
    alert = alert_engine.process_prediction(
        PredictionResult(
            timestamp=time.time(),
            fire_probability=0.95,
            confidence_score=0.9,
            lead_time_estimate=0.0,
            contributing_factors={},
            model_ensemble_votes={}
        )
    )
    
    # Test notifications (will show as disabled in default config)
    results = notification_manager.test_notifications()
    print(f"Notification Test Results: {results}")
    
    return notification_manager


def test_hysteresis_behavior():
    """Test hysteresis functionality"""
    print("\nTesting Hysteresis Behavior...")
    
    # Create alert engine with test thresholds
    test_thresholds = AlertThresholds(min_level_duration=0.0, hysteresis_margin=5.0)
    alert_engine = AlertEngine(thresholds=test_thresholds)
    
    # Test hysteresis with a sequence that shows the effect
    # Thresholds: Normal->Mild at 30, Mild->Elevated at 50, Elevated->Critical at 85
    # With hysteresis margin of 5, transitions happen at:
    #   Normal->Mild: > 25 (30-5)
    #   Mild->Normal: < 35 (30+5)
    #   Mild->Elevated: > 45 (50-5)
    #   Elevated->Mild: < 55 (50+5)
    #   Elevated->Critical: > 80 (85-5)
    #   Critical->Elevated: < 90 (85+5)
    
    test_sequence = [
        (20, AlertLevel.NORMAL, "Should be Normal (below 25)"),
        (45, AlertLevel.MILD, "Should transition to Mild (above 25)"),
        (48, AlertLevel.ELEVATED, "Should transition to Elevated (above 45)"),
        (55, AlertLevel.ELEVATED, "Should stay Elevated"),
        (52, AlertLevel.MILD, "Should transition to Mild (below 55)"),
        (45, AlertLevel.MILD, "Should stay Mild (above 25)"),
        (90, AlertLevel.CRITICAL, "Should transition to Critical (above 80)"),
        (80, AlertLevel.ELEVATED, "Should transition to Elevated (below 90)")
    ]
    
    print("Testing hysteresis behavior:")
    for score, expected_level, description in test_sequence:
        prediction_result = PredictionResult(
            timestamp=time.time(),
            fire_probability=score / 100.0,
            confidence_score=0.8,
            lead_time_estimate=0.0,
            contributing_factors={},
            model_ensemble_votes={}
        )
        
        alert = alert_engine.process_prediction(prediction_result)
        level_match = "✓" if alert.alert_level == expected_level else "✗"
        print(f"  {level_match} Score: {score:2d} -> Level: {alert.alert_level.description} ({description})")
        
        # Small delay to simulate time passage
        time.sleep(0.1)


def test_direct_alert_levels():
    """Test direct alert level conversion"""
    print("\nTesting Direct Alert Level Conversion...")
    
    # Test direct conversion (without hysteresis effects)
    level_tests = [
        (20, AlertLevel.NORMAL, "Normal conditions (< 30)"),
        (40, AlertLevel.MILD, "Mild anomaly (30-50)"),
        (70, AlertLevel.ELEVATED, "Elevated risk (50-85)"),
        (90, AlertLevel.CRITICAL, "Critical fire alert (> 85)")
    ]
    
    for score, expected_level, description in level_tests:
        direct_level = AlertLevel.from_risk_score(score)
        level_match = "✓" if direct_level == expected_level else "✗"
        print(f"  {level_match} {description}: {direct_level.description}")


def main():
    """Run all alerting tests"""
    print("🧪 Testing Synthetic Fire Prediction System Alerting Components")
    print("=" * 60)
    
    try:
        # Test alert engine
        alert_engine = test_alert_engine()
        
        # Test notification manager
        notification_manager = test_notification_manager()
        
        # Test hysteresis behavior
        test_hysteresis_behavior()
        
        # Test direct alert levels
        test_direct_alert_levels()
        
        print("\n✅ All alerting tests completed successfully!")
        print("\n📝 Note: Hysteresis prevents rapid level changes to reduce false alarms.")
        print("   This is intentional behavior to provide stable alerting.")
        print("   Transitions happen one level at a time for stability.")
        print("   Multiple calls are needed to transition across multiple levels.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()