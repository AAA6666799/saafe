"""
Debug script for the alerting system
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.alerting.engine import AlertLevel


def test_alert_levels():
    """Test alert level conversion"""
    print("Testing AlertLevel.from_risk_score:")
    
    test_scores = [15, 45, 75, 95]
    
    for score in test_scores:
        level = AlertLevel.from_risk_score(score)
        print(f"  Score {score} -> {level.name} ({level.description})")


if __name__ == "__main__":
    test_alert_levels()