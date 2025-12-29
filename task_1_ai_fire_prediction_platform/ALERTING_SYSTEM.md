# Synthetic Fire Prediction System - Alerting System

## Overview

The alerting system is a critical component of the Synthetic Fire Prediction System that converts AI model predictions into actionable alerts. It implements hysteresis-based logic to prevent false alarms while ensuring critical events are properly detected and communicated.

## Key Components

### 1. Alert Engine
The core of the alerting system, responsible for:
- Converting risk scores to alert levels
- Implementing hysteresis to prevent oscillation
- Managing alert history and statistics
- Formatting alert messages with context

### 2. Alert Levels
The system uses a 4-level alert hierarchy:
- **Normal** (🟢): System operating normally
- **Mild Anomaly** (🟡): Minor environmental variations detected
- **Elevated Risk** (🟠): Multiple sensors showing concerning patterns
- **Critical Fire Alert** (🔴): Immediate action required

### 3. Hysteresis Logic
To prevent rapid oscillation between alert levels, the system implements hysteresis with configurable thresholds:
- Normal → Mild: Risk score > 25 (30 - 5 hysteresis margin)
- Mild → Normal: Risk score < 35 (30 + 5 hysteresis margin)
- Mild → Elevated: Risk score > 45 (50 - 5 hysteresis margin)
- Elevated → Mild: Risk score < 55 (50 + 5 hysteresis margin)
- Elevated → Critical: Risk score > 80 (85 - 5 hysteresis margin)
- Critical → Elevated: Risk score < 90 (85 + 5 hysteresis margin)

### 4. Notification System
Supports multiple notification channels:
- Email notifications
- SMS alerts
- Webhook integration

## Integration with Main System

The alerting system is integrated into the main system manager and dashboard:

1. **System Manager**: Processes predictions and generates alerts
2. **Dashboard**: Displays current alert status and history
3. **Notifications**: Sends alerts through configured channels for elevated and critical alerts

## Configuration

The alerting system can be customized through thresholds:
- Hysteresis margins
- Minimum duration constraints
- Confidence-based adjustments
- Custom alert message templates

## Testing

The system includes comprehensive tests to verify:
- Alert level conversion accuracy
- Hysteresis behavior
- Notification functionality
- Statistics tracking

## Best Practices

1. **Stability**: Transitions happen one level at a time to ensure stability
2. **False Alarm Prevention**: Hysteresis prevents rapid oscillation
3. **Context Awareness**: Alert messages include contextual information
4. **Extensibility**: Notification channels can be easily extended
5. **History Tracking**: Maintains alert history for analysis

## Usage Examples

```python
# Create alert engine
from synthetic_fire_system.alerting.engine import create_alert_engine
alert_engine = create_alert_engine()

# Process prediction result
alert = alert_engine.process_prediction(prediction_result)

# Check alert level
if alert.alert_level.level >= 3:  # Elevated or Critical
    # Send notifications
    from synthetic_fire_system.alerting.notifications import create_notification_manager
    notification_manager = create_notification_manager()
    notification_manager.send_alert_notifications(alert)
```

The alerting system provides a robust foundation for fire detection alerts while minimizing false positives through intelligent hysteresis logic.