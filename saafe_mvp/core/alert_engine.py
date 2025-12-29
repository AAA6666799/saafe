"""
Alert generation and risk scoring system for fire detection.

This module implements the AlertEngine that converts risk scores to alert levels,
manages alert history, prevents oscillation, and formats alert messages with context.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .data_models import SensorReading, AlertData, AlertLevel

logger = logging.getLogger(__name__)




@dataclass
class AlertThresholds:
    """Configurable alert thresholds."""
    normal_max: float = 30.0
    mild_max: float = 50.0
    elevated_max: float = 85.0
    critical_min: float = 85.0
    
    # Oscillation prevention
    hysteresis_margin: float = 5.0  # Margin to prevent rapid level changes
    min_level_duration: float = 10.0  # Minimum seconds to stay at a level
    
    # Confidence thresholds
    min_confidence_for_critical: float = 0.7
    min_confidence_for_elevated: float = 0.5
    
    def get_level_with_hysteresis(self, 
                                risk_score: float, 
                                current_level: AlertLevel,
                                time_at_current: float) -> AlertLevel:
        """
        Get alert level with hysteresis to prevent oscillation.
        
        Args:
            risk_score (float): Current risk score
            current_level (AlertLevel): Current alert level
            time_at_current (float): Time spent at current level (seconds)
            
        Returns:
            AlertLevel: New alert level considering hysteresis
        """
        new_level = AlertLevel.from_risk_score(risk_score)
        
        # If we haven't been at current level long enough, be conservative
        if time_at_current < self.min_level_duration:
            # Only allow escalation, not de-escalation
            if new_level.level > current_level.level:
                return new_level
            else:
                return current_level
        
        # Apply hysteresis margins
        if current_level == AlertLevel.CRITICAL:
            # Require score to drop significantly below threshold to de-escalate
            if risk_score > (self.critical_min - self.hysteresis_margin):
                return AlertLevel.CRITICAL
        elif current_level == AlertLevel.ELEVATED:
            # Hysteresis for elevated level
            if (self.elevated_max - self.hysteresis_margin) < risk_score <= (self.elevated_max + self.hysteresis_margin):
                return AlertLevel.ELEVATED
        elif current_level == AlertLevel.MILD:
            # Hysteresis for mild level
            if (self.mild_max - self.hysteresis_margin) < risk_score <= (self.mild_max + self.hysteresis_margin):
                return AlertLevel.MILD
        
        return new_level


class AlertHistory:
    """Manages alert history and provides analytics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize alert history.
        
        Args:
            max_history (int): Maximum number of alerts to keep in history
        """
        self.max_history = max_history
        self.alerts: List[AlertData] = []
        self.level_transitions: List[Tuple[datetime, AlertLevel, AlertLevel]] = []
        
    def add_alert(self, alert: AlertData, previous_level: Optional[AlertLevel] = None):
        """
        Add alert to history.
        
        Args:
            alert (AlertData): Alert to add
            previous_level (AlertLevel): Previous alert level for transition tracking
        """
        self.alerts.append(alert)
        
        # Track level transitions
        if previous_level and previous_level != alert.alert_level:
            self.level_transitions.append((alert.timestamp, previous_level, alert.alert_level))
        
        # Maintain history size
        if len(self.alerts) > self.max_history:
            self.alerts = self.alerts[-self.max_history:]
        
        if len(self.level_transitions) > self.max_history:
            self.level_transitions = self.level_transitions[-self.max_history:]
    
    def get_recent_alerts(self, minutes: int = 60) -> List[AlertData]:
        """Get alerts from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_alerts_by_level(self, level: AlertLevel, hours: int = 24) -> List[AlertData]:
        """Get alerts of specific level from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts 
            if alert.alert_level == level and alert.timestamp >= cutoff_time
        ]
    
    def get_level_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get statistics about alert levels over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
        
        if not recent_alerts:
            return {'total_alerts': 0}
        
        level_counts = {}
        total_time_at_levels = {}
        
        for level in AlertLevel:
            level_alerts = [a for a in recent_alerts if a.alert_level == level]
            level_counts[level.description] = len(level_alerts)
        
        # Calculate time spent at each level
        if len(recent_alerts) > 1:
            for i in range(len(recent_alerts) - 1):
                current_alert = recent_alerts[i]
                next_alert = recent_alerts[i + 1]
                duration = (next_alert.timestamp - current_alert.timestamp).total_seconds()
                
                level_desc = current_alert.alert_level.description
                if level_desc not in total_time_at_levels:
                    total_time_at_levels[level_desc] = 0
                total_time_at_levels[level_desc] += duration
        
        return {
            'total_alerts': len(recent_alerts),
            'level_counts': level_counts,
            'time_at_levels': total_time_at_levels,
            'transitions': len([t for t in self.level_transitions if t[0] >= cutoff_time])
        }


class AlertMessageFormatter:
    """Formats alert messages with context information."""
    
    def __init__(self):
        """Initialize the message formatter."""
        self.message_templates = {
            AlertLevel.NORMAL: {
                'default': "All systems normal. Environment conditions are stable.",
                'cooking': "Cooking activity detected - No fire risk identified.",
                'recovery': "Alert level returned to normal. Conditions have stabilized."
            },
            AlertLevel.MILD: {
                'default': "Mild anomaly detected. Monitoring elevated sensor readings.",
                'cooking': "Cooking activity detected with elevated readings. No immediate concern.",
                'temperature': "Slight temperature increase detected. Continuing to monitor.",
                'pm25': "Elevated particulate matter detected. Possible cooking or dust."
            },
            AlertLevel.ELEVATED: {
                'default': "Elevated risk detected. Multiple sensors showing concerning readings.",
                'temperature': "Significant temperature increase detected. Investigating potential fire risk.",
                'multiple': "Multiple environmental indicators elevated. Enhanced monitoring active.",
                'pattern': "Unusual sensor pattern detected. Analyzing for fire signatures."
            },
            AlertLevel.CRITICAL: {
                'default': "🚨 FIRE EMERGENCY DETECTED! Immediate action required!",
                'confirmed': "🚨 FIRE CONFIRMED! Multiple indicators confirm fire emergency!",
                'temperature': "🚨 CRITICAL TEMPERATURE SPIKE! Fire emergency detected!",
                'multiple': "🚨 FIRE EMERGENCY! All fire indicators active - evacuate immediately!"
            }
        }
    
    def format_alert_message(self, 
                           alert_level: AlertLevel,
                           risk_score: float,
                           prediction_result: Optional[Any] = None,
                           sensor_reading: Optional[SensorReading] = None,
                           context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format alert message with context information.
        
        Args:
            alert_level (AlertLevel): Alert level
            risk_score (float): Risk score
            prediction_result (PredictionResult): Prediction details
            sensor_reading (SensorReading): Current sensor reading
            context (Dict): Additional context information
            
        Returns:
            str: Formatted alert message
        """
        context = context or {}
        
        # Determine message type based on context
        message_type = self._determine_message_type(
            alert_level, prediction_result, sensor_reading, context
        )
        
        # Get base message
        templates = self.message_templates[alert_level]
        base_message = templates.get(message_type, templates['default'])
        
        # Add context details
        details = self._generate_context_details(
            risk_score, prediction_result, sensor_reading, context
        )
        
        if details:
            return f"{base_message} {details}"
        else:
            return base_message
    
    def _determine_message_type(self, 
                               alert_level: AlertLevel,
                               prediction_result: Optional[Any],
                               sensor_reading: Optional[SensorReading],
                               context: Dict[str, Any]) -> str:
        """Determine the appropriate message type based on context."""
        
        # Check for cooking detection
        if prediction_result and hasattr(prediction_result, 'anti_hallucination'):
            if prediction_result.anti_hallucination and hasattr(prediction_result.anti_hallucination, 'cooking_detected'):
                if prediction_result.anti_hallucination.cooking_detected:
                    return 'cooking'
        
        # Check for recovery (previous level was higher)
        if context.get('previous_level_higher', False):
            return 'recovery'
        
        # Check for specific sensor triggers
        if sensor_reading:
            if sensor_reading.temperature > 50:
                return 'temperature'
            elif sensor_reading.pm25 > 75:
                return 'pm25'
        
        # Check for multiple indicators
        if prediction_result and hasattr(prediction_result, 'feature_importance'):
            if prediction_result.feature_importance:
                high_importance_features = [
                    feature for feature, importance in prediction_result.feature_importance.items()
                    if importance > 0.3
                ]
                if len(high_importance_features) >= 3:
                    return 'multiple'
        
        # Check for confirmed fire (high confidence critical alert)
        if alert_level == AlertLevel.CRITICAL:
            if prediction_result and hasattr(prediction_result, 'confidence'):
                if prediction_result.confidence > 0.9:
                    return 'confirmed'
            if prediction_result and hasattr(prediction_result, 'anti_hallucination'):
                if prediction_result.anti_hallucination and hasattr(prediction_result.anti_hallucination, 'fire_signatures_confirmed'):
                    if prediction_result.anti_hallucination.fire_signatures_confirmed:
                        return 'confirmed'
        
        return 'default'
    
    def _generate_context_details(self, 
                                risk_score: float,
                                prediction_result: Optional[Any],
                                sensor_reading: Optional[SensorReading],
                                context: Dict[str, Any]) -> str:
        """Generate context details for the alert message."""
        details = []
        
        # Add risk score
        details.append(f"Risk Score: {risk_score:.0f}/100")
        
        # Add confidence if available
        if prediction_result and hasattr(prediction_result, 'confidence'):
            details.append(f"Confidence: {prediction_result.confidence:.0%}")
        
        # Add key sensor readings
        if sensor_reading:
            if sensor_reading.temperature > 30:
                details.append(f"Temp: {sensor_reading.temperature:.1f}°C")
            if sensor_reading.pm25 > 25:
                details.append(f"PM2.5: {sensor_reading.pm25:.0f}μg/m³")
            if sensor_reading.co2 > 600:
                details.append(f"CO₂: {sensor_reading.co2:.0f}ppm")
        
        # Add processing time if available
        if prediction_result and hasattr(prediction_result, 'processing_time'):
            if prediction_result.processing_time:
                details.append(f"Response: {prediction_result.processing_time:.0f}ms")
        
        return f"({', '.join(details)})" if details else ""


class AlertEngine:
    """
    Main alert engine that converts risk scores to alert levels with comprehensive
    alert management, history tracking, and oscillation prevention.
    """
    
    def __init__(self, 
                 thresholds: Optional[AlertThresholds] = None,
                 max_history: int = 1000):
        """
        Initialize the alert engine.
        
        Args:
            thresholds (AlertThresholds): Alert threshold configuration
            max_history (int): Maximum alerts to keep in history
        """
        self.thresholds = thresholds or AlertThresholds()
        self.history = AlertHistory(max_history)
        self.formatter = AlertMessageFormatter()
        
        # Current state tracking
        self.current_level = AlertLevel.NORMAL
        self.level_start_time = time.time()
        self.last_alert: Optional[AlertData] = None
        
        # Performance tracking
        self.alert_stats = {
            'total_alerts': 0,
            'level_changes': 0,
            'false_alarm_preventions': 0
        }
        
        logger.info("AlertEngine initialized")
    
    def process_prediction(self, 
                         prediction_result: Any,
                         sensor_reading: Optional[SensorReading] = None) -> AlertData:
        """
        Process prediction result and generate appropriate alert.
        
        Args:
            prediction_result (Any): Model prediction result with risk_score and confidence
            sensor_reading (SensorReading): Current sensor reading
            
        Returns:
            AlertData: Generated alert with context
        """
        # Extract risk score and confidence with safe attribute access
        risk_score = getattr(prediction_result, 'risk_score', 0.0)
        confidence = getattr(prediction_result, 'confidence', 0.0)
        
        # Apply confidence-based adjustments
        adjusted_risk_score = self._apply_confidence_adjustments(risk_score, confidence)
        
        # Determine alert level with hysteresis
        time_at_current = time.time() - self.level_start_time
        new_level = self.thresholds.get_level_with_hysteresis(
            adjusted_risk_score, self.current_level, time_at_current
        )
        
        # Check for level change
        previous_level = self.current_level
        level_changed = new_level != self.current_level
        
        if level_changed:
            self.current_level = new_level
            self.level_start_time = time.time()
            self.alert_stats['level_changes'] += 1
            logger.info(f"Alert level changed: {previous_level.description} -> {new_level.description}")
        
        # Generate alert message
        context = {
            'previous_level_higher': previous_level.level > new_level.level,
            'level_changed': level_changed,
            'time_at_previous': time_at_current
        }
        
        message = self.formatter.format_alert_message(
            new_level, adjusted_risk_score, prediction_result, sensor_reading, context
        )
        
        # Create alert data
        alert = AlertData(
            alert_level=new_level,
            risk_score=adjusted_risk_score,
            confidence=confidence,
            message=message,
            timestamp=datetime.now(),
            sensor_readings=sensor_reading,
            prediction_result=prediction_result,
            context_info=context
        )
        
        # Add to history
        self.history.add_alert(alert, previous_level if level_changed else None)
        self.last_alert = alert
        self.alert_stats['total_alerts'] += 1
        
        return alert
    
    def _apply_confidence_adjustments(self, risk_score: float, confidence: float) -> float:
        """Apply confidence-based risk score adjustments."""
        
        # Reduce risk score for low confidence predictions
        if confidence < self.thresholds.min_confidence_for_critical and risk_score > 85:
            # Prevent critical alerts with low confidence
            adjusted_score = min(risk_score, 80.0)
            self.alert_stats['false_alarm_preventions'] += 1
            logger.debug(f"Critical alert prevented due to low confidence: {confidence:.2f}")
            return adjusted_score
        
        elif confidence < self.thresholds.min_confidence_for_elevated and risk_score > 50:
            # Reduce elevated alerts with low confidence
            adjusted_score = risk_score * (0.5 + confidence * 0.5)
            return min(adjusted_score, 75.0)
        
        return risk_score
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current alert status and system information.
        
        Returns:
            Dict[str, Any]: Current status information
        """
        time_at_current = time.time() - self.level_start_time
        
        status = {
            'current_level': {
                'level': self.current_level.level,
                'description': self.current_level.description,
                'icon': self.current_level.icon
            },
            'time_at_current_level': time_at_current,
            'last_alert': self.last_alert.to_dict() if self.last_alert else None,
            'alert_stats': self.alert_stats.copy(),
            'thresholds': {
                'normal_max': self.thresholds.normal_max,
                'mild_max': self.thresholds.mild_max,
                'elevated_max': self.thresholds.elevated_max,
                'critical_min': self.thresholds.critical_min
            }
        }
        
        return status
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alert history for the specified time period.
        
        Args:
            hours (int): Number of hours of history to retrieve
            
        Returns:
            List[Dict]: List of alert dictionaries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.history.alerts 
            if alert.timestamp >= cutoff_time
        ]
        
        return [alert.to_dict() for alert in recent_alerts]
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive alert statistics.
        
        Args:
            hours (int): Time period for statistics
            
        Returns:
            Dict[str, Any]: Alert statistics
        """
        stats = self.history.get_level_statistics(hours)
        stats.update(self.alert_stats)
        
        # Add current status
        stats['current_level'] = self.current_level.description
        stats['time_at_current'] = time.time() - self.level_start_time
        
        return stats
    
    def update_thresholds(self, new_thresholds: AlertThresholds):
        """
        Update alert thresholds.
        
        Args:
            new_thresholds (AlertThresholds): New threshold configuration
        """
        self.thresholds = new_thresholds
        logger.info("Alert thresholds updated")
    
    def reset_current_level(self, level: AlertLevel = AlertLevel.NORMAL):
        """
        Reset current alert level (for testing or manual override).
        
        Args:
            level (AlertLevel): Level to reset to
        """
        self.current_level = level
        self.level_start_time = time.time()
        logger.info(f"Alert level manually reset to: {level.description}")


# Convenience functions
def create_alert_engine(thresholds: Optional[AlertThresholds] = None) -> AlertEngine:
    """
    Create an alert engine with default or custom configuration.
    
    Args:
        thresholds (AlertThresholds): Custom thresholds (uses defaults if None)
        
    Returns:
        AlertEngine: Configured alert engine
    """
    return AlertEngine(thresholds=thresholds)


def create_custom_thresholds(normal_max: float = 30.0,
                           mild_max: float = 50.0,
                           elevated_max: float = 85.0,
                           hysteresis_margin: float = 5.0) -> AlertThresholds:
    """
    Create custom alert thresholds.
    
    Args:
        normal_max (float): Maximum risk score for normal level
        mild_max (float): Maximum risk score for mild level
        elevated_max (float): Maximum risk score for elevated level
        hysteresis_margin (float): Hysteresis margin to prevent oscillation
        
    Returns:
        AlertThresholds: Custom threshold configuration
    """
    return AlertThresholds(
        normal_max=normal_max,
        mild_max=mild_max,
        elevated_max=elevated_max,
        critical_min=elevated_max,
        hysteresis_margin=hysteresis_margin
    )