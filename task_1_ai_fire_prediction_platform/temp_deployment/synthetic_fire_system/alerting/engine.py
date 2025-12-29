"""
Alert engine for Synthetic Fire Prediction System

This module implements the AlertEngine that converts risk scores to alert levels,
manages alert history, prevents oscillation, and formats alert messages with context.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from synthetic_fire_system.core.interfaces import PredictionResult, RiskAssessment
from synthetic_fire_system.hardware.abstraction import SensorData

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert level enumeration with numeric values."""
    NORMAL = (1, "Normal", "🟢")
    MILD = (2, "Mild Anomaly", "🟡")
    ELEVATED = (3, "Elevated Risk", "🟠")
    CRITICAL = (4, "CRITICAL FIRE ALERT", "🔴")
    
    def __init__(self, level: int, description: str, icon: str):
        self.level = level
        self.description = description
        self.icon = icon
    
    @classmethod
    def from_risk_score(cls, risk_score: float) -> 'AlertLevel':
        """Convert risk score to alert level."""
        if risk_score <= 30:
            return cls.NORMAL
        elif risk_score <= 50:
            return cls.MILD
        elif risk_score <= 85:
            return cls.ELEVATED
        else:
            return cls.CRITICAL


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
            AlertLevel: New alert level with hysteresis applied
        """
        # Apply hysteresis margin to prevent rapid oscillation
        if current_level == AlertLevel.NORMAL and risk_score > self.normal_max - self.hysteresis_margin:
            target_level = AlertLevel.MILD
        elif current_level == AlertLevel.MILD:
            if risk_score < self.normal_max + self.hysteresis_margin:
                target_level = AlertLevel.NORMAL
            elif risk_score > self.mild_max - self.hysteresis_margin:
                target_level = AlertLevel.ELEVATED
            else:
                target_level = current_level
        elif current_level == AlertLevel.ELEVATED:
            if risk_score < self.mild_max + self.hysteresis_margin:
                target_level = AlertLevel.MILD
            elif risk_score > self.elevated_max - self.hysteresis_margin:
                target_level = AlertLevel.CRITICAL
            else:
                target_level = current_level
        elif current_level == AlertLevel.CRITICAL:
            if risk_score < self.elevated_max + self.hysteresis_margin:
                target_level = AlertLevel.ELEVATED
            else:
                target_level = current_level
        else:
            # Default case - use direct mapping
            target_level = AlertLevel.from_risk_score(risk_score)
        
        # Apply minimum duration constraint
        if time_at_current < self.min_level_duration and target_level != current_level:
            logger.debug(f"Preventing level change due to minimum duration constraint: "
                        f"{time_at_current:.1f}s < {self.min_level_duration:.1f}s")
            return current_level
        
        return target_level


@dataclass
class AlertData:
    """Comprehensive alert information."""
    alert_level: AlertLevel
    risk_score: float
    confidence: float
    message: str
    timestamp: datetime
    sensor_data: Optional[SensorData] = None
    prediction_result: Optional[PredictionResult] = None
    risk_assessment: Optional[RiskAssessment] = None
    context_info: Dict[str, Any] = field(default_factory=dict)
    alert_id: str = ""
    
    def __post_init__(self):
        """Generate alert ID if not provided."""
        if not self.alert_id:
            self.alert_id = f"alert_{int(self.timestamp.timestamp() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'alert_level': {
                'level': self.alert_level.level,
                'description': self.alert_level.description,
                'icon': self.alert_level.icon
            },
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'context_info': self.context_info
        }


class AlertHistory:
    """Manages alert history with size limits and statistics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize alert history.
        
        Args:
            max_history (int): Maximum number of alerts to keep
        """
        self.max_history = max_history
        self.alerts: List[AlertData] = []
        self.level_changes: List[Dict[str, Any]] = []
    
    def add_alert(self, alert: AlertData, previous_level: Optional[AlertLevel] = None):
        """
        Add alert to history.
        
        Args:
            alert (AlertData): Alert to add
            previous_level (AlertLevel): Previous alert level (if changed)
        """
        self.alerts.append(alert)
        
        # Track level changes
        if previous_level and previous_level != alert.alert_level:
            self.level_changes.append({
                'timestamp': alert.timestamp,
                'from_level': previous_level.description,
                'to_level': alert.alert_level.description,
                'risk_score': alert.risk_score
            })
        
        # Maintain size limit
        if len(self.alerts) > self.max_history:
            self.alerts = self.alerts[-self.max_history:]
        
        if len(self.level_changes) > self.max_history:
            self.level_changes = self.level_changes[-self.max_history:]
    
    def get_level_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get level statistics for specified time period.
        
        Args:
            hours (int): Number of hours to analyze
            
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alerts 
            if alert.timestamp >= cutoff_time
        ]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'level_distribution': {},
                'most_common_level': 'None'
            }
        
        # Calculate level distribution
        level_counts = {}
        for alert in recent_alerts:
            level_name = alert.alert_level.description
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
        
        # Find most common level
        most_common_level = max(level_counts, key=level_counts.get)
        
        return {
            'total_alerts': len(recent_alerts),
            'level_distribution': level_counts,
            'most_common_level': most_common_level
        }


class AlertMessageFormatter:
    """Formats alert messages with context and severity information."""
    
    def __init__(self):
        """Initialize message formatter."""
        self.message_templates = {
            AlertLevel.NORMAL: {
                'default': "System operating normally. All sensors within expected ranges.",
                'stable': "Stable environment. No significant anomalies detected.",
                'low_risk': "Low risk conditions. Environmental parameters within normal range."
            },
            AlertLevel.MILD: {
                'default': "Mild anomaly detected. Monitoring for pattern development.",
                'temperature': "Slight temperature elevation observed. Continuing monitoring.",
                'gas': "Minor gas concentration changes detected. Monitoring trend.",
                'environmental': "Environmental parameter variations observed. No immediate concern."
            },
            AlertLevel.ELEVATED: {
                'default': "Elevated risk detected. Multiple sensors showing concerning patterns.",
                'temperature': "Elevated temperature levels detected. Verify sensor readings.",
                'gas': "Increased gas concentrations detected. Check ventilation systems.",
                'combined': "Multiple sensor anomalies detected. Increased monitoring recommended."
            },
            AlertLevel.CRITICAL: {
                'default': "🚨 CRITICAL FIRE ALERT! Immediate action required!",
                'confirmed': "🚨 FIRE CONFIRMED! Multiple indicators confirm fire emergency!",
                'temperature': "🚨 CRITICAL TEMPERATURE SPIKE! Fire emergency detected!",
                'multiple': "🚨 FIRE EMERGENCY! All fire indicators active - evacuate immediately!"
            }
        }
    
    def format_alert_message(self, 
                           alert_level: AlertLevel,
                           risk_score: float,
                           prediction_result: Optional[PredictionResult] = None,
                           sensor_data: Optional[SensorData] = None,
                           context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format alert message with context information.
        
        Args:
            alert_level (AlertLevel): Alert level
            risk_score (float): Risk score
            prediction_result (PredictionResult): Prediction details
            sensor_data (SensorData): Current sensor data
            context (Dict): Additional context information
            
        Returns:
            str: Formatted alert message
        """
        context = context or {}
        template_key = 'default'
        
        # Select appropriate template based on context
        if alert_level == AlertLevel.CRITICAL:
            if context.get('previous_level_higher', False):
                template_key = 'confirmed'
            elif sensor_data and sensor_data.thermal_frame is not None:
                max_temp = np.max(sensor_data.thermal_frame)
                if max_temp > 70:
                    template_key = 'temperature'
                else:
                    template_key = 'multiple'
        elif alert_level == AlertLevel.ELEVATED:
            if sensor_data:
                if sensor_data.thermal_frame is not None:
                    max_temp = np.max(sensor_data.thermal_frame)
                    if max_temp > 40:
                        template_key = 'temperature'
                elif sensor_data.gas_readings:
                    template_key = 'gas'
                else:
                    template_key = 'combined'
        elif alert_level == AlertLevel.MILD:
            if sensor_data:
                if sensor_data.thermal_frame is not None:
                    template_key = 'temperature'
                elif sensor_data.gas_readings:
                    template_key = 'gas'
                elif sensor_data.environmental_data:
                    template_key = 'environmental'
        
        # Get template message
        templates = self.message_templates.get(alert_level, {})
        message = templates.get(template_key, templates.get('default', 'System alert'))
        
        # Add risk score information
        if risk_score is not None:
            message += f" (Risk Score: {risk_score:.1f})"
        
        return message


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
                         prediction_result: PredictionResult,
                         sensor_data: Optional[SensorData] = None,
                         risk_assessment: Optional[RiskAssessment] = None) -> AlertData:
        """
        Process prediction result and generate appropriate alert.
        
        Args:
            prediction_result (PredictionResult): AI model prediction
            sensor_data (SensorData): Current sensor data
            risk_assessment (RiskAssessment): Risk assessment (if available)
            
        Returns:
            AlertData: Generated alert
        """
        # Extract risk score and confidence
        risk_score = getattr(prediction_result, 'fire_probability', 50.0) * 100
        confidence = getattr(prediction_result, 'confidence_score', 0.5)
        
        # Apply confidence adjustments
        adjusted_risk_score = self._apply_confidence_adjustments(risk_score, confidence)
        
        # Get current time and duration at current level
        current_time = time.time()
        time_at_current = current_time - self.level_start_time
        previous_level = self.current_level
        
        # Determine new alert level with hysteresis
        new_level = self.thresholds.get_level_with_hysteresis(
            adjusted_risk_score, self.current_level, time_at_current
        )
        
        # Check if level changed
        level_changed = new_level != self.current_level
        
        # Update state if level changed
        if level_changed:
            self.current_level = new_level
            self.level_start_time = current_time
            self.alert_stats['level_changes'] += 1
            logger.info(f"Alert level changed: {previous_level.description} -> {new_level.description}")
        
        # Generate alert message
        context = {
            'previous_level_higher': previous_level.level > new_level.level,
            'level_changed': level_changed,
            'time_at_previous': time_at_current
        }
        
        message = self.formatter.format_alert_message(
            new_level, adjusted_risk_score, prediction_result, sensor_data, context
        )
        
        # Create alert data
        alert = AlertData(
            alert_level=new_level,
            risk_score=adjusted_risk_score,
            confidence=confidence,
            message=message,
            timestamp=datetime.now(),
            sensor_data=sensor_data,
            prediction_result=prediction_result,
            risk_assessment=risk_assessment,
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
            adjusted_score = risk_score * confidence / self.thresholds.min_confidence_for_critical
            logger.debug(f"Adjusted critical risk score due to low confidence: "
                        f"{risk_score:.1f} -> {adjusted_score:.1f}")
            return adjusted_score
        elif confidence < self.thresholds.min_confidence_for_elevated and risk_score > 50:
            adjusted_score = risk_score * confidence / self.thresholds.min_confidence_for_elevated
            logger.debug(f"Adjusted elevated risk score due to low confidence: "
                        f"{risk_score:.1f} -> {adjusted_score:.1f}")
            return adjusted_score
        
        return risk_score
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts for specified time period.
        
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