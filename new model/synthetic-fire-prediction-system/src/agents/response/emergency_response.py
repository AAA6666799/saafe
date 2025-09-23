"""
Emergency Response Agent for the multi-agent fire prediction system.

This agent handles emergency response coordination, alert generation, and 
action recommendations based on fire detection and risk assessments.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid

from ..base import ResponseAgent, Message


class ResponseLevel(Enum):
    """Enumeration of response levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertType(Enum):
    """Enumeration of alert types."""
    INFO = "info"
    WARNING = "warning"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


class EmergencyResponseAgent(ResponseAgent):
    """
    Emergency Response Agent for coordinating fire response actions.
    
    This agent determines appropriate response levels, generates alerts,
    and provides action recommendations based on risk assessments.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the Emergency Response Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
        """
        super().__init__(agent_id, config)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Response configuration
        self.response_thresholds = config.get('response_thresholds', {
            ResponseLevel.LOW: 0.3,
            ResponseLevel.MEDIUM: 0.5,
            ResponseLevel.HIGH: 0.7,
            ResponseLevel.CRITICAL: 0.9
        })
        
        self.alert_channels = config.get('alert_channels', ['system', 'email', 'sms'])
        self.emergency_contacts = config.get('emergency_contacts', [])
        self.response_protocols = config.get('response_protocols', {})
        
        # State tracking
        self.active_alerts = {}
        self.response_history = []
        self.alert_count_by_level = {level: 0 for level in ResponseLevel}
        self.suppression_windows = {}  # To prevent alert spam
        
        # Performance metrics
        self.total_responses = 0
        self.false_alarm_rate = 0.0
        self.response_time_avg = 0.0
        
        self.logger.info(f"Initialized Emergency Response Agent: {agent_id}")
    
    def validate_config(self) -> None:
        """Validate the configuration parameters."""
        if 'response_thresholds' not in self.config:
            raise ValueError("Missing response_thresholds in configuration")
        
        thresholds = self.config['response_thresholds']
        for level in ResponseLevel:
            if level not in thresholds:
                raise ValueError(f"Missing threshold for response level: {level}")
        
        # Validate threshold ordering
        threshold_values = [thresholds[level] for level in ResponseLevel]
        if threshold_values != sorted(threshold_values):
            raise ValueError("Response thresholds must be in ascending order")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process risk assessment data and coordinate emergency response.
        
        Args:
            data: Input data containing risk assessment and analysis results
            
        Returns:
            Dictionary containing response coordination results
        """
        try:
            self.total_responses += 1
            response_timestamp = datetime.now()
            
            # Extract risk assessment data
            risk_assessment = data.get('risk_assessment', {})
            analysis_results = data.get('analysis_results', {})
            # FLIR + SCD41 features are now extracted within the methods from risk_assessment
            
            # Determine response level
            response_level_int = self.determine_response_level(risk_assessment)
            response_level = ResponseLevel(response_level_int)
            
            # Generate alerts based on response level
            alerts = self.generate_alerts(risk_assessment, response_level_int)
            
            # Generate action recommendations
            recommendations = self.generate_recommendations(risk_assessment, response_level_int)
            
            # Execute immediate response actions
            response_actions = self._execute_response_actions(response_level, risk_assessment)
            
            # Compile response coordination results
            response_result = {
                'timestamp': response_timestamp.isoformat(),
                'agent_id': self.agent_id,
                'response_id': f"response_{self.total_responses}",
                'response_level': response_level.name,
                'response_level_value': response_level.value,
                'alerts_generated': len(alerts),
                'alerts': alerts,
                'recommendations': recommendations,
                'response_actions': response_actions,
                'risk_summary': {
                    'risk_score': risk_assessment.get('risk_score', 0.0),
                    'confidence': risk_assessment.get('confidence', 0.0),
                    'fire_detected': risk_assessment.get('fire_detected', False)
                },
                'metadata': {
                    'processing_time_ms': (datetime.now() - response_timestamp).total_seconds() * 1000,
                    'active_alert_count': len(self.active_alerts),
                    'response_protocol': self.response_protocols.get(response_level.name, 'standard')
                }
            }
            
            # Update state and tracking
            self._update_state(response_result)
            self._track_response_metrics(response_result)
            
            self.logger.info(f"Response coordination complete: level={response_level.name}, alerts={len(alerts)}")
            
            return response_result
            
        except Exception as e:
            self.logger.error(f"Error in response coordination: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'agent_id': self.agent_id,
                'error': str(e),
                'response_level': ResponseLevel.NONE.name,
                'alerts': [],
                'recommendations': []
            }
    
    def determine_response_level(self, risk_assessment: Dict[str, Any]) -> int:
        """
        Determine the appropriate response level based on risk assessment.
        
        Args:
            risk_assessment: Risk assessment data
            
        Returns:
            ResponseLevel enum value
        """
        risk_score = risk_assessment.get('risk_score', 0.0)
        confidence = risk_assessment.get('confidence', 0.0)
        fire_detected = risk_assessment.get('fire_detected', False)
        
        # Extract FLIR + SCD41 features from risk assessment
        thermal_features = risk_assessment.get('thermal_features', {})
        gas_features = risk_assessment.get('gas_features', {})
        
        # Adjust risk score based on confidence
        adjusted_risk = risk_score * confidence
        
        # Apply FLIR + SCD41 specific risk adjustments
        if thermal_features:
            # High temperature risk factor
            t_max = thermal_features.get('t_max', 0.0)
            if t_max > 80:  # Critical temperature threshold
                adjusted_risk = min(1.0, adjusted_risk * 1.5)
            elif t_max > 60:  # Elevated temperature threshold
                adjusted_risk = min(1.0, adjusted_risk * 1.2)
            
            # Hot area percentage risk factor
            hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
            if hot_area_pct > 20:  # Large hot area
                adjusted_risk = min(1.0, adjusted_risk * 1.3)
        
        if gas_features:
            # CO₂ concentration risk factor
            gas_val = gas_features.get('gas_val', 400.0)
            if gas_val > 3000:  # Critical CO₂ level
                adjusted_risk = min(1.0, adjusted_risk * 1.4)
            elif gas_val > 1500:  # Elevated CO₂ level
                adjusted_risk = min(1.0, adjusted_risk * 1.1)
            
            # CO₂ change rate risk factor
            gas_delta = abs(gas_features.get('gas_delta', 0.0))
            if gas_delta > 200:  # Rapid CO₂ increase
                adjusted_risk = min(1.0, adjusted_risk * 1.2)
        
        # Determine base response level from thresholds
        response_level = ResponseLevel.NONE
        for level in reversed(ResponseLevel):
            if adjusted_risk >= self.response_thresholds.get(level, 1.0):
                response_level = level
                break
        
        # Apply additional escalation factors
        response_level = self._apply_escalation_factors(response_level, risk_assessment)
        
        # Check for suppression (to prevent alert spam)
        response_level = self._apply_suppression_logic(response_level, risk_assessment)
        
        return response_level.value
    
    def generate_alerts(self, risk_assessment: Dict[str, Any], response_level: int) -> List[Dict[str, Any]]:
        """
        Generate alerts based on risk assessment and response level.
        
        Args:
            risk_assessment: Risk assessment data
            response_level: Determined response level (0-4)
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if response_level == ResponseLevel.NONE.value:
            return alerts
        
        # Extract FLIR + SCD41 features from risk assessment
        thermal_features = risk_assessment.get('thermal_features', {})
        gas_features = risk_assessment.get('gas_features', {})
        
        # Determine alert type based on response level
        alert_type = self._determine_alert_type(response_level)
        
        # Generate primary alert
        primary_alert = self._create_alert(
            alert_type=alert_type,
            message=self._generate_alert_message(risk_assessment, response_level, thermal_features, gas_features),
            risk_assessment=risk_assessment,
            response_level=response_level
        )
        alerts.append(primary_alert)
        
        # Generate specialized alerts based on risk factors
        if risk_assessment.get('fire_detected', False):
            fire_alert = self._create_alert(
                alert_type=AlertType.EMERGENCY if response_level >= 3 else AlertType.WARNING,
                message=f"Fire detection confirmed with {risk_assessment.get('confidence', 0.0):.1%} confidence",
                risk_assessment=risk_assessment,
                response_level=response_level,
                alert_category='fire_detection'
            )
            alerts.append(fire_alert)
        
        # Generate evacuation alerts for high-level responses
        if response_level >= ResponseLevel.HIGH.value:
            evacuation_alert = self._create_alert(
                alert_type=AlertType.CRITICAL,
                message="Immediate evacuation recommended - fire hazard detected",
                risk_assessment=risk_assessment,
                response_level=response_level,
                alert_category='evacuation'
            )
            alerts.append(evacuation_alert)
        
        # Generate system alerts for monitoring
        if response_level >= ResponseLevel.MEDIUM.value:
            system_alert = self._create_alert(
                alert_type=AlertType.INFO,
                message=f"Fire prediction system activated - response level {response_level}",
                risk_assessment=risk_assessment,
                response_level=response_level,
                alert_category='system_status'
            )
            alerts.append(system_alert)
        
        # Store active alerts
        for alert in alerts:
            self.active_alerts[alert['alert_id']] = alert
        
        # Update alert counters
        response_enum = ResponseLevel(response_level)
        self.alert_count_by_level[response_enum] += len(alerts)
        
        return alerts
    
    def generate_recommendations(self, risk_assessment: Dict[str, Any], response_level: int) -> List[str]:
        """
        Generate action recommendations based on risk assessment and response level.
        
        Args:
            risk_assessment: Risk assessment data
            response_level: Determined response level (0-4)
            
        Returns:
            List of recommended actions as strings
        """
        recommendations = []
        
        if response_level == ResponseLevel.NONE.value:
            recommendations.append("Continue normal monitoring")
            return recommendations
        
        # Base recommendations by response level
        level_recommendations = {
            ResponseLevel.LOW.value: [
                "Increase monitoring frequency",
                "Check sensor calibration",
                "Review recent maintenance logs",
                "Notify maintenance team of elevated readings"
            ],
            ResponseLevel.MEDIUM.value: [
                "Activate enhanced monitoring protocols",
                "Deploy additional sensors if available",
                "Alert security personnel",
                "Prepare emergency response team",
                "Review evacuation procedures"
            ],
            ResponseLevel.HIGH.value: [
                "Activate emergency response protocols",
                "Contact fire department immediately",
                "Begin controlled evacuation of immediate area",
                "Activate fire suppression systems",
                "Establish incident command center",
                "Notify emergency contacts"
            ],
            ResponseLevel.CRITICAL.value: [
                "IMMEDIATE EVACUATION OF ALL PERSONNEL",
                "Contact fire department - EMERGENCY RESPONSE",
                "Activate all fire suppression systems",
                "Shut down electrical systems in affected areas",
                "Establish emergency perimeter",
                "Account for all personnel",
                "Notify executive leadership immediately"
            ]
        }
        
        recommendations.extend(level_recommendations.get(response_level, []))
        
        # Add specific recommendations based on FLIR + SCD41 features
        # Extract FLIR + SCD41 features from risk assessment
        thermal_features = risk_assessment.get('thermal_features', {})
        gas_features = risk_assessment.get('gas_features', {})
        
        if thermal_features:
            t_max = thermal_features.get('t_max', 0.0)
            hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
            flow_mag = thermal_features.get('flow_mag_mean', 0.0)
            
            if t_max > 70:
                recommendations.extend([
                    "Critical thermal hazard detected - immediate attention required",
                    "Deploy thermal protection equipment",
                    "Establish thermal safety perimeter"
                ])
            elif t_max > 50:
                recommendations.append("Monitor elevated temperature zones")
            
            if hot_area_pct > 15:
                recommendations.extend([
                    "Large thermal anomaly detected",
                    "Assess thermal spread patterns",
                    "Prepare thermal containment measures"
                ])
            
            if flow_mag > 1.0:
                recommendations.append("Thermal movement detected - monitor for spread")
        
        if gas_features:
            gas_val = gas_features.get('gas_val', 400.0)
            gas_delta = gas_features.get('gas_delta', 0.0)
            gas_vel = gas_features.get('gas_vel', 0.0)
            
            if gas_val > 2000:
                recommendations.extend([
                    "Elevated CO₂ levels detected",
                    "Ensure adequate ventilation",
                    "Monitor respiratory safety"
                ])
            
            if abs(gas_delta) > 150:
                recommendations.append("Rapid CO₂ concentration change - potential fire indicator")
            
            if abs(gas_vel) > 20:
                recommendations.append("Accelerating CO₂ release - fire progression likely")
        
        # Add specific recommendations based on FLIR + SCD41 features
        if thermal_features:
            t_max = thermal_features.get('t_max', 0.0)
            hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
            flow_mag = thermal_features.get('flow_mag_mean', 0.0)
            
            if t_max > 70:
                recommendations.extend([
                    "Critical thermal hazard detected - immediate attention required",
                    "Deploy thermal protection equipment",
                    "Establish thermal safety perimeter"
                ])
            elif t_max > 50:
                recommendations.append("Monitor elevated temperature zones")
            
            if hot_area_pct > 15:
                recommendations.extend([
                    "Large thermal anomaly detected",
                    "Assess thermal spread patterns",
                    "Prepare thermal containment measures"
                ])
            
            if flow_mag > 1.0:
                recommendations.append("Thermal movement detected - monitor for spread")
        
        if gas_features:
            gas_val = gas_features.get('gas_val', 400.0)
            gas_delta = gas_features.get('gas_delta', 0.0)
            gas_vel = gas_features.get('gas_vel', 0.0)
            
            if gas_val > 2000:
                recommendations.extend([
                    "Elevated CO₂ levels detected",
                    "Ensure adequate ventilation",
                    "Monitor respiratory safety"
                ])
            
            if abs(gas_delta) > 150:
                recommendations.append("Rapid CO₂ concentration change - potential fire indicator")
            
            if abs(gas_vel) > 20:
                recommendations.append("Accelerating CO₂ release - fire progression likely")
        
        # Add specific recommendations based on risk factors
        if risk_assessment.get('thermal_risk', 0.0) > 0.7:
            recommendations.extend([
                "Focus on thermal hazard areas",
                "Deploy thermal imaging equipment",
                "Monitor temperature trends closely"
            ])
        
        if risk_assessment.get('gas_risk', 0.0) > 0.7:
            recommendations.extend([
                "Monitor air quality continuously",
                "Consider respiratory protection for response team",
                "Ventilate affected areas if safe to do so"
            ])
        
        if risk_assessment.get('spread_risk', 0.0) > 0.7:
            recommendations.extend([
                "Establish firebreaks if possible",
                "Monitor adjacent areas for spread",
                "Prepare resources for extended response"
            ])
        
        # Add time-sensitive recommendations
        if response_level >= ResponseLevel.MEDIUM.value:
            recommendations.append(f"Response initiated at {datetime.now().strftime('%H:%M:%S')} - time-critical actions required")
        
        return recommendations
    
    def _execute_response_actions(self, response_level: ResponseLevel, risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute immediate automated response actions."""
        actions_executed = []
        
        try:
            # Notification actions
            if response_level.value >= ResponseLevel.MEDIUM.value:
                notification_action = {
                    'action_type': 'notification',
                    'action': 'emergency_contacts_notified',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'simulated',  # In real system, would actually send notifications
                    'details': f'Notified {len(self.emergency_contacts)} emergency contacts'
                }
                actions_executed.append(notification_action)
            
            # System integration actions
            if response_level.value >= ResponseLevel.HIGH.value:
                system_action = {
                    'action_type': 'system_integration',
                    'action': 'fire_suppression_activated',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'simulated',  # In real system, would interface with building systems
                    'details': 'Fire suppression systems prepared for activation'
                }
                actions_executed.append(system_action)
            
            # Emergency services contact
            if response_level.value >= ResponseLevel.CRITICAL.value:
                emergency_action = {
                    'action_type': 'emergency_services',
                    'action': 'fire_department_contacted',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'simulated',  # In real system, would auto-dial emergency services
                    'details': 'Emergency services notified of critical fire detection'
                }
                actions_executed.append(emergency_action)
            
        except Exception as e:
            self.logger.error(f"Error executing response actions: {str(e)}")
            actions_executed.append({
                'action_type': 'error',
                'action': 'action_execution_failed',
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
        
        return actions_executed
    
    def _apply_escalation_factors(self, base_level: ResponseLevel, risk_assessment: Dict[str, Any]) -> ResponseLevel:
        """Apply escalation factors to adjust response level."""
        # Factor in historical patterns
        if len(self.response_history) > 0:
            recent_responses = [r for r in self.response_history[-10:] if r.get('fire_detected', False)]
            if len(recent_responses) > 2:  # Multiple recent fire detections
                base_level = ResponseLevel(min(ResponseLevel.CRITICAL.value, base_level.value + 1))
        
        # Factor in time of day (higher escalation during off-hours)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Off hours
            if base_level.value >= ResponseLevel.MEDIUM.value:
                base_level = ResponseLevel(min(ResponseLevel.CRITICAL.value, base_level.value + 1))
        
        # Factor in environmental conditions
        if risk_assessment.get('environmental_risk', 0.0) > 0.8:
            base_level = ResponseLevel(min(ResponseLevel.CRITICAL.value, base_level.value + 1))
        
        return base_level
    
    def _apply_suppression_logic(self, response_level: ResponseLevel, risk_assessment: Dict[str, Any]) -> ResponseLevel:
        """Apply suppression logic to prevent alert spam."""
        current_time = datetime.now()
        
        # Check if we're in a suppression window for this level
        if response_level in self.suppression_windows:
            suppress_until = self.suppression_windows[response_level]
            if current_time < suppress_until:
                # Suppress to lower level unless it's critical
                if response_level.value < ResponseLevel.CRITICAL.value:
                    return ResponseLevel(max(ResponseLevel.NONE.value, response_level.value - 1))
        
        # Set new suppression window for frequent alerts
        if response_level.value >= ResponseLevel.MEDIUM.value:
            # Suppress similar-level alerts for next 5 minutes
            self.suppression_windows[response_level] = current_time + timedelta(minutes=5)
        
        return response_level
    
    def _determine_alert_type(self, response_level: int) -> AlertType:
        """Determine alert type based on response level."""
        if response_level >= ResponseLevel.CRITICAL.value:
            return AlertType.CRITICAL
        elif response_level >= ResponseLevel.HIGH.value:
            return AlertType.EMERGENCY
        elif response_level >= ResponseLevel.MEDIUM.value:
            return AlertType.WARNING
        else:
            return AlertType.INFO
    
    def _create_alert(self, alert_type: AlertType, message: str, risk_assessment: Dict[str, Any], 
                     response_level: int, alert_category: str = 'general') -> Dict[str, Any]:
        """Create a standardized alert dictionary."""
        return {
            'alert_id': str(uuid.uuid4()),
            'alert_type': alert_type.value,
            'alert_category': alert_category,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'response_level': response_level,
            'risk_score': risk_assessment.get('risk_score', 0.0),
            'confidence': risk_assessment.get('confidence', 0.0),
            'fire_detected': risk_assessment.get('fire_detected', False),
            'location': risk_assessment.get('location', 'Unknown'),
            'channels': self.alert_channels,
            'priority': response_level,
            'agent_id': self.agent_id
        }
    
    def _generate_alert_message(self, risk_assessment: Dict[str, Any], response_level: int,
                               thermal_features: Optional[Dict[str, Any]] = None,
                               gas_features: Optional[Dict[str, Any]] = None) -> str:
        """Generate alert message text based on risk assessment."""
        risk_score = risk_assessment.get('risk_score', 0.0)
        confidence = risk_assessment.get('confidence', 0.0)
        location = risk_assessment.get('location', 'Unknown location')
        
        # Extract FLIR + SCD41 features from risk assessment if not provided
        if thermal_features is None:
            thermal_features = risk_assessment.get('thermal_features', {})
        if gas_features is None:
            gas_features = risk_assessment.get('gas_features', {})
        
        # Create base message
        base_message = f"at {location} (Risk: {risk_score:.1%}, Confidence: {confidence:.1%})"
        
        # Add FLIR + SCD41 specific details if available
        details = []
        if thermal_features:
            t_max = thermal_features.get('t_max', 0.0)
            hot_area_pct = thermal_features.get('t_hot_area_pct', 0.0)
            details.append(f"Max Temp: {t_max:.1f}°C")
            if hot_area_pct > 5:
                details.append(f"Hot Area: {hot_area_pct:.1f}%")
        
        if gas_features:
            gas_val = gas_features.get('gas_val', 400.0)
            if gas_val > 600:
                details.append(f"CO₂: {gas_val:.0f} ppm")
        
        # Add details to message if available
        if details:
            base_message = f"at {location} (Risk: {risk_score:.1%}, Confidence: {confidence:.1%}) - {', '.join(details)}"
        
        if response_level >= ResponseLevel.CRITICAL.value:
            return f"CRITICAL FIRE ALERT: High fire risk detected {base_message}"
        elif response_level >= ResponseLevel.HIGH.value:
            return f"FIRE EMERGENCY: Fire detected {base_message}"
        elif response_level >= ResponseLevel.MEDIUM.value:
            return f"FIRE WARNING: Elevated fire risk {base_message}"
        else:
            return f"Fire monitoring alert: Increased fire indicators {base_message}"
    
    def _update_state(self, response_result: Dict[str, Any]) -> None:
        """Update agent state with response results."""
        self.response_history.append(response_result)
        
        # Keep only recent history to manage memory
        if len(self.response_history) > 1000:
            self.response_history = self.response_history[-500:]
        
        # Clean up old active alerts (older than 24 hours)
        current_time = datetime.now()
        expired_alerts = []
        for alert_id, alert in self.active_alerts.items():
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if current_time - alert_time > timedelta(hours=24):
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]
    
    def _track_response_metrics(self, response_result: Dict[str, Any]) -> None:
        """Track response performance metrics."""
        processing_time = response_result.get('metadata', {}).get('processing_time_ms', 0)
        
        # Update average response time (exponential moving average)
        if self.response_time_avg == 0:
            self.response_time_avg = processing_time
        else:
            self.response_time_avg = 0.9 * self.response_time_avg + 0.1 * processing_time
    
    def default_message_handler(self, message: Message) -> Optional[Message]:
        """Handle unknown message types."""
        self.logger.warning(f"Received unknown message type: {message.message_type}")
        return None
    
    def create_message(self, receiver_id: str, message_type: str, content: Dict[str, Any], priority: int = 0) -> Message:
        """Create a new message to send to another agent."""
        return Message(self.agent_id, receiver_id, message_type, content, priority)
    
    def save_state(self, filepath: str) -> None:
        """Save agent state to file."""
        state_data = {
            'agent_id': self.agent_id,
            'config': self.config,
            'total_responses': self.total_responses,
            'alert_count_by_level': {level.name: count for level, count in self.alert_count_by_level.items()},
            'false_alarm_rate': self.false_alarm_rate,
            'response_time_avg': self.response_time_avg,
            'active_alerts': self.active_alerts,
            'recent_responses': self.response_history[-100:]  # Save only recent responses
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
    
    def load_state(self, filepath: str) -> None:
        """Load agent state from file."""
        import json
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.total_responses = state_data.get('total_responses', 0)
        self.false_alarm_rate = state_data.get('false_alarm_rate', 0.0)
        self.response_time_avg = state_data.get('response_time_avg', 0.0)
        self.active_alerts = state_data.get('active_alerts', {})
        self.response_history = state_data.get('recent_responses', [])