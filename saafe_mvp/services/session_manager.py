"""
Session management for data collection and export coordination.

This module manages session data collection, coordinates between different
components, and provides a unified interface for export and performance tracking.
"""

import uuid
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

from ..core.data_models import SensorReading
from ..core.alert_engine import AlertData
from .export_service import ExportService, SessionData, ExportConfig, BatchExportManager
from .performance_monitor import PerformanceMonitor

# Handle optional torch dependency
try:
    from ..core.fire_detection_pipeline import PredictionResult
except ImportError:
    # Create mock PredictionResult for testing without torch
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    
    @dataclass
    class PredictionResult:
        risk_score: float
        confidence: float
        predicted_class: str
        feature_importance: Dict[str, float]
        processing_time: float
        ensemble_votes: Dict[str, float]
        anti_hallucination: Any
        timestamp: datetime
        model_metadata: Dict[str, Any]


@dataclass
class SessionConfig:
    """Configuration for session management."""
    max_session_duration_hours: int = 24
    auto_export_enabled: bool = False
    auto_export_interval_minutes: int = 60
    auto_export_formats: List[str] = field(default_factory=lambda: ['json', 'csv'])
    max_stored_readings: int = 10000
    max_stored_predictions: int = 10000
    max_stored_alerts: int = 1000
    performance_monitoring_enabled: bool = True


class SessionManager:
    """
    Manages data collection sessions and coordinates export/monitoring.
    
    This class serves as the central coordinator for:
    - Collecting sensor readings, predictions, and alerts
    - Managing session lifecycle
    - Coordinating exports and performance monitoring
    - Providing unified access to session data
    """
    
    def __init__(self, 
                 config: SessionConfig = None,
                 export_config: ExportConfig = None):
        """
        Initialize session manager.
        
        Args:
            config (SessionConfig): Session configuration
            export_config (ExportConfig): Export configuration
        """
        self.config = config or SessionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.export_service = ExportService(export_config)
        self.batch_export_manager = BatchExportManager(self.export_service)
        
        if self.config.performance_monitoring_enabled:
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.start_monitoring()
        else:
            self.performance_monitor = None
        
        # Current session data
        self.current_session_id = None
        self.session_start_time = None
        self.current_scenario = None
        
        # Data storage with thread-safe access
        self.sensor_readings = deque(maxlen=self.config.max_stored_readings)
        self.predictions = deque(maxlen=self.config.max_stored_predictions)
        self.alerts = deque(maxlen=self.config.max_stored_alerts)
        self.session_metadata = {}
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Auto-export management
        self.auto_export_timer = None
        self.last_export_time = None
        
        # Event callbacks
        self.event_callbacks = {
            'session_started': [],
            'session_ended': [],
            'data_recorded': [],
            'export_completed': [],
            'performance_alert': []
        }
        
        self.logger.info("SessionManager initialized")
    
    def start_session(self, 
                     scenario_type: str = "unknown",
                     metadata: Dict[str, Any] = None) -> str:
        """
        Start a new data collection session.
        
        Args:
            scenario_type (str): Type of scenario being run
            metadata (Dict[str, Any]): Additional session metadata
            
        Returns:
            str: Session ID
        """
        # End current session if active
        if self.current_session_id:
            self.end_session()
        
        # Start new session
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.current_scenario = scenario_type
        
        # Clear previous data
        with self.data_lock:
            self.sensor_readings.clear()
            self.predictions.clear()
            self.alerts.clear()
            
            self.session_metadata = {
                'session_id': self.current_session_id,
                'start_time': self.session_start_time,
                'scenario_type': scenario_type,
                'metadata': metadata or {}
            }
        
        # Setup auto-export if enabled
        if self.config.auto_export_enabled:
            self._schedule_auto_export()
        
        # Trigger callbacks
        self._trigger_callbacks('session_started', {
            'session_id': self.current_session_id,
            'scenario_type': scenario_type
        })
        
        self.logger.info(f"Session started: {self.current_session_id} ({scenario_type})")
        return self.current_session_id
    
    def end_session(self) -> Optional[SessionData]:
        """
        End the current session and return session data.
        
        Returns:
            Optional[SessionData]: Session data if session was active
        """
        if not self.current_session_id:
            return None
        
        end_time = datetime.now()
        
        # Collect session data
        with self.data_lock:
            session_data = SessionData(
                session_id=self.current_session_id,
                start_time=self.session_start_time,
                end_time=end_time,
                scenario_type=self.current_scenario,
                sensor_readings=list(self.sensor_readings),
                predictions=list(self.predictions),
                alerts=list(self.alerts),
                performance_metrics=self._get_performance_metrics(),
                configuration=self._get_configuration_snapshot()
            )
        
        # Cancel auto-export timer
        if self.auto_export_timer:
            self.auto_export_timer.cancel()
            self.auto_export_timer = None
        
        # Trigger callbacks
        self._trigger_callbacks('session_ended', {
            'session_id': self.current_session_id,
            'duration_minutes': (end_time - self.session_start_time).total_seconds() / 60,
            'data_points': len(session_data.sensor_readings)
        })
        
        self.logger.info(f"Session ended: {self.current_session_id}")
        
        # Reset session state
        self.current_session_id = None
        self.session_start_time = None
        self.current_scenario = None
        
        return session_data
    
    def record_sensor_reading(self, reading: SensorReading):
        """
        Record a sensor reading in the current session.
        
        Args:
            reading (SensorReading): Sensor reading to record
        """
        if not self.current_session_id:
            self.logger.warning("No active session - sensor reading not recorded")
            return
        
        with self.data_lock:
            self.sensor_readings.append(reading)
        
        self._trigger_callbacks('data_recorded', {
            'type': 'sensor_reading',
            'timestamp': reading.timestamp
        })
    
    def record_prediction(self, prediction: PredictionResult):
        """
        Record a prediction result in the current session.
        
        Args:
            prediction (PredictionResult): Prediction result to record
        """
        if not self.current_session_id:
            self.logger.warning("No active session - prediction not recorded")
            return
        
        with self.data_lock:
            self.predictions.append(prediction)
        
        # Record performance metrics
        if self.performance_monitor:
            self.performance_monitor.record_prediction(
                processing_time=prediction.processing_time,
                success=True,
                accuracy=prediction.confidence
            )
        
        self._trigger_callbacks('data_recorded', {
            'type': 'prediction',
            'timestamp': prediction.timestamp,
            'risk_score': prediction.risk_score
        })
    
    def record_alert(self, alert: AlertData):
        """
        Record an alert in the current session.
        
        Args:
            alert (AlertData): Alert to record
        """
        if not self.current_session_id:
            self.logger.warning("No active session - alert not recorded")
            return
        
        with self.data_lock:
            self.alerts.append(alert)
        
        self._trigger_callbacks('data_recorded', {
            'type': 'alert',
            'timestamp': alert.timestamp,
            'alert_level': alert.alert_level.description
        })
    
    def get_current_session_data(self) -> Optional[SessionData]:
        """
        Get current session data without ending the session.
        
        Returns:
            Optional[SessionData]: Current session data if session is active
        """
        if not self.current_session_id:
            return None
        
        with self.data_lock:
            return SessionData(
                session_id=self.current_session_id,
                start_time=self.session_start_time,
                end_time=datetime.now(),
                scenario_type=self.current_scenario,
                sensor_readings=list(self.sensor_readings),
                predictions=list(self.predictions),
                alerts=list(self.alerts),
                performance_metrics=self._get_performance_metrics(),
                configuration=self._get_configuration_snapshot()
            )
    
    def export_current_session(self, 
                              formats: List[str] = None,
                              immediate: bool = True) -> Dict[str, str]:
        """
        Export current session data.
        
        Args:
            formats (List[str]): Export formats
            immediate (bool): Whether to export immediately or schedule
            
        Returns:
            Dict[str, str]: Export results (format -> file path)
        """
        session_data = self.get_current_session_data()
        if not session_data:
            raise ValueError("No active session to export")
        
        if immediate:
            results = self.export_service.export_session_data(session_data, formats)
            self._trigger_callbacks('export_completed', {
                'session_id': self.current_session_id,
                'formats': formats,
                'results': results
            })
            return results
        else:
            # Schedule for later export
            export_time = datetime.now() + timedelta(minutes=1)
            job_id = self.batch_export_manager.schedule_export(
                session_data, export_time, formats
            )
            return {'scheduled_job_id': job_id}
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the current session.
        
        Returns:
            Dict[str, Any]: Session statistics
        """
        if not self.current_session_id:
            return {}
        
        with self.data_lock:
            current_time = datetime.now()
            duration = (current_time - self.session_start_time).total_seconds()
            
            # Calculate basic statistics
            stats = {
                'session_id': self.current_session_id,
                'scenario_type': self.current_scenario,
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'total_readings': len(self.sensor_readings),
                'total_predictions': len(self.predictions),
                'total_alerts': len(self.alerts),
                'data_rate_per_minute': len(self.sensor_readings) / max(duration / 60, 1)
            }
            
            # Add sensor statistics
            if self.sensor_readings:
                temperatures = [r.temperature for r in self.sensor_readings]
                pm25_values = [r.pm25 for r in self.sensor_readings]
                
                stats['sensor_stats'] = {
                    'avg_temperature': sum(temperatures) / len(temperatures),
                    'max_temperature': max(temperatures),
                    'min_temperature': min(temperatures),
                    'avg_pm25': sum(pm25_values) / len(pm25_values),
                    'max_pm25': max(pm25_values)
                }
            
            # Add prediction statistics
            if self.predictions:
                risk_scores = [p.risk_score for p in self.predictions]
                processing_times = [p.processing_time for p in self.predictions]
                
                stats['prediction_stats'] = {
                    'avg_risk_score': sum(risk_scores) / len(risk_scores),
                    'max_risk_score': max(risk_scores),
                    'avg_processing_time': sum(processing_times) / len(processing_times),
                    'max_processing_time': max(processing_times)
                }
            
            # Add alert statistics
            if self.alerts:
                alert_levels = [a.alert_level.level for a in self.alerts]
                critical_alerts = sum(1 for level in alert_levels if level >= 4)
                
                stats['alert_stats'] = {
                    'critical_alerts': critical_alerts,
                    'total_alerts': len(self.alerts),
                    'alert_rate_per_minute': len(self.alerts) / max(duration / 60, 1)
                }
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance monitoring summary.
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        if not self.performance_monitor:
            return {'performance_monitoring': 'disabled'}
        
        return self.performance_monitor.get_performance_summary()
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for session events.
        
        Args:
            event (str): Event name
            callback (Callable): Callback function
        """
        if event in self.event_callbacks:
            self.event_callbacks[event].append(callback)
        else:
            self.logger.warning(f"Unknown event type: {event}")
    
    def cleanup(self):
        """Clean up resources and stop monitoring."""
        # End current session
        if self.current_session_id:
            self.end_session()
        
        # Stop performance monitoring
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        # Cancel timers
        if self.auto_export_timer:
            self.auto_export_timer.cancel()
        
        self.logger.info("SessionManager cleanup completed")
    
    def _schedule_auto_export(self):
        """Schedule automatic export of session data."""
        if not self.config.auto_export_enabled:
            return
        
        def auto_export():
            try:
                if self.current_session_id:
                    self.export_current_session(
                        formats=self.config.auto_export_formats,
                        immediate=False
                    )
                    self.last_export_time = datetime.now()
                    
                    # Schedule next export
                    self._schedule_auto_export()
            except Exception as e:
                self.logger.error(f"Auto-export failed: {e}")
        
        # Schedule next export
        self.auto_export_timer = threading.Timer(
            self.config.auto_export_interval_minutes * 60,
            auto_export
        )
        self.auto_export_timer.start()
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.performance_monitor:
            return {}
        
        return self.performance_monitor.get_performance_summary()
    
    def _get_configuration_snapshot(self) -> Dict[str, Any]:
        """Get current configuration snapshot."""
        return {
            'session_config': {
                'max_session_duration_hours': self.config.max_session_duration_hours,
                'auto_export_enabled': self.config.auto_export_enabled,
                'performance_monitoring_enabled': self.config.performance_monitoring_enabled
            },
            'export_config': {
                'output_directory': self.export_service.config.output_directory,
                'include_charts': self.export_service.config.include_charts,
                'chart_resolution': self.export_service.config.chart_resolution
            }
        }
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]):
        """Trigger registered callbacks for an event."""
        for callback in self.event_callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error for {event}: {e}")