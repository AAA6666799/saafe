"""
Device Health Monitor for FLIR+SCD41 Sensors.

This module provides comprehensive health monitoring capabilities for
FLIR Lepton 3.5 thermal cameras and SCD41 CO₂ sensors.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time

from .base import SensorInterface
from ..agents.monitoring.system_health import SystemMetrics


class DeviceHealthMetrics:
    """
    Class for collecting and storing device health metrics.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the device health metrics collector.
        
        Args:
            max_history: Maximum number of historical data points to store
        """
        self.max_history = max_history
        
        # Device-specific metrics
        self.device_metrics = defaultdict(lambda: {
            'readings': deque(maxlen=max_history),
            'errors': deque(maxlen=max_history),
            'connection_status': deque(maxlen=max_history),
            'battery_levels': deque(maxlen=max_history),
            'temperature_readings': deque(maxlen=max_history),
            'response_times': deque(maxlen=max_history),
            'calibration_status': deque(maxlen=max_history),
            'data_quality_scores': deque(maxlen=max_history)
        })
        
        # Aggregated metrics
        self.overall_health = {}
        self.alerts = []
    
    def record_reading(self, device_id: str, reading: Dict[str, Any]) -> None:
        """
        Record a sensor reading.
        
        Args:
            device_id: ID of the device
            reading: Sensor reading data
        """
        timestamp = datetime.now()
        
        # Extract relevant metrics from reading
        metrics = {
            'timestamp': timestamp,
            'success': True,
            'data_points': len(reading) if isinstance(reading, dict) else 1
        }
        
        # Add temperature data for FLIR sensors
        if 'temperature_max' in reading:
            metrics['temperature_max'] = reading['temperature_max']
            self.device_metrics[device_id]['temperature_readings'].append({
                'timestamp': timestamp,
                'value': reading['temperature_max']
            })
        
        # Add CO2 data for SCD41 sensors
        if 'co2_concentration' in reading:
            metrics['co2_concentration'] = reading['co2_concentration']
        
        self.device_metrics[device_id]['readings'].append(metrics)
    
    def record_error(self, device_id: str, error: str, error_type: str = "unknown") -> None:
        """
        Record a device error.
        
        Args:
            device_id: ID of the device
            error: Error message
            error_type: Type of error
        """
        timestamp = datetime.now()
        
        error_record = {
            'timestamp': timestamp,
            'error': error,
            'error_type': error_type
        }
        
        self.device_metrics[device_id]['errors'].append(error_record)
    
    def record_connection_status(self, device_id: str, connected: bool, details: Dict[str, Any] = None) -> None:
        """
        Record device connection status.
        
        Args:
            device_id: ID of the device
            connected: Connection status
            details: Additional connection details
        """
        timestamp = datetime.now()
        
        status_record = {
            'timestamp': timestamp,
            'connected': connected,
            'details': details or {}
        }
        
        self.device_metrics[device_id]['connection_status'].append(status_record)
    
    def record_battery_level(self, device_id: str, battery_level: float) -> None:
        """
        Record device battery level.
        
        Args:
            device_id: ID of the device
            battery_level: Battery level as percentage (0-100)
        """
        timestamp = datetime.now()
        
        battery_record = {
            'timestamp': timestamp,
            'battery_level': battery_level
        }
        
        self.device_metrics[device_id]['battery_levels'].append(battery_record)
    
    def record_response_time(self, device_id: str, response_time: float) -> None:
        """
        Record device response time.
        
        Args:
            device_id: ID of the device
            response_time: Response time in seconds
        """
        timestamp = datetime.now()
        
        response_record = {
            'timestamp': timestamp,
            'response_time': response_time
        }
        
        self.device_metrics[device_id]['response_times'].append(response_record)
    
    def record_calibration_status(self, device_id: str, calibrated: bool, details: Dict[str, Any] = None) -> None:
        """
        Record device calibration status.
        
        Args:
            device_id: ID of the device
            calibrated: Calibration status
            details: Additional calibration details
        """
        timestamp = datetime.now()
        
        calibration_record = {
            'timestamp': timestamp,
            'calibrated': calibrated,
            'details': details or {}
        }
        
        self.device_metrics[device_id]['calibration_status'].append(calibration_record)
    
    def record_data_quality_score(self, device_id: str, quality_score: float, details: Dict[str, Any] = None) -> None:
        """
        Record data quality score.
        
        Args:
            device_id: ID of the device
            quality_score: Data quality score (0-100)
            details: Additional quality details
        """
        timestamp = datetime.now()
        
        quality_record = {
            'timestamp': timestamp,
            'quality_score': quality_score,
            'details': details or {}
        }
        
        self.device_metrics[device_id]['data_quality_scores'].append(quality_record)
    
    def get_device_health_summary(self, device_id: str, timespan: timedelta = None) -> Dict[str, Any]:
        """
        Get health summary for a specific device.
        
        Args:
            device_id: ID of the device
            timespan: Timespan to analyze (None for all data)
            
        Returns:
            Dictionary containing device health summary
        """
        if device_id not in self.device_metrics:
            return {
                'device_id': device_id,
                'status': 'unknown',
                'last_seen': None,
                'metrics': {}
            }
        
        device_data = self.device_metrics[device_id]
        
        # Filter data by timespan
        if timespan:
            cutoff_time = datetime.now() - timespan
        else:
            cutoff_time = None
        
        # Calculate metrics
        metrics = {}
        
        # Connection status
        connection_history = list(device_data['connection_status'])
        if connection_history:
            if cutoff_time:
                connection_history = [r for r in connection_history if r['timestamp'] >= cutoff_time]
            
            if connection_history:
                latest_connection = connection_history[-1]
                metrics['connected'] = latest_connection['connected']
                metrics['last_connection_check'] = latest_connection['timestamp'].isoformat()
            else:
                metrics['connected'] = False
                metrics['last_connection_check'] = None
        else:
            metrics['connected'] = None
            metrics['last_connection_check'] = None
        
        # Error rate
        error_history = list(device_data['errors'])
        if error_history:
            if cutoff_time:
                error_history = [r for r in error_history if r['timestamp'] >= cutoff_time]
            
            metrics['error_count'] = len(error_history)
            metrics['error_rate'] = len(error_history) / max(1, len(device_data['readings']))
            
            # Most recent error
            if error_history:
                metrics['last_error'] = {
                    'timestamp': error_history[-1]['timestamp'].isoformat(),
                    'error': error_history[-1]['error'],
                    'error_type': error_history[-1]['error_type']
                }
        else:
            metrics['error_count'] = 0
            metrics['error_rate'] = 0
            metrics['last_error'] = None
        
        # Battery level
        battery_history = list(device_data['battery_levels'])
        if battery_history:
            if cutoff_time:
                battery_history = [r for r in battery_history if r['timestamp'] >= cutoff_time]
            
            if battery_history:
                metrics['battery_level'] = battery_history[-1]['battery_level']
                metrics['battery_trend'] = self._calculate_trend([r['battery_level'] for r in battery_history])
            else:
                metrics['battery_level'] = None
                metrics['battery_trend'] = None
        else:
            metrics['battery_level'] = None
            metrics['battery_trend'] = None
        
        # Response time
        response_history = list(device_data['response_times'])
        if response_history:
            if cutoff_time:
                response_history = [r for r in response_history if r['timestamp'] >= cutoff_time]
            
            if response_history:
                response_times = [r['response_time'] for r in response_history]
                metrics['avg_response_time'] = np.mean(response_times)
                metrics['max_response_time'] = np.max(response_times)
                metrics['response_time_trend'] = self._calculate_trend(response_times)
            else:
                metrics['avg_response_time'] = None
                metrics['max_response_time'] = None
                metrics['response_time_trend'] = None
        else:
            metrics['avg_response_time'] = None
            metrics['max_response_time'] = None
            metrics['response_time_trend'] = None
        
        # Data quality
        quality_history = list(device_data['data_quality_scores'])
        if quality_history:
            if cutoff_time:
                quality_history = [r for r in quality_history if r['timestamp'] >= cutoff_time]
            
            if quality_history:
                quality_scores = [r['quality_score'] for r in quality_history]
                metrics['avg_data_quality'] = np.mean(quality_scores)
                metrics['min_data_quality'] = np.min(quality_scores)
                metrics['quality_trend'] = self._calculate_trend(quality_scores)
            else:
                metrics['avg_data_quality'] = None
                metrics['min_data_quality'] = None
                metrics['quality_trend'] = None
        else:
            metrics['avg_data_quality'] = None
            metrics['min_data_quality'] = None
            metrics['quality_trend'] = None
        
        # Reading frequency
        reading_history = list(device_data['readings'])
        if reading_history:
            if cutoff_time:
                reading_history = [r for r in reading_history if r['timestamp'] >= cutoff_time]
            
            if len(reading_history) > 1:
                timestamps = [r['timestamp'] for r in reading_history]
                time_deltas = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                              for i in range(1, len(timestamps))]
                metrics['avg_reading_interval'] = np.mean(time_deltas)
                metrics['reading_frequency'] = 1.0 / np.mean(time_deltas) if np.mean(time_deltas) > 0 else 0
            else:
                metrics['avg_reading_interval'] = None
                metrics['reading_frequency'] = None
            
            metrics['total_readings'] = len(reading_history)
            metrics['last_reading'] = reading_history[-1]['timestamp'].isoformat() if reading_history else None
        else:
            metrics['avg_reading_interval'] = None
            metrics['reading_frequency'] = None
            metrics['total_readings'] = 0
            metrics['last_reading'] = None
        
        # Determine overall status
        status = self._determine_device_status(metrics)
        
        return {
            'device_id': device_id,
            'status': status,
            'last_seen': metrics.get('last_reading'),
            'metrics': metrics
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend from a series of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            Trend description ('increasing', 'decreasing', 'stable')
        """
        if len(values) < 2:
            return 'stable'
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _determine_device_status(self, metrics: Dict[str, Any]) -> str:
        """
        Determine device status based on metrics.
        
        Args:
            metrics: Device metrics
            
        Returns:
            Status string ('healthy', 'degraded', 'critical', 'failed', 'unknown')
        """
        # Check connection status
        if metrics.get('connected') is False:
            return 'failed'
        
        # Check error rate
        error_rate = metrics.get('error_rate', 0)
        if error_rate > 0.5:  # More than 50% errors
            return 'failed'
        elif error_rate > 0.1:  # More than 10% errors
            return 'critical'
        elif error_rate > 0.01:  # More than 1% errors
            return 'degraded'
        
        # Check battery level
        battery_level = metrics.get('battery_level')
        if battery_level is not None:
            if battery_level < 10:  # Less than 10% battery
                return 'critical'
            elif battery_level < 20:  # Less than 20% battery
                return 'degraded'
        
        # Check data quality
        avg_quality = metrics.get('avg_data_quality')
        if avg_quality is not None:
            if avg_quality < 50:  # Less than 50% quality
                return 'degraded'
            elif avg_quality < 30:  # Less than 30% quality
                return 'critical'
        
        # Check response time
        avg_response = metrics.get('avg_response_time')
        if avg_response is not None:
            if avg_response > 5.0:  # More than 5 seconds average response
                return 'degraded'
            elif avg_response > 10.0:  # More than 10 seconds average response
                return 'critical'
        
        # If all checks pass, device is healthy
        return 'healthy'
    
    def get_overall_health_summary(self, timespan: timedelta = None) -> Dict[str, Any]:
        """
        Get overall health summary for all devices.
        
        Args:
            timespan: Timespan to analyze (None for all data)
            
        Returns:
            Dictionary containing overall health summary
        """
        device_summaries = {}
        statuses = defaultdict(int)
        
        for device_id in self.device_metrics.keys():
            summary = self.get_device_health_summary(device_id, timespan)
            device_summaries[device_id] = summary
            statuses[summary['status']] += 1
        
        # Determine overall system status
        if statuses['failed'] > 0:
            overall_status = 'failed'
        elif statuses['critical'] > 0:
            overall_status = 'critical'
        elif statuses['degraded'] > 0:
            overall_status = 'degraded'
        elif statuses['healthy'] > 0:
            overall_status = 'healthy'
        else:
            overall_status = 'unknown'
        
        return {
            'overall_status': overall_status,
            'device_count': len(device_summaries),
            'status_distribution': dict(statuses),
            'devices': device_summaries,
            'timestamp': datetime.now().isoformat()
        }


class DeviceHealthMonitor:
    """
    Comprehensive device health monitor for FLIR and SCD41 sensors.
    
    This class provides real-time monitoring of device health metrics,
    including connection status, error rates, battery levels, response times,
    and data quality scores.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the device health monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics collector
        self.metrics = DeviceHealthMetrics(
            max_history=self.config.get('max_history', 1000)
        )
        
        # Monitoring settings
        self.check_interval = self.config.get('check_interval', 60)  # seconds
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'error_rate': 0.1,
            'battery_level': 20,
            'data_quality': 50,
            'response_time': 5.0
        })
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        self.logger.info("Initialized Device Health Monitor")
    
    def start_monitoring(self) -> None:
        """
        Start the device health monitoring thread.
        """
        if self.is_monitoring:
            self.logger.warning("Device health monitoring already running")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Started device health monitoring")
    
    def stop_monitoring(self) -> None:
        """
        Stop the device health monitoring thread.
        """
        if not self.is_monitoring:
            self.logger.warning("Device health monitoring not running")
            return
        
        self.logger.info("Stopping device health monitoring...")
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Device health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop running in separate thread.
        """
        self.logger.info("Device health monitoring loop started")
        
        while self.is_monitoring and not self.stop_event.is_set():
            try:
                # Perform health checks
                self._perform_health_checks()
                
                # Check for alerts
                self._check_for_alerts()
                
                # Wait for next check
                if self.stop_event.wait(self.check_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in device health monitoring loop: {str(e)}")
                time.sleep(1.0)  # Short sleep on error
        
        self.logger.info("Device health monitoring loop stopped")
    
    def _perform_health_checks(self) -> None:
        """
        Perform health checks on all registered devices.
        """
        # This would typically interface with actual devices
        # For now, we'll just log that checks are being performed
        self.logger.debug("Performing device health checks")
    
    def _check_for_alerts(self) -> None:
        """
        Check for device health alerts.
        """
        # This would typically generate alerts based on health metrics
        # For now, we'll just log that alert checking is being performed
        self.logger.debug("Checking for device health alerts")
    
    def register_device(self, device_id: str, device_type: str, details: Dict[str, Any] = None) -> None:
        """
        Register a device for health monitoring.
        
        Args:
            device_id: Unique identifier for the device
            device_type: Type of device ('flir', 'scd41', etc.)
            details: Additional device details
        """
        self.logger.info(f"Registered device {device_id} of type {device_type}")
        # Device registration would typically involve setting up monitoring for that device
    
    def get_device_health(self, device_id: str, timespan: timedelta = None) -> Dict[str, Any]:
        """
        Get health information for a specific device.
        
        Args:
            device_id: ID of the device
            timespan: Timespan to analyze (None for all data)
            
        Returns:
            Dictionary containing device health information
        """
        return self.metrics.get_device_health_summary(device_id, timespan)
    
    def get_overall_health(self, timespan: timedelta = None) -> Dict[str, Any]:
        """
        Get overall system health.
        
        Args:
            timespan: Timespan to analyze (None for all data)
            
        Returns:
            Dictionary containing overall health information
        """
        return self.metrics.get_overall_health_summary(timespan)
    
    def record_device_reading(self, device_id: str, reading: Dict[str, Any]) -> None:
        """
        Record a device reading for health monitoring.
        
        Args:
            device_id: ID of the device
            reading: Sensor reading data
        """
        self.metrics.record_reading(device_id, reading)
    
    def record_device_error(self, device_id: str, error: str, error_type: str = "unknown") -> None:
        """
        Record a device error.
        
        Args:
            device_id: ID of the device
            error: Error message
            error_type: Type of error
        """
        self.metrics.record_error(device_id, error, error_type)
    
    def record_connection_status(self, device_id: str, connected: bool, details: Dict[str, Any] = None) -> None:
        """
        Record device connection status.
        
        Args:
            device_id: ID of the device
            connected: Connection status
            details: Additional connection details
        """
        self.metrics.record_connection_status(device_id, connected, details)
    
    def record_battery_level(self, device_id: str, battery_level: float) -> None:
        """
        Record device battery level.
        
        Args:
            device_id: ID of the device
            battery_level: Battery level as percentage (0-100)
        """
        self.metrics.record_battery_level(device_id, battery_level)
    
    def record_response_time(self, device_id: str, response_time: float) -> None:
        """
        Record device response time.
        
        Args:
            device_id: ID of the device
            response_time: Response time in seconds
        """
        self.metrics.record_response_time(device_id, response_time)
    
    def record_calibration_status(self, device_id: str, calibrated: bool, details: Dict[str, Any] = None) -> None:
        """
        Record device calibration status.
        
        Args:
            device_id: ID of the device
            calibrated: Calibration status
            details: Additional calibration details
        """
        self.metrics.record_calibration_status(device_id, calibrated, details)
    
    def record_data_quality_score(self, device_id: str, quality_score: float, details: Dict[str, Any] = None) -> None:
        """
        Record data quality score.
        
        Args:
            device_id: ID of the device
            quality_score: Data quality score (0-100)
            details: Additional quality details
        """
        self.metrics.record_data_quality_score(device_id, quality_score, details)


# Convenience function for creating device health monitor
def create_device_health_monitor(config: Optional[Dict[str, Any]] = None) -> DeviceHealthMonitor:
    """
    Create a device health monitor with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured DeviceHealthMonitor instance
    """
    default_config = {
        'max_history': 1000,
        'check_interval': 60,
        'alert_thresholds': {
            'error_rate': 0.1,
            'battery_level': 20,
            'data_quality': 50,
            'response_time': 5.0
        }
    }
    
    if config:
        # Merge configs
        for key, value in config.items():
            if key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return DeviceHealthMonitor(default_config)