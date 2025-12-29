"""
Alert Monitor for the synthetic fire prediction system.

This module implements the AlertMonitor class, which is responsible for
monitoring generated alerts and their outcomes.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import threading
import time
from collections import defaultdict

from ..base import MonitoringAgent, Message

# Configure logging
logger = logging.getLogger(__name__)


class AlertMetrics:
    """
    Class for collecting and storing alert metrics.
    """
    
    def __init__(self):
        """
        Initialize the alert metrics collector.
        """
        self.alerts = []
        self.alert_timestamps = []
        self.alert_responses = {}  # alert_id -> response
        self.alert_outcomes = {}  # alert_id -> outcome
        self.alert_metrics = defaultdict(list)  # metric_name -> [values]
        self.alert_types = defaultdict(int)  # alert_type -> count
        self.alert_severities = defaultdict(int)  # severity -> count
        self.false_positives = []
        self.false_negatives = []
    
    def record_alert(self, 
                    alert_id: str, 
                    alert_type: str,
                    severity: str,
                    message: str,
                    source_id: Optional[str] = None,
                    timestamp: Optional[datetime] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an alert.
        
        Args:
            alert_id: Unique identifier for the alert
            alert_type: Type of alert
            severity: Severity of the alert
            message: Alert message
            source_id: ID of the alert source
            timestamp: Timestamp of the alert (default: current time)
            metadata: Additional metadata about the alert
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        alert = {
            "alert_id": alert_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "source_id": source_id,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        self.alerts.append(alert)
        self.alert_timestamps.append(timestamp)
        self.alert_types[alert_type] += 1
        self.alert_severities[severity] += 1
        
        # Limit history size
        max_history = 10000
        if len(self.alerts) > max_history:
            removed_alert = self.alerts.pop(0)
            self.alert_timestamps.pop(0)
            
            # Remove associated data
            alert_id = removed_alert["alert_id"]
            if alert_id in self.alert_responses:
                del self.alert_responses[alert_id]
            if alert_id in self.alert_outcomes:
                del self.alert_outcomes[alert_id]
    
    def record_alert_response(self, 
                             alert_id: str, 
                             response_type: str,
                             response_time: float,
                             responder_id: str,
                             actions_taken: List[str],
                             timestamp: Optional[datetime] = None) -> None:
        """
        Record a response to an alert.
        
        Args:
            alert_id: ID of the alert
            response_type: Type of response
            response_time: Time taken to respond in seconds
            responder_id: ID of the responder
            actions_taken: List of actions taken
            timestamp: Timestamp of the response (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        response = {
            "alert_id": alert_id,
            "response_type": response_type,
            "response_time": response_time,
            "responder_id": responder_id,
            "actions_taken": actions_taken,
            "timestamp": timestamp
        }
        
        self.alert_responses[alert_id] = response
        
        # Record response time metric
        self.alert_metrics["response_time"].append(response_time)
        
        # Limit metrics history
        max_metrics = 10000
        if len(self.alert_metrics["response_time"]) > max_metrics:
            self.alert_metrics["response_time"] = self.alert_metrics["response_time"][-max_metrics:]
    
    def record_alert_outcome(self, 
                            alert_id: str, 
                            outcome: str,
                            was_valid: bool,
                            resolution_time: float,
                            notes: Optional[str] = None,
                            timestamp: Optional[datetime] = None) -> None:
        """
        Record the outcome of an alert.
        
        Args:
            alert_id: ID of the alert
            outcome: Outcome of the alert (e.g., "resolved", "escalated", "ignored")
            was_valid: Whether the alert was valid
            resolution_time: Time taken to resolve in seconds
            notes: Additional notes about the outcome
            timestamp: Timestamp of the outcome (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        outcome_data = {
            "alert_id": alert_id,
            "outcome": outcome,
            "was_valid": was_valid,
            "resolution_time": resolution_time,
            "notes": notes,
            "timestamp": timestamp
        }
        
        self.alert_outcomes[alert_id] = outcome_data
        
        # Record resolution time metric
        self.alert_metrics["resolution_time"].append(resolution_time)
        
        # Record false positives and false negatives
        alert = self._find_alert_by_id(alert_id)
        if alert:
            if not was_valid:
                self.false_positives.append({
                    "alert": alert,
                    "outcome": outcome_data
                })
            elif outcome == "missed":
                self.false_negatives.append({
                    "alert": alert,
                    "outcome": outcome_data
                })
        
        # Limit metrics history
        max_metrics = 10000
        if len(self.alert_metrics["resolution_time"]) > max_metrics:
            self.alert_metrics["resolution_time"] = self.alert_metrics["resolution_time"][-max_metrics:]
        
        # Limit false positives/negatives history
        max_fp_fn = 1000
        if len(self.false_positives) > max_fp_fn:
            self.false_positives = self.false_positives[-max_fp_fn:]
        if len(self.false_negatives) > max_fp_fn:
            self.false_negatives = self.false_negatives[-max_fp_fn:]
    
    def _find_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """
        Find an alert by ID.
        
        Args:
            alert_id: ID of the alert to find
            
        Returns:
            Alert dictionary if found, None otherwise
        """
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                return alert
        return None
    
    def get_alert_count(self, alert_type: Optional[str] = None, severity: Optional[str] = None, timespan: timedelta = None) -> int:
        """
        Get the count of alerts.
        
        Args:
            alert_type: Type of alerts to count (None for all)
            severity: Severity of alerts to count (None for all)
            timespan: Timespan to count alerts for (None for all)
            
        Returns:
            Alert count
        """
        if not self.alerts:
            return 0
        
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            filtered_alerts = [a for a, t in zip(self.alerts, self.alert_timestamps) if t >= cutoff_time]
        else:
            filtered_alerts = self.alerts
        
        if alert_type and severity:
            return sum(1 for a in filtered_alerts if a["alert_type"] == alert_type and a["severity"] == severity)
        elif alert_type:
            return sum(1 for a in filtered_alerts if a["alert_type"] == alert_type)
        elif severity:
            return sum(1 for a in filtered_alerts if a["severity"] == severity)
        else:
            return len(filtered_alerts)
    
    def get_average_response_time(self, alert_type: Optional[str] = None, timespan: timedelta = None) -> Optional[float]:
        """
        Get the average response time for alerts.
        
        Args:
            alert_type: Type of alerts to consider (None for all)
            timespan: Timespan to consider (None for all)
            
        Returns:
            Average response time in seconds, or None if no data
        """
        if not self.alert_responses:
            return None
        
        response_times = []
        
        for alert_id, response in self.alert_responses.items():
            alert = self._find_alert_by_id(alert_id)
            if not alert:
                continue
            
            if timespan:
                response_timestamp = response["timestamp"]
                if isinstance(response_timestamp, str):
                    response_timestamp = datetime.fromisoformat(response_timestamp)
                
                if response_timestamp < datetime.now() - timespan:
                    continue
            
            if alert_type and alert["alert_type"] != alert_type:
                continue
            
            response_times.append(response["response_time"])
        
        if not response_times:
            return None
        
        return sum(response_times) / len(response_times)
    
    def get_average_resolution_time(self, alert_type: Optional[str] = None, timespan: timedelta = None) -> Optional[float]:
        """
        Get the average resolution time for alerts.
        
        Args:
            alert_type: Type of alerts to consider (None for all)
            timespan: Timespan to consider (None for all)
            
        Returns:
            Average resolution time in seconds, or None if no data
        """
        if not self.alert_outcomes:
            return None
        
        resolution_times = []
        
        for alert_id, outcome in self.alert_outcomes.items():
            alert = self._find_alert_by_id(alert_id)
            if not alert:
                continue
            
            if timespan:
                outcome_timestamp = outcome["timestamp"]
                if isinstance(outcome_timestamp, str):
                    outcome_timestamp = datetime.fromisoformat(outcome_timestamp)
                
                if outcome_timestamp < datetime.now() - timespan:
                    continue
            
            if alert_type and alert["alert_type"] != alert_type:
                continue
            
            resolution_times.append(outcome["resolution_time"])
        
        if not resolution_times:
            return None
        
        return sum(resolution_times) / len(resolution_times)
    
    def get_false_positive_rate(self, alert_type: Optional[str] = None, timespan: timedelta = None) -> Optional[float]:
        """
        Get the false positive rate for alerts.
        
        Args:
            alert_type: Type of alerts to consider (None for all)
            timespan: Timespan to consider (None for all)
            
        Returns:
            False positive rate as a percentage, or None if no data
        """
        if not self.alert_outcomes:
            return None
        
        total_alerts = 0
        false_positives = 0
        
        for alert_id, outcome in self.alert_outcomes.items():
            alert = self._find_alert_by_id(alert_id)
            if not alert:
                continue
            
            if timespan:
                outcome_timestamp = outcome["timestamp"]
                if isinstance(outcome_timestamp, str):
                    outcome_timestamp = datetime.fromisoformat(outcome_timestamp)
                
                if outcome_timestamp < datetime.now() - timespan:
                    continue
            
            if alert_type and alert["alert_type"] != alert_type:
                continue
            
            total_alerts += 1
            if not outcome["was_valid"]:
                false_positives += 1
        
        if total_alerts == 0:
            return None
        
        return (false_positives / total_alerts) * 100
    
    def get_metrics_summary(self, timespan: timedelta = None) -> Dict[str, Any]:
        """
        Get a summary of alert metrics.
        
        Args:
            timespan: Timespan to summarize metrics for (None for all)
            
        Returns:
            Dictionary of metric summaries
        """
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            filtered_alerts = [a for a, t in zip(self.alerts, self.alert_timestamps) if t >= cutoff_time]
            
            # Count by type and severity
            alert_types = defaultdict(int)
            alert_severities = defaultdict(int)
            
            for alert in filtered_alerts:
                alert_types[alert["alert_type"]] += 1
                alert_severities[alert["severity"]] += 1
            
            # Get average response and resolution times
            avg_response_time = self.get_average_response_time(timespan=timespan)
            avg_resolution_time = self.get_average_resolution_time(timespan=timespan)
            
            # Get false positive rate
            false_positive_rate = self.get_false_positive_rate(timespan=timespan)
            
            return {
                "alert_count": len(filtered_alerts),
                "alert_types": dict(alert_types),
                "alert_severities": dict(alert_severities),
                "avg_response_time": avg_response_time,
                "avg_resolution_time": avg_resolution_time,
                "false_positive_rate": false_positive_rate
            }
        else:
            # Use all data
            return {
                "alert_count": len(self.alerts),
                "alert_types": dict(self.alert_types),
                "alert_severities": dict(self.alert_severities),
                "avg_response_time": self.get_average_response_time(),
                "avg_resolution_time": self.get_average_resolution_time(),
                "false_positive_rate": self.get_false_positive_rate()
            }


class AlertMonitor(MonitoringAgent):
    """
    Agent responsible for monitoring generated alerts and their outcomes.
    
    This agent tracks alert metrics, analyzes alert patterns, and
    provides insights into alert effectiveness.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the alert monitor.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Initialize alert metrics
        self.metrics = AlertMetrics()
        
        # Initialize monitoring thresholds
        self.false_positive_threshold = config.get("false_positive_threshold", 10.0)  # percentage
        self.response_time_threshold = config.get("response_time_threshold", 300)  # seconds
        self.resolution_time_threshold = config.get("resolution_time_threshold", 1800)  # seconds
        self.alert_rate_threshold = config.get("alert_rate_threshold", 10)  # alerts per hour
        
        # Initialize monitoring state
        self.last_check_time = None
        self.check_interval = config.get("check_interval_seconds", 300)  # 5 minutes default
        self.monitoring_alerts = []
        
        # Start monitoring thread if auto_start is enabled
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        if config.get("auto_start", False):
            self.start_monitoring()
        
        # Register message handlers
        self.register_message_handler("record_alert", self._handle_record_alert)
        self.register_message_handler("record_alert_response", self._handle_record_alert_response)
        self.register_message_handler("record_alert_outcome", self._handle_record_alert_outcome)
        self.register_message_handler("get_alert_metrics", self._handle_get_alert_metrics)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate threshold values
        if "false_positive_threshold" in self.config:
            threshold = self.config["false_positive_threshold"]
            if not 0 <= threshold <= 100:
                raise ValueError(f"false_positive_threshold must be between 0 and 100, got {threshold}")
        
        if "response_time_threshold" in self.config:
            threshold = self.config["response_time_threshold"]
            if threshold < 0:
                raise ValueError(f"response_time_threshold must be non-negative, got {threshold}")
        
        if "resolution_time_threshold" in self.config:
            threshold = self.config["resolution_time_threshold"]
            if threshold < 0:
                raise ValueError(f"resolution_time_threshold must be non-negative, got {threshold}")
        
        if "alert_rate_threshold" in self.config:
            threshold = self.config["alert_rate_threshold"]
            if threshold < 0:
                raise ValueError(f"alert_rate_threshold must be non-negative, got {threshold}")
        
        # Validate check interval
        if "check_interval_seconds" in self.config:
            interval = self.config["check_interval_seconds"]
            if interval < 1:
                raise ValueError(f"check_interval_seconds must be at least 1, got {interval}")
    
    def start_monitoring(self) -> None:
        """
        Start the monitoring thread.
        """
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Started alert monitoring thread")
    
    def stop_monitoring(self) -> None:
        """
        Stop the monitoring thread.
        """
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is not running")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5)
        logger.info("Stopped alert monitoring thread")
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that runs in a separate thread.
        """
        while not self.stop_monitoring.is_set():
            try:
                # Check alert metrics
                self._check_alert_metrics()
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep for a shorter time if there's an error
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and monitor alerts.
        
        Args:
            data: Input data containing alert information
            
        Returns:
            Dictionary containing alert monitoring results
        """
        try:
            # Record alert if provided
            if "alert" in data:
                alert_data = data["alert"]
                self.metrics.record_alert(
                    alert_id=alert_data["alert_id"],
                    alert_type=alert_data["alert_type"],
                    severity=alert_data["severity"],
                    message=alert_data["message"],
                    source_id=alert_data.get("source_id"),
                    timestamp=datetime.fromisoformat(alert_data.get("timestamp", datetime.now().isoformat())),
                    metadata=alert_data.get("metadata")
                )
            
            # Record alert response if provided
            if "response" in data:
                response_data = data["response"]
                self.metrics.record_alert_response(
                    alert_id=response_data["alert_id"],
                    response_type=response_data["response_type"],
                    response_time=response_data["response_time"],
                    responder_id=response_data["responder_id"],
                    actions_taken=response_data["actions_taken"],
                    timestamp=datetime.fromisoformat(response_data.get("timestamp", datetime.now().isoformat()))
                )
            
            # Record alert outcome if provided
            if "outcome" in data:
                outcome_data = data["outcome"]
                self.metrics.record_alert_outcome(
                    alert_id=outcome_data["alert_id"],
                    outcome=outcome_data["outcome"],
                    was_valid=outcome_data["was_valid"],
                    resolution_time=outcome_data["resolution_time"],
                    notes=outcome_data.get("notes"),
                    timestamp=datetime.fromisoformat(outcome_data.get("timestamp", datetime.now().isoformat()))
                )
            
            # Check if it's time to check alert metrics
            now = datetime.now()
            if self.last_check_time is None or (now - self.last_check_time).total_seconds() >= self.check_interval:
                # Check alert metrics
                metrics_results = self._check_alert_metrics()
                self.last_check_time = now
                
                return {
                    "timestamp": now.isoformat(),
                    "metrics_results": metrics_results
                }
            else:
                # Return basic processing results
                return {
                    "timestamp": datetime.now().isoformat(),
                    "message": "Alert data processed successfully"
                }
        except Exception as e:
            logger.error(f"Error in alert monitoring: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _check_alert_metrics(self) -> Dict[str, Any]:
        """
        Check alert metrics for issues.
        
        Returns:
            Dictionary containing metrics check results
        """
        issues = []
        
        # Get recent metrics (last hour)
        recent_metrics = self.metrics.get_metrics_summary(timedelta(hours=1))
        
        # Check false positive rate
        false_positive_rate = recent_metrics.get("false_positive_rate")
        if false_positive_rate is not None and false_positive_rate > self.false_positive_threshold:
            issues.append({
                "type": "high_false_positive_rate",
                "message": f"High false positive rate: {false_positive_rate:.2f}% (threshold: {self.false_positive_threshold:.2f}%)",
                "severity": "high"
            })
        
        # Check average response time
        avg_response_time = recent_metrics.get("avg_response_time")
        if avg_response_time is not None and avg_response_time > self.response_time_threshold:
            issues.append({
                "type": "high_response_time",
                "message": f"High average response time: {avg_response_time:.2f}s (threshold: {self.response_time_threshold}s)",
                "severity": "medium"
            })
        
        # Check average resolution time
        avg_resolution_time = recent_metrics.get("avg_resolution_time")
        if avg_resolution_time is not None and avg_resolution_time > self.resolution_time_threshold:
            issues.append({
                "type": "high_resolution_time",
                "message": f"High average resolution time: {avg_resolution_time:.2f}s (threshold: {self.resolution_time_threshold}s)",
                "severity": "medium"
            })
        
        # Check alert rate
        alert_count = recent_metrics.get("alert_count", 0)
        if alert_count > self.alert_rate_threshold:
            issues.append({
                "type": "high_alert_rate",
                "message": f"High alert rate: {alert_count} alerts/hour (threshold: {self.alert_rate_threshold})",
                "severity": "high"
            })
        
        # Generate monitoring alerts for issues
        for issue in issues:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "issue_type": issue["type"],
                "message": issue["message"],
                "severity": issue["severity"]
            }
            self.monitoring_alerts.append(alert)
            
            # Log alert
            log_method = logger.warning if issue["severity"] == "medium" else logger.error
            log_method(f"Alert monitoring issue: {issue['message']}")
        
        # Limit monitoring alerts history
        max_alerts = self.config.get("max_alerts", 1000)
        if len(self.monitoring_alerts) > max_alerts:
            self.monitoring_alerts = self.monitoring_alerts[-max_alerts:]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": recent_metrics,
            "issues": issues
        }
    
    def detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the input data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for alert anomalies
        if "alert" in data:
            alert_data = data["alert"]
            alert_type = alert_data.get("alert_type")
            severity = alert_data.get("severity")
            
            # Check for unusual alert patterns
            if alert_type and severity == "critical":
                # Check if there's a sudden increase in critical alerts of this type
                recent_count = self.metrics.get_alert_count(alert_type, "critical", timedelta(hours=1))
                if recent_count > 5:  # Arbitrary threshold for demonstration
                    anomalies.append({
                        "type": "high_critical_alert_rate",
                        "alert_type": alert_type,
                        "count": recent_count,
                        "message": f"Unusually high rate of critical {alert_type} alerts: {recent_count} in the last hour",
                        "severity": "high"
                    })
        
        # Check for response anomalies
        if "response" in data:
            response_data = data["response"]
            response_time = response_data.get("response_time")
            
            if response_time and response_time > self.response_time_threshold * 2:
                anomalies.append({
                    "type": "very_slow_response",
                    "response_time": response_time,
                    "threshold": self.response_time_threshold,
                    "message": f"Very slow response time: {response_time:.2f}s (threshold: {self.response_time_threshold}s)",
                    "severity": "high"
                })
        
        return anomalies
    
    def update_baseline(self, data: Dict[str, Any]) -> None:
        """
        Update the baseline model with new data.
        
        Args:
            data: New data to incorporate into the baseline
        """
        # This is a placeholder - actual implementation would depend on the baseline model
        pass
    
    def check_sensor_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the health of sensors based on their data.
        
        Args:
            data: Sensor data to analyze
            
        Returns:
            Dictionary containing sensor health information
        """
        # This method is not directly applicable to alert monitoring
        return {}
    
    def _handle_record_alert(self, message: Message) -> Optional[Message]:
        """
        Handle a message to record an alert.
        
        Args:
            message: Message containing alert information
            
        Returns:
            Optional response message
        """
        content = message.content
        
        required_fields = ["alert_id", "alert_type", "severity", "message"]
        for field in required_fields:
            if field not in content:
                logger.error(f"Missing {field} in record_alert message")
                return self.create_message(
                    message.sender_id,
                    "record_alert_nack",
                    {
                        "error": f"Missing {field}"
                    }
                )
        
        # Record the alert
        self.metrics.record_alert(
            alert_id=content["alert_id"],
            alert_type=content["alert_type"],
            severity=content["severity"],
            message=content["message"],
            source_id=content.get("source_id"),
            timestamp=datetime.fromisoformat(content.get("timestamp", datetime.now().isoformat())),
            metadata=content.get("metadata")
        )
        
        return self.create_message(
            message.sender_id,
            "record_alert_ack",
            {
                "alert_id": content["alert_id"],
                "status": "recorded"
            }
        )
    
    def _handle_record_alert_response(self, message: Message) -> Optional[Message]:
        """
        Handle a message to record an alert response.
        
        Args:
            message: Message containing alert response information
            
        Returns:
            Optional response message
        """
        content = message.content
        
        required_fields = ["alert_id", "response_type", "response_time", "responder_id", "actions_taken"]
        for field in required_fields:
            if field not in content:
                logger.error(f"Missing {field} in record_alert_response message")
                return self.create_message(
                    message.sender_id,
                    "record_response_nack",
                    {
                        "error": f"Missing {field}"
                    }
                )
        
        # Record the alert response
        self.metrics.record_alert_response(
            alert_id=content["alert_id"],
            response_type=content["response_type"],
            response_time=content["response_time"],
            responder_id=content["responder_id"],
            actions_taken=content["actions_taken"],
            timestamp=datetime.fromisoformat(content.get("timestamp", datetime.now().isoformat()))
        )
        
        return self.create_message(
            message.sender_id,
            "record_response_ack",
            {
                "alert_id": content["alert_id"],
                "status": "recorded"
            }
        )
    
    def _handle_record_alert_outcome(self, message: Message) -> Optional[Message]:
        """
        Handle a message to record an alert outcome.
        
        Args:
            message: Message containing alert outcome information
            
        Returns:
            Optional response message
        """
        content = message.content
        
        required_fields = ["alert_id", "outcome", "was_valid", "resolution_time"]
        for field in required_fields:
            if field not in content:
                logger.error(f"Missing {field} in record_alert_outcome message")
                return self.create_message(
                    message.sender_id,
                    "record_outcome_nack",
                    {
                        "error": f"Missing {field}"
                    }
                )
        
        # Record the alert outcome
        self.metrics.record_alert_outcome(
            alert_id=content["alert_id"],
            outcome=content["outcome"],
            was_valid=content["was_valid"],
            resolution_time=content["resolution_time"],
            notes=content.get("notes"),
            timestamp=datetime.fromisoformat(content.get("timestamp", datetime.now().isoformat()))
        )
        
        return self.create_message(
            message.sender_id,
            "record_outcome_ack",
            {
                "alert_id": content["alert_id"],
                "status": "recorded"
            }
        )
    
    def _handle_get_alert_metrics(self, message: Message) -> Optional[Message]:
        """
        Handle a message to get alert metrics.
        
        Args:
            message: Message requesting alert metrics
            
        Returns:
            Response message with alert metrics
        """
        content = message.content
        timespan_hours = content.get("timespan_hours", 24)
        alert_type = content.get("alert_type")
        
        # Get metrics
        if alert_type:
            # Get metrics for specific alert type
            metrics = self.metrics.get_metrics_summary(alert_type, timedelta(hours=timespan_hours))
        else:
            # Get metrics for all alert types
            metrics = self.metrics.get_overall_metrics(timedelta(hours=timespan_hours))
        
        return self.create_message(
            message.sender_id,
            "alert_metrics_info",
            {
                "metrics": metrics,
                "alert_type": alert_type,
                "timespan_hours": timespan_hours
            }
        )