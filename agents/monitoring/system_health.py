"""
System Health Monitor for the synthetic fire prediction system.

This module implements the SystemHealthMonitor class, which is responsible for
monitoring the health of the fire prediction system components.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import psutil  # For system metrics
import time
import threading

from ..base import MonitoringAgent, Message

# Configure logging
logger = logging.getLogger(__name__)


class SystemMetrics:
    """
    Class for collecting and storing system metrics.
    """
    
    def __init__(self):
        """
        Initialize the system metrics collector.
        """
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_usage = []
        self.network_io = []
        self.timestamps = []
        self.component_status = {}
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current system metrics.
        
        Returns:
            Dictionary of collected metrics
        """
        timestamp = datetime.now()
        
        # Collect CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Collect memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Collect disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Collect network I/O
        network = psutil.net_io_counters()
        
        # Store metrics
        self.timestamps.append(timestamp)
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_percent)
        self.disk_usage.append(disk_percent)
        self.network_io.append({
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv
        })
        
        # Limit history size
        max_history = 1000
        if len(self.timestamps) > max_history:
            self.timestamps = self.timestamps[-max_history:]
            self.cpu_usage = self.cpu_usage[-max_history:]
            self.memory_usage = self.memory_usage[-max_history:]
            self.disk_usage = self.disk_usage[-max_history:]
            self.network_io = self.network_io[-max_history:]
        
        # Return current metrics
        return {
            'timestamp': timestamp.isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
        }
    
    def update_component_status(self, component_id: str, status: str, details: Dict[str, Any] = None) -> None:
        """
        Update the status of a system component.
        
        Args:
            component_id: ID of the component
            status: Status of the component (e.g., "healthy", "degraded", "failed")
            details: Additional details about the component status
        """
        self.component_status[component_id] = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
    
    def get_metrics_history(self, timespan: timedelta = None) -> Dict[str, Any]:
        """
        Get historical metrics.
        
        Args:
            timespan: Timespan to retrieve metrics for (None for all)
            
        Returns:
            Dictionary of historical metrics
        """
        if not self.timestamps:
            return {
                'timestamps': [],
                'cpu_usage': [],
                'memory_usage': [],
                'disk_usage': [],
                'network_io': []
            }
        
        if timespan is None:
            return {
                'timestamps': [t.isoformat() for t in self.timestamps],
                'cpu_usage': self.cpu_usage,
                'memory_usage': self.memory_usage,
                'disk_usage': self.disk_usage,
                'network_io': self.network_io
            }
        
        # Filter by timespan
        cutoff_time = datetime.now() - timespan
        indices = [i for i, t in enumerate(self.timestamps) if t >= cutoff_time]
        
        return {
            'timestamps': [self.timestamps[i].isoformat() for i in indices],
            'cpu_usage': [self.cpu_usage[i] for i in indices],
            'memory_usage': [self.memory_usage[i] for i in indices],
            'disk_usage': [self.disk_usage[i] for i in indices],
            'network_io': [self.network_io[i] for i in indices]
        }
    
    def get_component_status(self) -> Dict[str, Any]:
        """
        Get the status of all system components.
        
        Returns:
            Dictionary of component statuses
        """
        return self.component_status


class SystemHealthMonitor(MonitoringAgent):
    """
    Agent responsible for monitoring the health of the fire prediction system.
    
    This agent collects system metrics, monitors component health, and detects
    anomalies in system behavior.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the system health monitor.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Initialize system metrics
        self.metrics = SystemMetrics()
        
        # Initialize monitoring thresholds
        self.cpu_threshold = config.get("cpu_threshold", 80)
        self.memory_threshold = config.get("memory_threshold", 80)
        self.disk_threshold = config.get("disk_threshold", 90)
        
        # Initialize monitoring state
        self.last_check_time = None
        self.check_interval = config.get("check_interval_seconds", 60)
        self.alerts = []
        
        # Initialize component registry
        self.components = {}
        self._register_default_components()
        
        # Start monitoring thread if auto_start is enabled
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        if config.get("auto_start", False):
            self.start_monitoring()
        
        # Register message handlers
        self.register_message_handler("register_component", self._handle_register_component)
        self.register_message_handler("update_component_status", self._handle_update_component_status)
        self.register_message_handler("get_system_health", self._handle_get_system_health)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate threshold values
        for threshold_name in ["cpu_threshold", "memory_threshold", "disk_threshold"]:
            if threshold_name in self.config:
                threshold = self.config[threshold_name]
                if not 0 <= threshold <= 100:
                    raise ValueError(f"{threshold_name} must be between 0 and 100, got {threshold}")
        
        # Validate check interval
        if "check_interval_seconds" in self.config:
            interval = self.config["check_interval_seconds"]
            if interval < 1:
                raise ValueError(f"check_interval_seconds must be at least 1, got {interval}")
    
    def _register_default_components(self) -> None:
        """
        Register default system components for monitoring.
        """
        default_components = [
            {
                "id": "main_application",
                "name": "Main Application",
                "type": "application",
                "critical": True
            },
            {
                "id": "data_pipeline",
                "name": "Data Processing Pipeline",
                "type": "pipeline",
                "critical": True
            },
            {
                "id": "model_service",
                "name": "ML Model Service",
                "type": "service",
                "critical": True
            },
            {
                "id": "alert_service",
                "name": "Alert Service",
                "type": "service",
                "critical": True
            },
            {
                "id": "database",
                "name": "Database",
                "type": "database",
                "critical": True
            }
        ]
        
        for component in default_components:
            self.components[component["id"]] = component
            self.metrics.update_component_status(component["id"], "unknown")
    
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
        logger.info("Started system health monitoring thread")
    
    def stop_monitoring(self) -> None:
        """
        Stop the monitoring thread.
        """
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is not running")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5)
        logger.info("Stopped system health monitoring thread")
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that runs in a separate thread.
        """
        while not self.stop_monitoring.is_set():
            try:
                # Collect metrics
                self.metrics.collect_metrics()
                
                # Check system health
                self._check_system_health()
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Sleep for a shorter time if there's an error
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and check system health.
        
        Args:
            data: Input data containing system metrics or component status updates
            
        Returns:
            Dictionary containing system health check results
        """
        try:
            # Update component status if provided
            if "component_status" in data:
                component_id = data["component_status"]["id"]
                status = data["component_status"]["status"]
                details = data["component_status"].get("details", {})
                self.metrics.update_component_status(component_id, status, details)
            
            # Check if it's time to collect metrics
            now = datetime.now()
            if self.last_check_time is None or (now - self.last_check_time).total_seconds() >= self.check_interval:
                # Collect metrics
                current_metrics = self.metrics.collect_metrics()
                self.last_check_time = now
                
                # Check system health
                health_status = self._check_system_health()
                
                return {
                    "timestamp": now.isoformat(),
                    "metrics": current_metrics,
                    "health_status": health_status
                }
            else:
                # Return last known health status
                return {
                    "timestamp": now.isoformat(),
                    "health_status": self._get_system_health()
                }
        except Exception as e:
            logger.error(f"Error in system health monitoring: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "health_status": "unknown"
            }
    
    def _check_system_health(self) -> str:
        """
        Check the health of the system based on current metrics.
        
        Returns:
            System health status ("healthy", "degraded", or "critical")
        """
        if not self.metrics.cpu_usage:
            return "unknown"
        
        # Get latest metrics
        cpu_percent = self.metrics.cpu_usage[-1]
        memory_percent = self.metrics.memory_usage[-1]
        disk_percent = self.metrics.disk_usage[-1]
        
        # Check for critical issues
        critical_issues = []
        if cpu_percent >= self.cpu_threshold:
            critical_issues.append(f"CPU usage at {cpu_percent}% (threshold: {self.cpu_threshold}%)")
        
        if memory_percent >= self.memory_threshold:
            critical_issues.append(f"Memory usage at {memory_percent}% (threshold: {self.memory_threshold}%)")
        
        if disk_percent >= self.disk_threshold:
            critical_issues.append(f"Disk usage at {disk_percent}% (threshold: {self.disk_threshold}%)")
        
        # Check component status
        component_status = self.metrics.get_component_status()
        failed_components = [
            comp_id for comp_id, status in component_status.items()
            if status["status"] == "failed" and self.components.get(comp_id, {}).get("critical", False)
        ]
        
        degraded_components = [
            comp_id for comp_id, status in component_status.items()
            if status["status"] == "degraded"
        ]
        
        if failed_components:
            critical_issues.append(f"Failed critical components: {', '.join(failed_components)}")
        
        # Determine overall health status
        if critical_issues:
            health_status = "critical"
            self._generate_health_alert("critical", critical_issues)
        elif degraded_components:
            health_status = "degraded"
            self._generate_health_alert("warning", [f"Degraded components: {', '.join(degraded_components)}"])
        else:
            health_status = "healthy"
        
        return health_status
    
    def _get_system_health(self) -> Dict[str, Any]:
        """
        Get the current system health status.
        
        Returns:
            Dictionary containing system health information
        """
        # Get latest metrics if available
        if self.metrics.cpu_usage:
            cpu_percent = self.metrics.cpu_usage[-1]
            memory_percent = self.metrics.memory_usage[-1]
            disk_percent = self.metrics.disk_usage[-1]
        else:
            cpu_percent = None
            memory_percent = None
            disk_percent = None
        
        # Get component status
        component_status = self.metrics.get_component_status()
        
        # Count components by status
        status_counts = {
            "healthy": 0,
            "degraded": 0,
            "failed": 0,
            "unknown": 0
        }
        
        for comp_id, status in component_status.items():
            if status["status"] in status_counts:
                status_counts[status["status"]] += 1
        
        # Determine overall health status
        if any(comp.get("critical", False) and component_status.get(comp_id, {}).get("status") == "failed" 
               for comp_id, comp in self.components.items()):
            health_status = "critical"
        elif status_counts["degraded"] > 0:
            health_status = "degraded"
        elif status_counts["unknown"] == len(component_status):
            health_status = "unknown"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "component_status": component_status,
            "status_counts": status_counts
        }
    
    def _generate_health_alert(self, level: str, issues: List[str]) -> None:
        """
        Generate a health alert.
        
        Args:
            level: Alert level ("warning" or "critical")
            issues: List of issues detected
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "issues": issues,
            "message": f"System health {level}: {'; '.join(issues)}"
        }
        
        self.alerts.append(alert)
        
        # Limit alerts history
        max_alerts = self.config.get("max_alerts", 1000)
        if len(self.alerts) > max_alerts:
            self.alerts = self.alerts[-max_alerts:]
        
        # Log alert
        log_method = logger.warning if level == "warning" else logger.error
        log_method(f"System health alert: {alert['message']}")
    
    def detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the input data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for sudden spikes in CPU usage
        if len(self.metrics.cpu_usage) > 10:
            recent_cpu = self.metrics.cpu_usage[-10:]
            avg_cpu = sum(recent_cpu[:-1]) / len(recent_cpu[:-1])
            current_cpu = recent_cpu[-1]
            
            if current_cpu > avg_cpu * 1.5 and current_cpu > 50:
                anomalies.append({
                    "type": "cpu_spike",
                    "message": f"Sudden CPU spike detected: {current_cpu}% (avg: {avg_cpu:.1f}%)",
                    "severity": "high" if current_cpu > self.cpu_threshold else "medium"
                })
        
        # Check for memory leaks (steadily increasing memory usage)
        if len(self.metrics.memory_usage) > 30:
            recent_memory = self.metrics.memory_usage[-30:]
            if all(recent_memory[i] <= recent_memory[i+1] for i in range(len(recent_memory)-1)):
                start_memory = recent_memory[0]
                end_memory = recent_memory[-1]
                if end_memory > start_memory + 10:  # 10% increase
                    anomalies.append({
                        "type": "memory_leak",
                        "message": f"Possible memory leak detected: {start_memory}% -> {end_memory}%",
                        "severity": "high" if end_memory > self.memory_threshold else "medium"
                    })
        
        # Check for component status changes
        component_status = self.metrics.get_component_status()
        for comp_id, status in component_status.items():
            if status["status"] in ["failed", "degraded"] and comp_id in self.components:
                anomalies.append({
                    "type": "component_issue",
                    "component_id": comp_id,
                    "component_name": self.components[comp_id].get("name", comp_id),
                    "status": status["status"],
                    "message": f"Component {self.components[comp_id].get('name', comp_id)} is {status['status']}",
                    "severity": "high" if status["status"] == "failed" and self.components[comp_id].get("critical", False) else "medium"
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
        # This is a placeholder - actual implementation would depend on the sensor data format
        return {
            "sensors": {
                "thermal": data.get("thermal_sensor_health", "unknown"),
                "gas": data.get("gas_sensor_health", "unknown"),
                "environmental": data.get("environmental_sensor_health", "unknown")
            }
        }
    
    def _handle_register_component(self, message: Message) -> Optional[Message]:
        """
        Handle a message to register a component.
        
        Args:
            message: Message containing component information
            
        Returns:
            Optional response message
        """
        content = message.content
        
        if "component" not in content:
            logger.error("Missing component in register_component message")
            return self.create_message(
                message.sender_id,
                "register_component_nack",
                {
                    "error": "Missing component information"
                }
            )
        
        component = content["component"]
        if "id" not in component:
            logger.error("Missing component ID in register_component message")
            return self.create_message(
                message.sender_id,
                "register_component_nack",
                {
                    "error": "Missing component ID"
                }
            )
        
        # Register the component
        component_id = component["id"]
        self.components[component_id] = component
        
        # Initialize component status if not already present
        if component_id not in self.metrics.component_status:
            self.metrics.update_component_status(component_id, "unknown")
        
        logger.info(f"Registered component: {component_id}")
        
        return self.create_message(
            message.sender_id,
            "register_component_ack",
            {
                "component_id": component_id,
                "status": "registered"
            }
        )
    
    def _handle_update_component_status(self, message: Message) -> Optional[Message]:
        """
        Handle a message to update component status.
        
        Args:
            message: Message containing component status update
            
        Returns:
            Optional response message
        """
        content = message.content
        
        if "component_id" not in content or "status" not in content:
            logger.error("Missing component_id or status in update_component_status message")
            return self.create_message(
                message.sender_id,
                "update_status_nack",
                {
                    "error": "Missing component_id or status"
                }
            )
        
        component_id = content["component_id"]
        status = content["status"]
        details = content.get("details", {})
        
        # Update component status
        self.metrics.update_component_status(component_id, status, details)
        
        logger.info(f"Updated component status: {component_id} -> {status}")
        
        # Check system health after status update
        health_status = self._check_system_health()
        
        return self.create_message(
            message.sender_id,
            "update_status_ack",
            {
                "component_id": component_id,
                "status": status,
                "system_health": health_status
            }
        )
    
    def _handle_get_system_health(self, message: Message) -> Optional[Message]:
        """
        Handle a message to get system health information.
        
        Args:
            message: Message requesting system health
            
        Returns:
            Response message with system health information
        """
        content = message.content
        timespan_seconds = content.get("timespan_seconds")
        
        # Get system health
        health_info = self._get_system_health()
        
        # Get metrics history if requested
        if timespan_seconds:
            timespan = timedelta(seconds=timespan_seconds)
            metrics_history = self.metrics.get_metrics_history(timespan)
            health_info["metrics_history"] = metrics_history
        
        return self.create_message(
            message.sender_id,
            "system_health_info",
            health_info
        )
    
    def default_message_handler(self, message: Message) -> Optional[Message]:
        """
        Default handler for message types without a specific handler.
        
        Args:
            message: Incoming message
            
        Returns:
            Optional response message
        """
        logger.warning(f"Received unhandled message type: {message.message_type}")
        return None
    
    def create_message(self, receiver_id: str, message_type: str, content: Dict[str, Any], priority: int = 0) -> Message:
        """
        Create a new message to send to another agent.
        
        Args:
            receiver_id: ID of the receiving agent
            message_type: Type of message
            content: Message content
            priority: Message priority
            
        Returns:
            Created message
        """
        return Message(self.agent_id, receiver_id, message_type, content, priority)
    
    def save_state(self, filepath: str) -> None:
        """
        Save the agent's state to a file.
        
        Args:
            filepath: Path to save the state
        """
        # Get metrics history
        metrics_history = self.metrics.get_metrics_history()
        
        state = {
            "cpu_threshold": self.cpu_threshold,
            "memory_threshold": self.memory_threshold,
            "disk_threshold": self.disk_threshold,
            "check_interval": self.check_interval,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "components": self.components,
            "alerts": self.alerts,
            "metrics_history": {
                "timestamps": metrics_history["timestamps"],
                "cpu_usage": metrics_history["cpu_usage"],
                "memory_usage": metrics_history["memory_usage"],
                "disk_usage": metrics_history["disk_usage"]
            },
            "component_status": self.metrics.get_component_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            filepath: Path to load the state from
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.cpu_threshold = state["cpu_threshold"]
        self.memory_threshold = state["memory_threshold"]
        self.disk_threshold = state["disk_threshold"]
        self.check_interval = state["check_interval"]
        self.last_check_time = datetime.fromisoformat(state["last_check_time"]) if state["last_check_time"] else None
        self.components = state["components"]
        self.alerts = state["alerts"]
        
        # Restore component status
        for comp_id, status in state["component_status"].items():
            self.metrics.update_component_status(comp_id, status["status"], status.get("details", {}))