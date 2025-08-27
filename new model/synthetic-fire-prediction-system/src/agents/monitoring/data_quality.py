"""
Data Quality Monitor for the synthetic fire prediction system.

This module implements the DataQualityMonitor class, which is responsible for
monitoring the quality of incoming data used by the system.
"""

from typing import Dict, Any, List, Optional, Union, Set
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


class DataQualityMetrics:
    """
    Class for collecting and storing data quality metrics.
    """
    
    def __init__(self):
        """
        Initialize the data quality metrics collector.
        """
        self.data_metrics = defaultdict(lambda: defaultdict(list))
        self.data_timestamps = defaultdict(list)
        self.missing_values = defaultdict(lambda: defaultdict(list))
        self.out_of_range_values = defaultdict(lambda: defaultdict(list))
        self.anomalies = defaultdict(lambda: defaultdict(list))
        self.data_volumes = defaultdict(list)
        self.data_sources = set()
    
    def record_data_metrics(self, 
                           source_id: str, 
                           metrics: Dict[str, Any],
                           timestamp: Optional[datetime] = None) -> None:
        """
        Record data quality metrics.
        
        Args:
            source_id: ID of the data source
            metrics: Dictionary of metrics to record
            timestamp: Timestamp of the metrics (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store timestamp
        self.data_timestamps[source_id].append(timestamp)
        
        # Store metrics
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.data_metrics[source_id][metric_name].append(value)
        
        # Add to data sources
        self.data_sources.add(source_id)
        
        # Store data volume if provided
        if "data_volume" in metrics:
            self.data_volumes[source_id].append(metrics["data_volume"])
        
        # Limit history size
        max_history = 10000
        if len(self.data_timestamps[source_id]) > max_history:
            self.data_timestamps[source_id] = self.data_timestamps[source_id][-max_history:]
            
            for metric_name in self.data_metrics[source_id]:
                self.data_metrics[source_id][metric_name] = self.data_metrics[source_id][metric_name][-max_history:]
            
            if self.data_volumes[source_id]:
                self.data_volumes[source_id] = self.data_volumes[source_id][-max_history:]
    
    def record_missing_values(self, 
                             source_id: str, 
                             field_name: str, 
                             count: int,
                             total: int,
                             timestamp: Optional[datetime] = None) -> None:
        """
        Record missing values for a data field.
        
        Args:
            source_id: ID of the data source
            field_name: Name of the data field
            count: Number of missing values
            total: Total number of values
            timestamp: Timestamp of the metrics (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.missing_values[source_id][field_name].append({
            "timestamp": timestamp,
            "count": count,
            "total": total,
            "percentage": (count / total) * 100 if total > 0 else 0
        })
        
        # Limit history size
        max_history = 1000
        if len(self.missing_values[source_id][field_name]) > max_history:
            self.missing_values[source_id][field_name] = self.missing_values[source_id][field_name][-max_history:]
    
    def record_out_of_range(self, 
                           source_id: str, 
                           field_name: str, 
                           count: int,
                           total: int,
                           min_value: Optional[float] = None,
                           max_value: Optional[float] = None,
                           timestamp: Optional[datetime] = None) -> None:
        """
        Record out-of-range values for a data field.
        
        Args:
            source_id: ID of the data source
            field_name: Name of the data field
            count: Number of out-of-range values
            total: Total number of values
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            timestamp: Timestamp of the metrics (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.out_of_range_values[source_id][field_name].append({
            "timestamp": timestamp,
            "count": count,
            "total": total,
            "percentage": (count / total) * 100 if total > 0 else 0,
            "min_value": min_value,
            "max_value": max_value
        })
        
        # Limit history size
        max_history = 1000
        if len(self.out_of_range_values[source_id][field_name]) > max_history:
            self.out_of_range_values[source_id][field_name] = self.out_of_range_values[source_id][field_name][-max_history:]
    
    def record_anomaly(self, 
                      source_id: str, 
                      field_name: str, 
                      anomaly_type: str,
                      description: str,
                      value: Any = None,
                      timestamp: Optional[datetime] = None) -> None:
        """
        Record a data anomaly.
        
        Args:
            source_id: ID of the data source
            field_name: Name of the data field
            anomaly_type: Type of anomaly
            description: Description of the anomaly
            value: Anomalous value
            timestamp: Timestamp of the anomaly (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.anomalies[source_id][field_name].append({
            "timestamp": timestamp,
            "anomaly_type": anomaly_type,
            "description": description,
            "value": value
        })
        
        # Limit history size
        max_history = 1000
        if len(self.anomalies[source_id][field_name]) > max_history:
            self.anomalies[source_id][field_name] = self.anomalies[source_id][field_name][-max_history:]
    
    def get_missing_values_rate(self, source_id: str, field_name: str, timespan: timedelta = None) -> Optional[float]:
        """
        Get the rate of missing values for a data field.
        
        Args:
            source_id: ID of the data source
            field_name: Name of the data field
            timespan: Timespan to calculate rate for (None for all)
            
        Returns:
            Missing values rate as a percentage, or None if no data
        """
        if source_id not in self.missing_values or field_name not in self.missing_values[source_id]:
            return None
        
        missing_values = self.missing_values[source_id][field_name]
        
        if not missing_values:
            return None
        
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            filtered_values = [v for v in missing_values if v["timestamp"] >= cutoff_time]
            
            if not filtered_values:
                return None
            
            total_missing = sum(v["count"] for v in filtered_values)
            total_values = sum(v["total"] for v in filtered_values)
            
            return (total_missing / total_values) * 100 if total_values > 0 else 0
        else:
            total_missing = sum(v["count"] for v in missing_values)
            total_values = sum(v["total"] for v in missing_values)
            
            return (total_missing / total_values) * 100 if total_values > 0 else 0
    
    def get_out_of_range_rate(self, source_id: str, field_name: str, timespan: timedelta = None) -> Optional[float]:
        """
        Get the rate of out-of-range values for a data field.
        
        Args:
            source_id: ID of the data source
            field_name: Name of the data field
            timespan: Timespan to calculate rate for (None for all)
            
        Returns:
            Out-of-range values rate as a percentage, or None if no data
        """
        if source_id not in self.out_of_range_values or field_name not in self.out_of_range_values[source_id]:
            return None
        
        out_of_range_values = self.out_of_range_values[source_id][field_name]
        
        if not out_of_range_values:
            return None
        
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            filtered_values = [v for v in out_of_range_values if v["timestamp"] >= cutoff_time]
            
            if not filtered_values:
                return None
            
            total_out_of_range = sum(v["count"] for v in filtered_values)
            total_values = sum(v["total"] for v in filtered_values)
            
            return (total_out_of_range / total_values) * 100 if total_values > 0 else 0
        else:
            total_out_of_range = sum(v["count"] for v in out_of_range_values)
            total_values = sum(v["total"] for v in out_of_range_values)
            
            return (total_out_of_range / total_values) * 100 if total_values > 0 else 0
    
    def get_anomaly_count(self, source_id: str, field_name: Optional[str] = None, timespan: timedelta = None) -> int:
        """
        Get the count of anomalies for a data source or field.
        
        Args:
            source_id: ID of the data source
            field_name: Name of the data field (None for all fields)
            timespan: Timespan to count anomalies for (None for all)
            
        Returns:
            Anomaly count
        """
        if source_id not in self.anomalies:
            return 0
        
        if field_name is not None:
            if field_name not in self.anomalies[source_id]:
                return 0
            
            anomalies = self.anomalies[source_id][field_name]
        else:
            # Combine anomalies from all fields
            anomalies = []
            for field_anomalies in self.anomalies[source_id].values():
                anomalies.extend(field_anomalies)
        
        if not anomalies:
            return 0
        
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            return sum(1 for a in anomalies if a["timestamp"] >= cutoff_time)
        else:
            return len(anomalies)
    
    def get_data_volume(self, source_id: str, timespan: timedelta = None) -> Optional[float]:
        """
        Get the average data volume for a data source.
        
        Args:
            source_id: ID of the data source
            timespan: Timespan to calculate volume for (None for all)
            
        Returns:
            Average data volume, or None if no data
        """
        if source_id not in self.data_volumes or not self.data_volumes[source_id]:
            return None
        
        volumes = self.data_volumes[source_id]
        timestamps = self.data_timestamps[source_id]
        
        if not volumes or len(volumes) != len(timestamps):
            return None
        
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            indices = [i for i, t in enumerate(timestamps) if t >= cutoff_time]
            
            filtered_volumes = [volumes[i] for i in indices]
            
            if not filtered_volumes:
                return None
            
            return sum(filtered_volumes) / len(filtered_volumes)
        else:
            return sum(volumes) / len(volumes)
    
    def get_metrics_summary(self, source_id: str, timespan: timedelta = None) -> Dict[str, Any]:
        """
        Get a summary of data quality metrics for a data source.
        
        Args:
            source_id: ID of the data source
            timespan: Timespan to summarize metrics for (None for all)
            
        Returns:
            Dictionary of metric summaries
        """
        # Get timestamps for the source
        timestamps = self.data_timestamps.get(source_id, [])
        
        if not timestamps:
            return {
                "source_id": source_id,
                "data_points": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "metrics": {},
                "missing_values": {},
                "out_of_range_values": {},
                "anomaly_count": 0,
                "avg_data_volume": None
            }
        
        # Filter by timespan if provided
        if timespan:
            cutoff_time = datetime.now() - timespan
            indices = [i for i, t in enumerate(timestamps) if t >= cutoff_time]
            
            if not indices:
                return {
                    "source_id": source_id,
                    "data_points": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "metrics": {},
                    "missing_values": {},
                    "out_of_range_values": {},
                    "anomaly_count": 0,
                    "avg_data_volume": None
                }
            
            filtered_timestamps = [timestamps[i] for i in indices]
            first_timestamp = min(filtered_timestamps)
            last_timestamp = max(filtered_timestamps)
            data_points = len(filtered_timestamps)
        else:
            first_timestamp = min(timestamps)
            last_timestamp = max(timestamps)
            data_points = len(timestamps)
        
        # Calculate average metrics
        metrics = {}
        for metric_name, values in self.data_metrics.get(source_id, {}).items():
            if timespan:
                # Filter by timespan
                filtered_values = [values[i] for i in indices if i < len(values)]
                if filtered_values:
                    metrics[metric_name] = sum(filtered_values) / len(filtered_values)
            else:
                if values:
                    metrics[metric_name] = sum(values) / len(values)
        
        # Get missing values rates for all fields
        missing_values = {}
        for field_name in self.missing_values.get(source_id, {}):
            rate = self.get_missing_values_rate(source_id, field_name, timespan)
            if rate is not None:
                missing_values[field_name] = rate
        
        # Get out-of-range values rates for all fields
        out_of_range_values = {}
        for field_name in self.out_of_range_values.get(source_id, {}):
            rate = self.get_out_of_range_rate(source_id, field_name, timespan)
            if rate is not None:
                out_of_range_values[field_name] = rate
        
        # Get anomaly count
        anomaly_count = self.get_anomaly_count(source_id, None, timespan)
        
        # Get average data volume
        avg_data_volume = self.get_data_volume(source_id, timespan)
        
        return {
            "source_id": source_id,
            "data_points": data_points,
            "first_timestamp": first_timestamp.isoformat() if first_timestamp else None,
            "last_timestamp": last_timestamp.isoformat() if last_timestamp else None,
            "metrics": metrics,
            "missing_values": missing_values,
            "out_of_range_values": out_of_range_values,
            "anomaly_count": anomaly_count,
            "avg_data_volume": avg_data_volume
        }
    
    def get_all_sources(self) -> Set[str]:
        """
        Get a set of all data source IDs.
        
        Returns:
            Set of data source IDs
        """
        return self.data_sources


class DataQualityMonitor(MonitoringAgent):
    """
    Agent responsible for monitoring the quality of incoming data.
    
    This agent tracks data quality metrics, detects anomalies, and
    ensures that data meets quality standards.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the data quality monitor.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Initialize data quality metrics
        self.metrics = DataQualityMetrics()
        
        # Initialize quality thresholds
        self.missing_values_threshold = config.get("missing_values_threshold", 5.0)  # percentage
        self.out_of_range_threshold = config.get("out_of_range_threshold", 2.0)  # percentage
        self.anomaly_threshold = config.get("anomaly_threshold", 10)  # count
        
        # Initialize monitoring state
        self.last_check_time = None
        self.check_interval = config.get("check_interval_seconds", 300)  # 5 minutes default
        self.quality_alerts = []
        
        # Initialize data source registry
        self.data_sources = {}
        self._register_default_data_sources()
        
        # Initialize field definitions
        self.field_definitions = {}
        self._load_field_definitions()
        
        # Start monitoring thread if auto_start is enabled
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        if config.get("auto_start", False):
            self.start_monitoring()
        
        # Register message handlers
        self.register_message_handler("register_data_source", self._handle_register_data_source)
        self.register_message_handler("record_data_metrics", self._handle_record_data_metrics)
        self.register_message_handler("get_data_quality", self._handle_get_data_quality)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate threshold values
        if "missing_values_threshold" in self.config:
            threshold = self.config["missing_values_threshold"]
            if not 0 <= threshold <= 100:
                raise ValueError(f"missing_values_threshold must be between 0 and 100, got {threshold}")
        
        if "out_of_range_threshold" in self.config:
            threshold = self.config["out_of_range_threshold"]
            if not 0 <= threshold <= 100:
                raise ValueError(f"out_of_range_threshold must be between 0 and 100, got {threshold}")
        
        if "anomaly_threshold" in self.config:
            threshold = self.config["anomaly_threshold"]
            if threshold < 0:
                raise ValueError(f"anomaly_threshold must be non-negative, got {threshold}")
        
        # Validate check interval
        if "check_interval_seconds" in self.config:
            interval = self.config["check_interval_seconds"]
            if interval < 1:
                raise ValueError(f"check_interval_seconds must be at least 1, got {interval}")
    
    def _register_default_data_sources(self) -> None:
        """
        Register default data sources for monitoring.
        """
        default_sources = [
            {
                "id": "thermal_sensor",
                "name": "Thermal Sensor",
                "type": "sensor",
                "critical": True
            },
            {
                "id": "gas_sensor",
                "name": "Gas Sensor",
                "type": "sensor",
                "critical": True
            },
            {
                "id": "environmental_sensor",
                "name": "Environmental Sensor",
                "type": "sensor",
                "critical": False
            }
        ]
        
        for source in default_sources:
            self.data_sources[source["id"]] = source
    
    def _load_field_definitions(self) -> None:
        """
        Load field definitions from configuration.
        """
        # Try to load from config
        field_defs_path = self.config.get("field_definitions_path")
        if field_defs_path:
            try:
                with open(field_defs_path, 'r') as f:
                    self.field_definitions = json.load(f)
                logger.info(f"Loaded field definitions from {field_defs_path}")
                return
            except Exception as e:
                logger.error(f"Error loading field definitions: {e}")
        
        # Use default definitions
        self.field_definitions = {
            "thermal_sensor": {
                "max_temperature": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 1000,
                    "unit": "celsius"
                },
                "mean_temperature": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 500,
                    "unit": "celsius"
                },
                "hotspot_area_percentage": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 100,
                    "unit": "percentage"
                }
            },
            "gas_sensor": {
                "methane_ppm": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 10000,
                    "unit": "ppm"
                },
                "propane_ppm": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 10000,
                    "unit": "ppm"
                },
                "hydrogen_ppm": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 10000,
                    "unit": "ppm"
                },
                "co_ppm": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 5000,
                    "unit": "ppm"
                }
            },
            "environmental_sensor": {
                "temperature": {
                    "type": "float",
                    "min_value": -50,
                    "max_value": 100,
                    "unit": "celsius"
                },
                "humidity": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 100,
                    "unit": "percentage"
                },
                "pressure": {
                    "type": "float",
                    "min_value": 800,
                    "max_value": 1200,
                    "unit": "hPa"
                },
                "voc_level": {
                    "type": "float",
                    "min_value": 0,
                    "max_value": 1000,
                    "unit": "ppb"
                }
            }
        }
    
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
        logger.info("Started data quality monitoring thread")
    
    def stop_monitoring(self) -> None:
        """
        Stop the monitoring thread.
        """
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is not running")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5)
        logger.info("Stopped data quality monitoring thread")
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that runs in a separate thread.
        """
        while not self.stop_monitoring.is_set():
            try:
                # Check data quality
                self._check_data_quality()
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep for a shorter time if there's an error
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and monitor data quality.
        
        Args:
            data: Input data containing sensor readings or data metrics
            
        Returns:
            Dictionary containing data quality monitoring results
        """
        try:
            # Process data metrics if provided
            if "source_id" in data and "fields" in data:
                source_id = data["source_id"]
                fields = data["fields"]
                timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
                
                # Record data metrics
                metrics = {
                    "data_volume": len(fields),
                    "timestamp": timestamp.isoformat()
                }
                self.metrics.record_data_metrics(source_id, metrics, timestamp)
                
                # Check for missing values
                missing_count = sum(1 for f, v in fields.items() if v is None)
                if missing_count > 0:
                    self.metrics.record_missing_values(
                        source_id=source_id,
                        field_name="all_fields",
                        count=missing_count,
                        total=len(fields),
                        timestamp=timestamp
                    )
                
                # Check field values against definitions
                if source_id in self.field_definitions:
                    field_defs = self.field_definitions[source_id]
                    
                    for field_name, value in fields.items():
                        if field_name in field_defs and value is not None:
                            field_def = field_defs[field_name]
                            
                            # Check type
                            expected_type = field_def.get("type")
                            if expected_type == "float" and not isinstance(value, (int, float)):
                                self.metrics.record_anomaly(
                                    source_id=source_id,
                                    field_name=field_name,
                                    anomaly_type="type_mismatch",
                                    description=f"Expected float, got {type(value).__name__}",
                                    value=value,
                                    timestamp=timestamp
                                )
                            
                            # Check range
                            min_value = field_def.get("min_value")
                            max_value = field_def.get("max_value")
                            
                            if (min_value is not None or max_value is not None) and isinstance(value, (int, float)):
                                out_of_range = False
                                
                                if min_value is not None and value < min_value:
                                    out_of_range = True
                                
                                if max_value is not None and value > max_value:
                                    out_of_range = True
                                
                                if out_of_range:
                                    self.metrics.record_out_of_range(
                                        source_id=source_id,
                                        field_name=field_name,
                                        count=1,
                                        total=1,
                                        min_value=min_value,
                                        max_value=max_value,
                                        timestamp=timestamp
                                    )
                                    
                                    self.metrics.record_anomaly(
                                        source_id=source_id,
                                        field_name=field_name,
                                        anomaly_type="out_of_range",
                                        description=f"Value {value} outside range [{min_value}, {max_value}]",
                                        value=value,
                                        timestamp=timestamp
                                    )
            
            # Check if it's time to check data quality
            now = datetime.now()
            if self.last_check_time is None or (now - self.last_check_time).total_seconds() >= self.check_interval:
                # Check data quality
                quality_results = self._check_data_quality()
                self.last_check_time = now
                
                return {
                    "timestamp": now.isoformat(),
                    "quality_results": quality_results
                }
            else:
                # Return basic processing results
                return {
                    "timestamp": datetime.now().isoformat(),
                    "message": "Data processed successfully"
                }
        except Exception as e:
            logger.error(f"Error in data quality monitoring: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _check_data_quality(self) -> Dict[str, Any]:
        """
        Check the quality of data from all sources.
        
        Returns:
            Dictionary containing quality check results
        """
        results = {}
        alerts = []
        
        # Get all data source IDs
        source_ids = self.metrics.get_all_sources()
        
        # Add registered sources that don't have metrics yet
        for source_id in self.data_sources:
            if source_id not in source_ids:
                source_ids.add(source_id)
        
        # Check each data source
        for source_id in source_ids:
            # Get source info
            source_info = self.data_sources.get(source_id, {"id": source_id, "name": source_id, "critical": False})
            
            # Get recent metrics (last hour)
            recent_metrics = self.metrics.get_metrics_summary(source_id, timedelta(hours=1))
            
            # Check for quality issues
            issues = []
            
            # Check missing values
            for field_name, rate in recent_metrics.get("missing_values", {}).items():
                if rate > self.missing_values_threshold:
                    issues.append({
                        "type": "high_missing_values",
                        "field_name": field_name,
                        "message": f"High missing values rate for {field_name}: {rate:.2f}% (threshold: {self.missing_values_threshold:.2f}%)",
                        "severity": "high" if source_info.get("critical", False) else "medium"
                    })
            
            # Check out-of-range values
            for field_name, rate in recent_metrics.get("out_of_range_values", {}).items():
                if rate > self.out_of_range_threshold:
                    issues.append({
                        "type": "high_out_of_range",
                        "field_name": field_name,
                        "message": f"High out-of-range values rate for {field_name}: {rate:.2f}% (threshold: {self.out_of_range_threshold:.2f}%)",
                        "severity": "high" if source_info.get("critical", False) else "medium"
                    })
            
            # Check anomaly count
            anomaly_count = recent_metrics.get("anomaly_count", 0)
            if anomaly_count > self.anomaly_threshold:
                issues.append({
                    "type": "high_anomaly_count",
                    "field_name": "all_fields",
                    "message": f"High anomaly count: {anomaly_count} (threshold: {self.anomaly_threshold})",
                    "severity": "high" if source_info.get("critical", False) else "medium"
                })
            
            # Store results for this source
            results[source_id] = {
                "source_info": source_info,
                "metrics": recent_metrics,
                "issues": issues
            }
            
            # Add alerts for high severity issues
            for issue in issues:
                if issue["severity"] == "high":
                    alerts.append({
                        "source_id": source_id,
                        "issue": issue
                    })
        
        # Store alerts
        self.alerts = alerts
        
        return {
            "timestamp": datetime.now().isoformat(),
            "sources": results,
            "alerts": alerts,
            "total_sources": len(source_ids),
            "sources_with_issues": len([s for s in results.values() if s["issues"]]),
            "total_alerts": len(alerts)
        }