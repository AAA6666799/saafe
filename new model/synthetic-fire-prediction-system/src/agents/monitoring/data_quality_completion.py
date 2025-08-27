"""
Completion file for DataQualityMonitor class.
This contains the missing methods that need to be added to data_quality.py
"""

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
                "message": f"High anomaly count: {anomaly_count} (threshold: {self.anomaly_threshold})",
                "severity": "high" if source_info.get("critical", False) else "medium"
            })
        
        # Check data freshness
        last_timestamp = recent_metrics.get("last_timestamp")
        if last_timestamp:
            last_time = datetime.fromisoformat(last_timestamp)
            time_since_last = (datetime.now() - last_time).total_seconds()
            max_staleness = self.config.get("max_staleness_seconds", 3600)  # 1 hour default
            
            if time_since_last > max_staleness:
                issues.append({
                    "type": "stale_data",
                    "message": f"Stale data: Last update {time_since_last:.0f} seconds ago (threshold: {max_staleness} seconds)",
                    "severity": "high" if source_info.get("critical", False) else "medium"
                })
        
        # Store results
        results[source_id] = {
            "source_info": source_info,
            "metrics": recent_metrics,
            "issues": issues,
            "status": "degraded" if issues else "healthy"
        }
        
        # Generate alerts for issues
        if issues:
            for issue in issues:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "source_id": source_id,
                    "source_name": source_info.get("name", source_id),
                    "issue_type": issue["type"],
                    "message": issue["message"],
                    "severity": issue["severity"]
                }
                alerts.append(alert)
                self.quality_alerts.append(alert)
    
    # Limit alerts history
    max_alerts = self.config.get("max_alerts", 1000)
    if len(self.quality_alerts) > max_alerts:
        self.quality_alerts = self.quality_alerts[-max_alerts:]
    
    # Log alerts
    for alert in alerts:
        log_method = logger.warning if alert["severity"] == "medium" else logger.error
        log_method(f"Data quality alert: {alert['source_name']} - {alert['message']}")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "sources": results,
        "alerts": alerts
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
    
    if "source_id" in data and "fields" in data:
        source_id = data["source_id"]
        fields = data["fields"]
        
        # Check field values against definitions
        if source_id in self.field_definitions:
            field_defs = self.field_definitions[source_id]
            
            for field_name, value in fields.items():
                if field_name in field_defs and value is not None:
                    field_def = field_defs[field_name]
                    
                    # Check type
                    expected_type = field_def.get("type")
                    if expected_type == "float" and not isinstance(value, (int, float)):
                        anomalies.append({
                            "type": "type_mismatch",
                            "source_id": source_id,
                            "field_name": field_name,
                            "expected_type": expected_type,
                            "actual_type": type(value).__name__,
                            "value": value,
                            "message": f"Type mismatch for {field_name}: expected {expected_type}, got {type(value).__name__}",
                            "severity": "medium"
                        })
                    
                    # Check range
                    min_value = field_def.get("min_value")
                    max_value = field_def.get("max_value")
                    
                    if (min_value is not None or max_value is not None) and isinstance(value, (int, float)):
                        if min_value is not None and value < min_value:
                            anomalies.append({
                                "type": "below_min_value",
                                "source_id": source_id,
                                "field_name": field_name,
                                "min_value": min_value,
                                "value": value,
                                "message": f"Value {value} below minimum {min_value} for {field_name}",
                                "severity": "high"
                            })
                        
                        if max_value is not None and value > max_value:
                            anomalies.append({
                                "type": "above_max_value",
                                "source_id": source_id,
                                "field_name": field_name,
                                "max_value": max_value,
                                "value": value,
                                "message": f"Value {value} above maximum {max_value} for {field_name}",
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
    health_info = {}
    
    if "source_id" in data:
        source_id = data["source_id"]
        
        # Get recent metrics
        recent_metrics = self.metrics.get_metrics_summary(source_id, timedelta(hours=1))
        
        # Determine health status
        issues = []
        
        # Check missing values
        for field_name, rate in recent_metrics.get("missing_values", {}).items():
            if rate > self.missing_values_threshold:
                issues.append(f"High missing values rate for {field_name}: {rate:.2f}%")
        
        # Check out-of-range values
        for field_name, rate in recent_metrics.get("out_of_range_values", {}).items():
            if rate > self.out_of_range_threshold:
                issues.append(f"High out-of-range values rate for {field_name}: {rate:.2f}%")
        
        # Check anomaly count
        anomaly_count = recent_metrics.get("anomaly_count", 0)
        if anomaly_count > self.anomaly_threshold:
            issues.append(f"High anomaly count: {anomaly_count}")
        
        # Check data freshness
        last_timestamp = recent_metrics.get("last_timestamp")
        if last_timestamp:
            last_time = datetime.fromisoformat(last_timestamp)
            time_since_last = (datetime.now() - last_time).total_seconds()
            max_staleness = self.config.get("max_staleness_seconds", 3600)  # 1 hour default
            
            if time_since_last > max_staleness:
                issues.append(f"Stale data: Last update {time_since_last:.0f} seconds ago")
        
        # Determine overall health
        if issues:
            health_status = "unhealthy"
        else:
            health_status = "healthy"
        
        health_info = {
            "source_id": source_id,
            "status": health_status,
            "issues": issues,
            "metrics": recent_metrics
        }
    
    return health_info

def _handle_register_data_source(self, message: Message) -> Optional[Message]:
    """
    Handle a message to register a data source.
    
    Args:
        message: Message containing data source information
        
    Returns:
        Optional response message
    """
    content = message.content
    
    if "source" not in content:
        logger.error("Missing source in register_data_source message")
        return self.create_message(
            message.sender_id,
            "register_source_nack",
            {
                "error": "Missing source information"
            }
        )
    
    source = content["source"]
    if "id" not in source:
        logger.error("Missing source ID in register_data_source message")
        return self.create_message(
            message.sender_id,
            "register_source_nack",
            {
                "error": "Missing source ID"
            }
        )
    
    # Register the data source
    source_id = source["id"]
    self.data_sources[source_id] = source
    
    # Register field definitions if provided
    if "field_definitions" in content:
        field_defs = content["field_definitions"]
        self.field_definitions[source_id] = field_defs
    
    logger.info(f"Registered data source: {source_id}")
    
    return self.create_message(
        message.sender_id,
        "register_source_ack",
        {
            "source_id": source_id,
            "status": "registered"
        }
    )

def _handle_record_data_metrics(self, message: Message) -> Optional[Message]:
    """
    Handle a message to record data metrics.
    
    Args:
        message: Message containing data metrics
        
    Returns:
        Optional response message
    """
    content = message.content
    
    if "source_id" not in content or "fields" not in content:
        logger.error("Missing source_id or fields in record_data_metrics message")
        return self.create_message(
            message.sender_id,
            "record_metrics_nack",
            {
                "error": "Missing source_id or fields"
            }
        )
    
    source_id = content["source_id"]
    fields = content["fields"]
    timestamp = datetime.fromisoformat(content.get("timestamp", datetime.now().isoformat()))
    
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
    anomalies = []
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
                    anomalies.append({
                        "field_name": field_name,
                        "anomaly_type": "type_mismatch"
                    })
                
                # Check range
                min_value = field_def.get("min_value")
                max_value = field_def.get("max_value")
                
                if (min_value is not None or max_value is not None) and isinstance(value, (int, float)):
                    out_of_range = False
                    
                    if min_value is not None and value < min_value:
                        out_of_range = True
                        anomalies.append({
                            "field_name": field_name,
                            "anomaly_type": "below_min_value"
                        })
                    
                    if max_value is not None and value > max_value:
                        out_of_range = True
                        anomalies.append({
                            "field_name": field_name,
                            "anomaly_type": "above_max_value"
                        })
                    
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
    
    return self.create_message(
        message.sender_id,
        "record_metrics_ack",
        {
            "source_id": source_id,
            "status": "recorded",
            "anomalies": anomalies
        }
    )

def _handle_get_data_quality(self, message: Message) -> Optional[Message]:
    """
    Handle a message to get data quality information.
    
    Args:
        message: Message requesting data quality
        
    Returns:
        Response message with data quality information
    """
    content = message.content
    source_id = content.get("source_id")
    timespan_hours = content.get("timespan_hours", 24)
    
    # Get quality metrics
    if source_id:
        # Get metrics for a specific source
        if source_id not in self.data_sources and source_id not in self.metrics.get_all_sources():
            return self.create_message(
                message.sender_id,
                "data_quality_nack",
                {
                    "error": f"Unknown source ID: {source_id}"
                }
            )
        
        metrics = self.metrics.get_metrics_summary(source_id, timedelta(hours=timespan_hours))
        source_info = self.data_sources.get(source_id, {"id": source_id, "name": source_id})
        
        # Check for quality issues
        issues = []
        
        # Check missing values
        for field_name, rate in metrics.get("missing_values", {}).items():
            if rate > self.missing_values_threshold:
                issues.append({
                    "type": "high_missing_values",
                    "field_name": field_name,
                    "rate": rate,
                    "threshold": self.missing_values_threshold
                })
        
        # Check out-of-range values
        for field_name, rate in metrics.get("out_of_range_values", {}).items():
            if rate > self.out_of_range_threshold:
                issues.append({
                    "type": "high_out_of_range",
                    "field_name": field_name,
                    "rate": rate,
                    "threshold": self.out_of_range_threshold
                })
        
        # Check anomaly count
        anomaly_count = metrics.get("anomaly_count", 0)
        if anomaly_count > self.anomaly_threshold:
            issues.append({
                "type": "high_anomaly_count",
                "count": anomaly_count,
                "threshold": self.anomaly_threshold
            })
        
        return self.create_message(
            message.sender_id,
            "data_quality_info",
            {
                "source_id": source_id,
                "source_info": source_info,
                "metrics": metrics,
                "issues": issues,
                "status": "degraded" if issues else "healthy",
                "timespan_hours": timespan_hours
            }
        )
    else:
        # Get metrics for all sources
        all_metrics = {}
        all_issues = {}
        
        for sid in self.metrics.get_all_sources():
            metrics = self.metrics.get_metrics_summary(sid, timedelta(hours=timespan_hours))
            all_metrics[sid] = metrics
            
            # Check for quality issues
            issues = []
            
            # Check missing values
            for field_name, rate in metrics.get("missing_values", {}).items():
                if rate > self.missing_values_threshold:
                    issues.append({
                        "type": "high_missing_values",
                        "field_name": field_name,
                        "rate": rate,
                        "threshold": self.missing_values_threshold
                    })
            
            # Check out-of-range values
            for field_name, rate in metrics.get("out_of_range_values", {}).items():
                if rate > self.out_of_range_threshold:
                    issues.append({
                        "type": "high_out_of_range",
                        "field_name": field_name,
                        "rate": rate,
                        "threshold": self.out_of_range_threshold
                    })
            
            # Check anomaly count
            anomaly_count = metrics.get("anomaly_count", 0)
            if anomaly_count > self.anomaly_threshold:
                issues.append({
                    "type": "high_anomaly_count",
                    "count": anomaly_count,
                    "threshold": self.anomaly_threshold
                })
            
            all_issues[sid] = issues
        
        return self.create_message(
            message.sender_id,
            "data_quality_info",
            {
                "sources": all_metrics,
                "issues": all_issues,
                "timespan_hours": timespan_hours
            }
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
    # Get metrics for all sources
    metrics_by_source = {}
    for source_id in self.metrics.get_all_sources():
        metrics_by_source[source_id] = self.metrics.get_metrics_summary(source_id)
    
    state = {
        "missing_values_threshold": self.missing_values_threshold,
        "out_of_range_threshold": self.out_of_range_threshold,
        "anomaly_threshold": self.anomaly_threshold,
        "check_interval": self.check_interval,
        "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
        "data_sources": self.data_sources,
        "field_definitions": self.field_definitions,
        "quality_alerts": self.quality_alerts,
        "metrics_summary": metrics_by_source
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
    
    self.missing_values_threshold = state["missing_values_threshold"]
    self.out_of_range_threshold = state["out_of_range_threshold"]
    self.anomaly_threshold = state["anomaly_threshold"]
    self.check_interval = state["check_interval"]
    self.last_check_time = datetime.fromisoformat(state["last_check_time"]) if state["last_check_time"] else None
    self.data_sources = state["data_sources"]
    self.field_definitions = state["field_definitions"]
    self.quality_alerts = state["quality_alerts"]
    
    # Note: actual metrics data cannot be fully restored from the summary
    # This would require storing the raw data in a more complete format