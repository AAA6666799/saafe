"""
Model Performance Monitor for the synthetic fire prediction system.

This module implements the ModelPerformanceMonitor class, which is responsible for
monitoring the performance of the machine learning models used in the system.
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


class ModelMetrics:
    """
    Class for collecting and storing model performance metrics.
    """
    
    def __init__(self):
        """
        Initialize the model metrics collector.
        """
        self.model_metrics = defaultdict(lambda: defaultdict(list))
        self.model_timestamps = defaultdict(list)
        self.model_predictions = defaultdict(list)
        self.model_ground_truth = defaultdict(list)
        self.model_latencies = defaultdict(list)
        self.model_errors = defaultdict(list)
    
    def record_prediction(self, 
                         model_id: str, 
                         prediction: Any, 
                         ground_truth: Any = None, 
                         confidence: float = None,
                         latency_ms: float = None,
                         features: Dict[str, Any] = None,
                         metadata: Dict[str, Any] = None) -> None:
        """
        Record a model prediction.
        
        Args:
            model_id: ID of the model
            prediction: Model prediction
            ground_truth: Ground truth value (if available)
            confidence: Prediction confidence (if available)
            latency_ms: Prediction latency in milliseconds
            features: Input features used for prediction
            metadata: Additional metadata about the prediction
        """
        timestamp = datetime.now()
        
        # Store prediction data
        self.model_timestamps[model_id].append(timestamp)
        self.model_predictions[model_id].append(prediction)
        
        if ground_truth is not None:
            self.model_ground_truth[model_id].append(ground_truth)
        
        if latency_ms is not None:
            self.model_latencies[model_id].append(latency_ms)
        
        # Store additional metrics
        if confidence is not None:
            self.model_metrics[model_id]["confidence"].append(confidence)
        
        # Store metadata as separate metrics if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    self.model_metrics[model_id][key].append(value)
        
        # Limit history size
        max_history = 10000
        if len(self.model_timestamps[model_id]) > max_history:
            self.model_timestamps[model_id] = self.model_timestamps[model_id][-max_history:]
            self.model_predictions[model_id] = self.model_predictions[model_id][-max_history:]
            
            if self.model_ground_truth[model_id]:
                self.model_ground_truth[model_id] = self.model_ground_truth[model_id][-max_history:]
            
            if self.model_latencies[model_id]:
                self.model_latencies[model_id] = self.model_latencies[model_id][-max_history:]
            
            for metric_name in self.model_metrics[model_id]:
                self.model_metrics[model_id][metric_name] = self.model_metrics[model_id][metric_name][-max_history:]
    
    def record_error(self, model_id: str, error_type: str, error_message: str, context: Dict[str, Any] = None) -> None:
        """
        Record a model error.
        
        Args:
            model_id: ID of the model
            error_type: Type of error
            error_message: Error message
            context: Additional context about the error
        """
        timestamp = datetime.now()
        
        error = {
            "timestamp": timestamp,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        self.model_errors[model_id].append(error)
        
        # Limit history size
        max_errors = 1000
        if len(self.model_errors[model_id]) > max_errors:
            self.model_errors[model_id] = self.model_errors[model_id][-max_errors:]
    
    def calculate_accuracy(self, model_id: str, timespan: timedelta = None) -> Optional[float]:
        """
        Calculate model accuracy over a timespan.
        
        Args:
            model_id: ID of the model
            timespan: Timespan to calculate accuracy for (None for all)
            
        Returns:
            Accuracy as a float between 0 and 1, or None if no data
        """
        predictions = self.model_predictions.get(model_id, [])
        ground_truth = self.model_ground_truth.get(model_id, [])
        timestamps = self.model_timestamps.get(model_id, [])
        
        if not predictions or not ground_truth or len(predictions) != len(ground_truth):
            return None
        
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            indices = [i for i, t in enumerate(timestamps) if t >= cutoff_time]
            
            filtered_predictions = [predictions[i] for i in indices]
            filtered_ground_truth = [ground_truth[i] for i in indices]
            
            if not filtered_predictions:
                return None
            
            correct = sum(1 for p, g in zip(filtered_predictions, filtered_ground_truth) if p == g)
            return correct / len(filtered_predictions)
        else:
            correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
            return correct / len(predictions)
    
    def calculate_precision_recall(self, model_id: str, positive_class: Any = 1, timespan: timedelta = None) -> Dict[str, float]:
        """
        Calculate precision and recall for binary classification.
        
        Args:
            model_id: ID of the model
            positive_class: Value representing the positive class
            timespan: Timespan to calculate metrics for (None for all)
            
        Returns:
            Dictionary with precision and recall values
        """
        predictions = self.model_predictions.get(model_id, [])
        ground_truth = self.model_ground_truth.get(model_id, [])
        timestamps = self.model_timestamps.get(model_id, [])
        
        if not predictions or not ground_truth or len(predictions) != len(ground_truth):
            return {"precision": None, "recall": None, "f1_score": None}
        
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            indices = [i for i, t in enumerate(timestamps) if t >= cutoff_time]
            
            filtered_predictions = [predictions[i] for i in indices]
            filtered_ground_truth = [ground_truth[i] for i in indices]
            
            if not filtered_predictions:
                return {"precision": None, "recall": None, "f1_score": None}
            
            predictions = filtered_predictions
            ground_truth = filtered_ground_truth
        
        # Calculate true positives, false positives, false negatives
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == positive_class and g == positive_class)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == positive_class and g != positive_class)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != positive_class and g == positive_class)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    def calculate_average_latency(self, model_id: str, timespan: timedelta = None) -> Optional[float]:
        """
        Calculate average prediction latency.
        
        Args:
            model_id: ID of the model
            timespan: Timespan to calculate latency for (None for all)
            
        Returns:
            Average latency in milliseconds, or None if no data
        """
        latencies = self.model_latencies.get(model_id, [])
        timestamps = self.model_timestamps.get(model_id, [])
        
        if not latencies:
            return None
        
        if timespan and timestamps:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            indices = [i for i, t in enumerate(timestamps) if t >= cutoff_time]
            
            filtered_latencies = [latencies[i] for i in indices if i < len(latencies)]
            
            if not filtered_latencies:
                return None
            
            return sum(filtered_latencies) / len(filtered_latencies)
        else:
            return sum(latencies) / len(latencies)
    
    def get_error_count(self, model_id: str, timespan: timedelta = None) -> int:
        """
        Get the count of errors for a model.
        
        Args:
            model_id: ID of the model
            timespan: Timespan to count errors for (None for all)
            
        Returns:
            Error count
        """
        errors = self.model_errors.get(model_id, [])
        
        if not errors:
            return 0
        
        if timespan:
            # Filter by timespan
            cutoff_time = datetime.now() - timespan
            return sum(1 for e in errors if e["timestamp"] >= cutoff_time)
        else:
            return len(errors)
    
    def get_metrics_summary(self, model_id: str, timespan: timedelta = None) -> Dict[str, Any]:
        """
        Get a summary of model metrics.
        
        Args:
            model_id: ID of the model
            timespan: Timespan to summarize metrics for (None for all)
            
        Returns:
            Dictionary of metric summaries
        """
        # Calculate accuracy if ground truth is available
        accuracy = self.calculate_accuracy(model_id, timespan)
        
        # Calculate precision and recall for binary classification
        pr_metrics = self.calculate_precision_recall(model_id, timespan=timespan)
        
        # Calculate average latency
        avg_latency = self.calculate_average_latency(model_id, timespan)
        
        # Get error count
        error_count = self.get_error_count(model_id, timespan)
        
        # Get prediction count
        if timespan and self.model_timestamps.get(model_id):
            cutoff_time = datetime.now() - timespan
            prediction_count = sum(1 for t in self.model_timestamps[model_id] if t >= cutoff_time)
        else:
            prediction_count = len(self.model_timestamps.get(model_id, []))
        
        # Calculate average confidence if available
        confidence_values = self.model_metrics.get(model_id, {}).get("confidence", [])
        if confidence_values:
            if timespan and self.model_timestamps.get(model_id):
                cutoff_time = datetime.now() - timespan
                indices = [i for i, t in enumerate(self.model_timestamps[model_id]) if t >= cutoff_time]
                filtered_confidence = [confidence_values[i] for i in indices if i < len(confidence_values)]
                avg_confidence = sum(filtered_confidence) / len(filtered_confidence) if filtered_confidence else None
            else:
                avg_confidence = sum(confidence_values) / len(confidence_values)
        else:
            avg_confidence = None
        
        return {
            "model_id": model_id,
            "prediction_count": prediction_count,
            "accuracy": accuracy,
            "precision": pr_metrics["precision"],
            "recall": pr_metrics["recall"],
            "f1_score": pr_metrics["f1_score"],
            "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence,
            "error_count": error_count
        }
    
    def get_all_models(self) -> List[str]:
        """
        Get a list of all model IDs.
        
        Returns:
            List of model IDs
        """
        return list(self.model_timestamps.keys())


class ModelPerformanceMonitor(MonitoringAgent):
    """
    Agent responsible for monitoring the performance of machine learning models.
    
    This agent tracks model predictions, calculates performance metrics, and
    detects degradation in model performance.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the model performance monitor.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Initialize model metrics
        self.metrics = ModelMetrics()
        
        # Initialize performance thresholds
        self.accuracy_threshold = config.get("accuracy_threshold", 0.9)
        self.latency_threshold_ms = config.get("latency_threshold_ms", 100)
        self.error_rate_threshold = config.get("error_rate_threshold", 0.01)
        
        # Initialize monitoring state
        self.last_check_time = None
        self.check_interval = config.get("check_interval_seconds", 300)  # 5 minutes default
        self.performance_alerts = []
        
        # Initialize model registry
        self.models = {}
        self._register_default_models()
        
        # Start monitoring thread if auto_start is enabled
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        if config.get("auto_start", False):
            self.start_monitoring()
        
        # Register message handlers
        self.register_message_handler("register_model", self._handle_register_model)
        self.register_message_handler("record_prediction", self._handle_record_prediction)
        self.register_message_handler("record_error", self._handle_record_error)
        self.register_message_handler("get_model_performance", self._handle_get_model_performance)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate threshold values
        if "accuracy_threshold" in self.config:
            threshold = self.config["accuracy_threshold"]
            if not 0 <= threshold <= 1:
                raise ValueError(f"accuracy_threshold must be between 0 and 1, got {threshold}")
        
        if "latency_threshold_ms" in self.config:
            threshold = self.config["latency_threshold_ms"]
            if threshold <= 0:
                raise ValueError(f"latency_threshold_ms must be positive, got {threshold}")
        
        if "error_rate_threshold" in self.config:
            threshold = self.config["error_rate_threshold"]
            if not 0 <= threshold <= 1:
                raise ValueError(f"error_rate_threshold must be between 0 and 1, got {threshold}")
        
        # Validate check interval
        if "check_interval_seconds" in self.config:
            interval = self.config["check_interval_seconds"]
            if interval < 1:
                raise ValueError(f"check_interval_seconds must be at least 1, got {interval}")
    
    def _register_default_models(self) -> None:
        """
        Register default models for monitoring.
        """
        default_models = [
            {
                "id": "fire_detection",
                "name": "Fire Detection Model",
                "type": "binary_classification",
                "version": "1.0",
                "critical": True
            },
            {
                "id": "fire_classification",
                "name": "Fire Classification Model",
                "type": "multiclass_classification",
                "version": "1.0",
                "critical": True
            },
            {
                "id": "fire_progression",
                "name": "Fire Progression Model",
                "type": "regression",
                "version": "1.0",
                "critical": False
            }
        ]
        
        for model in default_models:
            self.models[model["id"]] = model
    
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
        logger.info("Started model performance monitoring thread")
    
    def stop_monitoring(self) -> None:
        """
        Stop the monitoring thread.
        """
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is not running")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5)
        logger.info("Stopped model performance monitoring thread")
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that runs in a separate thread.
        """
        while not self.stop_monitoring.is_set():
            try:
                # Check model performance
                self._check_model_performance()
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep for a shorter time if there's an error
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and monitor model performance.
        
        Args:
            data: Input data containing model predictions or performance metrics
            
        Returns:
            Dictionary containing performance monitoring results
        """
        try:
            # Record prediction if provided
            if "prediction" in data:
                pred_data = data["prediction"]
                self.metrics.record_prediction(
                    model_id=pred_data["model_id"],
                    prediction=pred_data["prediction"],
                    ground_truth=pred_data.get("ground_truth"),
                    confidence=pred_data.get("confidence"),
                    latency_ms=pred_data.get("latency_ms"),
                    features=pred_data.get("features"),
                    metadata=pred_data.get("metadata")
                )
            
            # Record error if provided
            if "error" in data:
                error_data = data["error"]
                self.metrics.record_error(
                    model_id=error_data["model_id"],
                    error_type=error_data["error_type"],
                    error_message=error_data["error_message"],
                    context=error_data.get("context")
                )
            
            # Check if it's time to check performance
            now = datetime.now()
            if self.last_check_time is None or (now - self.last_check_time).total_seconds() >= self.check_interval:
                # Check model performance
                performance_results = self._check_model_performance()
                self.last_check_time = now
                
                return {
                    "timestamp": now.isoformat(),
                    "performance_results": performance_results
                }
            else:
                # Return last known performance results
                return {
                    "timestamp": now.isoformat(),
                    "message": "Performance check not due yet"
                }
        except Exception as e:
            logger.error(f"Error in model performance monitoring: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _check_model_performance(self) -> Dict[str, Any]:
        """
        Check the performance of all registered models.
        
        Returns:
            Dictionary containing performance check results
        """
        results = {}
        alerts = []
        
        # Get all model IDs
        model_ids = self.metrics.get_all_models()
        
        # Add registered models that don't have metrics yet
        for model_id in self.models:
            if model_id not in model_ids:
                model_ids.append(model_id)
        
        # Check each model
        for model_id in model_ids:
            # Get model info
            model_info = self.models.get(model_id, {"id": model_id, "name": model_id, "critical": False})
            
            # Get recent metrics (last hour)
            recent_metrics = self.metrics.get_metrics_summary(model_id, timedelta(hours=1))
            
            # Get long-term metrics (last day)
            long_term_metrics = self.metrics.get_metrics_summary(model_id, timedelta(days=1))
            
            # Check for performance issues
            issues = []
            
            # Check accuracy if available
            if recent_metrics["accuracy"] is not None and recent_metrics["accuracy"] < self.accuracy_threshold:
                issues.append({
                    "type": "low_accuracy",
                    "message": f"Low accuracy: {recent_metrics['accuracy']:.2f} (threshold: {self.accuracy_threshold:.2f})",
                    "severity": "high" if model_info.get("critical", False) else "medium"
                })
            
            # Check latency if available
            if recent_metrics["avg_latency_ms"] is not None and recent_metrics["avg_latency_ms"] > self.latency_threshold_ms:
                issues.append({
                    "type": "high_latency",
                    "message": f"High latency: {recent_metrics['avg_latency_ms']:.2f}ms (threshold: {self.latency_threshold_ms}ms)",
                    "severity": "medium"
                })
            
            # Check error rate if predictions are available
            if recent_metrics["prediction_count"] > 0:
                error_rate = recent_metrics["error_count"] / recent_metrics["prediction_count"]
                if error_rate > self.error_rate_threshold:
                    issues.append({
                        "type": "high_error_rate",
                        "message": f"High error rate: {error_rate:.2%} (threshold: {self.error_rate_threshold:.2%})",
                        "severity": "high" if model_info.get("critical", False) else "medium"
                    })
            
            # Check for performance degradation compared to long-term metrics
            if (recent_metrics["accuracy"] is not None and long_term_metrics["accuracy"] is not None and
                recent_metrics["accuracy"] < long_term_metrics["accuracy"] * 0.9):  # 10% degradation
                issues.append({
                    "type": "accuracy_degradation",
                    "message": f"Accuracy degradation: {long_term_metrics['accuracy']:.2f} -> {recent_metrics['accuracy']:.2f}",
                    "severity": "high" if model_info.get("critical", False) else "medium"
                })
            
            # Store results
            results[model_id] = {
                "model_info": model_info,
                "recent_metrics": recent_metrics,
                "long_term_metrics": long_term_metrics,
                "issues": issues,
                "status": "degraded" if issues else "healthy"
            }
            
            # Generate alerts for issues
            if issues:
                for issue in issues:
                    alert = {
                        "timestamp": datetime.now().isoformat(),
                        "model_id": model_id,
                        "model_name": model_info.get("name", model_id),
                        "issue_type": issue["type"],
                        "message": issue["message"],
                        "severity": issue["severity"]
                    }
                    alerts.append(alert)
                    self.performance_alerts.append(alert)
        
        # Limit alerts history
        max_alerts = self.config.get("max_alerts", 1000)
        if len(self.performance_alerts) > max_alerts:
            self.performance_alerts = self.performance_alerts[-max_alerts:]
        
        # Log alerts
        for alert in alerts:
            log_method = logger.warning if alert["severity"] == "medium" else logger.error
            log_method(f"Model performance alert: {alert['model_name']} - {alert['message']}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "models": results,
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
        
        # Check for model performance anomalies
        if "model_id" in data and "prediction" in data:
            model_id = data["model_id"]
            prediction = data["prediction"]
            confidence = data.get("confidence")
            latency_ms = data.get("latency_ms")
            
            # Check for unusually low confidence
            if confidence is not None and confidence < 0.5:
                anomalies.append({
                    "type": "low_confidence",
                    "model_id": model_id,
                    "confidence": confidence,
                    "message": f"Unusually low confidence: {confidence:.2f}",
                    "severity": "medium"
                })
            
            # Check for unusually high latency
            if latency_ms is not None and latency_ms > self.latency_threshold_ms * 2:
                anomalies.append({
                    "type": "very_high_latency",
                    "model_id": model_id,
                    "latency_ms": latency_ms,
                    "message": f"Very high latency: {latency_ms:.2f}ms",
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
        # This method is not directly applicable to model performance monitoring
        return {}
    
    def _handle_register_model(self, message: Message) -> Optional[Message]:
        """
        Handle a message to register a model.
        
        Args:
            message: Message containing model information
            
        Returns:
            Optional response message
        """
        content = message.content
        
        if "model" not in content:
            logger.error("Missing model in register_model message")
            return self.create_message(
                message.sender_id,
                "register_model_nack",
                {
                    "error": "Missing model information"
                }
            )
        
        model = content["model"]
        if "id" not in model:
            logger.error("Missing model ID in register_model message")
            return self.create_message(
                message.sender_id,
                "register_model_nack",
                {
                    "error": "Missing model ID"
                }
            )
        
        # Register the model
        model_id = model["id"]
        self.models[model_id] = model
        
        logger.info(f"Registered model: {model_id}")
        
        return self.create_message(
            message.sender_id,
            "register_model_ack",
            {
                "model_id": model_id,
                "status": "registered"
            }
        )
    
    def _handle_record_prediction(self, message: Message) -> Optional[Message]:
        """
        Handle a message to record a model prediction.
        
        Args:
            message: Message containing prediction information
            
        Returns:
            Optional response message
        """
        content = message.content
        
        if "model_id" not in content or "prediction" not in content:
            logger.error("Missing model_id or prediction in record_prediction message")
            return self.create_message(
                message.sender_id,
                "record_prediction_nack",
                {
                    "error": "Missing model_id or prediction"
                }
            )
        
        # Record the prediction
        self.metrics.record_prediction(
            model_id=content["model_id"],
            prediction=content["prediction"],
            ground_truth=content.get("ground_truth"),
            confidence=content.get("confidence"),
            latency_ms=content.get("latency_ms"),
            features=content.get("features"),
            metadata=content.get("metadata")
        )
        
        return self.create_message(
            message.sender_id,
            "record_prediction_ack",
            {
                "model_id": content["model_id"],
                "status": "recorded"
            }
        )
    
    def _handle_record_error(self, message: Message) -> Optional[Message]:
        """
        Handle a message to record a model error.
        
        Args:
            message: Message containing error information
            
        Returns:
            Optional response message
        """
        content = message.content
        
        if "model_id" not in content or "error_type" not in content or "error_message" not in content:
            logger.error("Missing required fields in record_error message")
            return self.create_message(
                message.sender_id,
                "record_error_nack",
                {
                    "error": "Missing required fields"
                }
            )
        
        # Record the error
        self.metrics.record_error(
            model_id=content["model_id"],
            error_type=content["error_type"],
            error_message=content["error_message"],
            context=content.get("context")
        )
        
        return self.create_message(
            message.sender_id,
            "record_error_ack",
            {
                "model_id": content["model_id"],
                "status": "recorded"
            }
        )
    
    def _handle_get_model_performance(self, message: Message) -> Optional[Message]:
        """
        Handle a message to get model performance information.
        
        Args:
            message: Message requesting model performance
            
        Returns:
            Response message with model performance information
        """
        content = message.content
        model_id = content.get("model_id")
        timespan_hours = content.get("timespan_hours", 24)
        
        # Get performance metrics
        if model_id:
            # Get metrics for a specific model
            if model_id not in self.models and model_id not in self.metrics.get_all_models():
                return self.create_message(
                    message.sender_id,
                    "model_performance_nack",
                    {
                        "error": f"Unknown model ID: {model_id}"
                    }
                )
            
            metrics = self.metrics.get_metrics_summary(model_id, timedelta(hours=timespan_hours))
            model_info = self.models.get(model_id, {"id": model_id, "name": model_id})
            
            return self.create_message(
                message.sender_id,
                "model_performance_info",
                {
                    "model_id": model_id,
                    "model_info": model_info,
                    "metrics": metrics,
                    "timespan_hours": timespan_hours
                }
            )
        else:
            # Get metrics for all models
            all_metrics = {}
            for mid in self.metrics.get_all_models():
                all_metrics[mid] = self.metrics.get_metrics_summary(mid, timedelta(hours=timespan_hours))
            
            return self.create_message(
                message.sender_id,
                "model_performance_info",
                {
                    "models": all_metrics,
                    "timespan_hours": timespan_hours
                }
            )
