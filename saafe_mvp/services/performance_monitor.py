"""
Performance monitoring and metrics tracking system.

This module implements comprehensive system performance monitoring,
reliability metrics calculation, processing time tracking, and
diagnostic information collection for troubleshooting.
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from statistics import mean, median, stdev
import json
import os


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """System resource snapshot at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    active_threads: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_utilization_percent': self.gpu_utilization_percent,
            'active_threads': self.active_threads
        }


@dataclass
class ReliabilityMetrics:
    """System reliability and accuracy metrics."""
    uptime_seconds: float
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    average_processing_time: float
    prediction_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    system_availability: float
    error_rate: float
    
    @property
    def success_rate(self) -> float:
        """Calculate prediction success rate."""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'uptime_seconds': self.uptime_seconds,
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'success_rate': self.success_rate,
            'average_processing_time': self.average_processing_time,
            'prediction_accuracy': self.prediction_accuracy,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'system_availability': self.system_availability,
            'error_rate': self.error_rate
        }


class ProcessingTimeTracker:
    """Tracks processing times for different operations."""
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize processing time tracker.
        
        Args:
            max_samples (int): Maximum number of samples to keep in memory
        """
        self.max_samples = max_samples
        self.processing_times = defaultdict(lambda: deque(maxlen=max_samples))
        self.operation_counts = defaultdict(int)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def record_processing_time(self, operation: str, processing_time: float):
        """
        Record processing time for an operation.
        
        Args:
            operation (str): Operation name
            processing_time (float): Processing time in milliseconds
        """
        with self.lock:
            self.processing_times[operation].append(processing_time)
            self.operation_counts[operation] += 1
    
    def get_statistics(self, operation: str) -> Dict[str, float]:
        """
        Get processing time statistics for an operation.
        
        Args:
            operation (str): Operation name
            
        Returns:
            Dict[str, float]: Statistics including mean, median, std, min, max
        """
        with self.lock:
            times = list(self.processing_times[operation])
            
            if not times:
                return {
                    'count': 0,
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'p95': 0.0,
                    'p99': 0.0
                }
            
            sorted_times = sorted(times)
            n = len(sorted_times)
            
            return {
                'count': n,
                'mean': mean(times),
                'median': median(times),
                'std': stdev(times) if n > 1 else 0.0,
                'min': min(times),
                'max': max(times),
                'p95': sorted_times[int(0.95 * n)] if n > 0 else 0.0,
                'p99': sorted_times[int(0.99 * n)] if n > 0 else 0.0
            }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked operations."""
        return {op: self.get_statistics(op) for op in self.processing_times.keys()}


class SystemResourceMonitor:
    """Monitors system resource usage."""
    
    def __init__(self, sample_interval: float = 1.0, max_samples: int = 3600):
        """
        Initialize system resource monitor.
        
        Args:
            sample_interval (float): Sampling interval in seconds
            max_samples (int): Maximum number of samples to keep
        """
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.snapshots = deque(maxlen=max_samples)
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # GPU monitoring (optional)
        self.gpu_available = False
        try:
            import GPUtil
            self.gpu_available = True
            self.gputil = GPUtil
        except ImportError:
            self.logger.info("GPUtil not available - GPU monitoring disabled")
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System resource monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("System resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                with self.lock:
                    self.snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def _take_snapshot(self) -> SystemSnapshot:
        """Take a system resource snapshot."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU information (if available)
        gpu_memory_used = None
        gpu_utilization = None
        
        if self.gpu_available:
            try:
                gpus = self.gputil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_memory_used = gpu.memoryUsed
                    gpu_utilization = gpu.load * 100
            except Exception as e:
                self.logger.debug(f"GPU monitoring error: {e}")
        
        # Thread count
        active_threads = threading.active_count()
        
        return SystemSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_utilization_percent=gpu_utilization,
            active_threads=active_threads
        )
    
    def get_current_snapshot(self) -> SystemSnapshot:
        """Get current system snapshot."""
        return self._take_snapshot()
    
    def get_recent_snapshots(self, minutes: int = 5) -> List[SystemSnapshot]:
        """Get snapshots from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            return [
                snapshot for snapshot in self.snapshots
                if snapshot.timestamp >= cutoff_time
            ]
    
    def get_resource_statistics(self, minutes: int = 5) -> Dict[str, Dict[str, float]]:
        """Get resource usage statistics for the last N minutes."""
        snapshots = self.get_recent_snapshots(minutes)
        
        if not snapshots:
            return {}
        
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]
        
        stats = {
            'cpu': {
                'mean': mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'current': snapshots[-1].cpu_percent
            },
            'memory': {
                'mean': mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'current': snapshots[-1].memory_percent
            }
        }
        
        # Add GPU stats if available
        gpu_memory_values = [s.gpu_memory_used_mb for s in snapshots if s.gpu_memory_used_mb is not None]
        gpu_util_values = [s.gpu_utilization_percent for s in snapshots if s.gpu_utilization_percent is not None]
        
        if gpu_memory_values:
            stats['gpu_memory'] = {
                'mean': mean(gpu_memory_values),
                'max': max(gpu_memory_values),
                'min': min(gpu_memory_values),
                'current': snapshots[-1].gpu_memory_used_mb
            }
        
        if gpu_util_values:
            stats['gpu_utilization'] = {
                'mean': mean(gpu_util_values),
                'max': max(gpu_util_values),
                'min': min(gpu_util_values),
                'current': snapshots[-1].gpu_utilization_percent
            }
        
        return stats


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 log_file: str = "performance.log"):
        """
        Initialize performance monitor.
        
        Args:
            config (Dict[str, Any]): Configuration options
            log_file (str): Log file path
        """
        self.config = config or {}
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.processing_tracker = ProcessingTimeTracker()
        self.resource_monitor = SystemResourceMonitor()
        self.metrics_history = deque(maxlen=10000)
        
        # Reliability tracking
        self.start_time = datetime.now()
        self.prediction_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'processing_times': []
        }
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.last_errors = deque(maxlen=100)
        
        # Performance thresholds
        self.thresholds = {
            'max_processing_time': self.config.get('max_processing_time', 100.0),  # ms
            'max_memory_usage': self.config.get('max_memory_usage', 80.0),  # %
            'max_cpu_usage': self.config.get('max_cpu_usage', 90.0),  # %
            'min_accuracy': self.config.get('min_accuracy', 0.85)  # 85%
        }
        
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start all monitoring components."""
        self.resource_monitor.start_monitoring()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.resource_monitor.stop_monitoring()
        self.logger.info("Performance monitoring stopped")
    
    def record_prediction(self, 
                         processing_time: float,
                         success: bool = True,
                         accuracy: float = None,
                         error: str = None):
        """
        Record a prediction operation.
        
        Args:
            processing_time (float): Processing time in milliseconds
            success (bool): Whether prediction was successful
            accuracy (float): Prediction accuracy (if available)
            error (str): Error message (if failed)
        """
        with self.lock:
            self.prediction_stats['total'] += 1
            
            if success:
                self.prediction_stats['successful'] += 1
                self.processing_tracker.record_processing_time('prediction', processing_time)
                self.prediction_stats['processing_times'].append(processing_time)
                
                # Keep only recent processing times
                if len(self.prediction_stats['processing_times']) > 1000:
                    self.prediction_stats['processing_times'] = self.prediction_stats['processing_times'][-1000:]
            else:
                self.prediction_stats['failed'] += 1
                if error:
                    self.error_counts[error] += 1
                    self.last_errors.append({
                        'timestamp': datetime.now(),
                        'error': error,
                        'processing_time': processing_time
                    })
        
        # Record metric
        self.record_metric(
            name='prediction_processing_time',
            value=processing_time,
            unit='ms',
            category='performance'
        )
        
        # Check thresholds
        self._check_performance_thresholds(processing_time, success)
    
    def record_metric(self, 
                     name: str,
                     value: float,
                     unit: str = "",
                     category: str = "general",
                     metadata: Dict[str, Any] = None):
        """
        Record a custom performance metric.
        
        Args:
            name (str): Metric name
            value (float): Metric value
            unit (str): Unit of measurement
            category (str): Metric category
            metadata (Dict[str, Any]): Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics_history.append(metric)
    
    def get_reliability_metrics(self) -> ReliabilityMetrics:
        """Get current reliability metrics."""
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate averages
            avg_processing_time = 0.0
            if self.prediction_stats['processing_times']:
                avg_processing_time = mean(self.prediction_stats['processing_times'])
            
            # Calculate error rate
            total_errors = sum(self.error_counts.values())
            error_rate = total_errors / max(self.prediction_stats['total'], 1)
            
            # System availability (simplified - based on uptime and error rate)
            availability = max(0.0, 1.0 - error_rate)
            
            return ReliabilityMetrics(
                uptime_seconds=uptime,
                total_predictions=self.prediction_stats['total'],
                successful_predictions=self.prediction_stats['successful'],
                failed_predictions=self.prediction_stats['failed'],
                average_processing_time=avg_processing_time,
                prediction_accuracy=0.95,  # Placeholder - would need ground truth
                false_positive_rate=0.02,  # Placeholder
                false_negative_rate=0.01,  # Placeholder
                system_availability=availability,
                error_rate=error_rate
            )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_snapshot = self.resource_monitor.get_current_snapshot()
        reliability = self.get_reliability_metrics()
        
        return {
            'memory_usage_mb': current_snapshot.memory_usage_mb,
            'cpu_percent': current_snapshot.cpu_percent,
            'disk_usage_percent': current_snapshot.disk_usage_percent,
            'network_bytes_sent': current_snapshot.network_bytes_sent,
            'network_bytes_recv': current_snapshot.network_bytes_recv,
            'active_threads': current_snapshot.active_threads,
            'prediction_success_rate': reliability.success_rate,
            'avg_processing_time': reliability.avg_processing_time,
            'total_predictions': reliability.total_predictions,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        reliability = self.get_reliability_metrics()
        processing_stats = self.processing_tracker.get_all_statistics()
        resource_stats = self.resource_monitor.get_resource_statistics()
        current_snapshot = self.resource_monitor.get_current_snapshot()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'reliability': reliability.to_dict(),
            'processing_times': processing_stats,
            'resource_usage': resource_stats,
            'current_system': current_snapshot.to_dict(),
            'error_summary': dict(self.error_counts),
            'thresholds': self.thresholds,
            'alerts': self._get_current_alerts()
        }
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information for troubleshooting."""
        recent_errors = list(self.last_errors)[-10:]  # Last 10 errors
        
        diagnostic_info = {
            'system_info': {
                'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
                'platform': psutil.os.name,
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_available': self.resource_monitor.gpu_available
            },
            'recent_errors': [
                {
                    'timestamp': error['timestamp'].isoformat(),
                    'error': error['error'],
                    'processing_time': error['processing_time']
                }
                for error in recent_errors
            ],
            'error_frequency': dict(self.error_counts),
            'performance_issues': self._identify_performance_issues(),
            'resource_warnings': self._get_resource_warnings(),
            'recommendations': self._get_performance_recommendations()
        }
        
        return diagnostic_info
    
    def export_metrics(self, 
                      filepath: str,
                      format: str = 'json',
                      time_range_hours: int = 24) -> bool:
        """
        Export performance metrics to file.
        
        Args:
            filepath (str): Output file path
            format (str): Export format ('json', 'csv')
            time_range_hours (int): Time range to export
            
        Returns:
            bool: Success status
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            
            with self.lock:
                recent_metrics = [
                    metric for metric in self.metrics_history
                    if metric.timestamp >= cutoff_time
                ]
            
            if format.lower() == 'json':
                export_data = {
                    'export_time': datetime.now().isoformat(),
                    'time_range_hours': time_range_hours,
                    'performance_summary': self.get_performance_summary(),
                    'diagnostic_info': self.get_diagnostic_info(),
                    'metrics': [
                        {
                            'name': m.name,
                            'value': m.value,
                            'unit': m.unit,
                            'timestamp': m.timestamp.isoformat(),
                            'category': m.category,
                            'metadata': m.metadata
                        }
                        for m in recent_metrics
                    ]
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format.lower() == 'csv':
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'name', 'value', 'unit', 'category'])
                    
                    for metric in recent_metrics:
                        writer.writerow([
                            metric.timestamp.isoformat(),
                            metric.name,
                            metric.value,
                            metric.unit,
                            metric.category
                        ])
            
            self.logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def _check_performance_thresholds(self, processing_time: float, success: bool):
        """Check if performance thresholds are exceeded."""
        if processing_time > self.thresholds['max_processing_time']:
            self.logger.warning(f"Processing time threshold exceeded: {processing_time}ms")
        
        if not success:
            self.logger.warning("Prediction failed")
        
        # Check resource usage
        current_snapshot = self.resource_monitor.get_current_snapshot()
        if current_snapshot.memory_percent > self.thresholds['max_memory_usage']:
            self.logger.warning(f"Memory usage high: {current_snapshot.memory_percent}%")
        
        if current_snapshot.cpu_percent > self.thresholds['max_cpu_usage']:
            self.logger.warning(f"CPU usage high: {current_snapshot.cpu_percent}%")
    
    def _get_current_alerts(self) -> List[Dict[str, Any]]:
        """Get current performance alerts."""
        alerts = []
        current_snapshot = self.resource_monitor.get_current_snapshot()
        
        if current_snapshot.memory_percent > self.thresholds['max_memory_usage']:
            alerts.append({
                'type': 'memory_high',
                'message': f"Memory usage is {current_snapshot.memory_percent:.1f}%",
                'severity': 'warning'
            })
        
        if current_snapshot.cpu_percent > self.thresholds['max_cpu_usage']:
            alerts.append({
                'type': 'cpu_high',
                'message': f"CPU usage is {current_snapshot.cpu_percent:.1f}%",
                'severity': 'warning'
            })
        
        # Check recent processing times
        recent_stats = self.processing_tracker.get_statistics('prediction')
        if recent_stats['mean'] > self.thresholds['max_processing_time']:
            alerts.append({
                'type': 'processing_slow',
                'message': f"Average processing time is {recent_stats['mean']:.1f}ms",
                'severity': 'warning'
            })
        
        return alerts
    
    def _identify_performance_issues(self) -> List[str]:
        """Identify potential performance issues."""
        issues = []
        
        # Check error rate
        reliability = self.get_reliability_metrics()
        if reliability.error_rate > 0.05:  # 5% error rate
            issues.append(f"High error rate: {reliability.error_rate:.2%}")
        
        # Check processing time trends
        processing_stats = self.processing_tracker.get_statistics('prediction')
        if processing_stats['p95'] > self.thresholds['max_processing_time'] * 2:
            issues.append(f"95th percentile processing time is high: {processing_stats['p95']:.1f}ms")
        
        # Check resource usage trends
        resource_stats = self.resource_monitor.get_resource_statistics()
        if 'memory' in resource_stats and resource_stats['memory']['mean'] > 70:
            issues.append(f"Average memory usage is high: {resource_stats['memory']['mean']:.1f}%")
        
        return issues
    
    def _get_resource_warnings(self) -> List[str]:
        """Get resource usage warnings."""
        warnings = []
        current_snapshot = self.resource_monitor.get_current_snapshot()
        
        if current_snapshot.memory_percent > 85:
            warnings.append("Memory usage is critically high")
        
        if current_snapshot.disk_usage_percent > 90:
            warnings.append("Disk space is running low")
        
        if current_snapshot.active_threads > 50:
            warnings.append("High number of active threads")
        
        return warnings
    
    def _get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        # Analyze processing times
        processing_stats = self.processing_tracker.get_statistics('prediction')
        if processing_stats['mean'] > 50:  # 50ms average
            recommendations.append("Consider optimizing model inference or using GPU acceleration")
        
        # Analyze memory usage
        current_snapshot = self.resource_monitor.get_current_snapshot()
        if current_snapshot.memory_percent > 70:
            recommendations.append("Consider reducing batch size or optimizing memory usage")
        
        # Analyze error patterns
        if self.error_counts:
            most_common_error = max(self.error_counts.items(), key=lambda x: x[1])
            if most_common_error[1] > 10:
                recommendations.append(f"Address recurring error: {most_common_error[0]}")
        
        return recommendations