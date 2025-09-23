"""
Performance Tracker for FLIR+SCD41 Fire Detection System.

This module implements comprehensive performance monitoring, AUC score tracking,
and automated alerting for performance degradation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import json
import os

# Try to import sklearn for AUC calculation
try:
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Using simplified metrics.")

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.auc_score = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.false_positive_rate = 0.0
        self.false_negative_rate = 0.0
        self.processing_time = 0.0
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'auc_score': self.auc_score,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }

class PerformanceTracker:
    """Tracks and monitors system performance metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize performance tracker.
        
        Args:
            config: Configuration dictionary with performance thresholds
        """
        self.config = config or {}
        self.baseline_auc = self.config.get('baseline_auc', 0.7658)
        self.auc_threshold = self.config.get('auc_threshold', 0.85)
        self.performance_history = deque(maxlen=1000)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'auc_degradation': 0.05,
            'accuracy_drop': 0.05,
            'fpr_increase': 0.05,
            'processing_time_increase': 0.1
        })
        self.metrics_file = self.config.get('metrics_file', 'performance_metrics.json')
        
        # Load historical metrics if file exists
        self._load_metrics()
        
        logger.info(f"Performance Tracker initialized with baseline AUC: {self.baseline_auc}")
    
    def _load_metrics(self):
        """Load historical metrics from file."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    for metric_dict in data.get('metrics', []):
                        metrics = PerformanceMetrics()
                        metrics.auc_score = metric_dict.get('auc_score', 0.0)
                        metrics.accuracy = metric_dict.get('accuracy', 0.0)
                        metrics.precision = metric_dict.get('precision', 0.0)
                        metrics.recall = metric_dict.get('recall', 0.0)
                        metrics.f1_score = metric_dict.get('f1_score', 0.0)
                        metrics.false_positive_rate = metric_dict.get('false_positive_rate', 0.0)
                        metrics.false_negative_rate = metric_dict.get('false_negative_rate', 0.0)
                        metrics.processing_time = metric_dict.get('processing_time', 0.0)
                        timestamp_str = metric_dict.get('timestamp', datetime.now().isoformat())
                        metrics.timestamp = datetime.fromisoformat(timestamp_str)
                        self.performance_history.append(metrics)
                logger.info(f"Loaded {len(self.performance_history)} historical metrics")
            except Exception as e:
                logger.warning(f"Failed to load metrics from {self.metrics_file}: {e}")
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            data = {
                'metrics': [m.to_dict() for m in self.performance_history],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics to {self.metrics_file}: {e}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: Optional[np.ndarray] = None,
                         processing_time: float = 0.0) -> PerformanceMetrics:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC calculation)
            processing_time: Processing time in seconds
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        metrics.processing_time = processing_time
        
        try:
            # Calculate basic metrics
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred, zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, zero_division=0)
            metrics.f1_score = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate false positive and false negative rates
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            total_negatives = tn + fp
            total_positives = fn + tp
            
            metrics.false_positive_rate = fp / total_negatives if total_negatives > 0 else 0.0
            metrics.false_negative_rate = fn / total_positives if total_positives > 0 else 0.0
            
            # Calculate AUC score if probabilities are provided
            if y_proba is not None and SKLEARN_AVAILABLE:
                metrics.auc_score = roc_auc_score(y_true, y_proba)
            elif SKLEARN_AVAILABLE:
                # If no probabilities, use predicted labels (less accurate)
                metrics.auc_score = roc_auc_score(y_true, y_pred)
            else:
                # Fallback calculation
                metrics.auc_score = (metrics.recall + (1 - metrics.false_positive_rate)) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            # Set default values
            metrics.auc_score = 0.0
            metrics.accuracy = 0.0
            metrics.precision = 0.0
            metrics.recall = 0.0
            metrics.f1_score = 0.0
            metrics.false_positive_rate = 0.0
            metrics.false_negative_rate = 0.0
        
        # Add to history
        self.performance_history.append(metrics)
        
        # Save metrics
        self._save_metrics()
        
        return metrics
    
    def check_performance_degradation(self) -> Dict[str, Any]:
        """
        Check for performance degradation against baseline and recent history.
        
        Returns:
            Dictionary with degradation alerts and metrics
        """
        if len(self.performance_history) < 2:
            return {'status': 'insufficient_data', 'alerts': []}
        
        # Get latest metrics
        latest_metrics = self.performance_history[-1]
        
        # Get historical average (last 10 measurements)
        recent_metrics = list(self.performance_history)[-10:]
        avg_auc = np.mean([m.auc_score for m in recent_metrics])
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        avg_fpr = np.mean([m.false_positive_rate for m in recent_metrics])
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        
        alerts = []
        
        # Check AUC degradation
        if latest_metrics.auc_score < (self.baseline_auc - self.alert_thresholds['auc_degradation']):
            alerts.append({
                'type': 'auc_degradation',
                'severity': 'warning',
                'message': f"AUC score {latest_metrics.auc_score:.4f} below baseline threshold {self.baseline_auc - self.alert_thresholds['auc_degradation']:.4f}",
                'current': latest_metrics.auc_score,
                'baseline': self.baseline_auc,
                'threshold': self.baseline_auc - self.alert_thresholds['auc_degradation']
            })
        
        # Check accuracy drop
        if latest_metrics.accuracy < (avg_accuracy - self.alert_thresholds['accuracy_drop']):
            alerts.append({
                'type': 'accuracy_drop',
                'severity': 'warning',
                'message': f"Accuracy {latest_metrics.accuracy:.4f} below recent average {avg_accuracy:.4f}",
                'current': latest_metrics.accuracy,
                'average': avg_accuracy,
                'threshold': avg_accuracy - self.alert_thresholds['accuracy_drop']
            })
        
        # Check false positive rate increase
        if latest_metrics.false_positive_rate > (avg_fpr + self.alert_thresholds['fpr_increase']):
            alerts.append({
                'type': 'fpr_increase',
                'severity': 'warning',
                'message': f"False positive rate {latest_metrics.false_positive_rate:.4f} above recent average {avg_fpr:.4f}",
                'current': latest_metrics.false_positive_rate,
                'average': avg_fpr,
                'threshold': avg_fpr + self.alert_thresholds['fpr_increase']
            })
        
        # Check processing time increase
        if latest_metrics.processing_time > (avg_processing_time * (1 + self.alert_thresholds['processing_time_increase'])):
            alerts.append({
                'type': 'processing_time_increase',
                'severity': 'info',
                'message': f"Processing time {latest_metrics.processing_time:.4f}s above recent average {avg_processing_time:.4f}s",
                'current': latest_metrics.processing_time,
                'average': avg_processing_time,
                'threshold': avg_processing_time * (1 + self.alert_thresholds['processing_time_increase'])
            })
        
        # Check if AUC is below critical threshold
        if latest_metrics.auc_score < self.auc_threshold:
            alerts.append({
                'type': 'critical_auc',
                'severity': 'critical',
                'message': f"AUC score {latest_metrics.auc_score:.4f} below critical threshold {self.auc_threshold:.4f}",
                'current': latest_metrics.auc_score,
                'threshold': self.auc_threshold
            })
        
        status = 'degraded' if alerts else 'normal'
        
        return {
            'status': status,
            'latest_metrics': latest_metrics.to_dict(),
            'recent_average': {
                'auc_score': avg_auc,
                'accuracy': avg_accuracy,
                'false_positive_rate': avg_fpr,
                'processing_time': avg_processing_time
            },
            'alerts': alerts
        }
    
    def get_performance_trend(self, window_size: int = 30) -> Dict[str, Any]:
        """
        Get performance trend over a time window.
        
        Args:
            window_size: Number of recent measurements to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.performance_history) < window_size:
            return {'status': 'insufficient_data'}
        
        # Get recent metrics
        recent_metrics = list(self.performance_history)[-window_size:]
        
        # Calculate trends
        auc_scores = [m.auc_score for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics]
        fpr_rates = [m.false_positive_rate for m in recent_metrics]
        processing_times = [m.processing_time for m in recent_metrics]
        
        # Calculate slopes (simple linear regression)
        x = np.arange(len(auc_scores))
        
        auc_slope = np.polyfit(x, auc_scores, 1)[0] if len(auc_scores) > 1 else 0
        accuracy_slope = np.polyfit(x, accuracies, 1)[0] if len(accuracies) > 1 else 0
        fpr_slope = np.polyfit(x, fpr_rates, 1)[0] if len(fpr_rates) > 1 else 0
        time_slope = np.polyfit(x, processing_times, 1)[0] if len(processing_times) > 1 else 0
        
        # Determine trend directions
        auc_trend = 'improving' if auc_slope > 0.001 else 'degrading' if auc_slope < -0.001 else 'stable'
        accuracy_trend = 'improving' if accuracy_slope > 0.001 else 'degrading' if accuracy_slope < -0.001 else 'stable'
        fpr_trend = 'improving' if fpr_slope < -0.001 else 'degrading' if fpr_slope > 0.001 else 'stable'
        time_trend = 'improving' if time_slope < -0.001 else 'degrading' if time_slope > 0.001 else 'stable'
        
        return {
            'status': 'success',
            'window_size': window_size,
            'trends': {
                'auc': {
                    'slope': auc_slope,
                    'trend': auc_trend,
                    'current': auc_scores[-1],
                    'average': np.mean(auc_scores)
                },
                'accuracy': {
                    'slope': accuracy_slope,
                    'trend': accuracy_trend,
                    'current': accuracies[-1],
                    'average': np.mean(accuracies)
                },
                'false_positive_rate': {
                    'slope': fpr_slope,
                    'trend': fpr_trend,
                    'current': fpr_rates[-1],
                    'average': np.mean(fpr_rates)
                },
                'processing_time': {
                    'slope': time_slope,
                    'trend': time_trend,
                    'current': processing_times[-1],
                    'average': np.mean(processing_times)
                }
            }
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with performance report
        """
        if len(self.performance_history) == 0:
            return {'status': 'no_data'}
        
        # Get latest metrics
        latest = self.performance_history[-1]
        
        # Get historical statistics
        all_auc_scores = [m.auc_score for m in self.performance_history]
        all_accuracies = [m.accuracy for m in self.performance_history]
        all_fpr_rates = [m.false_positive_rate for m in self.performance_history]
        all_processing_times = [m.processing_time for m in self.performance_history]
        
        report = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_measurements': len(self.performance_history),
            'latest_metrics': latest.to_dict(),
            'historical_statistics': {
                'auc_score': {
                    'current': latest.auc_score,
                    'baseline': self.baseline_auc,
                    'min': np.min(all_auc_scores),
                    'max': np.max(all_auc_scores),
                    'mean': np.mean(all_auc_scores),
                    'std': np.std(all_auc_scores),
                    'improvement': latest.auc_score - self.baseline_auc
                },
                'accuracy': {
                    'current': latest.accuracy,
                    'min': np.min(all_accuracies),
                    'max': np.max(all_accuracies),
                    'mean': np.mean(all_accuracies),
                    'std': np.std(all_accuracies)
                },
                'false_positive_rate': {
                    'current': latest.false_positive_rate,
                    'min': np.min(all_fpr_rates),
                    'max': np.max(all_fpr_rates),
                    'mean': np.mean(all_fpr_rates),
                    'std': np.std(all_fpr_rates)
                },
                'processing_time': {
                    'current': latest.processing_time,
                    'min': np.min(all_processing_times),
                    'max': np.max(all_processing_times),
                    'mean': np.mean(all_processing_times),
                    'std': np.std(all_processing_times)
                }
            },
            'performance_status': self.check_performance_degradation(),
            'trend_analysis': self.get_performance_trend()
        }
        
        return report

# Convenience function
def create_performance_tracker(config: Dict[str, Any] = None) -> PerformanceTracker:
    """
    Create a performance tracker instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PerformanceTracker instance
    """
    return PerformanceTracker(config)

__all__ = ['PerformanceMetrics', 'PerformanceTracker', 'create_performance_tracker']