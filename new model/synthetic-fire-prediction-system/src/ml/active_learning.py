"""
Active Learning Framework for FLIR+SCD41 Fire Detection System.

This module implements an active learning loop that includes:
1. Feedback mechanism for continuous improvement
2. Uncertainty sampling for active learning
3. Model update pipeline without full retraining
4. Performance monitoring dashboard
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
import json
import sqlite3
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FeedbackMechanism:
    """
    Feedback mechanism for collecting and processing user/system feedback.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feedback mechanism.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feedback_storage = deque(maxlen=1000)  # Store last 1000 feedback items
        self.feedback_db_path = self.config.get('feedback_db_path', '/tmp/feedback.db')
        
        # Initialize feedback database
        self._initialize_feedback_db()
        
        logger.info("Feedback mechanism initialized")
    
    def _initialize_feedback_db(self):
        """Initialize feedback database."""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            # Create feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    prediction_id TEXT,
                    predicted_label INTEGER,
                    predicted_confidence REAL,
                    true_label INTEGER,
                    feedback_type TEXT,
                    feedback_data TEXT,
                    model_version TEXT
                )
            ''')
            
            # Create feedback statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_version TEXT,
                    total_feedback INTEGER,
                    correct_predictions INTEGER,
                    incorrect_predictions INTEGER,
                    accuracy REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Feedback database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing feedback database: {str(e)}")
    
    def collect_feedback(self, prediction_id: str, predicted_label: int, 
                        predicted_confidence: float, true_label: int,
                        feedback_type: str = 'manual', feedback_data: Optional[Dict] = None,
                        model_version: str = 'unknown') -> bool:
        """
        Collect feedback on a prediction.
        
        Args:
            prediction_id: Unique identifier for the prediction
            predicted_label: Predicted label from the model
            predicted_confidence: Confidence score of the prediction
            true_label: True/correct label
            feedback_type: Type of feedback ('manual', 'system', 'validation')
            feedback_data: Additional feedback data
            model_version: Version of the model that made the prediction
            
        Returns:
            True if feedback was successfully collected
        """
        try:
            feedback_item = {
                'timestamp': datetime.now().isoformat(),
                'prediction_id': prediction_id,
                'predicted_label': predicted_label,
                'predicted_confidence': predicted_confidence,
                'true_label': true_label,
                'feedback_type': feedback_type,
                'feedback_data': feedback_data or {},
                'model_version': model_version
            }
            
            # Store in memory
            self.feedback_storage.append(feedback_item)
            
            # Store in database
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback 
                (timestamp, prediction_id, predicted_label, predicted_confidence, 
                 true_label, feedback_type, feedback_data, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_item['timestamp'],
                feedback_item['prediction_id'],
                feedback_item['predicted_label'],
                feedback_item['predicted_confidence'],
                feedback_item['true_label'],
                feedback_item['feedback_type'],
                json.dumps(feedback_item['feedback_data']),
                feedback_item['model_version']
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Feedback collected for prediction {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
            return False
    
    def get_feedback_summary(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of collected feedback.
        
        Args:
            model_version: Optional model version to filter by
            
        Returns:
            Dictionary with feedback summary
        """
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            if model_version:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_feedback,
                        SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END) as correct_predictions,
                        SUM(CASE WHEN predicted_label != true_label THEN 1 ELSE 0 END) as incorrect_predictions
                    FROM feedback 
                    WHERE model_version = ?
                ''', (model_version,))
            else:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_feedback,
                        SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END) as correct_predictions,
                        SUM(CASE WHEN predicted_label != true_label THEN 1 ELSE 0 END) as incorrect_predictions
                    FROM feedback
                ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                total_feedback, correct_predictions, incorrect_predictions = result
                accuracy = correct_predictions / total_feedback if total_feedback > 0 else 0.0
                
                return {
                    'total_feedback': total_feedback,
                    'correct_predictions': correct_predictions,
                    'incorrect_predictions': incorrect_predictions,
                    'accuracy': accuracy,
                    'model_version': model_version
                }
            else:
                return {
                    'total_feedback': 0,
                    'correct_predictions': 0,
                    'incorrect_predictions': 0,
                    'accuracy': 0.0,
                    'model_version': model_version
                }
                
        except Exception as e:
            logger.error(f"Error getting feedback summary: {str(e)}")
            return {
                'total_feedback': 0,
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'accuracy': 0.0,
                'error': str(e)
            }

class UncertaintySampler:
    """
    Uncertainty sampling for active learning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize uncertainty sampler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sampling_strategy = self.config.get('sampling_strategy', 'margin')
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.2)
        
        logger.info("Uncertainty sampler initialized")
    
    def calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate uncertainty scores for predictions.
        
        Args:
            probabilities: Array of prediction probabilities (n_samples, n_classes)
            
        Returns:
            Array of uncertainty scores
        """
        if len(probabilities.shape) == 1:
            # Binary classification case
            uncertainty = 1 - np.abs(probabilities - 0.5) * 2
        else:
            # Multi-class case
            sorted_probs = np.sort(probabilities, axis=1)
            # Margin sampling: 1 - (p_max - p_second_max)
            if sorted_probs.shape[1] >= 2:
                uncertainty = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
            else:
                uncertainty = 1 - np.abs(sorted_probs[:, -1] - 0.5) * 2
        
        return uncertainty
    
    def select_samples_for_labeling(self, X: pd.DataFrame, probabilities: np.ndarray, 
                                  n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select samples for labeling based on uncertainty.
        
        Args:
            X: Feature data
            probabilities: Prediction probabilities
            n_samples: Number of samples to select
            
        Returns:
            Tuple of (selected_indices, uncertainty_scores)
        """
        # Calculate uncertainty scores
        uncertainty_scores = self.calculate_uncertainty(probabilities)
        
        # Select samples based on strategy
        if self.sampling_strategy == 'margin':
            # Select samples with lowest margin (highest uncertainty)
            selected_indices = np.argsort(uncertainty_scores)[-n_samples:]
        elif self.sampling_strategy == 'entropy':
            # Select samples with highest entropy
            if len(probabilities.shape) == 1:
                entropy = - (probabilities * np.log(probabilities + 1e-10) + 
                           (1 - probabilities) * np.log(1 - probabilities + 1e-10))
            else:
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            selected_indices = np.argsort(entropy)[-n_samples:]
        elif self.sampling_strategy == 'random':
            # Random selection (baseline)
            selected_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        else:
            # Default to margin sampling
            selected_indices = np.argsort(uncertainty_scores)[-n_samples:]
        
        return selected_indices, uncertainty_scores[selected_indices]

class ModelUpdatePipeline:
    """
    Model update pipeline for continuous improvement without full retraining.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model update pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.update_frequency = self.config.get('update_frequency', 100)  # Update after 100 feedback items
        self.min_update_samples = self.config.get('min_update_samples', 50)
        self.incremental_learning_enabled = self.config.get('incremental_learning_enabled', True)
        
        # Storage for new training data
        self.new_training_data = []
        self.new_training_labels = []
        
        logger.info("Model update pipeline initialized")
    
    def add_feedback_data(self, X_feedback: pd.DataFrame, y_feedback: pd.Series) -> bool:
        """
        Add feedback data to the update pipeline.
        
        Args:
            X_feedback: Feedback feature data
            y_feedback: Feedback labels
            
        Returns:
            True if data was successfully added
        """
        try:
            # Convert to numpy arrays
            X_np = X_feedback.values if hasattr(X_feedback, 'values') else np.array(X_feedback)
            y_np = y_feedback.values if hasattr(y_feedback, 'values') else np.array(y_feedback)
            
            # Store new data
            self.new_training_data.extend(X_np)
            self.new_training_labels.extend(y_np)
            
            logger.info(f"Added {len(X_np)} feedback samples to update pipeline")
            return True
            
        except Exception as e:
            logger.error(f"Error adding feedback data: {str(e)}")
            return False
    
    def should_update_model(self) -> bool:
        """
        Check if model should be updated based on accumulated feedback.
        
        Returns:
            True if model should be updated
        """
        return (len(self.new_training_data) >= self.update_frequency and 
                len(self.new_training_data) >= self.min_update_samples)
    
    def prepare_incremental_update(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for incremental model update.
        
        Returns:
            Tuple of (X_update, y_update) arrays
        """
        if len(self.new_training_data) == 0:
            return np.array([]), np.array([])
        
        X_update = np.array(self.new_training_data)
        y_update = np.array(self.new_training_labels)
        
        # Clear stored data after preparing update
        self.new_training_data.clear()
        self.new_training_labels.clear()
        
        return X_update, y_update
    
    def apply_model_update(self, model: Any, X_update: np.ndarray, y_update: np.ndarray) -> Any:
        """
        Apply incremental update to the model.
        
        Args:
            model: Current model
            X_update: New training data
            y_update: New training labels
            
        Returns:
            Updated model
        """
        if not self.incremental_learning_enabled:
            logger.warning("Incremental learning is disabled")
            return model
        
        try:
            # Check if model supports partial fit
            if hasattr(model, 'partial_fit'):
                # Use incremental learning
                model.partial_fit(X_update, y_update)
                logger.info(f"Applied incremental update with {len(X_update)} samples")
            else:
                # Fallback: Combine with existing data and retrain
                logger.warning("Model doesn't support incremental learning, using fallback approach")
                # This would typically involve combining with existing training data
                # and retraining, but we'll just log for now
                pass
                
            return model
            
        except Exception as e:
            logger.error(f"Error applying model update: {str(e)}")
            return model

class PerformanceMonitoringDashboard:
    """
    Performance monitoring dashboard for tracking model performance over time.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitoring dashboard.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.monitoring_window = self.config.get('monitoring_window', 1000)  # Track last 1000 predictions
        self.performance_history = deque(maxlen=self.monitoring_window)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.75,
            'f1_score': 0.75
        })
        
        logger.info("Performance monitoring dashboard initialized")
    
    def log_prediction_result(self, prediction_id: str, predicted_label: int, 
                            true_label: int, confidence: float, 
                            processing_time: float, model_version: str) -> bool:
        """
        Log prediction result for monitoring.
        
        Args:
            prediction_id: Unique identifier for the prediction
            predicted_label: Predicted label
            true_label: True label
            confidence: Prediction confidence
            processing_time: Time taken for prediction
            model_version: Model version
            
        Returns:
            True if result was successfully logged
        """
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'prediction_id': prediction_id,
                'predicted_label': predicted_label,
                'true_label': true_label,
                'confidence': confidence,
                'processing_time': processing_time,
                'model_version': model_version,
                'correct': predicted_label == true_label
            }
            
            self.performance_history.append(result)
            logger.info(f"Logged prediction result for {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging prediction result: {str(e)}")
            return False
    
    def get_performance_metrics(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate performance metrics over a time window.
        
        Args:
            window_size: Number of recent predictions to consider
            
        Returns:
            Dictionary with performance metrics
        """
        if len(self.performance_history) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'avg_confidence': 0.0,
                'avg_processing_time': 0.0,
                'total_predictions': 0
            }
        
        # Determine window
        if window_size is None:
            window_size = len(self.performance_history)
        
        recent_results = list(self.performance_history)[-window_size:]
        
        # Extract labels
        y_true = [r['true_label'] for r in recent_results]
        y_pred = [r['predicted_label'] for r in recent_results]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate averages
        avg_confidence = np.mean([r['confidence'] for r in recent_results])
        avg_processing_time = np.mean([r['processing_time'] for r in recent_results])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'total_predictions': len(recent_results)
        }
    
    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for performance alerts based on thresholds.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        metrics = self.get_performance_metrics()
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                if current_value < threshold:
                    alerts.append({
                        'type': 'performance_degradation',
                        'metric': metric_name,
                        'current_value': current_value,
                        'threshold': threshold,
                        'severity': 'high' if current_value < threshold * 0.8 else 'medium',
                        'timestamp': datetime.now().isoformat()
                    })
        
        return alerts
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with performance report
        """
        # Get metrics for different time windows
        recent_metrics = self.get_performance_metrics(100)  # Last 100 predictions
        medium_metrics = self.get_performance_metrics(500)  # Last 500 predictions
        all_metrics = self.get_performance_metrics()        # All available predictions
        
        # Check for alerts
        alerts = self.check_performance_alerts()
        
        # Generate trend analysis
        trend_analysis = self._analyze_performance_trends()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'recent_100': recent_metrics,
                'recent_500': medium_metrics,
                'all_time': all_metrics
            },
            'alerts': alerts,
            'trend_analysis': trend_analysis,
            'total_logged_predictions': len(self.performance_history)
        }
        
        return report
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.performance_history) < 20:
            return {'status': 'insufficient_data'}
        
        # Divide history into segments for trend analysis
        history_list = list(self.performance_history)
        segment_size = max(20, len(history_list) // 5)
        
        segments = []
        for i in range(0, len(history_list), segment_size):
            segment = history_list[i:i + segment_size]
            if len(segment) >= 10:  # Minimum segment size
                y_true = [r['true_label'] for r in segment]
                y_pred = [r['predicted_label'] for r in segment]
                accuracy = accuracy_score(y_true, y_pred)
                segments.append({
                    'start_index': i,
                    'end_index': i + len(segment) - 1,
                    'accuracy': accuracy,
                    'size': len(segment)
                })
        
        if len(segments) < 2:
            return {'status': 'insufficient_segments'}
        
        # Calculate trend
        accuracies = [s['accuracy'] for s in segments]
        trend_slope = np.polyfit(range(len(accuracies)), accuracies, 1)[0] if len(accuracies) > 1 else 0
        
        trend_direction = 'improving' if trend_slope > 0.01 else 'degrading' if trend_slope < -0.01 else 'stable'
        
        return {
            'status': 'analyzed',
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'segments_analyzed': len(segments),
            'latest_accuracy': accuracies[-1] if accuracies else 0.0,
            'oldest_accuracy': accuracies[0] if accuracies else 0.0
        }

class ActiveLearningLoop:
    """
    Complete active learning loop integrating all components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize active learning loop.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.feedback_mechanism = FeedbackMechanism(self.config.get('feedback_config'))
        self.uncertainty_sampler = UncertaintySampler(self.config.get('uncertainty_config'))
        self.model_update_pipeline = ModelUpdatePipeline(self.config.get('update_config'))
        self.performance_monitor = PerformanceMonitoringDashboard(self.config.get('monitoring_config'))
        
        # Model version tracking
        self.current_model_version = self.config.get('initial_model_version', 'v1.0')
        self.model_update_count = 0
        
        logger.info("Active learning loop initialized")
    
    def process_prediction_feedback(self, prediction_id: str, predicted_label: int,
                                  predicted_confidence: float, true_label: int,
                                  X_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Process feedback for a prediction and potentially trigger active learning.
        
        Args:
            prediction_id: Unique identifier for the prediction
            predicted_label: Predicted label from the model
            predicted_confidence: Confidence score of the prediction
            true_label: True/correct label
            X_sample: Feature data for the sample
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'feedback_collected': False,
            'sample_selected_for_labeling': False,
            'model_update_triggered': False,
            'performance_logged': False
        }
        
        # 1. Collect feedback
        feedback_success = self.feedback_mechanism.collect_feedback(
            prediction_id=prediction_id,
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            true_label=true_label,
            model_version=self.current_model_version
        )
        results['feedback_collected'] = feedback_success
        
        # 2. Log performance
        performance_success = self.performance_monitor.log_prediction_result(
            prediction_id=prediction_id,
            predicted_label=predicted_label,
            true_label=true_label,
            confidence=predicted_confidence,
            processing_time=0.0,  # This would come from actual timing
            model_version=self.current_model_version
        )
        results['performance_logged'] = performance_success
        
        # 3. Check if sample should be selected for labeling (uncertainty sampling)
        # This would typically be done in batches, but for demo we'll simulate
        uncertainty_score = 1 - abs(predicted_confidence - 0.5) * 2  # Simplified uncertainty
        if uncertainty_score > self.uncertainty_sampler.uncertainty_threshold:
            results['sample_selected_for_labeling'] = True
            logger.info(f"Sample {prediction_id} selected for labeling (uncertainty: {uncertainty_score:.3f})")
        
        # 4. Add to model update pipeline
        self.model_update_pipeline.add_feedback_data(X_sample, pd.Series([true_label]))
        
        # 5. Check if model should be updated
        if self.model_update_pipeline.should_update_model():
            X_update, y_update = self.model_update_pipeline.prepare_incremental_update()
            logger.info(f"Triggering model update with {len(X_update)} new samples")
            results['model_update_triggered'] = True
            results['update_samples'] = len(X_update)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current status of the active learning system.
        
        Returns:
            Dictionary with system status
        """
        feedback_summary = self.feedback_mechanism.get_feedback_summary(self.current_model_version)
        performance_metrics = self.performance_monitor.get_performance_metrics()
        alerts = self.performance_monitor.check_performance_alerts()
        
        return {
            'model_version': self.current_model_version,
            'model_updates': self.model_update_count,
            'feedback_summary': feedback_summary,
            'performance_metrics': performance_metrics,
            'active_alerts': len(alerts),
            'system_health': 'healthy' if len(alerts) == 0 else 'degraded',
            'total_logged_predictions': len(self.performance_monitor.performance_history)
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report of the active learning system.
        
        Returns:
            Dictionary with comprehensive report
        """
        system_status = self.get_system_status()
        performance_report = self.performance_monitor.generate_performance_report()
        feedback_summary = self.feedback_mechanism.get_feedback_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': system_status,
            'performance_report': performance_report,
            'feedback_summary': feedback_summary,
            'model_version': self.current_model_version
        }

# Convenience functions
def create_active_learning_system(config: Optional[Dict[str, Any]] = None) -> ActiveLearningLoop:
    """
    Create a complete active learning system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized active learning system
    """
    return ActiveLearningLoop(config)

def create_feedback_mechanism(config: Optional[Dict[str, Any]] = None) -> FeedbackMechanism:
    """
    Create feedback mechanism.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized feedback mechanism
    """
    return FeedbackMechanism(config)

def create_uncertainty_sampler(config: Optional[Dict[str, Any]] = None) -> UncertaintySampler:
    """
    Create uncertainty sampler.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized uncertainty sampler
    """
    return UncertaintySampler(config)

def create_model_update_pipeline(config: Optional[Dict[str, Any]] = None) -> ModelUpdatePipeline:
    """
    Create model update pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model update pipeline
    """
    return ModelUpdatePipeline(config)

def create_performance_monitor(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitoringDashboard:
    """
    Create performance monitoring dashboard.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized performance monitoring dashboard
    """
    return PerformanceMonitoringDashboard(config)

__all__ = [
    'ActiveLearningLoop',
    'FeedbackMechanism',
    'UncertaintySampler',
    'ModelUpdatePipeline',
    'PerformanceMonitoringDashboard',
    'create_active_learning_system',
    'create_feedback_mechanism',
    'create_uncertainty_sampler',
    'create_model_update_pipeline',
    'create_performance_monitor'
]