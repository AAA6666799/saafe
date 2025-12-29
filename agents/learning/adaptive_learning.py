"""
Adaptive Learning Agent for the multi-agent fire prediction system.

This agent continuously learns from system performance, analyzes errors,
and recommends improvements to enhance fire detection accuracy and reduce false alarms.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json

from ..base import LearningAgent, Message


class AdaptiveLearningAgent(LearningAgent):
    """
    Adaptive Learning Agent for continuous system improvement.
    
    This agent tracks system performance, analyzes errors and patterns,
    and provides recommendations for improving fire detection accuracy.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the Adaptive Learning Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
        """
        super().__init__(agent_id, config)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Learning configuration
        self.learning_window_size = config.get('learning_window_size', 1000)
        self.performance_threshold = config.get('performance_threshold', 0.85)
        self.error_analysis_interval = config.get('error_analysis_interval', 100)  # analyze every N predictions
        self.improvement_confidence_threshold = config.get('improvement_confidence_threshold', 0.7)
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.learning_window_size)
        self.error_database = []
        self.model_performance_by_type = defaultdict(list)
        self.false_alarm_patterns = []
        self.missed_detection_patterns = []
        
        # Learning metrics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.false_alarms = 0
        self.missed_detections = 0
        self.current_accuracy = 0.0
        self.accuracy_trend = 'stable'
        
        # Improvement recommendations tracking
        self.active_recommendations = {}
        self.implemented_improvements = []
        self.recommendation_success_rate = 0.0
        
        # Analysis components
        self.pattern_analyzer = PatternAnalyzer(config.get('pattern_config', {}))
        self.performance_predictor = PerformancePredictor(config.get('predictor_config', {}))
        self.improvement_generator = ImprovementGenerator(config.get('improvement_config', {}))
        
        self.logger.info(f"Initialized Adaptive Learning Agent: {agent_id}")
    
    def validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_keys = ['learning_window_size', 'performance_threshold']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config['learning_window_size'] <= 0:
            raise ValueError("learning_window_size must be positive")
        
        if not 0 < self.config['performance_threshold'] <= 1:
            raise ValueError("performance_threshold must be between 0 and 1")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process performance data and generate learning insights.
        
        Args:
            data: Input data containing prediction results and ground truth
            
        Returns:
            Dictionary containing learning analysis and recommendations
        """
        try:
            processing_timestamp = datetime.now()
            
            # Track performance metrics
            performance_metrics = self._extract_performance_metrics(data)
            self.track_performance(performance_metrics)
            
            # Analyze errors if present
            error_analysis = {}
            if 'prediction_errors' in data:
                error_analysis = self.analyze_errors(data['prediction_errors'])
            
            # Generate improvement recommendations
            improvement_recommendations = self.recommend_improvements(
                performance_metrics, 
                error_analysis
            )
            
            # Analyze learning trends
            learning_trends = self._analyze_learning_trends()
            
            # Predict future performance
            performance_prediction = self._predict_performance()
            
            # Compile learning results
            learning_result = {
                'timestamp': processing_timestamp.isoformat(),
                'agent_id': self.agent_id,
                'learning_cycle': len(self.performance_history),
                'performance_metrics': performance_metrics,
                'error_analysis': error_analysis,
                'improvement_recommendations': improvement_recommendations,
                'learning_trends': learning_trends,
                'performance_prediction': performance_prediction,
                'system_health': {
                    'accuracy': self.current_accuracy,
                    'accuracy_trend': self.accuracy_trend,
                    'false_alarm_rate': self.false_alarms / max(1, self.total_predictions),
                    'missed_detection_rate': self.missed_detections / max(1, self.total_predictions),
                    'recommendation_success_rate': self.recommendation_success_rate
                },
                'metadata': {
                    'processing_time_ms': (datetime.now() - processing_timestamp).total_seconds() * 1000,
                    'total_predictions_analyzed': self.total_predictions,
                    'learning_data_points': len(self.performance_history),
                    'active_recommendations': len(self.active_recommendations)
                }
            }
            
            # Update learning state
            self._update_learning_state(learning_result)
            
            self.logger.debug(f"Learning analysis complete: accuracy={self.current_accuracy:.3f}, trend={self.accuracy_trend}")
            
            return learning_result
            
        except Exception as e:
            self.logger.error(f"Error in learning analysis: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'agent_id': self.agent_id,
                'error': str(e),
                'learning_cycle': len(self.performance_history)
            }
    
    def track_performance(self, metrics: Dict[str, float]) -> None:
        """
        Track system performance metrics.
        
        Args:
            metrics: Performance metrics to track
        """
        # Store performance data
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'accuracy': metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0)
        }
        
        self.performance_history.append(performance_data)
        
        # Update running statistics
        self.total_predictions = metrics.get('total_predictions', self.total_predictions)
        self.correct_predictions = metrics.get('correct_predictions', self.correct_predictions)
        self.false_alarms = metrics.get('false_alarms', self.false_alarms)
        self.missed_detections = metrics.get('missed_detections', self.missed_detections)
        
        # Calculate current accuracy
        if self.total_predictions > 0:
            self.current_accuracy = self.correct_predictions / self.total_predictions
        
        # Determine accuracy trend
        self._update_accuracy_trend()
        
        # Track performance by model type
        model_type = metrics.get('model_type', 'unknown')
        self.model_performance_by_type[model_type].append(performance_data)
    
    def analyze_errors(self, error_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze system errors to identify patterns.
        
        Args:
            error_data: List of error data to analyze
            
        Returns:
            Dictionary containing error analysis results
        """
        if not error_data:
            return {'total_errors': 0, 'patterns': [], 'recommendations': []}
        
        # Store errors in database
        self.error_database.extend(error_data)
        
        # Keep database manageable
        if len(self.error_database) > 5000:
            self.error_database = self.error_database[-3000:]
        
        # Categorize errors
        false_alarms = [e for e in error_data if e.get('error_type') == 'false_alarm']
        missed_detections = [e for e in error_data if e.get('error_type') == 'missed_detection']
        system_errors = [e for e in error_data if e.get('error_type') == 'system_error']
        
        # Analyze false alarm patterns
        false_alarm_analysis = self._analyze_false_alarms(false_alarms)
        
        # Analyze missed detection patterns
        missed_detection_analysis = self._analyze_missed_detections(missed_detections)
        
        # Analyze system errors
        system_error_analysis = self._analyze_system_errors(system_errors)
        
        # Identify common error patterns
        error_patterns = self._identify_error_patterns(error_data)
        
        # Generate error-specific recommendations
        error_recommendations = self._generate_error_recommendations(
            false_alarm_analysis,
            missed_detection_analysis,
            system_error_analysis
        )
        
        return {
            'total_errors': len(error_data),
            'false_alarms': {
                'count': len(false_alarms),
                'analysis': false_alarm_analysis
            },
            'missed_detections': {
                'count': len(missed_detections),
                'analysis': missed_detection_analysis
            },
            'system_errors': {
                'count': len(system_errors),
                'analysis': system_error_analysis
            },
            'error_patterns': error_patterns,
            'error_recommendations': error_recommendations,
            'error_rate': len(error_data) / max(1, self.total_predictions),
            'critical_error_rate': len([e for e in error_data if e.get('severity') == 'critical']) / max(1, len(error_data))
        }
    
    def recommend_improvements(self, performance_data: Dict[str, Any], error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend system improvements based on performance and error analysis.
        
        Args:
            performance_data: Performance tracking data
            error_analysis: Error analysis results
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Performance-based recommendations
        current_accuracy = performance_data.get('accuracy', 0.0)
        
        if current_accuracy < self.performance_threshold:
            recommendations.append({
                'type': 'model_improvement',
                'priority': 'high',
                'title': 'Improve model accuracy',
                'description': f'Current accuracy ({current_accuracy:.2%}) is below threshold ({self.performance_threshold:.2%})',
                'suggested_actions': [
                    'Retrain models with recent data',
                    'Adjust model hyperparameters',
                    'Consider ensemble methods',
                    'Increase training data quality'
                ],
                'confidence': 0.9,
                'expected_impact': 'accuracy_improvement'
            })
        
        # False alarm reduction recommendations
        false_alarm_rate = self.false_alarms / max(1, self.total_predictions)
        if false_alarm_rate > 0.1:  # More than 10% false alarms
            recommendations.append({
                'type': 'false_alarm_reduction',
                'priority': 'medium',
                'title': 'Reduce false alarm rate',
                'description': f'False alarm rate ({false_alarm_rate:.2%}) is too high',
                'suggested_actions': [
                    'Adjust confidence thresholds',
                    'Improve sensor calibration',
                    'Add temporal correlation checks',
                    'Implement multi-sensor validation'
                ],
                'confidence': 0.8,
                'expected_impact': 'false_alarm_reduction'
            })
        
        # Data quality recommendations
        if error_analysis.get('system_errors', {}).get('count', 0) > 0:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'title': 'Improve data quality',
                'description': 'System errors detected in data processing',
                'suggested_actions': [
                    'Check sensor connectivity',
                    'Validate data preprocessing pipeline',
                    'Implement data quality monitoring',
                    'Add data validation checkpoints'
                ],
                'confidence': 0.7,
                'expected_impact': 'error_reduction'
            })
        
        # Temporal pattern recommendations
        if len(self.performance_history) > 10:
            recent_performance = [p['metrics'].get('accuracy', 0) for p in list(self.performance_history)[-10:]]
            if len(recent_performance) > 1 and statistics.stdev(recent_performance) > 0.1:
                recommendations.append({
                    'type': 'stability_improvement',
                    'priority': 'medium',
                    'title': 'Improve prediction stability',
                    'description': 'High variance in recent predictions detected',
                    'suggested_actions': [
                        'Implement prediction smoothing',
                        'Add temporal consistency checks',
                        'Review model regularization',
                        'Check for data drift'
                    ],
                    'confidence': 0.6,
                    'expected_impact': 'stability_improvement'
                })
        
        # Generate model-specific recommendations
        model_recommendations = self._generate_model_specific_recommendations()
        recommendations.extend(model_recommendations)
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda r: (
            {'high': 3, 'medium': 2, 'low': 1}.get(r['priority'], 0),
            r['confidence']
        ), reverse=True)
        
        # Update active recommendations
        for rec in recommendations:
            rec['id'] = f"rec_{len(self.active_recommendations)}_{datetime.now().timestamp()}"
            rec['created_at'] = datetime.now().isoformat()
            rec['status'] = 'active'
            self.active_recommendations[rec['id']] = rec
        
        return recommendations
    
    def _extract_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from input data."""
        metrics = {}
        
        # Direct metrics if available
        if 'performance_metrics' in data:
            return data['performance_metrics']
        
        # Calculate from prediction results
        if 'predictions' in data and 'ground_truth' in data:
            predictions = data['predictions']
            ground_truth = data['ground_truth']
            
            if len(predictions) == len(ground_truth):
                correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
                total = len(predictions)
                
                metrics['accuracy'] = correct / total if total > 0 else 0.0
                metrics['total_predictions'] = total
                metrics['correct_predictions'] = correct
        
        # Default metrics if not available
        metrics.setdefault('accuracy', 0.0)
        metrics.setdefault('precision', 0.0)
        metrics.setdefault('recall', 0.0)
        metrics.setdefault('f1_score', 0.0)
        
        return metrics
    
    def _update_accuracy_trend(self) -> None:
        """Update the accuracy trend based on recent performance."""
        if len(self.performance_history) < 5:
            self.accuracy_trend = 'insufficient_data'
            return
        
        recent_accuracies = [p['accuracy'] for p in list(self.performance_history)[-5:]]
        
        if len(recent_accuracies) < 2:
            self.accuracy_trend = 'stable'
            return
        
        # Calculate trend using linear regression slope
        x = list(range(len(recent_accuracies)))
        n = len(recent_accuracies)
        
        if n < 2:
            self.accuracy_trend = 'stable'
            return
        
        # Simple slope calculation
        slope = (recent_accuracies[-1] - recent_accuracies[0]) / (n - 1)
        
        if slope > 0.01:
            self.accuracy_trend = 'improving'
        elif slope < -0.01:
            self.accuracy_trend = 'declining'
        else:
            self.accuracy_trend = 'stable'
    
    def _analyze_learning_trends(self) -> Dict[str, Any]:
        """Analyze learning trends from historical data."""
        if len(self.performance_history) < 10:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        recent_data = list(self.performance_history)[-50:]  # Last 50 data points
        accuracies = [p['accuracy'] for p in recent_data]
        
        # Calculate moving averages
        window_size = min(10, len(accuracies) // 2)
        if window_size > 0:
            moving_avg = []
            for i in range(window_size, len(accuracies)):
                avg = statistics.mean(accuracies[i-window_size:i])
                moving_avg.append(avg)
        else:
            moving_avg = accuracies
        
        # Determine overall trend
        if len(moving_avg) >= 2:
            trend_slope = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
            
            if trend_slope > 0.001:
                trend = 'improving'
                confidence = min(0.9, abs(trend_slope) * 100)
            elif trend_slope < -0.001:
                trend = 'declining'
                confidence = min(0.9, abs(trend_slope) * 100)
            else:
                trend = 'stable'
                confidence = 0.7
        else:
            trend = 'stable'
            confidence = 0.5
        
        return {
            'trend': trend,
            'confidence': confidence,
            'recent_performance': moving_avg[-5:] if len(moving_avg) >= 5 else moving_avg,
            'performance_variance': statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        }
    
    def _predict_performance(self) -> Dict[str, Any]:
        """Predict future performance based on current trends."""
        if len(self.performance_history) < 20:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0,
                'predicted_accuracy': self.current_accuracy
            }
        
        recent_accuracies = [p['accuracy'] for p in list(self.performance_history)[-20:]]
        
        # Simple trend-based prediction
        if len(recent_accuracies) >= 2:
            trend = (recent_accuracies[-1] - recent_accuracies[0]) / len(recent_accuracies)
            predicted_accuracy = self.current_accuracy + (trend * 10)  # Project 10 steps ahead
            predicted_accuracy = max(0.0, min(1.0, predicted_accuracy))  # Clamp to valid range
            
            confidence = 0.7 if abs(trend) < 0.01 else min(0.9, 0.5 + abs(trend) * 10)
            
            if predicted_accuracy > self.current_accuracy * 1.05:
                prediction = 'improvement_expected'
            elif predicted_accuracy < self.current_accuracy * 0.95:
                prediction = 'decline_expected'
            else:
                prediction = 'stable_performance'
        else:
            predicted_accuracy = self.current_accuracy
            prediction = 'stable_performance'
            confidence = 0.5
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'predicted_accuracy': predicted_accuracy,
            'current_accuracy': self.current_accuracy
        }
    
    def _analyze_false_alarms(self, false_alarms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in false alarms."""
        if not false_alarms:
            return {'patterns': [], 'common_causes': []}
        
        # Analyze by time of day
        time_patterns = defaultdict(int)
        for fa in false_alarms:
            if 'timestamp' in fa:
                try:
                    dt = datetime.fromisoformat(fa['timestamp'])
                    hour = dt.hour
                    time_patterns[f"{hour:02d}:00"] += 1
                except:
                    pass
        
        # Analyze by sensor type
        sensor_patterns = defaultdict(int)
        for fa in false_alarms:
            sensor_type = fa.get('sensor_type', 'unknown')
            sensor_patterns[sensor_type] += 1
        
        return {
            'patterns': {
                'time_of_day': dict(time_patterns),
                'sensor_type': dict(sensor_patterns)
            },
            'common_causes': [
                'Sensor calibration drift',
                'Environmental interference',
                'Threshold settings too sensitive'
            ]
        }
    
    def _analyze_missed_detections(self, missed_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in missed detections."""
        if not missed_detections:
            return {'patterns': [], 'common_causes': []}
        
        # Similar analysis to false alarms but for missed detections
        return {
            'patterns': {'analysis_pending': True},
            'common_causes': [
                'Insufficient sensor sensitivity',
                'Model training gaps',
                'Unusual fire patterns not in training data'
            ]
        }
    
    def _analyze_system_errors(self, system_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze system errors."""
        if not system_errors:
            return {'error_types': {}, 'recommendations': []}
        
        error_types = defaultdict(int)
        for error in system_errors:
            error_type = error.get('error_category', 'unknown')
            error_types[error_type] += 1
        
        return {
            'error_types': dict(error_types),
            'recommendations': [
                'Implement better error handling',
                'Add system health monitoring',
                'Improve data validation'
            ]
        }
    
    def _identify_error_patterns(self, error_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify common patterns across all error types."""
        patterns = []
        
        # Example pattern detection - in real system would be more sophisticated
        if len(error_data) > 5:
            patterns.append({
                'pattern_type': 'high_error_frequency',
                'description': f'{len(error_data)} errors detected in recent period',
                'confidence': 0.8,
                'suggested_action': 'Investigate system stability'
            })
        
        return patterns
    
    def _generate_error_recommendations(self, fa_analysis: Dict[str, Any], 
                                      md_analysis: Dict[str, Any], 
                                      se_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        # False alarm recommendations
        if fa_analysis.get('patterns'):
            recommendations.append('Adjust detection thresholds to reduce false alarms')
            recommendations.append('Implement multi-sensor correlation for validation')
        
        # Missed detection recommendations
        if md_analysis.get('patterns'):
            recommendations.append('Increase sensor sensitivity in problematic areas')
            recommendations.append('Retrain models with additional edge cases')
        
        # System error recommendations
        if se_analysis.get('error_types'):
            recommendations.append('Implement robust error handling mechanisms')
            recommendations.append('Add comprehensive system health monitoring')
        
        return recommendations
    
    def _generate_model_specific_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations specific to model performance."""
        recommendations = []
        
        # Analyze performance by model type
        for model_type, performances in self.model_performance_by_type.items():
            if len(performances) > 5:
                recent_accuracy = statistics.mean([p['accuracy'] for p in performances[-5:]])
                
                if recent_accuracy < 0.8:
                    recommendations.append({
                        'type': 'model_specific',
                        'priority': 'medium',
                        'title': f'Improve {model_type} model performance',
                        'description': f'{model_type} model accuracy ({recent_accuracy:.2%}) needs improvement',
                        'suggested_actions': [
                            f'Retrain {model_type} model',
                            f'Adjust {model_type} hyperparameters',
                            f'Collect more training data for {model_type}'
                        ],
                        'confidence': 0.7,
                        'expected_impact': 'model_accuracy_improvement'
                    })
        
        return recommendations
    
    def _update_learning_state(self, learning_result: Dict[str, Any]) -> None:
        """Update learning state with new results."""
        # Update recommendation success tracking
        # In a real system, this would track which recommendations were implemented
        # and measure their effectiveness
        
        # Clean up old recommendations
        current_time = datetime.now()
        expired_recs = []
        for rec_id, rec in self.active_recommendations.items():
            rec_time = datetime.fromisoformat(rec['created_at'])
            if current_time - rec_time > timedelta(days=7):  # Expire after 7 days
                expired_recs.append(rec_id)
        
        for rec_id in expired_recs:
            del self.active_recommendations[rec_id]
    
    def default_message_handler(self, message: Message) -> Optional[Message]:
        """Handle unknown message types."""
        self.logger.warning(f"Received unknown message type: {message.message_type}")
        return None
    
    def create_message(self, receiver_id: str, message_type: str, content: Dict[str, Any], priority: int = 0) -> Message:
        """Create a new message to send to another agent."""
        return Message(self.agent_id, receiver_id, message_type, content, priority)
    
    def save_state(self, filepath: str) -> None:
        """Save agent state to file."""
        state_data = {
            'agent_id': self.agent_id,
            'config': self.config,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'false_alarms': self.false_alarms,
            'missed_detections': self.missed_detections,
            'current_accuracy': self.current_accuracy,
            'accuracy_trend': self.accuracy_trend,
            'active_recommendations': self.active_recommendations,
            'recommendation_success_rate': self.recommendation_success_rate,
            'recent_performance': list(self.performance_history)[-100:],  # Save recent performance only
            'recent_errors': self.error_database[-100:] if self.error_database else []
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
    
    def load_state(self, filepath: str) -> None:
        """Load agent state from file."""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.total_predictions = state_data.get('total_predictions', 0)
        self.correct_predictions = state_data.get('correct_predictions', 0)
        self.false_alarms = state_data.get('false_alarms', 0)
        self.missed_detections = state_data.get('missed_detections', 0)
        self.current_accuracy = state_data.get('current_accuracy', 0.0)
        self.accuracy_trend = state_data.get('accuracy_trend', 'stable')
        self.active_recommendations = state_data.get('active_recommendations', {})
        self.recommendation_success_rate = state_data.get('recommendation_success_rate', 0.0)
        
        # Restore performance history
        recent_performance = state_data.get('recent_performance', [])
        self.performance_history = deque(recent_performance, maxlen=self.learning_window_size)
        
        # Restore error database
        self.error_database = state_data.get('recent_errors', [])


# Helper classes for specialized learning components
class PatternAnalyzer:
    """Analyze patterns in system behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def analyze_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in data."""
        return {'patterns_detected': 0, 'analysis_confidence': 0.5}


class PerformancePredictor:
    """Predict future performance based on trends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def predict_performance(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict future performance."""
        return {'predicted_accuracy': 0.85, 'prediction_confidence': 0.6}


class ImprovementGenerator:
    """Generate specific improvement recommendations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def generate_improvements(self, performance_data: Dict[str, Any], error_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate improvement recommendations."""
        return []