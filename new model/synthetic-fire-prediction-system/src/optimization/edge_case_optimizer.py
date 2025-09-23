"""
Edge Case Optimization for FLIR+SCD41 Fire Detection System.

This module implements edge case optimization including:
1. Systematic identification of challenging scenarios
2. Generation of synthetic data for edge cases
3. Specialized handling for identified edge cases
4. Robustness testing framework
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class EdgeCaseIdentifier:
    """
    Systematic identification of challenging scenarios and edge cases.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize edge case identifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.edge_case_thresholds = self.config.get('edge_case_thresholds', {
            'low_confidence_threshold': 0.6,
            'high_uncertainty_threshold': 0.8,
            'prediction_variance_threshold': 0.3,
            'rare_pattern_threshold': 0.05
        })
        
        # Store identified edge cases
        self.identified_edge_cases = []
        
        logger.info("Edge case identifier initialized")
    
    def identify_model_edge_cases(self, X: pd.DataFrame, y_true: pd.Series, 
                                y_pred: pd.Series, y_proba: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify edge cases based on model predictions and performance.
        
        Args:
            X: Feature data
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            List of identified edge cases
        """
        edge_cases = []
        
        # Convert to numpy arrays if needed
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
            
        # Calculate prediction confidence
        if len(y_proba.shape) == 1:
            # Binary classification case
            confidence = np.abs(y_proba - 0.5) * 2
        else:
            # Multi-class case - use maximum probability
            confidence = np.max(y_proba, axis=1)
        
        # Calculate prediction correctness
        correct = (y_pred == y_true)
        
        # Identify low confidence predictions
        low_confidence_mask = confidence < self.edge_case_thresholds['low_confidence_threshold']
        
        # Identify incorrect predictions
        incorrect_mask = ~correct
        
        # Identify high uncertainty cases (incorrect + low confidence)
        high_uncertainty_mask = incorrect_mask & low_confidence_mask
        
        # Process identified edge cases
        for i in range(len(X)):
            case_features = {}
            
            # Check for different types of edge cases
            is_low_confidence = low_confidence_mask[i]
            is_incorrect = incorrect_mask[i]
            is_high_uncertainty = high_uncertainty_mask[i]
            
            if is_high_uncertainty or is_low_confidence or is_incorrect:
                # Extract feature values for this sample
                if isinstance(X, pd.DataFrame):
                    sample_features = X.iloc[i].to_dict()
                else:
                    sample_features = {f'feature_{j}': X[i, j] for j in range(X.shape[1])}
                
                edge_case = {
                    'index': i,
                    'timestamp': datetime.now().isoformat(),
                    'true_label': int(y_true[i]),
                    'predicted_label': int(y_pred[i]),
                    'confidence': float(confidence[i]),
                    'correct': bool(correct[i]),
                    'features': sample_features,
                    'edge_case_types': []
                }
                
                # Classify edge case types
                if is_high_uncertainty:
                    edge_case['edge_case_types'].append('high_uncertainty')
                if is_low_confidence:
                    edge_case['edge_case_types'].append('low_confidence')
                if is_incorrect:
                    edge_case['edge_case_types'].append('incorrect_prediction')
                
                # Add specific feature-based edge cases
                feature_edge_cases = self._identify_feature_edge_cases(sample_features)
                edge_case['edge_case_types'].extend(feature_edge_cases)
                
                edge_cases.append(edge_case)
        
        # Store identified edge cases
        self.identified_edge_cases.extend(edge_cases)
        
        logger.info(f"Identified {len(edge_cases)} edge cases")
        return edge_cases
    
    def _identify_feature_edge_cases(self, features: Dict[str, Any]) -> List[str]:
        """
        Identify edge cases based on feature values.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            List of edge case types
        """
        edge_case_types = []
        
        # Check for extreme thermal values
        if 't_max' in features:
            if features['t_max'] > 100:  # Very high temperature
                edge_case_types.append('extreme_high_temperature')
            elif features['t_max'] < 0:  # Very low temperature
                edge_case_types.append('extreme_low_temperature')
        
        if 't_hot_area_pct' in features:
            if features['t_hot_area_pct'] > 50:  # Very large hot area
                edge_case_types.append('large_hot_area')
            elif features['t_hot_area_pct'] < 0.1:  # Very small hot area
                edge_case_types.append('small_hot_area')
        
        # Check for extreme gas values
        if 'gas_val' in features:
            if features['gas_val'] > 5000:  # Very high gas concentration
                edge_case_types.append('extreme_high_gas')
            elif features['gas_val'] < 300:  # Very low gas concentration
                edge_case_types.append('extreme_low_gas')
        
        # Check for rapid changes
        if 'tproxy_delta' in features:
            if abs(features['tproxy_delta']) > 20:  # Rapid temperature change
                edge_case_types.append('rapid_temperature_change')
        
        if 'gas_delta' in features:
            if abs(features['gas_delta']) > 500:  # Rapid gas change
                edge_case_types.append('rapid_gas_change')
        
        # Check for unusual combinations
        if 't_max' in features and 'gas_val' in features:
            # High temperature but low gas (electrical fire)
            if features['t_max'] > 70 and features['gas_val'] < 600:
                edge_case_types.append('electrical_fire_pattern')
            # Low temperature but high gas (smoldering fire)
            elif features['t_max'] < 30 and features['gas_val'] > 1500:
                edge_case_types.append('smoldering_fire_pattern')
        
        return edge_case_types
    
    def get_edge_case_summary(self) -> Dict[str, Any]:
        """
        Get summary of identified edge cases.
        
        Returns:
            Dictionary with edge case summary
        """
        if not self.identified_edge_cases:
            return {
                'total_edge_cases': 0,
                'edge_case_types': {},
                'confidence_stats': {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
            }
        
        # Count edge case types
        type_counts = defaultdict(int)
        confidences = []
        
        for case in self.identified_edge_cases:
            for edge_type in case['edge_case_types']:
                type_counts[edge_type] += 1
            confidences.append(case['confidence'])
        
        # Calculate confidence statistics
        confidences = np.array(confidences)
        confidence_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
        
        return {
            'total_edge_cases': len(self.identified_edge_cases),
            'edge_case_types': dict(type_counts),
            'confidence_stats': confidence_stats
        }

class EdgeCaseDataGenerator:
    """
    Generate synthetic data for edge cases.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize edge case data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.base_feature_names = self.config.get('feature_names', [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel',
            'gas_val', 'gas_delta', 'gas_vel'
        ])
        
        logger.info("Edge case data generator initialized")
    
    def generate_edge_case_data(self, edge_case_types: List[str], 
                              n_samples: int = 100) -> pd.DataFrame:
        """
        Generate synthetic data for specific edge case types.
        
        Args:
            edge_case_types: List of edge case types to generate
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with generated edge case data
        """
        data = []
        
        for i in range(n_samples):
            sample = {}
            
            # Generate base features with edge case characteristics
            for edge_type in edge_case_types:
                if edge_type == 'extreme_high_temperature':
                    sample.update(self._generate_extreme_high_temperature())
                elif edge_type == 'extreme_low_temperature':
                    sample.update(self._generate_extreme_low_temperature())
                elif edge_type == 'large_hot_area':
                    sample.update(self._generate_large_hot_area())
                elif edge_type == 'small_hot_area':
                    sample.update(self._generate_small_hot_area())
                elif edge_type == 'extreme_high_gas':
                    sample.update(self._generate_extreme_high_gas())
                elif edge_type == 'extreme_low_gas':
                    sample.update(self._generate_extreme_low_gas())
                elif edge_type == 'rapid_temperature_change':
                    sample.update(self._generate_rapid_temperature_change())
                elif edge_type == 'rapid_gas_change':
                    sample.update(self._generate_rapid_gas_change())
                elif edge_type == 'electrical_fire_pattern':
                    sample.update(self._generate_electrical_fire_pattern())
                elif edge_type == 'smoldering_fire_pattern':
                    sample.update(self._generate_smoldering_fire_pattern())
                else:
                    # Generate normal baseline
                    sample.update(self._generate_baseline())
            
            # If no specific edge case was generated, create baseline
            if not sample:
                sample.update(self._generate_baseline())
            
            data.append(sample)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure all feature columns are present
        for feature in self.base_feature_names:
            if feature not in df.columns:
                df[feature] = np.random.normal(0, 1, len(df))
        
        # Add labels (this would typically be determined by domain experts)
        df['fire_detected'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
        
        logger.info(f"Generated {len(df)} edge case samples for types: {edge_case_types}")
        return df
    
    def _generate_extreme_high_temperature(self) -> Dict[str, float]:
        """Generate data with extreme high temperature."""
        return {
            't_mean': np.random.normal(60, 10),    # Very high mean temperature
            't_max': np.random.normal(120, 20),    # Very high max temperature
            't_p95': np.random.normal(100, 15),    # High 95th percentile
            't_hot_area_pct': np.random.exponential(30),  # Large hot area
            'tproxy_val': np.random.normal(110, 25)  # High proxy value
        }
    
    def _generate_extreme_low_temperature(self) -> Dict[str, float]:
        """Generate data with extreme low temperature."""
        return {
            't_mean': np.random.normal(-5, 3),     # Very low mean temperature
            't_max': np.random.normal(5, 5),       # Low max temperature
            't_p95': np.random.normal(2, 3),       # Low 95th percentile
            'tproxy_val': np.random.normal(0, 5)   # Low proxy value
        }
    
    def _generate_large_hot_area(self) -> Dict[str, float]:
        """Generate data with large hot area."""
        return {
            't_hot_area_pct': np.random.exponential(40),  # Very large hot area
            't_hot_largest_blob_pct': np.random.exponential(25),  # Large blob
            't_grad_mean': np.random.exponential(8)       # High gradients
        }
    
    def _generate_small_hot_area(self) -> Dict[str, float]:
        """Generate data with small hot area."""
        return {
            't_hot_area_pct': np.random.exponential(0.5),  # Very small hot area
            't_hot_largest_blob_pct': np.random.exponential(0.2),  # Small blob
            't_grad_mean': np.random.exponential(0.5)      # Low gradients
        }
    
    def _generate_extreme_high_gas(self) -> Dict[str, float]:
        """Generate data with extreme high gas concentration."""
        return {
            'gas_val': np.random.normal(3000, 500),  # Very high gas value
            'gas_delta': np.random.normal(800, 200), # Large gas change
            'gas_vel': np.random.normal(400, 100)    # High gas velocity
        }
    
    def _generate_extreme_low_gas(self) -> Dict[str, float]:
        """Generate data with extreme low gas concentration."""
        return {
            'gas_val': np.random.normal(200, 50),    # Very low gas value
            'gas_delta': np.random.normal(10, 20),   # Small gas change
            'gas_vel': np.random.normal(5, 10)       # Low gas velocity
        }
    
    def _generate_rapid_temperature_change(self) -> Dict[str, float]:
        """Generate data with rapid temperature change."""
        return {
            'tproxy_delta': np.random.normal(0, 30),  # Large temperature delta
            'tproxy_vel': np.random.normal(0, 15),    # High temperature velocity
            't_diff_mean': np.random.normal(0, 8),    # Large temperature differences
            't_diff_std': np.random.normal(0, 4)      # High temperature diff std
        }
    
    def _generate_rapid_gas_change(self) -> Dict[str, float]:
        """Generate data with rapid gas change."""
        return {
            'gas_delta': np.random.normal(0, 600),    # Large gas delta
            'gas_vel': np.random.normal(0, 300)       # High gas velocity
        }
    
    def _generate_electrical_fire_pattern(self) -> Dict[str, float]:
        """Generate data with electrical fire pattern (high temp, low gas)."""
        return {
            't_mean': np.random.normal(50, 15),      # High temperature
            't_max': np.random.normal(90, 25),       # Very high max temp
            't_p95': np.random.normal(75, 20),       # High 95th percentile
            'gas_val': np.random.normal(400, 100),   # Normal/low gas
            'gas_delta': np.random.normal(20, 50),   # Small gas change
            't_grad_mean': np.random.exponential(6)  # High gradients
        }
    
    def _generate_smoldering_fire_pattern(self) -> Dict[str, float]:
        """Generate data with smoldering fire pattern (low temp, high gas)."""
        return {
            't_mean': np.random.normal(25, 5),       # Normal/low temperature
            't_max': np.random.normal(35, 10),       # Low max temp
            't_p95': np.random.normal(30, 8),        # Low 95th percentile
            'gas_val': np.random.normal(2000, 400),  # High gas concentration
            'gas_delta': np.random.normal(500, 200), # Large gas change
            't_grad_mean': np.random.exponential(2)  # Lower gradients
        }
    
    def _generate_baseline(self) -> Dict[str, float]:
        """Generate baseline data."""
        return {
            't_mean': np.random.normal(22, 5),
            't_std': np.random.lognormal(0.8, 0.3),
            't_max': np.random.normal(25, 10),
            't_p95': np.random.normal(24, 8),
            't_hot_area_pct': np.random.exponential(2),
            't_hot_largest_blob_pct': np.random.exponential(1),
            't_grad_mean': np.random.exponential(1),
            't_grad_std': np.random.lognormal(0.3, 0.1),
            't_diff_mean': np.random.normal(0.5, 0.2),
            't_diff_std': np.random.lognormal(0.1, 0.05),
            'flow_mag_mean': np.random.exponential(0.5),
            'flow_mag_std': np.random.lognormal(0.1, 0.05),
            'tproxy_val': np.random.normal(25, 5),
            'tproxy_delta': np.random.normal(0.5, 0.2),
            'tproxy_vel': np.random.normal(0.2, 0.1),
            'gas_val': np.random.normal(450, 100),
            'gas_delta': np.random.normal(0, 20),
            'gas_vel': np.random.normal(0, 10)
        }

class EdgeCaseHandler:
    """
    Specialized handling for identified edge cases.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize edge case handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.edge_case_rules = self.config.get('edge_case_rules', {})
        self.specialized_models = {}
        
        logger.info("Edge case handler initialized")
    
    def register_specialized_model(self, edge_case_type: str, model: Any) -> bool:
        """
        Register a specialized model for a specific edge case type.
        
        Args:
            edge_case_type: Type of edge case
            model: Specialized model for handling this edge case
            
        Returns:
            True if model was successfully registered
        """
        self.specialized_models[edge_case_type] = model
        logger.info(f"Registered specialized model for edge case type: {edge_case_type}")
        return True
    
    def handle_edge_case(self, sample: pd.DataFrame, edge_case_types: List[str]) -> Dict[str, Any]:
        """
        Handle a sample with identified edge cases.
        
        Args:
            sample: DataFrame with sample data
            edge_case_types: List of identified edge case types
            
        Returns:
            Dictionary with handling results
        """
        results = {
            'original_prediction': None,
            'specialized_predictions': {},
            'final_prediction': None,
            'confidence': 0.0,
            'handling_method': 'default'
        }
        
        # Check if we have specialized models for any of the edge case types
        applicable_models = {}
        for edge_type in edge_case_types:
            if edge_type in self.specialized_models:
                applicable_models[edge_type] = self.specialized_models[edge_type]
        
        # If we have specialized models, use them
        if applicable_models:
            results['handling_method'] = 'specialized_models'
            
            # Get predictions from all applicable specialized models
            for edge_type, model in applicable_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(sample)
                        if len(pred_proba.shape) == 1:
                            pred = int(pred_proba > 0.5)
                            conf = float(abs(pred_proba - 0.5) * 2)
                        else:
                            pred = int(np.argmax(pred_proba))
                            conf = float(np.max(pred_proba))
                    else:
                        pred = int(model.predict(sample)[0])
                        conf = 0.8  # Default confidence for models without probability
                    
                    results['specialized_predictions'][edge_type] = {
                        'prediction': pred,
                        'confidence': conf
                    }
                except Exception as e:
                    logger.warning(f"Error using specialized model for {edge_type}: {str(e)}")
            
            # Combine specialized predictions (simple voting)
            if results['specialized_predictions']:
                predictions = [p['prediction'] for p in results['specialized_predictions'].values()]
                confidences = [p['confidence'] for p in results['specialized_predictions'].values()]
                
                # Weighted voting
                weighted_sum = sum(p * c for p, c in zip(predictions, confidences))
                total_weight = sum(confidences)
                
                if total_weight > 0:
                    final_pred = int(weighted_sum / total_weight > 0.5)
                    final_conf = weighted_sum / total_weight
                else:
                    final_pred = int(np.mean(predictions) > 0.5)
                    final_conf = np.mean(confidences)
                
                results['final_prediction'] = final_pred
                results['confidence'] = float(final_conf)
            else:
                results['handling_method'] = 'fallback'
        else:
            # Use default handling (no specialized models available)
            results['handling_method'] = 'default'
        
        return results

class RobustnessTestingFramework:
    """
    Robustness testing framework for edge case validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize robustness testing framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.test_results = []
        self.performance_thresholds = self.config.get('performance_thresholds', {
            'min_accuracy': 0.8,
            'min_precision': 0.75,
            'min_recall': 0.75,
            'max_false_positive_rate': 0.15
        })
        
        logger.info("Robustness testing framework initialized")
    
    def run_edge_case_tests(self, model: Any, edge_case_data: pd.DataFrame, 
                          edge_case_labels: pd.Series) -> Dict[str, Any]:
        """
        Run comprehensive edge case tests.
        
        Args:
            model: Model to test
            edge_case_data: Edge case test data
            edge_case_labels: True labels for edge case data
            
        Returns:
            Dictionary with test results
        """
        logger.info("Running edge case robustness tests")
        
        try:
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(edge_case_data)
                if len(y_proba.shape) == 1:
                    y_pred = (y_proba > 0.5).astype(int)
                else:
                    y_pred = np.argmax(y_proba, axis=1)
            else:
                y_pred = model.predict(edge_case_data)
                y_proba = None
            
            # Convert to numpy arrays
            y_true = edge_case_labels.values if hasattr(edge_case_labels, 'values') else np.array(edge_case_labels)
            y_pred = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate per-class metrics
            if len(np.unique(y_true)) > 1:
                class_report = classification_report(y_true, y_pred, output_dict=True)
            else:
                class_report = {}
            
            # Check performance thresholds
            performance_checks = {
                'accuracy_meets_threshold': accuracy >= self.performance_thresholds['min_accuracy'],
                'precision_meets_threshold': precision >= self.performance_thresholds['min_precision'],
                'recall_meets_threshold': recall >= self.performance_thresholds['min_recall']
            }
            
            # Calculate false positive rate
            if len(cm) >= 2:
                fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
                performance_checks['fpr_within_limit'] = fp_rate <= self.performance_thresholds['max_false_positive_rate']
            else:
                fp_rate = 0
                performance_checks['fpr_within_limit'] = True
            
            # Overall robustness score
            passed_checks = sum(1 for check in performance_checks.values() if check)
            total_checks = len(performance_checks)
            robustness_score = passed_checks / total_checks if total_checks > 0 else 0
            
            test_result = {
                'timestamp': datetime.now().isoformat(),
                'test_data_size': len(edge_case_data),
                'metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'false_positive_rate': float(fp_rate)
                },
                'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
                'class_report': class_report,
                'performance_checks': performance_checks,
                'robustness_score': float(robustness_score),
                'overall_status': 'passed' if robustness_score >= 0.8 else 'failed'
            }
            
            # Store test result
            self.test_results.append(test_result)
            
            logger.info(f"Edge case tests completed. Robustness score: {robustness_score:.2f}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error running edge case tests: {str(e)}")
            return {
                'error': str(e),
                'overall_status': 'error'
            }
    
    def generate_robustness_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive robustness report.
        
        Returns:
            Dictionary with robustness report
        """
        if not self.test_results:
            return {
                'status': 'no_tests_run',
                'message': 'No robustness tests have been run yet'
            }
        
        # Aggregate results from all tests
        all_metrics = []
        all_scores = []
        
        for result in self.test_results:
            if 'metrics' in result:
                all_metrics.append(result['metrics'])
            if 'robustness_score' in result:
                all_scores.append(result['robustness_score'])
        
        if not all_metrics:
            return {
                'status': 'no_valid_results',
                'message': 'No valid test results available'
            }
        
        # Calculate aggregate metrics
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            avg_metrics[metric_name] = float(np.mean([m[metric_name] for m in all_metrics]))
        
        avg_robustness_score = float(np.mean(all_scores)) if all_scores else 0.0
        std_robustness_score = float(np.std(all_scores)) if all_scores else 0.0
        
        # Determine overall system robustness
        if avg_robustness_score >= 0.9:
            system_robustness = 'excellent'
        elif avg_robustness_score >= 0.8:
            system_robustness = 'good'
        elif avg_robustness_score >= 0.7:
            system_robustness = 'acceptable'
        else:
            system_robustness = 'needs_improvement'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.test_results),
            'average_metrics': avg_metrics,
            'robustness_statistics': {
                'mean_score': avg_robustness_score,
                'std_score': std_robustness_score,
                'min_score': float(np.min(all_scores)) if all_scores else 0.0,
                'max_score': float(np.max(all_scores)) if all_scores else 0.0
            },
            'system_robustness': system_robustness,
            'recommendations': self._generate_recommendations(avg_metrics, avg_robustness_score)
        }
        
        return report
    
    def _generate_recommendations(self, avg_metrics: Dict[str, float], 
                                robustness_score: float) -> List[str]:
        """
        Generate recommendations based on test results.
        
        Args:
            avg_metrics: Average metrics from tests
            robustness_score: Average robustness score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check individual metrics
        if avg_metrics.get('accuracy', 0) < self.performance_thresholds['min_accuracy']:
            recommendations.append("Improve overall model accuracy for edge cases")
        
        if avg_metrics.get('precision', 0) < self.performance_thresholds['min_precision']:
            recommendations.append("Reduce false positive rate in edge case scenarios")
        
        if avg_metrics.get('recall', 0) < self.performance_thresholds['min_recall']:
            recommendations.append("Improve detection rate for edge case fire events")
        
        if avg_metrics.get('false_positive_rate', 0) > self.performance_thresholds['max_false_positive_rate']:
            recommendations.append("Implement additional filtering for false positives")
        
        # Check robustness score
        if robustness_score < 0.8:
            recommendations.append("Develop specialized models for identified edge cases")
            recommendations.append("Expand edge case training data")
            recommendations.append("Implement ensemble methods for improved robustness")
        elif robustness_score < 0.9:
            recommendations.append("Continue monitoring edge case performance")
            recommendations.append("Consider active learning for edge case improvement")
        
        if not recommendations:
            recommendations.append("System robustness is excellent - continue monitoring")
        
        return recommendations

class EdgeCaseOptimizer:
    """
    Complete edge case optimization system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize edge case optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.identifier = EdgeCaseIdentifier(self.config.get('identifier_config'))
        self.generator = EdgeCaseDataGenerator(self.config.get('generator_config'))
        self.handler = EdgeCaseHandler(self.config.get('handler_config'))
        self.robustness_tester = RobustnessTestingFramework(self.config.get('tester_config'))
        
        logger.info("Edge case optimizer initialized")
    
    def optimize_for_edge_cases(self, model: Any, X_train: pd.DataFrame, 
                              y_train: pd.Series, X_val: pd.DataFrame, 
                              y_val: pd.Series) -> Dict[str, Any]:
        """
        Complete edge case optimization workflow.
        
        Args:
            model: Model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting edge case optimization workflow")
        
        results = {
            'edge_case_identification': {},
            'data_generation': {},
            'model_enhancement': {},
            'robustness_testing': {}
        }
        
        try:
            # Step 1: Identify edge cases in validation data
            logger.info("Step 1: Identifying edge cases in validation data")
            
            # Get model predictions on validation data
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)
                if len(y_val_proba.shape) == 1:
                    y_val_pred = (y_val_proba > 0.5).astype(int)
                else:
                    y_val_pred = np.argmax(y_val_proba, axis=1)
            else:
                y_val_pred = model.predict(X_val)
                y_val_proba = None
            
            # Identify edge cases
            edge_cases = self.identifier.identify_model_edge_cases(
                X_val, y_val, y_val_pred, y_val_proba
            )
            
            results['edge_case_identification'] = {
                'identified_cases': len(edge_cases),
                'summary': self.identifier.get_edge_case_summary()
            }
            
            # Step 2: Generate synthetic data for edge cases
            logger.info("Step 2: Generating synthetic data for edge cases")
            
            # Get unique edge case types
            all_edge_types = set()
            for case in edge_cases:
                all_edge_types.update(case['edge_case_types'])
            
            if all_edge_types:
                # Generate data for each edge case type
                generated_datasets = {}
                for edge_type in list(all_edge_types)[:5]:  # Limit to top 5 types
                    generated_data = self.generator.generate_edge_case_data(
                        [edge_type], n_samples=200
                    )
                    generated_datasets[edge_type] = generated_data
                
                results['data_generation'] = {
                    'generated_datasets': len(generated_datasets),
                    'total_samples': sum(len(df) for df in generated_datasets.values())
                }
                
                # Combine generated data with training data for model enhancement
                if generated_datasets:
                    # Combine all generated data
                    all_generated_data = pd.concat(list(generated_datasets.values()), ignore_index=True)
                    
                    # Combine with original training data
                    enhanced_X = pd.concat([X_train, all_generated_data.drop('fire_detected', axis=1)], ignore_index=True)
                    enhanced_y = pd.concat([y_train, all_generated_data['fire_detected']], ignore_index=True)
                    
                    results['model_enhancement'] = {
                        'original_training_size': len(X_train),
                        'enhanced_training_size': len(enhanced_X),
                        'additional_samples': len(all_generated_data)
                    }
            
            # Step 3: Test robustness with edge cases
            logger.info("Step 3: Testing robustness with edge cases")
            
            # Create edge case test dataset
            edge_case_samples = []
            edge_case_labels = []
            
            for case in edge_cases[:100]:  # Limit to first 100 edge cases
                edge_case_samples.append(case['features'])
                edge_case_labels.append(case['true_label'])
            
            if edge_case_samples:
                # Convert to DataFrame
                edge_case_df = pd.DataFrame(edge_case_samples)
                edge_case_labels_series = pd.Series(edge_case_labels)
                
                # Run robustness tests
                robustness_results = self.robustness_tester.run_edge_case_tests(
                    model, edge_case_df, edge_case_labels_series
                )
                
                results['robustness_testing'] = robustness_results
            
            # Step 4: Generate final report
            logger.info("Step 4: Generating optimization report")
            
            robustness_report = self.robustness_tester.generate_robustness_report()
            results['final_report'] = robustness_report
            
            logger.info("Edge case optimization workflow completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in edge case optimization: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed'
            }

# Convenience functions
def create_edge_case_optimizer(config: Optional[Dict[str, Any]] = None) -> EdgeCaseOptimizer:
    """
    Create a complete edge case optimization system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized edge case optimizer
    """
    return EdgeCaseOptimizer(config)

def create_edge_case_identifier(config: Optional[Dict[str, Any]] = None) -> EdgeCaseIdentifier:
    """
    Create edge case identifier.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized edge case identifier
    """
    return EdgeCaseIdentifier(config)

def create_edge_case_generator(config: Optional[Dict[str, Any]] = None) -> EdgeCaseDataGenerator:
    """
    Create edge case data generator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized edge case data generator
    """
    return EdgeCaseDataGenerator(config)

def create_edge_case_handler(config: Optional[Dict[str, Any]] = None) -> EdgeCaseHandler:
    """
    Create edge case handler.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized edge case handler
    """
    return EdgeCaseHandler(config)

def create_robustness_tester(config: Optional[Dict[str, Any]] = None) -> RobustnessTestingFramework:
    """
    Create robustness testing framework.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized robustness testing framework
    """
    return RobustnessTestingFramework(config)

__all__ = [
    'EdgeCaseOptimizer',
    'EdgeCaseIdentifier',
    'EdgeCaseDataGenerator',
    'EdgeCaseHandler',
    'RobustnessTestingFramework',
    'create_edge_case_optimizer',
    'create_edge_case_identifier',
    'create_edge_case_generator',
    'create_edge_case_handler',
    'create_robustness_tester'
]