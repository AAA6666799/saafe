#!/usr/bin/env python3
"""
Test script for Task 8: Active Learning Loop.
This script tests feedback mechanism, uncertainty sampling, model update pipeline, and performance monitoring.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import tempfile
import shutil

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(n_samples=1000, n_features=18):
    """Generate synthetic test data for active learning."""
    logger.info(f"Generating test data: {n_samples} samples, {n_features} features")
    
    np.random.seed(42)
    
    # Generate base features (similar to FLIR+SCD41 features)
    feature_names = [
        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
        'gas_val', 'gas_delta', 'gas_vel'
    ][:n_features]
    
    # Create data with some patterns
    data = []
    labels = []
    
    for i in range(n_samples):
        # Base values with some temporal correlation
        base_temp = 20 + 5 * np.sin(i * 0.1) + np.random.normal(0, 2)
        base_gas = 450 + 50 * np.sin(i * 0.05) + np.random.normal(0, 20)
        
        # Determine if this is a fire scenario
        fire_probability = 0.1  # Base probability
        if 200 <= i < 300 or 500 <= i < 550 or 700 <= i < 800:
            fire_probability = 0.8  # Higher probability during fire events
        
        is_fire = np.random.random() < fire_probability
        
        if is_fire:
            # Fire scenario - elevated values
            sample_data = {
                't_mean': base_temp + np.random.normal(15, 5),
                't_std': np.random.lognormal(1.5, 0.5),
                't_max': base_temp + np.random.normal(40, 10),
                't_p95': base_temp + np.random.normal(30, 8),
                't_hot_area_pct': np.random.exponential(15),
                't_hot_largest_blob_pct': np.random.exponential(8),
                't_grad_mean': np.random.exponential(5),
                't_grad_std': np.random.lognormal(0.8, 0.3),
                't_diff_mean': np.random.normal(3, 1),
                't_diff_std': np.random.lognormal(0.5, 0.2),
                'flow_mag_mean': np.random.exponential(2),
                'flow_mag_std': np.random.lognormal(0.3, 0.1),
                'tproxy_val': base_temp + np.random.normal(35, 8),
                'tproxy_delta': np.random.normal(10, 3),
                'tproxy_vel': np.random.normal(5, 2),
                'gas_val': base_gas + np.random.normal(800, 200),
                'gas_delta': np.random.normal(200, 50),
                'gas_vel': np.random.normal(100, 25)
            }
            label = 1  # Fire detected
        else:
            # Normal scenario
            sample_data = {
                't_mean': base_temp + np.random.normal(0, 1),
                't_std': np.random.lognormal(0.8, 0.3),
                't_max': base_temp + np.random.normal(5, 2),
                't_p95': base_temp + np.random.normal(3, 1),
                't_hot_area_pct': np.random.exponential(2),
                't_hot_largest_blob_pct': np.random.exponential(1),
                't_grad_mean': np.random.exponential(1),
                't_grad_std': np.random.lognormal(0.3, 0.1),
                't_diff_mean': np.random.normal(0.5, 0.2),
                't_diff_std': np.random.lognormal(0.1, 0.05),
                'flow_mag_mean': np.random.exponential(0.5),
                'flow_mag_std': np.random.lognormal(0.1, 0.05),
                'tproxy_val': base_temp + np.random.normal(2, 1),
                'tproxy_delta': np.random.normal(0.5, 0.2),
                'tproxy_vel': np.random.normal(0.2, 0.1),
                'gas_val': base_gas + np.random.normal(0, 30),
                'gas_delta': np.random.normal(0, 10),
                'gas_vel': np.random.normal(0, 5)
            }
            label = 0  # No fire
        
        data.append(sample_data)
        labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.columns = feature_names
    y = pd.Series(labels, name='fire_detected')
    
    logger.info(f"Generated {len(df)} samples with {sum(labels)} fire events")
    return df, y

def test_feedback_mechanism():
    """Test feedback mechanism for continuous improvement."""
    logger.info("Testing feedback mechanism for continuous improvement...")
    
    try:
        from src.ml.active_learning import FeedbackMechanism
        
        # Create temporary database for testing
        temp_dir = tempfile.mkdtemp()
        feedback_db_path = os.path.join(temp_dir, 'test_feedback.db')
        
        # Create feedback mechanism
        feedback_config = {
            'feedback_db_path': feedback_db_path
        }
        feedback_mechanism = FeedbackMechanism(feedback_config)
        logger.info("✓ Feedback mechanism created successfully")
        
        # Test collecting feedback
        test_feedbacks = [
            ('pred_001', 1, 0.85, 1, 'manual', {'notes': 'clear fire'}, 'v1.0'),
            ('pred_002', 0, 0.30, 0, 'manual', {'notes': 'normal'}, 'v1.0'),
            ('pred_003', 1, 0.65, 0, 'manual', {'notes': 'false positive'}, 'v1.0'),  # Incorrect
            ('pred_004', 0, 0.20, 1, 'manual', {'notes': 'missed fire'}, 'v1.0'),     # Incorrect
        ]
        
        for feedback in test_feedbacks:
            success = feedback_mechanism.collect_feedback(*feedback)
            assert success, f"Failed to collect feedback for {feedback[0]}"
        
        logger.info("✓ Feedback collection successful")
        
        # Test feedback summary
        summary = feedback_mechanism.get_feedback_summary('v1.0')
        logger.info(f"✓ Feedback summary: {summary}")
        
        assert summary['total_feedback'] == 4
        assert summary['correct_predictions'] == 2
        assert summary['incorrect_predictions'] == 2
        assert abs(summary['accuracy'] - 0.5) < 0.001
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing feedback mechanism: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_uncertainty_sampling():
    """Test uncertainty sampling for active learning."""
    logger.info("Testing uncertainty sampling for active learning...")
    
    try:
        from src.ml.active_learning import UncertaintySampler
        
        # Generate test data
        X, y = generate_test_data(n_samples=200, n_features=18)
        
        # Create mock probabilities (simulating model predictions)
        np.random.seed(42)
        # Mix of confident and uncertain predictions
        probabilities = np.random.beta(2, 2, len(X))  # Beta distribution for realistic probabilities
        
        # Create uncertainty sampler
        uncertainty_config = {
            'sampling_strategy': 'margin',
            'uncertainty_threshold': 0.2
        }
        uncertainty_sampler = UncertaintySampler(uncertainty_config)
        logger.info("✓ Uncertainty sampler created successfully")
        
        # Test uncertainty calculation
        uncertainty_scores = uncertainty_sampler.calculate_uncertainty(probabilities)
        logger.info(f"✓ Uncertainty scores calculated: {len(uncertainty_scores)} scores")
        logger.info(f"  Min uncertainty: {uncertainty_scores.min():.3f}")
        logger.info(f"  Max uncertainty: {uncertainty_scores.max():.3f}")
        logger.info(f"  Mean uncertainty: {uncertainty_scores.mean():.3f}")
        
        # Test sample selection
        selected_indices, selected_uncertainties = uncertainty_sampler.select_samples_for_labeling(
            X, probabilities.reshape(-1, 1), n_samples=20
        )
        logger.info(f"✓ Selected {len(selected_indices)} samples for labeling")
        logger.info(f"  Selected uncertainties: {selected_uncertainties}")
        
        # Test different sampling strategies
        for strategy in ['margin', 'entropy', 'random']:
            uncertainty_config['sampling_strategy'] = strategy
            sampler = UncertaintySampler(uncertainty_config)
            indices, _ = sampler.select_samples_for_labeling(X, probabilities.reshape(-1, 1), n_samples=10)
            assert len(indices) == 10, f"Strategy {strategy} failed to select correct number of samples"
        
        logger.info("✓ All sampling strategies work correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing uncertainty sampling: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_update_pipeline():
    """Test model update pipeline without full retraining."""
    logger.info("Testing model update pipeline without full retraining...")
    
    try:
        from src.ml.active_learning import ModelUpdatePipeline
        
        # Create model update pipeline
        update_config = {
            'update_frequency': 5,  # Small frequency for testing
            'min_update_samples': 3,
            'incremental_learning_enabled': True
        }
        update_pipeline = ModelUpdatePipeline(update_config)
        logger.info("✓ Model update pipeline created successfully")
        
        # Generate test feedback data
        X_feedback, y_feedback = generate_test_data(n_samples=10, n_features=18)
        
        # Add feedback data
        for i in range(7):  # Add 7 samples
            success = update_pipeline.add_feedback_data(
                X_feedback.iloc[[i]], 
                y_feedback.iloc[[i]]
            )
            assert success, f"Failed to add feedback sample {i}"
        
        logger.info("✓ Feedback data added successfully")
        
        # Check if update should be triggered
        should_update = update_pipeline.should_update_model()
        logger.info(f"✓ Update trigger check: {should_update}")
        
        # Prepare update data
        if should_update:
            X_update, y_update = update_pipeline.prepare_incremental_update()
            logger.info(f"✓ Prepared update data: {X_update.shape[0]} samples")
            assert X_update.shape[0] == 7, "Incorrect number of update samples"
        
        # Test with simple mock model
        class MockModel:
            def __init__(self):
                self.weights = np.random.rand(18)
                self.trained = True
            
            def partial_fit(self, X, y):
                # Simple weight update
                self.weights = np.mean(X, axis=0)
                return self
        
        mock_model = MockModel()
        original_weights = mock_model.weights.copy()
        
        # Apply model update
        if 'X_update' in locals():
            updated_model = update_pipeline.apply_model_update(mock_model, X_update, y_update)
            logger.info("✓ Model update applied successfully")
            
            # Check that weights changed (indicating update occurred)
            weights_changed = not np.allclose(original_weights, updated_model.weights)
            logger.info(f"✓ Model weights updated: {weights_changed}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing model update pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_monitoring():
    """Test performance monitoring dashboard."""
    logger.info("Testing performance monitoring dashboard...")
    
    try:
        from src.ml.active_learning import PerformanceMonitoringDashboard
        
        # Create performance monitor
        monitor_config = {
            'monitoring_window': 100,
            'alert_thresholds': {
                'accuracy': 0.8,
                'precision': 0.75,
                'recall': 0.75,
                'f1_score': 0.75
            }
        }
        performance_monitor = PerformanceMonitoringDashboard(monitor_config)
        logger.info("✓ Performance monitoring dashboard created successfully")
        
        # Log some prediction results
        test_predictions = [
            ('pred_001', 1, 1, 0.95, 0.05, 'v1.0'),  # Correct fire detection
            ('pred_002', 0, 0, 0.20, 0.03, 'v1.0'),  # Correct normal
            ('pred_003', 1, 0, 0.65, 0.04, 'v1.0'),  # False positive
            ('pred_004', 0, 1, 0.30, 0.06, 'v1.0'),  # Missed fire
            ('pred_005', 1, 1, 0.88, 0.05, 'v1.0'),  # Correct fire detection
        ]
        
        for pred in test_predictions:
            success = performance_monitor.log_prediction_result(*pred)
            assert success, f"Failed to log prediction {pred[0]}"
        
        logger.info("✓ Prediction results logged successfully")
        
        # Test performance metrics
        metrics = performance_monitor.get_performance_metrics()
        logger.info(f"✓ Performance metrics: {metrics}")
        
        # Check specific metrics
        assert metrics['total_predictions'] == 5
        assert abs(metrics['accuracy'] - 0.6) < 0.001  # 3 correct out of 5
        assert metrics['avg_confidence'] > 0.4
        assert metrics['avg_processing_time'] > 0.03
        
        # Test alerts
        alerts = performance_monitor.check_performance_alerts()
        logger.info(f"✓ Performance alerts: {len(alerts)} alerts")
        
        # Test performance report
        report = performance_monitor.generate_performance_report()
        logger.info(f"✓ Performance report generated: {list(report.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing performance monitoring: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_active_learning_loop():
    """Test complete active learning loop."""
    logger.info("Testing complete active learning loop...")
    
    try:
        from src.ml.active_learning import ActiveLearningLoop
        
        # Create temporary directory for database
        temp_dir = tempfile.mkdtemp()
        feedback_db_path = os.path.join(temp_dir, 'test_active_learning.db')
        
        # Create active learning loop
        active_learning_config = {
            'feedback_config': {
                'feedback_db_path': feedback_db_path
            },
            'uncertainty_config': {
                'sampling_strategy': 'margin',
                'uncertainty_threshold': 0.3
            },
            'update_config': {
                'update_frequency': 3,
                'min_update_samples': 2,
                'incremental_learning_enabled': True
            },
            'monitoring_config': {
                'monitoring_window': 50,
                'alert_thresholds': {
                    'accuracy': 0.7,
                    'precision': 0.65,
                    'recall': 0.65,
                    'f1_score': 0.65
                }
            },
            'initial_model_version': 'test_v1.0'
        }
        
        active_learning_loop = ActiveLearningLoop(active_learning_config)
        logger.info("✓ Active learning loop created successfully")
        
        # Generate test data
        X_test, y_test = generate_test_data(n_samples=10, n_features=18)
        
        # Process several prediction feedbacks
        test_feedbacks = [
            ('test_pred_001', 1, 0.92, 1, X_test.iloc[[0]]),  # Correct fire
            ('test_pred_002', 0, 0.25, 0, X_test.iloc[[1]]),  # Correct normal
            ('test_pred_003', 1, 0.70, 0, X_test.iloc[[2]]),  # False positive
            ('test_pred_004', 0, 0.40, 1, X_test.iloc[[3]]),  # Missed fire
            ('test_pred_005', 1, 0.88, 1, X_test.iloc[[4]]),  # Correct fire
        ]
        
        results_summary = []
        for feedback in test_feedbacks:
            result = active_learning_loop.process_prediction_feedback(*feedback)
            results_summary.append(result)
            logger.info(f"✓ Processed feedback {feedback[0]}: {result}")
        
        # Check system status
        system_status = active_learning_loop.get_system_status()
        logger.info(f"✓ System status: {system_status}")
        
        # Generate comprehensive report
        comprehensive_report = active_learning_loop.generate_comprehensive_report()
        logger.info(f"✓ Comprehensive report generated: {list(comprehensive_report.keys())}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing active learning loop: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run all Task 8 active learning tests."""
    logger.info("Starting Task 8: Active Learning Loop Tests")
    logger.info("=" * 50)
    
    results = {}
    
    # Test 1: Feedback mechanism for continuous improvement
    logger.info("Test 1: Feedback mechanism for continuous improvement")
    results['feedback_mechanism'] = test_feedback_mechanism()
    
    # Test 2: Uncertainty sampling for active learning
    logger.info("\nTest 2: Uncertainty sampling for active learning")
    results['uncertainty_sampling'] = test_uncertainty_sampling()
    
    # Test 3: Model update pipeline without full retraining
    logger.info("\nTest 3: Model update pipeline without full retraining")
    results['model_update_pipeline'] = test_model_update_pipeline()
    
    # Test 4: Performance monitoring dashboard
    logger.info("\nTest 4: Performance monitoring dashboard")
    results['performance_monitoring'] = test_performance_monitoring()
    
    # Test 5: Complete active learning loop
    logger.info("\nTest 5: Complete active learning loop")
    results['active_learning_loop'] = test_active_learning_loop()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Task 8 Active Learning Loop Test Results:")
    
    success_count = sum(1 for test in results.values() if test)
    total_tests = len(results)
    
    for test_name, test_result in results.items():
        status = "✅ PASSED" if test_result else "❌ FAILED"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        logger.info("🎉 Task 8 Active Learning Loop COMPLETED SUCCESSFULLY")
        logger.info("✅ Feedback mechanism for continuous improvement implemented")
        logger.info("✅ Uncertainty sampling for active learning implemented")
        logger.info("✅ Model update pipeline without full retraining implemented")
        logger.info("✅ Performance monitoring dashboard implemented")
        return 0
    else:
        logger.info("❌ Task 8 Active Learning Loop had some failures")
        return 1

if __name__ == "__main__":
    sys.exit(main())