#!/usr/bin/env python3
"""
Test script for Task 7: Temporal Modeling.
This script tests LSTM layers, Transformer-based temporal pattern recognition,
sliding window analysis, and early fire detection algorithms.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_temporal_fire_data(n_samples=1000, n_features=18):
    """Generate synthetic temporal fire detection data."""
    logger.info(f"Generating temporal fire data: {n_samples} samples, {n_features} features")
    
    np.random.seed(42)
    
    # Generate base features (similar to FLIR+SCD41 features)
    feature_names = [
        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
        'gas_val', 'gas_delta', 'gas_vel'
    ][:n_features]
    
    # Create temporal patterns
    data = []
    labels = []
    
    for i in range(n_samples):
        # Base values with some temporal correlation
        base_temp = 20 + 5 * np.sin(i * 0.1) + np.random.normal(0, 2)
        base_gas = 450 + 50 * np.sin(i * 0.05) + np.random.normal(0, 20)
        
        # Determine if this is a fire scenario
        # Fire events occur in bursts
        fire_probability = 0.1  # Base probability
        if 200 <= i < 300 or 500 <= i < 550 or 700 <= i < 800:
            fire_probability = 0.8  # Higher probability during fire events
        
        is_fire = np.random.random() < fire_probability
        
        if is_fire:
            # Fire scenario - elevated values
            thermal_features = {
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
                'tproxy_vel': np.random.normal(5, 2)
            }
            
            gas_features = {
                'gas_val': base_gas + np.random.normal(800, 200),
                'gas_delta': np.random.normal(200, 50),
                'gas_vel': np.random.normal(100, 25)
            }
            
            label = 1  # Fire detected
        else:
            # Normal scenario
            thermal_features = {
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
                'tproxy_vel': np.random.normal(0.2, 0.1)
            }
            
            gas_features = {
                'gas_val': base_gas + np.random.normal(0, 30),
                'gas_delta': np.random.normal(0, 10),
                'gas_vel': np.random.normal(0, 5)
            }
            
            label = 0  # No fire
        
        # Combine features
        sample_data = {**thermal_features, **gas_features}
        data.append(sample_data)
        labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.columns = feature_names
    df['fire_detected'] = labels
    df['timestamp'] = [datetime.now() + timedelta(seconds=i) for i in range(len(df))]
    
    logger.info(f"Generated {len(df)} samples with {sum(labels)} fire events")
    return df

def test_lstm_temporal_model():
    """Test LSTM layers for sequence analysis."""
    logger.info("Testing LSTM layers for sequence analysis...")
    
    try:
        from src.ml.temporal_modeling import LSTMTemporalModel
        
        # Generate test data
        df = generate_temporal_fire_data(n_samples=500, n_features=18)
        feature_columns = [col for col in df.columns if col not in ['fire_detected', 'timestamp']]
        
        X = df[feature_columns]
        y = df['fire_detected']
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create and train LSTM model
        lstm_config = {
            'sequence_length': 20,
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 16,
            'num_epochs': 5
        }
        
        lstm_model = LSTMTemporalModel(lstm_config)
        logger.info("✓ LSTM model created successfully")
        
        # Train model
        train_results = lstm_model.train(X_train, y_train)
        logger.info(f"✓ LSTM model trained: {train_results}")
        
        # Test predictions
        predictions = lstm_model.predict(X_test)
        probabilities = lstm_model.predict_proba(X_test)
        
        logger.info(f"✓ LSTM predictions generated: {len(predictions)} predictions")
        logger.info(f"✓ LSTM probabilities generated: {probabilities.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing LSTM temporal model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_transformer_temporal_model():
    """Test Transformer-based temporal pattern recognition."""
    logger.info("Testing Transformer-based temporal pattern recognition...")
    
    try:
        from src.ml.temporal_modeling import TransformerTemporalModel
        
        # Generate test data
        df = generate_temporal_fire_data(n_samples=500, n_features=18)
        feature_columns = [col for col in df.columns if col not in ['fire_detected', 'timestamp']]
        
        X = df[feature_columns]
        y = df['fire_detected']
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create and train Transformer model
        transformer_config = {
            'sequence_length': 20,
            'd_model': 32,
            'nhead': 4,
            'num_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 16,
            'num_epochs': 5
        }
        
        transformer_model = TransformerTemporalModel(transformer_config)
        logger.info("✓ Transformer model created successfully")
        
        # Train model
        train_results = transformer_model.train(X_train, y_train)
        logger.info(f"✓ Transformer model trained: {train_results}")
        
        # Test predictions
        predictions = transformer_model.predict(X_test)
        probabilities = transformer_model.predict_proba(X_test)
        
        logger.info(f"✓ Transformer predictions generated: {len(predictions)} predictions")
        logger.info(f"✓ Transformer probabilities generated: {probabilities.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Transformer temporal model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_sliding_window_analysis():
    """Test sliding window analysis for continuous monitoring."""
    logger.info("Testing sliding window analysis for continuous monitoring...")
    
    try:
        from src.ml.temporal_modeling import SlidingWindowAnalyzer, LSTMTemporalModel
        
        # Generate test data
        df = generate_temporal_fire_data(n_samples=200, n_features=18)
        feature_columns = [col for col in df.columns if col not in ['fire_detected', 'timestamp']]
        X = df[feature_columns]
        
        # Create sliding window analyzer
        window_config = {
            'window_size': 30,
            'slide_step': 10
        }
        window_analyzer = SlidingWindowAnalyzer(window_config)
        logger.info("✓ Sliding window analyzer created successfully")
        
        # Create a simple temporal model for testing
        class SimpleTemporalModel:
            def predict(self, data):
                # Simple mock prediction
                return np.random.choice([0, 1], size=max(1, len(data) - 29))
            
            def predict_proba(self, data):
                # Simple mock probabilities
                n_preds = max(1, len(data) - 29)
                proba_0 = np.random.rand(n_preds)
                proba_1 = 1 - proba_0
                return np.column_stack([proba_0, proba_1])
        
        # Add model to analyzer
        simple_model = SimpleTemporalModel()
        window_analyzer.add_temporal_model('simple_model', simple_model)
        
        # Test window analysis
        window_result = window_analyzer.analyze_window(X.head(50))
        logger.info(f"✓ Window analysis completed: {window_result}")
        
        # Test continuous monitoring
        monitoring_results = window_analyzer.continuous_monitoring(X)
        logger.info(f"✓ Continuous monitoring completed: {len(monitoring_results)} windows analyzed")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing sliding window analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_early_fire_detection():
    """Test early fire detection algorithms based on temporal patterns."""
    logger.info("Testing early fire detection algorithms based on temporal patterns...")
    
    try:
        from src.ml.temporal_modeling import EarlyFireDetectionSystem
        
        # Generate test data with clear temporal patterns
        df = generate_temporal_fire_data(n_samples=100, n_features=18)
        feature_columns = [col for col in df.columns if col not in ['fire_detected', 'timestamp']]
        X = df[feature_columns]
        
        # Create early detection system
        early_detection_config = {
            'detection_threshold': 0.7,
            'early_warning_threshold': 0.5,
            'temporal_window_size': 15,
            'trend_analysis_window': 10
        }
        early_detector = EarlyFireDetectionSystem(early_detection_config)
        logger.info("✓ Early fire detection system created successfully")
        
        # Test early fire pattern detection
        detection_result = early_detector.detect_early_fire_patterns(X.tail(30))
        logger.info(f"✓ Early fire detection completed: {detection_result}")
        
        # Test detection time improvement
        baseline_time = 30.0  # 30 seconds baseline
        improvement_result = early_detector.get_detection_time_improvement(baseline_time)
        logger.info(f"✓ Detection time improvement calculated: {improvement_result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing early fire detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run all Task 7 temporal modeling tests."""
    logger.info("Starting Task 7: Temporal Modeling Tests")
    logger.info("=" * 50)
    
    results = {}
    
    # Test 1: LSTM layers for sequence analysis
    logger.info("Test 1: LSTM layers for sequence analysis")
    results['lstm_modeling'] = test_lstm_temporal_model()
    
    # Test 2: Transformer-based temporal pattern recognition
    logger.info("\nTest 2: Transformer-based temporal pattern recognition")
    results['transformer_modeling'] = test_transformer_temporal_model()
    
    # Test 3: Sliding window analysis for continuous monitoring
    logger.info("\nTest 3: Sliding window analysis for continuous monitoring")
    results['sliding_window'] = test_sliding_window_analysis()
    
    # Test 4: Early fire detection algorithms based on temporal patterns
    logger.info("\nTest 4: Early fire detection algorithms based on temporal patterns")
    results['early_detection'] = test_early_fire_detection()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Task 7 Temporal Modeling Test Results:")
    
    success_count = sum(1 for test in results.values() if test)
    total_tests = len(results)
    
    for test_name, test_result in results.items():
        status = "✅ PASSED" if test_result else "❌ FAILED"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        logger.info("🎉 Task 7 Temporal Modeling COMPLETED SUCCESSFULLY")
        logger.info("✅ LSTM layers for sequence analysis implemented")
        logger.info("✅ Transformer-based temporal pattern recognition implemented")
        logger.info("✅ Sliding window analysis for continuous monitoring implemented")
        logger.info("✅ Early fire detection algorithms based on temporal patterns implemented")
        return 0
    else:
        logger.info("❌ Task 7 Temporal Modeling had some failures")
        return 1

if __name__ == "__main__":
    sys.exit(main())