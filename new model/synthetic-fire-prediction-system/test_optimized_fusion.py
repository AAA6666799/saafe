#!/usr/bin/env python3
"""
Test script for the Optimized Fusion Model.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_optimized_fusion_model():
    """Test the Optimized Fusion Model implementation."""
    print("Testing Optimized Fusion Model...")
    
    try:
        from src.feature_engineering.fusion.optimized_fusion_model import OptimizedFusionModel
        
        # Create sample data with realistic feature names
        np.random.seed(42)
        n_samples = 1000
        
        # Create thermal features (15 features)
        thermal_data = {
            't_mean': np.random.normal(25, 5, n_samples),
            't_std': np.random.lognormal(1, 0.5, n_samples),
            't_max': np.random.normal(35, 8, n_samples),
            't_p95': np.random.normal(32, 7, n_samples),
            't_hot_area_pct': np.random.exponential(2, n_samples),
            't_hot_largest_blob_pct': np.random.exponential(1, n_samples),
            't_grad_mean': np.random.exponential(2, n_samples),
            't_grad_std': np.random.lognormal(0.5, 0.3, n_samples),
            't_diff_mean': np.random.normal(0, 1, n_samples),
            't_diff_std': np.random.lognormal(0.3, 0.2, n_samples),
            'flow_mag_mean': np.random.exponential(1, n_samples),
            'flow_mag_std': np.random.lognormal(0.2, 0.1, n_samples),
            'tproxy_val': np.random.normal(30, 6, n_samples),
            'tproxy_delta': np.random.normal(0, 2, n_samples),
            'tproxy_vel': np.random.normal(0, 1, n_samples)
        }
        
        # Create gas features (3 features)
        gas_data = {
            'gas_val': np.random.normal(500, 100, n_samples),
            'gas_delta': np.random.normal(0, 20, n_samples),
            'gas_vel': np.random.normal(0, 15, n_samples)
        }
        
        # Combine into DataFrame
        thermal_df = pd.DataFrame(thermal_data)
        gas_df = pd.DataFrame(gas_data)
        X = pd.concat([thermal_df, gas_df], axis=1)
        
        # Create sample labels with some correlation to features
        fire_score = (
            (thermal_df['t_max'] - 35) / 10 + 
            (gas_df['gas_val'] - 500) / 200 +
            np.random.normal(0, 0.5, n_samples)
        )
        y = pd.Series((fire_score > 0.5).astype(int))
        
        # Create and train optimized model
        model = OptimizedFusionModel({
            'thermal_features_count': 15,
            'gas_features_count': 3,
            'target_latency_ms': 5.0,  # 5ms target for real-time
            'enable_caching': True,
            'cache_size': 50
        })
        
        # Test training
        model.fit(X, y)
        print("  ✅ Optimized model trained successfully")
        
        # Test prediction
        predictions = model.predict(X.iloc[:10])  # Test on first 10 samples
        print(f"  ✅ Predictions generated: {len(predictions)} samples")
        print(f"  Prediction range: {np.min(predictions):.3f} to {np.max(predictions):.3f}")
        
        # Test probability prediction
        probabilities = model.predict_proba(X.iloc[:10])
        print(f"  ✅ Probabilities generated: shape {probabilities.shape}")
        
        # Test performance stats
        performance_stats = model.get_performance_stats()
        print(f"  ✅ Performance stats retrieved")
        print(f"  Average latency: {performance_stats.get('avg_latency_ms', 0):.2f}ms")
        
        # Test model info
        model_info = model.get_model_info()
        print(f"  ✅ Model info retrieved: {model_info['model_type']}")
        print(f"  Meets latency target: {model_info['performance_targets']['current_avg_latency_ms'] <= 5.0}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Optimized Fusion Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_real_time_optimizer():
    """Test the Real-Time Fusion Optimizer."""
    print("Testing Real-Time Fusion Optimizer...")
    
    try:
        from src.feature_engineering.fusion.optimized_fusion_model import OptimizedFusionModel, RealTimeFusionOptimizer
        
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        
        # Create features
        data = {
            't_mean': np.random.normal(25, 5, n_samples),
            't_max': np.random.normal(35, 8, n_samples),
            't_hot_area_pct': np.random.exponential(2, n_samples),
            'gas_val': np.random.normal(500, 100, n_samples),
            'gas_delta': np.random.normal(0, 20, n_samples)
        }
        
        X = pd.DataFrame(data)
        y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]))
        
        # Create model and optimizer
        model = OptimizedFusionModel({
            'thermal_features_count': 3,
            'gas_features_count': 2,
            'target_latency_ms': 2.0
        })
        
        # Train model
        model.fit(X, y)
        
        optimizer = RealTimeFusionOptimizer()
        
        # Test optimization
        optimization_results = optimizer.optimize_for_latency(model, X.iloc[:50], target_latency_ms=2.0)
        print(f"  ✅ Real-time optimization completed")
        print(f"  Initial latency: {optimization_results['initial_latency_ms']:.2f}ms")
        print(f"  Final latency: {optimization_results['final_latency_ms']:.2f}ms")
        print(f"  Optimizations applied: {len(optimization_results['optimizations_applied'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Real-Time Fusion Optimizer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Optimized Fusion Model for Real-Time Performance")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_optimized_fusion_model,
        test_real_time_optimizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Optimized fusion model is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())