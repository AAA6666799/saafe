#!/usr/bin/env python3
"""
Test script for the Attention-based Fusion Model.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_attention_fusion_model():
    """Test the Attention Fusion Model implementation."""
    print("Testing Attention Fusion Model...")
    
    try:
        from src.feature_engineering.fusion.attention_fusion_model import AttentionFusionModel
        
        # Create sample data
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
        
        # Create sample labels
        y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]))
        
        # Create and train model
        model = AttentionFusionModel({
            'thermal_features_count': 15,
            'gas_features_count': 3,
            'attention_type': 'cross_sensor'
        })
        
        # Test training
        model.fit(X, y)
        print("  ✅ Model trained successfully")
        
        # Test prediction
        predictions = model.predict(X.iloc[:10])  # Test on first 10 samples
        print(f"  ✅ Predictions generated: {len(predictions)} samples")
        print(f"  Prediction range: {np.min(predictions):.3f} to {np.max(predictions):.3f}")
        
        # Test probability prediction
        probabilities = model.predict_proba(X.iloc[:10])
        print(f"  ✅ Probabilities generated: shape {probabilities.shape}")
        
        # Test attention weights
        attention_weights = model.get_attention_weights()
        print(f"  ✅ Attention weights retrieved: {len(attention_weights)} components")
        print(f"  Thermal attention: {attention_weights['thermal_attention']:.3f}")
        print(f"  Gas attention: {attention_weights['gas_attention']:.3f}")
        
        # Test feature importance
        feature_importance = model.get_feature_importance()
        print(f"  ✅ Feature importance retrieved: {len(feature_importance)} features")
        
        # Test cross-sensor analysis
        cross_sensor_analysis = model.analyze_cross_sensor_importance(X.iloc[:100])
        print(f"  ✅ Cross-sensor analysis completed")
        print(f"  Thermal importance: {cross_sensor_analysis['thermal_sensor_importance']:.3f}")
        print(f"  Gas importance: {cross_sensor_analysis['gas_sensor_importance']:.3f}")
        
        # Test dynamic feature selection
        selected_features = model.dynamic_feature_selection(X.iloc[:100], threshold=0.001)
        print(f"  ✅ Dynamic feature selection completed: {len(selected_features)} features selected")
        
        # Test model info
        model_info = model.get_model_info()
        print(f"  ✅ Model info retrieved: {model_info['model_type']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Attention Fusion Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_sensor_analyzer():
    """Test the Cross-Sensor Feature Analyzer."""
    print("Testing Cross-Sensor Feature Analyzer...")
    
    try:
        from src.feature_engineering.fusion.attention_fusion_model import CrossSensorFeatureAnalyzer
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        
        # Create thermal features
        thermal_data = {
            't_mean': np.random.normal(25, 5, n_samples),
            't_max': np.random.normal(35, 8, n_samples),
            't_hot_area_pct': np.random.exponential(2, n_samples)
        }
        
        # Create gas features
        gas_data = {
            'gas_val': np.random.normal(500, 100, n_samples),
            'gas_delta': np.random.normal(0, 20, n_samples)
        }
        
        thermal_df = pd.DataFrame(thermal_data)
        gas_df = pd.DataFrame(gas_data)
        
        # Create analyzer
        analyzer = CrossSensorFeatureAnalyzer({
            'correlation_threshold': 0.1,
            'importance_threshold': 0.001
        })
        
        # Test feature interactions analysis
        interaction_analysis = analyzer.analyze_feature_interactions(thermal_df, gas_df)
        print(f"  ✅ Feature interactions analysis completed")
        print(f"  Cross correlations found: {interaction_analysis['total_cross_correlations']}")
        print(f"  Important pairs found: {interaction_analysis['total_important_pairs']}")
        
        # Test dynamic feature weights
        dynamic_weights = analyzer.compute_dynamic_feature_weights(thermal_df, gas_df)
        print(f"  ✅ Dynamic feature weights computed")
        print(f"  Thermal weight: {dynamic_weights['sensor_weights']['thermal']:.3f}")
        print(f"  Gas weight: {dynamic_weights['sensor_weights']['gas']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Cross-Sensor Feature Analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Attention-based Fusion Model Features")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_attention_fusion_model,
        test_cross_sensor_analyzer
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
        print("🎉 All tests passed! Attention-based fusion model is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())