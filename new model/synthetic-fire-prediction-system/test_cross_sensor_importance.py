#!/usr/bin/env python3
"""
Test script for the Cross-Sensor Feature Importance Analyzer.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cross_sensor_importance_analyzer():
    """Test the Cross-Sensor Importance Analyzer implementation."""
    print("Testing Cross-Sensor Importance Analyzer...")
    
    try:
        from src.feature_engineering.fusion.cross_sensor_importance_analyzer import CrossSensorImportanceAnalyzer
        
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
        # Make labels somewhat correlated with high temperatures and gas values
        fire_score = (
            (thermal_df['t_max'] - 35) / 10 + 
            (gas_df['gas_val'] - 500) / 200 +
            np.random.normal(0, 0.5, n_samples)
        )
        y = pd.Series((fire_score > 0.5).astype(int))
        
        # Create and train analyzer
        analyzer = CrossSensorImportanceAnalyzer({
            'thermal_feature_prefixes': ['t_', 'temp', 'thermal', 'tproxy'],
            'gas_feature_prefixes': ['gas'],
            'selection_threshold': 0.001
        })
        
        # Test training
        analyzer.fit(X, y)
        print("  ✅ Analyzer trained successfully")
        
        # Test feature importance
        mi_importance = analyzer.get_feature_importance('mutual_info')
        print(f"  ✅ Mutual information importance computed: {len(mi_importance)} features")
        
        stat_importance = analyzer.get_feature_importance('statistical')
        print(f"  ✅ Statistical importance computed: {len(stat_importance)} features")
        
        combined_importance = analyzer.get_feature_importance('combined')
        print(f"  ✅ Combined importance computed: {len(combined_importance)} features")
        
        # Test sensor importance
        sensor_importance = analyzer.get_sensor_importance()
        print(f"  ✅ Sensor importance computed")
        print(f"  Thermal sensor importance: {sensor_importance['thermal_sensor_importance']:.3f}")
        print(f"  Gas sensor importance: {sensor_importance['gas_sensor_importance']:.3f}")
        
        # Test feature selection
        selected_features = analyzer.select_features(X, threshold=0.001)
        print(f"  ✅ Feature selection completed: {len(selected_features)} features selected")
        
        # Test cross-sensor analysis
        cross_analysis = analyzer.get_cross_sensor_analysis()
        print(f"  ✅ Cross-sensor analysis completed")
        print(f"  Total cross-correlations: {cross_analysis['cross_correlations']['total']}")
        
        # Test validation
        validation_results = analyzer.validate_feature_selection(X, y, selected_features)
        print(f"  ✅ Feature selection validation completed")
        print(f"  CV mean score: {validation_results['cv_mean_score']:.3f}")
        
        # Test analysis report
        analysis_report = analyzer.get_analysis_report()
        print(f"  ✅ Analysis report generated")
        print(f"  Top feature: {analysis_report['top_features'][0]['feature']} "
              f"({analysis_report['top_features'][0]['importance']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Cross-Sensor Importance Analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_feature_selector():
    """Test the Dynamic Feature Selector."""
    print("Testing Dynamic Feature Selector...")
    
    try:
        from src.feature_engineering.fusion.cross_sensor_importance_analyzer import DynamicFeatureSelector
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        
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
        
        # Create selector
        selector = DynamicFeatureSelector({
            'selection_threshold': 0.001,
            'context_window': 10
        })
        
        # Initialize analyzer
        selector.initialize_analyzer(X, y)
        print("  ✅ Dynamic selector initialized with analyzer")
        
        # Test dynamic feature selection
        selected_features = selector.select_features_dynamic(X.iloc[:100])
        print(f"  ✅ Dynamic feature selection completed: {len(selected_features)} features")
        
        # Test adaptation insights
        insights = selector.get_adaptation_insights()
        print(f"  ✅ Adaptation insights computed")
        print(f"  Average selected features: {insights.get('average_selected_features', 0):.1f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dynamic Feature Selector test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Cross-Sensor Feature Importance Analysis")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_cross_sensor_importance_analyzer,
        test_dynamic_feature_selector
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
        print("🎉 All tests passed! Cross-sensor importance analysis is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())