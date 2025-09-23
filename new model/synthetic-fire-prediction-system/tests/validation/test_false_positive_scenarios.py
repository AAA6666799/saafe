"""
Validation tests for false positive scenarios in FLIR+SCD41 fire detection system.

This module contains comprehensive validation tests for false positive reduction,
including specific scenario testing and discrimination effectiveness.
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

try:
    from src.optimization.false_positive_reducer import FalsePositiveReducer, create_false_positive_reducer
    from src.feature_engineering.fusion.false_positive_discriminator import FalsePositiveDiscriminator
    FALSE_POSITIVE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import false positive components: {e}")
    FALSE_POSITIVE_AVAILABLE = False


class TestFalsePositiveScenarios(unittest.TestCase):
    """Test cases for false positive scenario validation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        if not FALSE_POSITIVE_AVAILABLE:
            self.skipTest("False positive components not available")
        
        # Create false positive reducer
        self.fp_config = {
            'baseline_fpr': 0.182,
            'target_reduction': 0.5,
            'discrimination_threshold': 0.7,
            'category_thresholds': {
                'sunlight_heating': 0.8,
                'hvac_effect': 0.75,
                'cooking': 0.7,
                'steam_dust': 0.65,
                'other': 0.6
            }
        }
        
        self.fp_reducer = create_false_positive_reducer(self.fp_config)
        self.fp_discriminator = FalsePositiveDiscriminator()
    
    def test_sunlight_heating_discrimination(self):
        """Test discrimination of sunlight heating false positives."""
        # Create sunlight heating scenario data
        n_samples = 100
        feature_data = {
            't_mean': np.random.normal(35.0, 5.0, n_samples),  # High temperature
            't_std': np.random.normal(3.0, 1.0, n_samples),
            't_max': np.random.normal(50.0, 8.0, n_samples),   # Very high max temp
            't_p95': np.random.normal(45.0, 7.0, n_samples),
            't_hot_area_pct': np.random.uniform(25.0, 50.0, n_samples),  # Large hot area (sunlight)
            't_hot_largest_blob_pct': np.random.uniform(15.0, 30.0, n_samples),
            't_grad_mean': np.random.normal(1.5, 0.5, n_samples),
            't_grad_std': np.random.normal(0.8, 0.3, n_samples),
            't_diff_mean': np.random.normal(0.2, 0.1, n_samples),
            't_diff_std': np.random.normal(0.1, 0.05, n_samples),
            'flow_mag_mean': np.random.normal(0.3, 0.1, n_samples),
            'flow_mag_std': np.random.normal(0.2, 0.1, n_samples),
            'tproxy_val': np.random.normal(50.0, 8.0, n_samples),
            'tproxy_delta': np.random.normal(0.5, 0.2, n_samples),  # Low temporal change
            'tproxy_vel': np.random.normal(0.1, 0.05, n_samples),   # Low velocity
            'gas_val': np.random.normal(450.0, 50.0, n_samples),    # Normal CO2
            'gas_delta': np.random.normal(2.0, 1.0, n_samples),     # Very low CO2 change
            'gas_vel': np.random.normal(0.5, 0.2, n_samples),       # Low CO2 velocity
            'ground_truth': np.zeros(n_samples, dtype=int)          # All false positives
        }
        
        features_df = pd.DataFrame(feature_data)
        
        # Test discrimination
        discrimination_results = []
        for _, row in features_df.iterrows():
            result = self.fp_discriminator.discriminate_false_positives(
                thermal_features={
                    't_max': row['t_max'],
                    't_hot_area_pct': row['t_hot_area_pct'],
                    'tproxy_delta': row['tproxy_delta']
                },
                gas_features={
                    'gas_val': row['gas_val'],
                    'gas_delta': row['gas_delta'],
                    'gas_vel': row['gas_vel']
                }
            )
            discrimination_results.append(result)
        
        # Check that sunlight discrimination works
        sunlight_scores = [r.get('sunlight_discrimination_score', 0.0) for r in discrimination_results]
        high_sunlight_scores = [s for s in sunlight_scores if s > 0.6]
        
        # Most samples should be identified as sunlight heating
        self.assertGreater(len(high_sunlight_scores) / len(sunlight_scores), 0.7,
                          "Most sunlight heating scenarios should be discriminated")
        
        # Test filtering effectiveness
        predictions = np.ones(n_samples, dtype=int)  # All predicted as fire
        confidence_scores = np.random.uniform(0.6, 0.9, n_samples)  # High confidence false positives
        
        filtered_predictions = self.fp_reducer.apply_false_positive_filter(
            predictions, confidence_scores, features_df
        )
        
        # Check that some predictions were filtered out
        original_positive = np.sum(predictions)
        filtered_positive = np.sum(filtered_predictions)
        reduction = (original_positive - filtered_positive) / original_positive
        
        self.assertGreater(reduction, 0.3, 
                          "False positive filter should reduce sunlight heating false positives by >30%")
    
    def test_hvac_effect_discrimination(self):
        """Test discrimination of HVAC effect false positives."""
        # Create HVAC effect scenario data
        n_samples = 80
        feature_data = {
            't_mean': np.random.normal(25.0, 2.0, n_samples),   # Normal temperature
            't_std': np.random.normal(2.5, 0.8, n_samples),
            't_max': np.random.normal(35.0, 5.0, n_samples),    # Moderate high temp
            't_p95': np.random.normal(32.0, 4.0, n_samples),
            't_hot_area_pct': np.random.uniform(3.0, 8.0, n_samples),   # Small hot areas
            't_hot_largest_blob_pct': np.random.uniform(1.0, 4.0, n_samples),
            't_grad_mean': np.random.normal(1.2, 0.4, n_samples),
            't_grad_std': np.random.normal(0.6, 0.2, n_samples),
            't_diff_mean': np.random.normal(0.3, 0.1, n_samples),
            't_diff_std': np.random.normal(0.15, 0.08, n_samples),
            'flow_mag_mean': np.random.normal(0.4, 0.2, n_samples),
            'flow_mag_std': np.random.normal(0.25, 0.1, n_samples),
            'tproxy_val': np.random.normal(35.0, 5.0, n_samples),
            'tproxy_delta': np.random.normal(8.0, 3.0, n_samples),      # Moderate temporal change
            'tproxy_vel': np.random.normal(2.0, 1.0, n_samples),        # Moderate velocity
            'gas_val': np.random.normal(700.0, 100.0, n_samples),       # Normal CO2
            'gas_delta': np.random.normal(50.0, 20.0, n_samples),       # Moderate CO2 change
            'gas_vel': np.random.normal(15.0, 8.0, n_samples),          # Moderate CO2 velocity
            'ground_truth': np.zeros(n_samples, dtype=int)              # All false positives
        }
        
        features_df = pd.DataFrame(feature_data)
        
        # Test discrimination
        discrimination_results = []
        for _, row in features_df.iterrows():
            result = self.fp_discriminator.discriminate_false_positives(
                thermal_features={
                    't_max': row['t_max'],
                    't_mean': row['t_mean'],
                    't_hot_area_pct': row['t_hot_area_pct']
                },
                gas_features={
                    'gas_val': row['gas_val'],
                    'gas_delta': row['gas_delta']
                }
            )
            discrimination_results.append(result)
        
        # Check that HVAC discrimination works
        hvac_scores = [r.get('hvac_discrimination_score', 0.0) for r in discrimination_results]
        high_hvac_scores = [s for s in hvac_scores if s > 0.5]
        
        # Most samples should be identified as HVAC effect
        self.assertGreater(len(high_hvac_scores) / len(hvac_scores), 0.6,
                          "Most HVAC effect scenarios should be discriminated")
    
    def test_cooking_discrimination(self):
        """Test discrimination of cooking false positives."""
        # Create cooking scenario data
        n_samples = 60
        feature_data = {
            't_mean': np.random.normal(30.0, 3.0, n_samples),   # Elevated temperature
            't_std': np.random.normal(4.0, 1.2, n_samples),
            't_max': np.random.normal(55.0, 10.0, n_samples),   # High max temp
            't_p95': np.random.normal(48.0, 8.0, n_samples),
            't_hot_area_pct': np.random.uniform(2.0, 6.0, n_samples),   # Localized heating
            't_hot_largest_blob_pct': np.random.uniform(1.0, 3.0, n_samples),
            't_grad_mean': np.random.normal(2.0, 0.8, n_samples),
            't_grad_std': np.random.normal(1.0, 0.4, n_samples),
            't_diff_mean': np.random.normal(0.5, 0.2, n_samples),
            't_diff_std': np.random.normal(0.25, 0.1, n_samples),
            'flow_mag_mean': np.random.normal(0.8, 0.3, n_samples),
            'flow_mag_std': np.random.normal(0.4, 0.2, n_samples),
            'tproxy_val': np.random.normal(55.0, 10.0, n_samples),
            'tproxy_delta': np.random.normal(15.0, 5.0, n_samples),     # Significant temporal change
            'tproxy_vel': np.random.normal(5.0, 2.0, n_samples),        # High velocity
            'gas_val': np.random.normal(1200.0, 200.0, n_samples),      # Elevated CO2
            'gas_delta': np.random.normal(200.0, 50.0, n_samples),      # Significant CO2 change
            'gas_vel': np.random.normal(60.0, 20.0, n_samples),         # High CO2 velocity
            'ground_truth': np.zeros(n_samples, dtype=int)              # All false positives
        }
        
        features_df = pd.DataFrame(feature_data)
        
        # Test discrimination
        discrimination_results = []
        for _, row in features_df.iterrows():
            result = self.fp_discriminator.discriminate_false_positives(
                thermal_features={
                    't_max': row['t_max'],
                    't_hot_area_pct': row['t_hot_area_pct'],
                    'tproxy_delta': row['tproxy_delta']
                },
                gas_features={
                    'gas_val': row['gas_val'],
                    'gas_delta': row['gas_delta'],
                    'gas_vel': row['gas_vel']
                }
            )
            discrimination_results.append(result)
        
        # Check that cooking discrimination works
        cooking_scores = [r.get('cooking_discrimination_score', 0.0) for r in discrimination_results]
        high_cooking_scores = [s for s in cooking_scores if s > 0.5]
        
        # Most samples should be identified as cooking
        self.assertGreater(len(high_cooking_scores) / len(cooking_scores), 0.5,
                          "Most cooking scenarios should be discriminated")
    
    def test_steam_dust_discrimination(self):
        """Test discrimination of steam/dust false positives."""
        # Create steam/dust scenario data
        n_samples = 50
        feature_data = {
            't_mean': np.random.normal(32.0, 4.0, n_samples),   # Moderate temperature
            't_std': np.random.normal(3.5, 1.0, n_samples),
            't_max': np.random.normal(45.0, 8.0, n_samples),    # Moderate high temp
            't_p95': np.random.normal(40.0, 7.0, n_samples),
            't_hot_area_pct': np.random.uniform(5.0, 12.0, n_samples),  # Moderate hot areas
            't_hot_largest_blob_pct': np.random.uniform(2.0, 6.0, n_samples),
            't_grad_mean': np.random.normal(1.8, 0.6, n_samples),
            't_grad_std': np.random.normal(0.9, 0.3, n_samples),
            't_diff_mean': np.random.normal(0.4, 0.15, n_samples),
            't_diff_std': np.random.normal(0.2, 0.1, n_samples),
            'flow_mag_mean': np.random.normal(0.6, 0.25, n_samples),
            'flow_mag_std': np.random.normal(0.3, 0.15, n_samples),
            'tproxy_val': np.random.normal(45.0, 8.0, n_samples),
            'tproxy_delta': np.random.normal(12.0, 4.0, n_samples),     # Moderate temporal change
            'tproxy_vel': np.random.normal(4.0, 1.5, n_samples),        # Moderate velocity
            'gas_val': np.random.normal(800.0, 150.0, n_samples),       # Elevated CO2
            'gas_delta': np.random.normal(100.0, 30.0, n_samples),      # Moderate CO2 change
            'gas_vel': np.random.normal(35.0, 15.0, n_samples),         # Moderate CO2 velocity
            'ground_truth': np.zeros(n_samples, dtype=int)              # All false positives
        }
        
        features_df = pd.DataFrame(feature_data)
        
        # Test discrimination
        discrimination_results = []
        for _, row in features_df.iterrows():
            result = self.fp_discriminator.discriminate_false_positives(
                thermal_features={
                    't_max': row['t_max'],
                    't_hot_area_pct': row['t_hot_area_pct'],
                    'tproxy_val': row['tproxy_val']
                },
                gas_features={
                    'gas_val': row['gas_val'],
                    'gas_delta': row['gas_delta']
                }
            )
            discrimination_results.append(result)
        
        # Check that steam/dust discrimination works
        dust_scores = [r.get('dust_discrimination_score', 0.0) for r in discrimination_results]
        high_dust_scores = [s for s in dust_scores if s > 0.4]
        
        # Many samples should be identified as steam/dust
        self.assertGreater(len(high_dust_scores) / len(dust_scores), 0.4,
                          "Many steam/dust scenarios should be discriminated")
    
    def test_false_positive_rate_analysis(self):
        """Test false positive rate analysis and tracking."""
        # Generate test data with known false positive rates
        n_samples = 1000
        
        # Create predictions with 15% false positive rate (improved from 18.2% baseline)
        true_labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        predictions = true_labels.copy()
        
        # Introduce false positives (15% of negative samples)
        negative_indices = np.where(true_labels == 0)[0]
        n_false_positives = int(len(negative_indices) * 0.15)
        false_positive_indices = np.random.choice(negative_indices, size=n_false_positives, replace=False)
        predictions[false_positive_indices] = 1
        
        # Generate confidence scores
        confidence_scores = np.random.beta(2, 1, n_samples)  # Skewed toward higher confidence
        
        # Create feature data for analysis
        feature_names = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel',  # FLIR features
            'gas_val', 'gas_delta', 'gas_vel'  # SCD41 features
        ]
        
        feature_data = {}
        for feature in feature_names:
            feature_data[feature] = np.random.normal(0, 1, n_samples)
        
        feature_data['ground_truth'] = true_labels
        features_df = pd.DataFrame(feature_data)
        
        # Analyze false positives
        metrics = self.fp_reducer.analyze_false_positives(predictions, true_labels, features_df)
        
        # Check metrics
        self.assertIsInstance(metrics.overall_rate, float)
        self.assertGreaterEqual(metrics.overall_rate, 0.0)
        self.assertLessEqual(metrics.overall_rate, 1.0)
        
        # Should be close to our expected 15% rate
        self.assertAlmostEqual(metrics.overall_rate, 0.15, delta=0.05)
        
        # Check that rate is better than baseline
        self.assertLess(metrics.overall_rate, self.fp_reducer.baseline_false_positive_rate)
    
    def test_false_positive_filter_effectiveness(self):
        """Test effectiveness of false positive filtering."""
        # Create test data with high false positive rate
        n_samples = 500
        true_labels = np.zeros(n_samples, dtype=int)  # All negative samples
        predictions = np.ones(n_samples, dtype=int)   # All predicted as positive (false positives)
        
        # Create varying confidence scores
        confidence_scores = np.random.uniform(0.5, 0.95, n_samples)
        
        # Create feature data that should trigger discrimination
        feature_data = {
            't_mean': np.random.normal(45.0, 8.0, n_samples),   # High temperature
            't_std': np.random.normal(5.0, 2.0, n_samples),
            't_max': np.random.normal(65.0, 12.0, n_samples),   # Very high max temp
            't_p95': np.random.normal(58.0, 10.0, n_samples),
            't_hot_area_pct': np.random.uniform(30.0, 60.0, n_samples),  # Large hot area (sunlight)
            't_hot_largest_blob_pct': np.random.uniform(20.0, 40.0, n_samples),
            't_grad_mean': np.random.normal(2.0, 0.8, n_samples),
            't_grad_std': np.random.normal(1.2, 0.5, n_samples),
            't_diff_mean': np.random.normal(0.3, 0.1, n_samples),
            't_diff_std': np.random.normal(0.15, 0.08, n_samples),
            'flow_mag_mean': np.random.normal(0.5, 0.2, n_samples),
            'flow_mag_std': np.random.normal(0.3, 0.15, n_samples),
            'tproxy_val': np.random.normal(65.0, 12.0, n_samples),
            'tproxy_delta': np.random.normal(1.0, 0.5, n_samples),      # Low temporal change
            'tproxy_vel': np.random.normal(0.2, 0.1, n_samples),        # Low velocity
            'gas_val': np.random.normal(460.0, 60.0, n_samples),        # Normal CO2
            'gas_delta': np.random.normal(3.0, 1.5, n_samples),         # Very low CO2 change
            'gas_vel': np.random.normal(1.0, 0.5, n_samples),           # Low CO2 velocity
            'ground_truth': true_labels
        }
        
        features_df = pd.DataFrame(feature_data)
        
        # Apply filtering
        filtered_predictions = self.fp_reducer.apply_false_positive_filter(
            predictions, confidence_scores, features_df
        )
        
        # Check that filtering reduced false positives
        original_fp_count = np.sum(predictions)
        filtered_fp_count = np.sum(filtered_predictions)
        reduction_rate = (original_fp_count - filtered_fp_count) / original_fp_count
        
        # Should achieve significant reduction
        self.assertGreater(reduction_rate, 0.4, 
                          f"False positive filter should reduce false positives by >40% (actual: {reduction_rate:.2%})")
        
        # Should still have some predictions (not filter everything)
        self.assertGreater(filtered_fp_count, 0, 
                          "Should not filter all predictions")
        
        # Should have fewer false positives than original
        self.assertLess(filtered_fp_count, original_fp_count, 
                       "Filtered predictions should be fewer than original")
    
    def test_reduction_metrics(self):
        """Test false positive reduction metrics calculation."""
        # Generate historical data showing improvement
        baseline_rate = 0.182
        current_rate = 0.087  # 52.2% reduction (from our optimized system)
        
        # Simulate historical tracking
        for i in range(20):
            # Simulate gradual improvement
            rate = baseline_rate - (i * (baseline_rate - current_rate) / 20)
            self.fp_reducer.false_positive_history.append({
                'rate': rate,
                'timestamp': datetime.now() - timedelta(minutes=i*30),
                'sample_count': 1000
            })
        
        # Get reduction metrics
        metrics = self.fp_reducer.get_reduction_metrics()
        
        # Check metrics
        self.assertEqual(metrics['status'], 'success')
        self.assertEqual(metrics['baseline_rate'], baseline_rate)
        self.assertAlmostEqual(metrics['current_rate'], current_rate, places=3)
        
        # Check reduction percentage
        expected_reduction = (baseline_rate - current_rate) / baseline_rate * 100
        self.assertAlmostEqual(metrics['reduction_percentage'], expected_reduction, places=1)
        
        # Should have achieved target (50% reduction)
        self.assertTrue(metrics['achieved_target'], 
                       "Should have achieved 50% reduction target")
        
        # Should be improving (negative trend)
        self.assertTrue(metrics['improving'], 
                       "Trend should be improving (decreasing false positives)")
    
    def test_false_positive_report_generation(self):
        """Test false positive analysis report generation."""
        # Generate some historical data
        for i in range(15):
            rate = 0.182 - (i * 0.01)  # Gradually improving
            self.fp_reducer.false_positive_history.append({
                'rate': rate,
                'timestamp': datetime.now() - timedelta(hours=i),
                'sample_count': 500 + i * 50
            })
        
        # Generate report
        report = self.fp_reducer.generate_false_positive_report()
        
        # Check report structure
        self.assertEqual(report['status'], 'success')
        self.assertIn('timestamp', report)
        self.assertIn('current_metrics', report)
        self.assertIn('historical_statistics', report)
        self.assertIn('reduction_metrics', report)
        self.assertIn('target_metrics', report)
        
        # Check current metrics
        current = report['current_metrics']
        self.assertIn('overall_rate', current)
        self.assertIn('sample_count', current)
        
        # Check that current rate is better than baseline
        self.assertLess(current['overall_rate'], self.fp_reducer.baseline_false_positive_rate)


if __name__ == '__main__':
    unittest.main(verbosity=2)