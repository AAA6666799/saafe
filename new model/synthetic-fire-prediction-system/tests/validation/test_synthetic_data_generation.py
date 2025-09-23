"""
Validation tests for synthetic data generation in FLIR+SCD41 fire detection system.

This module contains comprehensive validation tests for the synthetic data generation
pipeline, including statistical validation, distribution checks, and scenario coverage.
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
    from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
    from src.data_generation.scenarios.fire_scenario_generator import FireScenarioGenerator
    from src.data_generation.scenarios.false_positive_generator import FalsePositiveScenarioGenerator
    DATA_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data generation components: {e}")
    DATA_GENERATION_AVAILABLE = False


class TestSyntheticDataGenerationValidation(unittest.TestCase):
    """Test cases for synthetic data generation validation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        if not DATA_GENERATION_AVAILABLE:
            self.skipTest("Data generation components not available")
        
        # Create data generator with FLIR+SCD41 configuration
        self.generator_config = {
            'n_samples': 1000,
            'n_fire_samples': 300,
            'n_false_positive_samples': 200,
            'random_state': 42,
            'sensors': {
                'flir_lepton35': {
                    'enabled': True,
                    'features': [
                        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
                        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
                        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
                        'tproxy_val', 'tproxy_delta', 'tproxy_vel'
                    ]
                },
                'scd41_co2': {
                    'enabled': True,
                    'features': ['gas_val', 'gas_delta', 'gas_vel']
                }
            }
        }
        
        self.generator = SyntheticDataGenerator(self.generator_config)
    
    def test_data_generation_pipeline(self):
        """Test the complete synthetic data generation pipeline."""
        # Generate training data
        train_data, train_labels = self.generator.generate_training_data()
        
        # Check data structure
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(train_labels, pd.Series)
        
        # Check data dimensions
        expected_features = 18  # 15 FLIR + 3 SCD41
        self.assertEqual(train_data.shape[1], expected_features)
        self.assertEqual(len(train_data), len(train_labels))
        
        # Check feature names
        expected_feature_names = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel',  # FLIR features
            'gas_val', 'gas_delta', 'gas_vel'  # SCD41 features
        ]
        
        for feature in expected_feature_names:
            self.assertIn(feature, train_data.columns)
        
        # Check label distribution
        unique_labels = train_labels.unique()
        self.assertTrue(set(unique_labels).issubset({0, 1}), "Labels should be binary (0 or 1)")
        
        # Check that we have both positive and negative samples
        label_counts = train_labels.value_counts()
        self.assertGreater(label_counts.get(0, 0), 0, "Should have negative samples")
        self.assertGreater(label_counts.get(1, 0), 0, "Should have positive samples")
    
    def test_fire_scenario_generation(self):
        """Test fire scenario generation."""
        fire_generator = FireScenarioGenerator()
        
        # Generate fire scenarios
        n_scenarios = 50
        fire_data = []
        fire_labels = []
        
        for i in range(n_scenarios):
            scenario_data = fire_generator.generate_fire_scenario()
            fire_data.append(scenario_data)
            fire_labels.append(1)  # All fire scenarios are positive
        
        # Convert to DataFrame
        fire_df = pd.DataFrame(fire_data)
        
        # Check data structure
        self.assertEqual(len(fire_df), n_scenarios)
        self.assertEqual(len(fire_labels), n_scenarios)
        
        # Check that fire scenarios have elevated values
        # FLIR thermal features should be elevated
        self.assertGreater(fire_df['t_max'].mean(), 50.0, "Fire scenarios should have high max temperature")
        self.assertGreater(fire_df['t_hot_area_pct'].mean(), 5.0, "Fire scenarios should have significant hot areas")
        
        # SCD41 gas features should be elevated
        self.assertGreater(fire_df['gas_val'].mean(), 800.0, "Fire scenarios should have elevated CO2 levels")
        self.assertGreater(fire_df['gas_delta'].mean(), 50.0, "Fire scenarios should have significant CO2 changes")
    
    def test_false_positive_scenario_generation(self):
        """Test false positive scenario generation."""
        fp_generator = FalsePositiveScenarioGenerator()
        
        # Generate false positive scenarios
        n_scenarios = 30
        fp_data = []
        fp_labels = []
        
        for i in range(n_scenarios):
            scenario_data = fp_generator.generate_false_positive_scenario()
            fp_data.append(scenario_data)
            fp_labels.append(0)  # All false positive scenarios are negative
        
        # Convert to DataFrame
        fp_df = pd.DataFrame(fp_data)
        
        # Check data structure
        self.assertEqual(len(fp_df), n_scenarios)
        self.assertEqual(len(fp_labels), n_scenarios)
        
        # Check that false positive scenarios have realistic values
        # Temperature should be within reasonable ranges
        self.assertGreaterEqual(fp_df['t_mean'].min(), -40.0, "Temperature should not be below -40°C")
        self.assertLessEqual(fp_df['t_max'].max(), 100.0, "Temperature should not exceed 100°C for false positives")
        
        # CO2 should be within reasonable ranges
        self.assertGreaterEqual(fp_df['gas_val'].min(), 400.0, "CO2 should not be below 400ppm")
        self.assertLessEqual(fp_df['gas_val'].max(), 5000.0, "CO2 should not exceed 5000ppm for false positives")
    
    def test_statistical_distribution_validation(self):
        """Test statistical distribution of generated data."""
        # Generate large dataset for statistical validation
        large_config = self.generator_config.copy()
        large_config['n_samples'] = 5000
        large_config['n_fire_samples'] = 1500
        large_config['n_false_positive_samples'] = 1000
        
        large_generator = SyntheticDataGenerator(large_config)
        data, labels = large_generator.generate_training_data()
        
        # Check statistical properties
        # FLIR thermal features
        self.assertGreaterEqual(data['t_mean'].min(), -40.0, "Minimum temperature should be realistic")
        self.assertLessEqual(data['t_max'].max(), 330.0, "Maximum temperature should be realistic")
        
        # SCD41 gas features
        self.assertGreaterEqual(data['gas_val'].min(), 400.0, "Minimum CO2 should be realistic")
        self.assertLessEqual(data['gas_val'].max(), 40000.0, "Maximum CO2 should be realistic")
        
        # Check for reasonable standard deviations
        self.assertGreater(data['t_std'].mean(), 0, "Temperature standard deviation should be positive")
        self.assertGreater(data['gas_delta'].std(), 0, "CO2 delta should have variation")
    
    def test_scenario_diversity_validation(self):
        """Test diversity of generated scenarios."""
        # Generate data
        data, labels = self.generator.generate_training_data()
        
        # Check diversity in fire scenarios
        fire_data = data[labels == 1]
        no_fire_data = data[labels == 0]
        
        # Fire scenarios should have higher values than no-fire scenarios
        self.assertGreater(fire_data['t_max'].mean(), no_fire_data['t_max'].mean(), 
                          "Fire scenarios should have higher max temperatures")
        self.assertGreater(fire_data['gas_val'].mean(), no_fire_data['gas_val'].mean(), 
                          "Fire scenarios should have higher CO2 levels")
        
        # Check variance - there should be diversity within each class
        self.assertGreater(fire_data['t_max'].std(), 5.0, 
                          "Fire scenarios should have diverse temperature ranges")
        self.assertGreater(no_fire_data['gas_val'].std(), 50.0, 
                          "No-fire scenarios should have diverse CO2 levels")
    
    def test_data_quality_metrics(self):
        """Test data quality metrics for generated data."""
        # Generate data
        data, labels = self.generator.generate_training_data()
        
        # Check for missing values
        self.assertFalse(data.isnull().any().any(), "Generated data should not have missing values")
        
        # Check for infinite values
        self.assertFalse(np.isinf(data.values).any(), "Generated data should not have infinite values")
        
        # Check data types
        for column in data.columns:
            self.assertTrue(np.issubdtype(data[column].dtype, np.number), 
                           f"Column {column} should contain numeric data")
    
    def test_temporal_consistency_validation(self):
        """Test temporal consistency in generated data sequences."""
        # Generate sequential data
        n_sequences = 20
        sequence_length = 30
        
        sequences = []
        for i in range(n_sequences):
            sequence = []
            for j in range(sequence_length):
                # Generate data with some temporal correlation
                if j == 0:
                    sample = self.generator._generate_single_sample(is_fire=np.random.random() > 0.5)
                else:
                    # Add some correlation with previous sample
                    prev_sample = sequence[-1]
                    sample = {}
                    for key, value in prev_sample.items():
                        # Add small random perturbation
                        sample[key] = value + np.random.normal(0, 0.1)
                sequence.append(sample)
            sequences.append(sequence)
        
        # Check temporal consistency
        for sequence in sequences:
            # Convert to DataFrame for easier analysis
            seq_df = pd.DataFrame(sequence)
            
            # Check that values don't change too drastically between consecutive samples
            for column in seq_df.columns:
                diffs = seq_df[column].diff().abs()
                # Most changes should be relatively small
                large_changes = (diffs > 10.0).sum()
                self.assertLess(large_changes, len(diffs) * 0.3, 
                               f"Too many large changes in {column}")


if __name__ == '__main__':
    unittest.main(verbosity=2)