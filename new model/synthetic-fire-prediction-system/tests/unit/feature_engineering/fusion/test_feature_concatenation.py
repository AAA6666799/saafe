"""
Unit tests for the FeatureConcatenation class.
"""

import unittest
import os
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from src.feature_engineering.fusion.early.feature_concatenation import FeatureConcatenation


class TestFeatureConcatenation(unittest.TestCase):
    """
    Test cases for the FeatureConcatenation class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock configuration
        self.config = {
            'normalization': 'min_max',
            'feature_selection': 'none',
            'include_metadata': True
        }
        
        # Create mock feature data
        self.thermal_features = {
            'max_temperature': 120.5,
            'mean_temperature': 85.2,
            'hotspot_count': 3,
            'temperature_rise': 15.3
        }
        
        self.gas_features = {
            'max_concentration': 75.8,
            'mean_concentration': 45.2,
            'concentration_slope': 5.7
        }
        
        self.environmental_features = {
            'temperature': 32.5,
            'humidity': 45.2,
            'pressure': 1013.2,
            'temperature_rise': 2.3
        }
    
    def test_init(self):
        """
        Test initialization of FeatureConcatenation.
        """
        # Create the fusion component
        fusion = FeatureConcatenation(self.config)
        
        # Verify that the configuration was stored
        self.assertEqual(fusion.config, self.config)
    
    def test_validate_config_valid(self):
        """
        Test validating a valid configuration.
        """
        # Create the fusion component
        fusion = FeatureConcatenation(self.config)
        
        # Validate the configuration (should not raise an exception)
        fusion.validate_config()
    
    def test_validate_config_invalid(self):
        """
        Test validating an invalid configuration.
        """
        # Create an invalid configuration (missing required parameter)
        invalid_config = {
            'feature_selection': 'none'
        }
        
        # Create the fusion component
        fusion = FeatureConcatenation(invalid_config)
        
        # Validate the configuration (should raise an exception)
        with self.assertRaises(ValueError):
            fusion.validate_config()
    
    def test_fuse_features(self):
        """
        Test fusing features.
        """
        # Create the fusion component
        fusion = FeatureConcatenation(self.config)
        
        # Fuse features
        result = fusion.fuse_features(
            self.thermal_features,
            self.gas_features,
            self.environmental_features
        )
        
        # Verify the result
        self.assertIn('concatenated_features', result)
        self.assertIn('feature_names', result)
        self.assertIn('feature_counts', result)
        
        # Check feature counts
        self.assertEqual(result['feature_counts']['thermal'], 4)
        self.assertEqual(result['feature_counts']['gas'], 3)
        self.assertEqual(result['feature_counts']['environmental'], 4)
        self.assertEqual(result['feature_counts']['total'], 11)
        
        # Check that all features are present in the concatenated features
        concatenated = result['concatenated_features']
        self.assertIn('thermal_max_temperature', concatenated)
        self.assertIn('gas_max_concentration', concatenated)
        self.assertIn('environmental_temperature', concatenated)
    
    def test_calculate_risk_score(self):
        """
        Test calculating risk score.
        """
        # Create the fusion component
        fusion = FeatureConcatenation(self.config)
        
        # Create mock fused features
        fused_features = {
            'concatenated_features': {
                'thermal_max_temperature': 120.5,
                'thermal_hotspot_count': 3,
                'gas_max_concentration': 75.8,
                'environmental_temperature_rise': 2.3
            }
        }
        
        # Calculate risk score
        risk_score = fusion.calculate_risk_score(fused_features)
        
        # Verify the result
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)
    
    def test_extract_feature_values(self):
        """
        Test extracting feature values.
        """
        # Create the fusion component
        fusion = FeatureConcatenation(self.config)
        
        # Extract feature values
        values = fusion._extract_feature_values(self.thermal_features, 'thermal')
        
        # Verify the result
        self.assertEqual(len(values), 4)
        self.assertEqual(values['thermal_max_temperature'], 120.5)
        self.assertEqual(values['thermal_mean_temperature'], 85.2)
        self.assertEqual(values['thermal_hotspot_count'], 3)
        self.assertEqual(values['thermal_temperature_rise'], 15.3)
    
    def test_normalize_features(self):
        """
        Test normalizing features.
        """
        # Create the fusion component
        fusion = FeatureConcatenation(self.config)
        
        # Create a test DataFrame
        df = pd.DataFrame({
            'feature1': [10.0],
            'feature2': [20.0],
            'feature3': [30.0]
        })
        
        # Normalize features
        normalized = fusion._normalize_features(df)
        
        # Verify the result
        self.assertEqual(normalized['feature1'].iloc[0], 0.0)  # min value
        self.assertEqual(normalized['feature3'].iloc[0], 1.0)  # max value
        self.assertEqual(normalized['feature2'].iloc[0], 0.5)  # middle value
    
    def test_select_features_all(self):
        """
        Test selecting all features.
        """
        # Create the fusion component with 'all' feature selection
        config = self.config.copy()
        config['feature_selection'] = 'all'
        fusion = FeatureConcatenation(config)
        
        # Create a test DataFrame
        df = pd.DataFrame({
            'feature1': [10.0],
            'feature2': [20.0],
            'feature3': [30.0]
        })
        
        # Select features
        selected = fusion._select_features(df)
        
        # Verify the result
        self.assertEqual(len(selected.columns), 3)
        self.assertIn('feature1', selected.columns)
        self.assertIn('feature2', selected.columns)
        self.assertIn('feature3', selected.columns)
    
    def test_select_features_top_k(self):
        """
        Test selecting top k features.
        """
        # Create the fusion component with 'top_k' feature selection
        config = self.config.copy()
        config['feature_selection'] = 'top_k'
        config['top_k'] = 2
        fusion = FeatureConcatenation(config)
        
        # Create a test DataFrame
        df = pd.DataFrame({
            'feature1': [10.0],
            'feature2': [20.0],
            'feature3': [30.0]
        })
        
        # Select features
        selected = fusion._select_features(df)
        
        # Verify the result
        self.assertEqual(len(selected.columns), 2)
        self.assertIn('feature3', selected.columns)  # highest variance
        self.assertIn('feature2', selected.columns)  # second highest variance
    
    def test_select_features_threshold(self):
        """
        Test selecting features above threshold.
        """
        # Create the fusion component with 'threshold' feature selection
        config = self.config.copy()
        config['feature_selection'] = 'threshold'
        config['threshold'] = 0.5
        fusion = FeatureConcatenation(config)
        
        # Create a test DataFrame with different variances
        df = pd.DataFrame({
            'feature1': [10.0],  # low variance
            'feature2': [20.0],  # medium variance
            'feature3': [30.0]   # high variance
        })
        
        # Mock the variance calculation
        with patch.object(df, 'var') as mock_var:
            mock_var.return_value = pd.Series({
                'feature1': 0.1,
                'feature2': 0.6,
                'feature3': 0.9
            })
            
            # Select features
            selected = fusion._select_features(df)
        
        # Verify the result
        self.assertEqual(len(selected.columns), 2)
        self.assertIn('feature2', selected.columns)
        self.assertIn('feature3', selected.columns)


if __name__ == '__main__':
    unittest.main()