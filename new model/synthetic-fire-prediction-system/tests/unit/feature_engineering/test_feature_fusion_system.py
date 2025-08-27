"""
Unit tests for the FeatureFusionSystem class.
"""

import unittest
import os
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from src.feature_engineering.fusion.feature_fusion_system import FeatureFusionSystem, FusionPipeline


class TestFeatureFusionSystem(unittest.TestCase):
    """
    Test cases for the FeatureFusionSystem class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a temporary output directory
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a mock configuration
        self.config = {
            'output_dir': self.output_dir,
            'fusion_components': [
                {
                    'type': 'early.FeatureConcatenation',
                    'config': {
                        'normalization': 'min_max',
                        'feature_selection': 'none',
                        'include_metadata': True
                    }
                }
            ],
            'log_level': 'INFO'
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
    
    def tearDown(self):
        """
        Clean up test fixtures.
        """
        # Remove the temporary output directory
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    @patch('src.feature_engineering.fusion.feature_fusion_system.importlib.import_module')
    def test_init(self, mock_import_module):
        """
        Test initialization of FeatureFusionSystem.
        """
        # Create a mock fusion component
        mock_component = MagicMock()
        mock_component_class = MagicMock(return_value=mock_component)
        
        # Configure the mock import_module to return a module with the component class
        mock_module = MagicMock()
        mock_module.FeatureConcatenation = mock_component_class
        mock_import_module.return_value = mock_module
        
        # Create the fusion system
        fusion_system = FeatureFusionSystem(self.config)
        
        # Verify that the fusion components were loaded
        self.assertEqual(len(fusion_system.fusion_components), 1)
        mock_import_module.assert_called_once()
        mock_component_class.assert_called_once()
    
    @patch('src.feature_engineering.fusion.feature_fusion_system.importlib.import_module')
    def test_fuse_features(self, mock_import_module):
        """
        Test fusing features.
        """
        # Create a mock fusion component
        mock_component = MagicMock()
        mock_component.fuse_features.return_value = {
            'fused_data': {'feature1': 1.0, 'feature2': 2.0},
            'risk_score': 0.75
        }
        mock_component_class = MagicMock(return_value=mock_component)
        
        # Configure the mock import_module to return a module with the component class
        mock_module = MagicMock()
        mock_module.FeatureConcatenation = mock_component_class
        mock_import_module.return_value = mock_module
        
        # Create the fusion system
        fusion_system = FeatureFusionSystem(self.config)
        
        # Fuse features
        result = fusion_system.fuse_features(
            self.thermal_features,
            self.gas_features,
            self.environmental_features
        )
        
        # Verify that the fusion component was called
        mock_component.fuse_features.assert_called_once_with(
            self.thermal_features,
            self.gas_features,
            self.environmental_features
        )
        
        # Verify the result
        self.assertIn('fused_features', result)
        self.assertIn('early.FeatureConcatenation', result['fused_features'])
        self.assertEqual(result['fused_features']['early.FeatureConcatenation']['risk_score'], 0.75)
    
    @patch('src.feature_engineering.fusion.feature_fusion_system.importlib.import_module')
    def test_get_available_components(self, mock_import_module):
        """
        Test getting available components.
        """
        # Create a mock fusion component
        mock_component = MagicMock()
        mock_component_class = MagicMock(return_value=mock_component)
        
        # Configure the mock import_module to return a module with the component class
        mock_module = MagicMock()
        mock_module.FeatureConcatenation = mock_component_class
        mock_import_module.return_value = mock_module
        
        # Create the fusion system
        fusion_system = FeatureFusionSystem(self.config)
        
        # Get available components
        components = fusion_system.get_available_components()
        
        # Verify the result
        self.assertEqual(len(components), 1)
        self.assertIn('early.FeatureConcatenation', components)
    
    @patch('src.feature_engineering.fusion.feature_fusion_system.importlib.import_module')
    def test_get_component_info(self, mock_import_module):
        """
        Test getting component information.
        """
        # Create a mock fusion component
        mock_component = MagicMock()
        mock_component_class = MagicMock(return_value=mock_component)
        
        # Configure the mock import_module to return a module with the component class
        mock_module = MagicMock()
        mock_module.FeatureConcatenation = mock_component_class
        mock_import_module.return_value = mock_module
        
        # Create the fusion system
        fusion_system = FeatureFusionSystem(self.config)
        
        # Get component info
        info = fusion_system.get_component_info('early.FeatureConcatenation')
        
        # Verify the result
        self.assertEqual(info['type'], 'early.FeatureConcatenation')
        self.assertEqual(info['config']['normalization'], 'min_max')
        self.assertEqual(info['config']['feature_selection'], 'none')
    
    @patch('src.feature_engineering.fusion.feature_fusion_system.FeatureExtractionFramework')
    @patch('src.feature_engineering.fusion.feature_fusion_system.FeatureFusionSystem')
    def test_fusion_pipeline(self, mock_fusion_system_class, mock_extraction_framework_class):
        """
        Test the FusionPipeline class.
        """
        # Create mock objects
        mock_extraction_framework = MagicMock()
        mock_extraction_framework.extract_features.return_value = {
            'metadata': {'dataset_path': 'test_dataset'},
            'features': {
                'thermal': self.thermal_features,
                'gas': self.gas_features,
                'environmental': self.environmental_features
            }
        }
        mock_extraction_framework_class.return_value = mock_extraction_framework
        
        mock_fusion_system = MagicMock()
        mock_fusion_system.fuse_features.return_value = {
            'metadata': {'fusion_time': '2025-01-01T00:00:00'},
            'fused_features': {
                'early.FeatureConcatenation': {
                    'fused_data': {'feature1': 1.0, 'feature2': 2.0},
                    'risk_score': 0.75
                }
            }
        }
        mock_fusion_system_class.return_value = mock_fusion_system
        
        # Create the pipeline
        pipeline_config = {
            'output_dir': self.output_dir,
            'extraction_config': {'extractors': []},
            'fusion_config': {'fusion_components': []}
        }
        pipeline = FusionPipeline(pipeline_config)
        
        # Process a dataset
        result = pipeline.process_dataset('test_dataset')
        
        # Verify that the extraction framework and fusion system were called
        mock_extraction_framework.extract_features.assert_called_once()
        mock_fusion_system.fuse_features.assert_called_once()
        
        # Verify the result
        self.assertIn('metadata', result)
        self.assertIn('extracted_features', result)
        self.assertIn('fused_features', result)
        self.assertEqual(result['metadata']['dataset_path'], 'test_dataset')


if __name__ == '__main__':
    unittest.main()