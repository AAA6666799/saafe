"""
Unit tests for the FalsePositiveGenerator class.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import json
import os
import tempfile
from datetime import datetime, timedelta

from src.data_generation.scenarios.false_positive_generator import FalsePositiveGenerator
from src.data_generation.thermal.thermal_image_generator import ThermalImageGenerator
from src.data_generation.gas.gas_concentration_generator import GasConcentrationGenerator
from src.data_generation.environmental.environmental_data_generator import EnvironmentalDataGenerator


class TestFalsePositiveGenerator(unittest.TestCase):
    """
    Test cases for the FalsePositiveGenerator class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create mock generators
        self.mock_thermal_generator = MagicMock(spec=ThermalImageGenerator)
        self.mock_gas_generator = MagicMock(spec=GasConcentrationGenerator)
        self.mock_env_generator = MagicMock(spec=EnvironmentalDataGenerator)
        
        # Configure mock thermal generator
        self.mock_thermal_generator.generate.return_value = {
            'frames': [np.ones((288, 384)) * 25.0],
            'timestamps': [datetime.now()],
            'metadata': {'fire_type': 'test'}
        }
        
        # Configure mock gas generator
        self.mock_gas_generator.generate.return_value = {
            'gas_data': {
                'methane': {
                    'measured_concentrations': np.array([1.0]),
                    'timestamps': [datetime.now()]
                }
            },
            'timestamps': [datetime.now()],
            'metadata': {}
        }
        self.mock_gas_generator.coordinate_with_thermal_data.return_value = self.mock_gas_generator.generate.return_value
        
        # Configure mock environmental generator
        self.mock_env_generator.generate.return_value = {
            'environmental_data': {
                'temperature': {
                    'values': np.array([22.0]),
                    'unit': '°C'
                },
                'humidity': {
                    'values': np.array([50.0]),
                    'unit': '%'
                }
            },
            'timestamps': [datetime.now()],
            'metadata': {}
        }
        
        # Create config
        self.config = {
            'aws_integration': False
        }
        
        # Create false positive generator
        self.fp_generator = FalsePositiveGenerator(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
    
    def test_get_scenario_type(self):
        """
        Test that the scenario type is correct.
        """
        self.assertEqual(self.fp_generator.get_scenario_type(), 'false_positive')
    
    def test_false_positive_types(self):
        """
        Test that the false positive types are initialized correctly.
        """
        fp_types = self.fp_generator.false_positive_types
        
        # Check that the expected types are present
        self.assertIn('cooking', fp_types)
        self.assertIn('heating', fp_types)
        self.assertIn('steam', fp_types)
        self.assertIn('sunlight', fp_types)
        self.assertIn('electronic_equipment', fp_types)
        
        # Check that each type has the expected parameters
        for fp_type, params in fp_types.items():
            self.assertIn('fire_params', params)
            self.assertIn('thermal_params', params)
            self.assertIn('environmental_params', params)
            self.assertIn('metadata', params)
            
            # Check that the metadata has the correct classification
            self.assertEqual(params['metadata']['classification'], 'false_positive')
            self.assertEqual(params['metadata']['false_positive_type'], fp_type)
    
    def test_generate_false_positive(self):
        """
        Test generating a false positive scenario.
        """
        # Mock the generate_scenario method
        self.fp_generator.generate_scenario = MagicMock()
        self.fp_generator.generate_scenario.return_value = {
            'metadata': {
                'scenario_type': 'false_positive'
            }
        }
        
        # Generate a false positive scenario
        start_time = datetime.now()
        duration_seconds = 60
        sample_rate_hz = 1.0
        
        result = self.fp_generator.generate_false_positive(
            false_positive_type='cooking',
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            seed=42
        )
        
        # Check that generate_scenario was called with the expected parameters
        self.fp_generator.generate_scenario.assert_called_once()
        call_args = self.fp_generator.generate_scenario.call_args[1]
        self.assertEqual(call_args['start_time'], start_time)
        self.assertEqual(call_args['duration_seconds'], duration_seconds)
        self.assertEqual(call_args['sample_rate_hz'], sample_rate_hz)
        self.assertEqual(call_args['scenario_type'], 'false_positive')
        self.assertEqual(call_args['seed'], 42)
        
        # Check that the scenario parameters include the false positive type
        scenario_params = call_args['scenario_params']
        self.assertIn('metadata', scenario_params)
        self.assertEqual(scenario_params['metadata']['false_positive_type'], 'cooking')
    
    def test_generate_false_positive_invalid_type(self):
        """
        Test generating a false positive scenario with an invalid type.
        """
        with self.assertRaises(ValueError):
            self.fp_generator.generate_false_positive(
                false_positive_type='invalid_type',
                start_time=datetime.now(),
                duration_seconds=60,
                sample_rate_hz=1.0
            )
    
    def test_generate_cooking_scenario(self):
        """
        Test generating a cooking false positive scenario.
        """
        # Mock the generate_false_positive method
        self.fp_generator.generate_false_positive = MagicMock()
        self.fp_generator.generate_false_positive.return_value = {
            'metadata': {
                'scenario_type': 'false_positive',
                'false_positive_type': 'cooking',
                'cooking_type': 'frying'
            }
        }
        
        # Generate a cooking scenario
        start_time = datetime.now()
        duration_seconds = 60
        sample_rate_hz = 1.0
        
        result = self.fp_generator.generate_cooking_scenario(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            cooking_type='frying',
            seed=42
        )
        
        # Check that generate_false_positive was called with the expected parameters
        self.fp_generator.generate_false_positive.assert_called_once()
        call_args = self.fp_generator.generate_false_positive.call_args[1]
        self.assertEqual(call_args['false_positive_type'], 'cooking')
        self.assertEqual(call_args['start_time'], start_time)
        self.assertEqual(call_args['duration_seconds'], duration_seconds)
        self.assertEqual(call_args['sample_rate_hz'], sample_rate_hz)
        self.assertEqual(call_args['seed'], 42)
        
        # Check that the scenario parameters include the cooking type
        scenario_params = call_args['scenario_params']
        self.assertIn('metadata', scenario_params)
        self.assertEqual(scenario_params['metadata']['cooking_type'], 'frying')
    
    def test_generate_heating_scenario(self):
        """
        Test generating a heating system false positive scenario.
        """
        # Mock the generate_false_positive method
        self.fp_generator.generate_false_positive = MagicMock()
        self.fp_generator.generate_false_positive.return_value = {
            'metadata': {
                'scenario_type': 'false_positive',
                'false_positive_type': 'heating',
                'heating_type': 'furnace'
            }
        }
        
        # Generate a heating scenario
        start_time = datetime.now()
        duration_seconds = 60
        sample_rate_hz = 1.0
        
        result = self.fp_generator.generate_heating_scenario(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            heating_type='furnace',
            seed=42
        )
        
        # Check that generate_false_positive was called with the expected parameters
        self.fp_generator.generate_false_positive.assert_called_once()
        call_args = self.fp_generator.generate_false_positive.call_args[1]
        self.assertEqual(call_args['false_positive_type'], 'heating')
        self.assertEqual(call_args['start_time'], start_time)
        self.assertEqual(call_args['duration_seconds'], duration_seconds)
        self.assertEqual(call_args['sample_rate_hz'], sample_rate_hz)
        self.assertEqual(call_args['seed'], 42)
        
        # Check that the scenario parameters include the heating type
        scenario_params = call_args['scenario_params']
        self.assertIn('metadata', scenario_params)
        self.assertEqual(scenario_params['metadata']['heating_type'], 'furnace')
    
    def test_generate_random_false_positive(self):
        """
        Test generating a random false positive scenario.
        """
        # Mock the generate_false_positive, generate_cooking_scenario, and generate_heating_scenario methods
        self.fp_generator.generate_false_positive = MagicMock()
        self.fp_generator.generate_cooking_scenario = MagicMock()
        self.fp_generator.generate_heating_scenario = MagicMock()
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Generate a random false positive scenario
        start_time = datetime.now()
        duration_seconds = 60
        sample_rate_hz = 1.0
        
        self.fp_generator.generate_random_false_positive(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            seed=42
        )
        
        # Check that one of the generate methods was called
        self.assertTrue(
            self.fp_generator.generate_false_positive.called or
            self.fp_generator.generate_cooking_scenario.called or
            self.fp_generator.generate_heating_scenario.called
        )


if __name__ == '__main__':
    unittest.main()