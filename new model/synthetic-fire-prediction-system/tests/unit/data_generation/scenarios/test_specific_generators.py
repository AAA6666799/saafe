"""
Unit tests for the specific scenario generators.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import json
import os
import tempfile
from datetime import datetime, timedelta

from src.data_generation.scenarios.specific_generators import (
    SpecificScenarioGenerator,
    NormalScenarioGenerator,
    ElectricalFireScenarioGenerator,
    ChemicalFireScenarioGenerator,
    SmolderingFireScenarioGenerator,
    RapidCombustionScenarioGenerator
)
from src.data_generation.thermal.thermal_image_generator import ThermalImageGenerator
from src.data_generation.gas.gas_concentration_generator import GasConcentrationGenerator
from src.data_generation.environmental.environmental_data_generator import EnvironmentalDataGenerator


class TestSpecificScenarioGenerator(unittest.TestCase):
    """
    Test cases for the SpecificScenarioGenerator class.
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
        
        # Create a concrete subclass for testing the abstract base class
        class ConcreteSpecificGenerator(SpecificScenarioGenerator):
            def get_scenario_type(self):
                return 'test_type'
        
        self.specific_generator = ConcreteSpecificGenerator(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
    
    def test_deep_update(self):
        """
        Test the _deep_update method.
        """
        d = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3
            }
        }
        
        u = {
            'a': 10,
            'b': {
                'c': 20,
                'e': 30
            },
            'f': 40
        }
        
        expected = {
            'a': 10,
            'b': {
                'c': 20,
                'd': 3,
                'e': 30
            },
            'f': 40
        }
        
        self.specific_generator._deep_update(d, u)
        self.assertEqual(d, expected)
    
    def test_generate_specific_scenario(self):
        """
        Test generating a specific scenario.
        """
        # Mock the generate_scenario method
        self.specific_generator.generate_scenario = MagicMock()
        self.specific_generator.generate_scenario.return_value = {'test': 'data'}
        
        # Generate a specific scenario
        start_time = datetime.now()
        duration_seconds = 60
        sample_rate_hz = 1.0
        scenario_params = {
            'thermal_params': {
                'max_temperature': 100.0
            }
        }
        
        result = self.specific_generator.generate_specific_scenario(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_params=scenario_params,
            seed=42
        )
        
        # Check that generate_scenario was called with the expected parameters
        self.specific_generator.generate_scenario.assert_called_once()
        call_args = self.specific_generator.generate_scenario.call_args[1]
        self.assertEqual(call_args['start_time'], start_time)
        self.assertEqual(call_args['duration_seconds'], duration_seconds)
        self.assertEqual(call_args['sample_rate_hz'], sample_rate_hz)
        self.assertEqual(call_args['scenario_type'], 'test_type')
        self.assertEqual(call_args['seed'], 42)
        
        # Check that the scenario parameters were combined correctly
        combined_params = call_args['scenario_params']
        self.assertIn('room_params', combined_params)
        self.assertIn('thermal_params', combined_params)
        self.assertEqual(combined_params['thermal_params']['max_temperature'], 100.0)
        
        # Check that the result is as expected
        self.assertEqual(result, {'test': 'data'})


class TestNormalScenarioGenerator(unittest.TestCase):
    """
    Test cases for the NormalScenarioGenerator class.
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
        
        # Create normal scenario generator
        self.normal_generator = NormalScenarioGenerator(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
    
    def test_get_scenario_type(self):
        """
        Test that the scenario type is correct.
        """
        self.assertEqual(self.normal_generator.get_scenario_type(), 'normal')
    
    def test_default_scenario_params(self):
        """
        Test that the default scenario parameters are set correctly.
        """
        params = self.normal_generator.default_scenario_params
        
        # Check that the expected parameters are present
        self.assertIn('thermal_params', params)
        self.assertIn('gas_params', params)
        self.assertIn('environmental_params', params)
        self.assertIn('metadata', params)
        
        # Check specific values
        self.assertEqual(params['thermal_params']['max_temperature'], 30.0)
        self.assertEqual(params['thermal_params']['hotspot_count'], 0)
        self.assertEqual(params['metadata']['classification'], 'normal')


class TestFireScenarioGenerators(unittest.TestCase):
    """
    Test cases for the fire-specific scenario generators.
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
        
        # Create fire scenario generators
        self.electrical_generator = ElectricalFireScenarioGenerator(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
        
        self.chemical_generator = ChemicalFireScenarioGenerator(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
        
        self.smoldering_generator = SmolderingFireScenarioGenerator(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
        
        self.rapid_combustion_generator = RapidCombustionScenarioGenerator(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
    
    def test_scenario_types(self):
        """
        Test that the scenario types are correct.
        """
        self.assertEqual(self.electrical_generator.get_scenario_type(), 'electrical_fire')
        self.assertEqual(self.chemical_generator.get_scenario_type(), 'chemical_fire')
        self.assertEqual(self.smoldering_generator.get_scenario_type(), 'smoldering_fire')
        self.assertEqual(self.rapid_combustion_generator.get_scenario_type(), 'rapid_combustion')
    
    def test_default_scenario_params(self):
        """
        Test that the default scenario parameters are set correctly.
        """
        # Test electrical fire parameters
        params = self.electrical_generator.default_scenario_params
        self.assertIn('fire_params', params)
        self.assertIn('thermal_params', params)
        self.assertIn('gas_params', params)
        self.assertIn('environmental_params', params)
        self.assertIn('metadata', params)
        
        self.assertEqual(params['thermal_params']['max_temperature'], 200.0)
        self.assertEqual(params['metadata']['fire_type'], 'electrical')
        
        # Test chemical fire parameters
        params = self.chemical_generator.default_scenario_params
        self.assertEqual(params['thermal_params']['max_temperature'], 300.0)
        self.assertEqual(params['metadata']['fire_type'], 'chemical')
        
        # Test smoldering fire parameters
        params = self.smoldering_generator.default_scenario_params
        self.assertEqual(params['thermal_params']['max_temperature'], 150.0)
        self.assertEqual(params['metadata']['fire_type'], 'smoldering')
        
        # Test rapid combustion parameters
        params = self.rapid_combustion_generator.default_scenario_params
        self.assertEqual(params['thermal_params']['max_temperature'], 500.0)
        self.assertEqual(params['metadata']['fire_type'], 'rapid_combustion')
    
    def test_generate_specific_scenario(self):
        """
        Test generating specific fire scenarios.
        """
        # Mock the generate_scenario method
        self.electrical_generator.generate_scenario = MagicMock()
        self.electrical_generator.generate_scenario.return_value = {'test': 'electrical_data'}
        
        # Generate an electrical fire scenario
        start_time = datetime.now()
        duration_seconds = 60
        sample_rate_hz = 1.0
        
        result = self.electrical_generator.generate_specific_scenario(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            seed=42
        )
        
        # Check that generate_scenario was called with the expected parameters
        self.electrical_generator.generate_scenario.assert_called_once()
        call_args = self.electrical_generator.generate_scenario.call_args[1]
        self.assertEqual(call_args['start_time'], start_time)
        self.assertEqual(call_args['duration_seconds'], duration_seconds)
        self.assertEqual(call_args['sample_rate_hz'], sample_rate_hz)
        self.assertEqual(call_args['scenario_type'], 'electrical_fire')
        self.assertEqual(call_args['seed'], 42)
        
        # Check that the result is as expected
        self.assertEqual(result, {'test': 'electrical_data'})


if __name__ == '__main__':
    unittest.main()