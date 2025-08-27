"""
Unit tests for the ScenarioGenerator class.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import json
import os
import tempfile
from datetime import datetime, timedelta

from src.data_generation.scenarios.scenario_generator import ScenarioGenerator
from src.data_generation.thermal.thermal_image_generator import ThermalImageGenerator
from src.data_generation.gas.gas_concentration_generator import GasConcentrationGenerator
from src.data_generation.environmental.environmental_data_generator import EnvironmentalDataGenerator


class TestScenarioGenerator(unittest.TestCase):
    """
    Test cases for the ScenarioGenerator class.
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
        
        # Create scenario generator
        self.config = {
            'aws_integration': False
        }
        self.scenario_generator = ScenarioGenerator(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
    
    def test_validate_scenario_definition_valid(self):
        """
        Test validating a valid scenario definition.
        """
        scenario_def = {
            'scenario_type': 'normal',
            'duration': 60,
            'sample_rate': 1.0,
            'room_params': {
                'room_volume': 50.0,
                'ventilation_rate': 0.5
            }
        }
        
        # This should not raise an exception
        self.scenario_generator.validate_scenario_definition(scenario_def)
    
    def test_validate_scenario_definition_invalid(self):
        """
        Test validating an invalid scenario definition.
        """
        # Missing required field
        scenario_def = {
            'scenario_type': 'normal',
            'duration': 60,
            # Missing sample_rate
            'room_params': {
                'room_volume': 50.0,
                'ventilation_rate': 0.5
            }
        }
        
        with self.assertRaises(Exception):
            self.scenario_generator.validate_scenario_definition(scenario_def)
        
        # Invalid scenario type
        scenario_def = {
            'scenario_type': 'invalid_type',
            'duration': 60,
            'sample_rate': 1.0,
            'room_params': {
                'room_volume': 50.0,
                'ventilation_rate': 0.5
            }
        }
        
        with self.assertRaises(Exception):
            self.scenario_generator.validate_scenario_definition(scenario_def)
    
    def test_generate_scenario(self):
        """
        Test generating a scenario.
        """
        start_time = datetime.now()
        duration_seconds = 60
        sample_rate_hz = 1.0
        scenario_type = 'normal'
        scenario_params = {
            'room_params': {
                'room_volume': 50.0,
                'ventilation_rate': 0.5
            }
        }
        
        # Generate scenario
        scenario_data = self.scenario_generator.generate_scenario(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_type=scenario_type,
            scenario_params=scenario_params,
            seed=42
        )
        
        # Check that the scenario data has the expected structure
        self.assertIn('thermal_data', scenario_data)
        self.assertIn('gas_data', scenario_data)
        self.assertIn('environmental_data', scenario_data)
        self.assertIn('metadata', scenario_data)
        self.assertIn('scenario_definition', scenario_data)
        
        # Check that the metadata has the expected fields
        metadata = scenario_data['metadata']
        self.assertEqual(metadata['scenario_type'], scenario_type)
        self.assertEqual(metadata['duration'], duration_seconds)
        self.assertEqual(metadata['sample_rate'], sample_rate_hz)
        self.assertEqual(metadata['seed'], 42)
        
        # Check that the scenario definition has the expected fields
        scenario_def = scenario_data['scenario_definition']
        self.assertEqual(scenario_def['scenario_type'], scenario_type)
        self.assertEqual(scenario_def['duration'], duration_seconds)
        self.assertEqual(scenario_def['sample_rate'], sample_rate_hz)
        self.assertEqual(scenario_def['room_params'], scenario_params['room_params'])
    
    def test_save_scenario(self):
        """
        Test saving a scenario to files.
        """
        # Create a simple scenario
        scenario_data = {
            'thermal_data': {
                'frames': [np.ones((288, 384)) * 25.0],
                'timestamps': [datetime.now()],
                'metadata': {'fire_type': 'test'}
            },
            'gas_data': {
                'gas_data': {
                    'methane': {
                        'measured_concentrations': np.array([1.0]),
                        'timestamps': [datetime.now()]
                    }
                },
                'timestamps': [datetime.now()],
                'metadata': {}
            },
            'environmental_data': {
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
            },
            'metadata': {
                'scenario_type': 'normal',
                'duration': 60,
                'sample_rate': 1.0,
                'seed': 42,
                'start_time': datetime.now().isoformat(),
                'end_time': (datetime.now() + timedelta(seconds=60)).isoformat()
            },
            'scenario_definition': {
                'scenario_type': 'normal',
                'duration': 60,
                'sample_rate': 1.0,
                'room_params': {
                    'room_volume': 50.0,
                    'ventilation_rate': 0.5
                }
            }
        }
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the scenario
            self.scenario_generator.save_scenario(scenario_data, temp_dir)
            
            # Check that the expected files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'metadata.json')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'scenario_definition.json')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'thermal')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'gas')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'environmental')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'combined_data.csv')))
    
    def test_generate_from_definition(self):
        """
        Test generating a scenario from a definition.
        """
        scenario_def = {
            'scenario_type': 'normal',
            'duration': 60,
            'sample_rate': 1.0,
            'room_params': {
                'room_volume': 50.0,
                'ventilation_rate': 0.5
            }
        }
        
        # Generate scenario
        scenario_data = self.scenario_generator.generate_from_definition(
            scenario_def=scenario_def,
            start_time=datetime.now(),
            seed=42
        )
        
        # Check that the scenario data has the expected structure
        self.assertIn('thermal_data', scenario_data)
        self.assertIn('gas_data', scenario_data)
        self.assertIn('environmental_data', scenario_data)
        self.assertIn('metadata', scenario_data)
        self.assertIn('scenario_definition', scenario_data)
        
        # Check that the scenario definition matches the input
        self.assertEqual(scenario_data['scenario_definition']['scenario_type'], scenario_def['scenario_type'])
        self.assertEqual(scenario_data['scenario_definition']['duration'], scenario_def['duration'])
        self.assertEqual(scenario_data['scenario_definition']['sample_rate'], scenario_def['sample_rate'])
        self.assertEqual(scenario_data['scenario_definition']['room_params'], scenario_def['room_params'])
    
    def test_generate_dataset(self):
        """
        Test generating a dataset of scenarios.
        """
        scenario_definitions = [
            {
                'scenario_type': 'normal',
                'duration': 60,
                'sample_rate': 1.0,
                'room_params': {
                    'room_volume': 50.0,
                    'ventilation_rate': 0.5
                }
            },
            {
                'scenario_type': 'electrical_fire',
                'duration': 60,
                'sample_rate': 1.0,
                'room_params': {
                    'room_volume': 50.0,
                    'ventilation_rate': 0.5
                },
                'fire_params': {
                    'fire_size': 5.0
                }
            }
        ]
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate dataset
            dataset_metadata = self.scenario_generator.generate_dataset(
                output_dir=temp_dir,
                scenario_definitions=scenario_definitions,
                num_variations=2,
                seed=42
            )
            
            # Check that the dataset metadata has the expected structure
            self.assertIn('dataset_name', dataset_metadata)
            self.assertIn('num_scenarios', dataset_metadata)
            self.assertIn('creation_date', dataset_metadata)
            self.assertIn('scenarios', dataset_metadata)
            
            # Check that the expected number of scenarios was generated
            self.assertEqual(dataset_metadata['num_scenarios'], len(scenario_definitions) * 2)
            self.assertEqual(len(dataset_metadata['scenarios']), len(scenario_definitions) * 2)
            
            # Check that the dataset metadata file was created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'dataset_metadata.json')))
            
            # Check that the scenario directories were created
            for i in range(len(scenario_definitions) * 2):
                self.assertTrue(os.path.exists(os.path.join(temp_dir, f'scenario_{i:04d}')))


if __name__ == '__main__':
    unittest.main()