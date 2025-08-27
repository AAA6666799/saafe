"""
Unit tests for the ScenarioMixer class.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import json
import os
import tempfile
from datetime import datetime, timedelta

from src.data_generation.scenarios.scenario_mixer import ScenarioMixer
from src.data_generation.scenarios.scenario_generator import ScenarioGenerator
from src.data_generation.scenarios.specific_generators import (
    NormalScenarioGenerator,
    ElectricalFireScenarioGenerator,
    ChemicalFireScenarioGenerator,
    SmolderingFireScenarioGenerator,
    RapidCombustionScenarioGenerator
)
from src.data_generation.scenarios.false_positive_generator import FalsePositiveGenerator
from src.data_generation.thermal.thermal_image_generator import ThermalImageGenerator
from src.data_generation.gas.gas_concentration_generator import GasConcentrationGenerator
from src.data_generation.environmental.environmental_data_generator import EnvironmentalDataGenerator


class TestScenarioMixer(unittest.TestCase):
    """
    Test cases for the ScenarioMixer class.
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
        
        # Create scenario mixer
        self.scenario_mixer = ScenarioMixer(
            self.mock_thermal_generator,
            self.mock_gas_generator,
            self.mock_env_generator,
            self.config
        )
        
        # Mock the specific generators
        self.scenario_mixer.scenario_generator = MagicMock(spec=ScenarioGenerator)
        self.scenario_mixer.normal_generator = MagicMock(spec=NormalScenarioGenerator)
        self.scenario_mixer.electrical_fire_generator = MagicMock(spec=ElectricalFireScenarioGenerator)
        self.scenario_mixer.chemical_fire_generator = MagicMock(spec=ChemicalFireScenarioGenerator)
        self.scenario_mixer.smoldering_fire_generator = MagicMock(spec=SmolderingFireScenarioGenerator)
        self.scenario_mixer.rapid_combustion_generator = MagicMock(spec=RapidCombustionScenarioGenerator)
        self.scenario_mixer.false_positive_generator = MagicMock(spec=FalsePositiveGenerator)
        
        # Update generator map
        self.scenario_mixer.generator_map = {
            'normal': self.scenario_mixer.normal_generator,
            'electrical_fire': self.scenario_mixer.electrical_fire_generator,
            'chemical_fire': self.scenario_mixer.chemical_fire_generator,
            'smoldering_fire': self.scenario_mixer.smoldering_fire_generator,
            'rapid_combustion': self.scenario_mixer.rapid_combustion_generator,
            'false_positive': self.scenario_mixer.false_positive_generator
        }
    
    def test_create_scenario_variation(self):
        """
        Test creating a variation of a scenario.
        """
        # Create a base scenario
        base_scenario = {
            'thermal_data': {
                'frames': [np.ones((288, 384)) * 25.0],
                'timestamps': [datetime.now()]
            },
            'gas_data': {
                'gas_data': {
                    'methane': {
                        'measured_concentrations': np.array([1.0])
                    }
                },
                'timestamps': [datetime.now()]
            },
            'environmental_data': {
                'environmental_data': {
                    'temperature': {
                        'values': np.array([22.0])
                    }
                },
                'timestamps': [datetime.now()]
            },
            'metadata': {
                'scenario_type': 'normal',
                'start_time': datetime.now().isoformat()
            },
            'scenario_definition': {
                'scenario_type': 'normal',
                'duration': 60,
                'sample_rate': 1.0,
                'thermal_params': {
                    'max_temperature': 30.0
                }
            }
        }
        
        # Define variation parameters
        variation_params = {
            'thermal_params.max_temperature': {'factor': 1.5}
        }
        
        # Mock the generator's generate_scenario method
        self.scenario_mixer.normal_generator.generate_scenario.return_value = {
            'metadata': {'scenario_type': 'normal', 'variation_of': 'unknown'}
        }
        
        # Create a variation
        result = self.scenario_mixer.create_scenario_variation(
            base_scenario=base_scenario,
            variation_params=variation_params,
            seed=42
        )
        
        # Check that generate_scenario was called with the expected parameters
        self.scenario_mixer.normal_generator.generate_scenario.assert_called_once()
        call_args = self.scenario_mixer.normal_generator.generate_scenario.call_args[1]
        self.assertEqual(call_args['scenario_type'], 'normal')
        self.assertEqual(call_args['seed'], 42)
        
        # Check that the variation was applied
        scenario_params = call_args['scenario_params']
        self.assertIn('thermal_params', scenario_params)
        self.assertEqual(scenario_params['thermal_params']['max_temperature'], 45.0)  # 30.0 * 1.5
    
    def test_apply_variations(self):
        """
        Test applying variations to a scenario definition.
        """
        # Create a scenario definition
        scenario_def = {
            'scenario_type': 'normal',
            'duration': 60,
            'sample_rate': 1.0,
            'thermal_params': {
                'max_temperature': 30.0,
                'hotspot_count': 0
            },
            'gas_params': {
                'gas_types': ['methane']
            }
        }
        
        # Define variation parameters
        variation_params = {
            'thermal_params.max_temperature': {'factor': 1.5},
            'thermal_params.hotspot_count': {'value': 1},
            'gas_params.gas_types': ['methane', 'carbon_monoxide'],
            'new_param': 'new_value'
        }
        
        # Apply variations
        self.scenario_mixer._apply_variations(scenario_def, variation_params)
        
        # Check that the variations were applied correctly
        self.assertEqual(scenario_def['thermal_params']['max_temperature'], 45.0)  # 30.0 * 1.5
        self.assertEqual(scenario_def['thermal_params']['hotspot_count'], 1)
        self.assertEqual(scenario_def['gas_params']['gas_types'], ['methane', 'carbon_monoxide'])
        self.assertEqual(scenario_def['new_param'], 'new_value')
    
    def test_combine_scenarios_weighted_average(self):
        """
        Test combining scenarios using weighted average.
        """
        # Create test scenarios
        scenarios = [
            {
                'thermal_data': {
                    'frames': [np.ones((2, 2)) * 20.0, np.ones((2, 2)) * 30.0],
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'gas_data': {
                    'gas_data': {
                        'methane': {
                            'measured_concentrations': np.array([1.0, 2.0])
                        }
                    },
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'environmental_data': {
                    'environmental_data': {
                        'temperature': {
                            'values': np.array([20.0, 22.0])
                        }
                    },
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'metadata': {
                    'scenario_type': 'normal',
                    'scenario_id': 'scenario_1'
                },
                'scenario_definition': {
                    'scenario_type': 'normal',
                    'duration': 2,
                    'sample_rate': 1.0
                }
            },
            {
                'thermal_data': {
                    'frames': [np.ones((2, 2)) * 40.0, np.ones((2, 2)) * 50.0],
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'gas_data': {
                    'gas_data': {
                        'methane': {
                            'measured_concentrations': np.array([3.0, 4.0])
                        }
                    },
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'environmental_data': {
                    'environmental_data': {
                        'temperature': {
                            'values': np.array([24.0, 26.0])
                        }
                    },
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'metadata': {
                    'scenario_type': 'electrical_fire',
                    'scenario_id': 'scenario_2'
                },
                'scenario_definition': {
                    'scenario_type': 'electrical_fire',
                    'duration': 2,
                    'sample_rate': 1.0
                }
            }
        ]
        
        # Define weights
        weights = [0.3, 0.7]
        
        # Combine scenarios
        result = self.scenario_mixer.combine_scenarios(
            scenarios=scenarios,
            weights=weights,
            combination_method='weighted_average',
            seed=42
        )
        
        # Check that the result has the expected structure
        self.assertIn('thermal_data', result)
        self.assertIn('gas_data', result)
        self.assertIn('environmental_data', result)
        self.assertIn('metadata', result)
        self.assertIn('scenario_definition', result)
        
        # Check that the metadata has the expected fields
        metadata = result['metadata']
        self.assertEqual(metadata['scenario_type'], 'combined')
        self.assertEqual(metadata['combination_method'], 'weighted_average')
        self.assertEqual(metadata['combined_from'], ['scenario_1', 'scenario_2'])
        self.assertEqual(metadata['weights'], weights)
        
        # Check that the thermal frames were combined correctly
        thermal_frames = result['thermal_data']['frames']
        self.assertEqual(len(thermal_frames), 2)
        
        # Expected values: 20.0 * 0.3 + 40.0 * 0.7 = 34.0, 30.0 * 0.3 + 50.0 * 0.7 = 44.0
        np.testing.assert_allclose(thermal_frames[0], np.ones((2, 2)) * 34.0)
        np.testing.assert_allclose(thermal_frames[1], np.ones((2, 2)) * 44.0)
    
    def test_combine_scenarios_max(self):
        """
        Test combining scenarios using max values.
        """
        # Create test scenarios
        scenarios = [
            {
                'thermal_data': {
                    'frames': [np.ones((2, 2)) * 20.0, np.ones((2, 2)) * 50.0],
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'gas_data': {
                    'gas_data': {
                        'methane': {
                            'measured_concentrations': np.array([1.0, 4.0])
                        }
                    },
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'environmental_data': {
                    'environmental_data': {
                        'temperature': {
                            'values': np.array([20.0, 26.0])
                        }
                    },
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'metadata': {
                    'scenario_type': 'normal',
                    'scenario_id': 'scenario_1'
                },
                'scenario_definition': {
                    'scenario_type': 'normal',
                    'duration': 2,
                    'sample_rate': 1.0
                }
            },
            {
                'thermal_data': {
                    'frames': [np.ones((2, 2)) * 40.0, np.ones((2, 2)) * 30.0],
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'gas_data': {
                    'gas_data': {
                        'methane': {
                            'measured_concentrations': np.array([3.0, 2.0])
                        }
                    },
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'environmental_data': {
                    'environmental_data': {
                        'temperature': {
                            'values': np.array([24.0, 22.0])
                        }
                    },
                    'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1)]
                },
                'metadata': {
                    'scenario_type': 'electrical_fire',
                    'scenario_id': 'scenario_2'
                },
                'scenario_definition': {
                    'scenario_type': 'electrical_fire',
                    'duration': 2,
                    'sample_rate': 1.0
                }
            }
        ]
        
        # Combine scenarios
        result = self.scenario_mixer.combine_scenarios(
            scenarios=scenarios,
            combination_method='max',
            seed=42
        )
        
        # Check that the result has the expected structure
        self.assertIn('thermal_data', result)
        self.assertIn('gas_data', result)
        self.assertIn('environmental_data', result)
        self.assertIn('metadata', result)
        self.assertIn('scenario_definition', result)
        
        # Check that the metadata has the expected fields
        metadata = result['metadata']
        self.assertEqual(metadata['scenario_type'], 'combined')
        self.assertEqual(metadata['combination_method'], 'max')
        self.assertEqual(metadata['combined_from'], ['scenario_1', 'scenario_2'])
        
        # Check that the thermal frames were combined correctly
        thermal_frames = result['thermal_data']['frames']
        self.assertEqual(len(thermal_frames), 2)
        
        # Expected values: max(20.0, 40.0) = 40.0, max(50.0, 30.0) = 50.0
        np.testing.assert_allclose(thermal_frames[0], np.ones((2, 2)) * 40.0)
        np.testing.assert_allclose(thermal_frames[1], np.ones((2, 2)) * 50.0)
    
    def test_generate_dataset_with_distribution(self):
        """
        Test generating a dataset with a controlled distribution of scenario types.
        """
        # Mock the generator methods
        for generator in self.scenario_mixer.generator_map.values():
            generator.generate_specific_scenario = MagicMock()
            generator.generate_specific_scenario.return_value = {
                'metadata': {'scenario_type': 'test'},
                'scenario_definition': {'scenario_type': 'test'}
            }
        
        self.scenario_mixer.false_positive_generator.generate_random_false_positive = MagicMock()
        self.scenario_mixer.false_positive_generator.generate_random_false_positive.return_value = {
            'metadata': {'scenario_type': 'false_positive'},
            'scenario_definition': {'scenario_type': 'false_positive'}
        }
        
        self.scenario_mixer.scenario_generator.save_scenario = MagicMock()
        
        # Define distribution
        distribution = {
            'normal': 0.5,
            'electrical_fire': 0.3,
            'false_positive': 0.2
        }
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate dataset
            dataset_metadata = self.scenario_mixer.generate_dataset_with_distribution(
                output_dir=temp_dir,
                distribution=distribution,
                num_scenarios=10,
                duration_seconds=60,
                sample_rate_hz=1.0,
                seed=42
            )
            
            # Check that the dataset metadata has the expected structure
            self.assertIn('dataset_name', dataset_metadata)
            self.assertIn('num_scenarios', dataset_metadata)
            self.assertIn('distribution', dataset_metadata)
            self.assertIn('scenario_counts', dataset_metadata)
            self.assertIn('scenarios', dataset_metadata)
            
            # Check that the distribution is as expected
            self.assertEqual(dataset_metadata['distribution'], distribution)
            self.assertEqual(dataset_metadata['scenario_counts']['normal'], 5)
            self.assertEqual(dataset_metadata['scenario_counts']['electrical_fire'], 3)
            self.assertEqual(dataset_metadata['scenario_counts']['false_positive'], 2)
            
            # Check that the expected number of scenarios was generated
            self.assertEqual(dataset_metadata['num_scenarios'], 10)
            self.assertEqual(len(dataset_metadata['scenarios']), 10)
            
            # Check that the generators were called the expected number of times
            self.assertEqual(self.scenario_mixer.normal_generator.generate_specific_scenario.call_count, 5)
            self.assertEqual(self.scenario_mixer.electrical_fire_generator.generate_specific_scenario.call_count, 3)
            self.assertEqual(self.scenario_mixer.false_positive_generator.generate_random_false_positive.call_count, 2)
            
            # Check that save_scenario was called for each scenario
            self.assertEqual(self.scenario_mixer.scenario_generator.save_scenario.call_count, 10)


if __name__ == '__main__':
    unittest.main()