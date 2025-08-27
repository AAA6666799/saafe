"""
Unit tests for the GasConcentrationGenerator class.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.gas.gas_concentration_generator import GasConcentrationGenerator


class TestGasConcentrationGenerator(unittest.TestCase):
    """Test cases for the GasConcentrationGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.config = {
            'gas_types': ['methane', 'propane', 'hydrogen'],
            'diffusion_config': {
                'diffusion_type': 'gaussian',
                'grid_resolution': (20, 20, 10),
                'time_step': 1.0
            },
            'temporal_config': {
                'default_release_pattern': 'gradual',
                'default_duration': 300,
                'default_sample_rate': 1.0
            },
            'sensor_configs': {
                'methane': {
                    'sensor_type': 'infrared',
                    'noise_level': 0.02,
                    'drift_rate': 0.5,
                    'response_time': 30.0
                },
                'propane': {
                    'sensor_type': 'catalytic',
                    'noise_level': 0.03,
                    'drift_rate': 0.7,
                    'response_time': 20.0
                }
            },
            'default_sensor_config': {
                'sensor_type': 'electrochemical',
                'noise_level': 0.02,
                'drift_rate': 0.5,
                'response_time': 30.0
            },
            'aws_integration': False
        }
        
        # Create the generator
        self.generator = GasConcentrationGenerator(self.config)
    
    def test_initialization(self):
        """Test initialization of the generator."""
        self.assertEqual(self.generator.config['gas_types'], ['methane', 'propane', 'hydrogen'])
        self.assertIsNotNone(self.generator.diffusion_model)
        self.assertIsNotNone(self.generator.temporal_model)
        self.assertIsNotNone(self.generator.sensor_models)
        self.assertIsNotNone(self.generator.default_sensor_model)
        self.assertIsNone(self.generator.s3_service)
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Test with valid config
        self.generator.validate_config()
        
        # Test with missing gas_types
        config_no_gas_types = self.config.copy()
        del config_no_gas_types['gas_types']
        generator = GasConcentrationGenerator(config_no_gas_types)
        generator.validate_config()
        self.assertEqual(generator.config['gas_types'], ['methane', 'propane', 'hydrogen', 'carbon_monoxide'])
        
        # Test with invalid sensor_configs
        config_invalid_sensors = self.config.copy()
        config_invalid_sensors['sensor_configs'] = "not a dict"
        with self.assertRaises(ValueError):
            GasConcentrationGenerator(config_invalid_sensors).validate_config()
        
        # Test with AWS integration but no config
        config_aws_no_config = self.config.copy()
        config_aws_no_config['aws_integration'] = True
        with self.assertRaises(ValueError):
            GasConcentrationGenerator(config_aws_no_config).validate_config()
    
    def test_generate(self):
        """Test generation of gas concentration data."""
        # Mock the temporal model and sensor models
        self.generator.temporal_model = MagicMock()
        self.generator.temporal_model.generate_concentration_time_series.return_value = {
            'concentrations': np.array([1.0, 2.0, 3.0]),
            'timestamps': [datetime.now(), datetime.now() + timedelta(seconds=1), datetime.now() + timedelta(seconds=2)],
            'metadata': {'gas_type': 'methane'}
        }
        
        self.generator.sensor_models = {
            'methane': MagicMock(),
            'propane': MagicMock(),
            'hydrogen': MagicMock()
        }
        
        for gas_type, model in self.generator.sensor_models.items():
            model.simulate_batch_response.return_value = np.array([1.1, 2.1, 3.1])
        
        # Generate data
        timestamp = datetime.now()
        duration = 2
        sample_rate = 1.0
        data = self.generator.generate(timestamp, duration, sample_rate, seed=42)
        
        # Check results
        self.assertIn('gas_data', data)
        self.assertIn('timestamps', data)
        self.assertIn('metadata', data)
        
        # Check gas data
        for gas_type in self.config['gas_types']:
            self.assertIn(gas_type, data['gas_data'])
            self.assertIn('true_concentrations', data['gas_data'][gas_type])
            self.assertIn('measured_concentrations', data['gas_data'][gas_type])
            self.assertIn('timestamps', data['gas_data'][gas_type])
            self.assertIn('metadata', data['gas_data'][gas_type])
        
        # Check metadata
        self.assertEqual(data['metadata']['gas_types'], self.config['gas_types'])
        self.assertEqual(data['metadata']['duration'], duration)
        self.assertEqual(data['metadata']['sample_rate'], sample_rate)
        self.assertEqual(data['metadata']['seed'], 42)
    
    def test_generate_concentration(self):
        """Test generation of a single gas concentration reading."""
        # Mock the sensor model
        sensor_model = MagicMock()
        sensor_model.simulate_response.return_value = 105.0
        self.generator.sensor_models = {'methane': sensor_model}
        
        # Generate concentration
        timestamp = datetime.now()
        gas_type = 'methane'
        baseline = 100.0
        anomaly_factor = 1.05
        
        concentration = self.generator.generate_concentration(timestamp, gas_type, baseline, anomaly_factor)
        
        # Check result
        self.assertEqual(concentration, 105.0)
        sensor_model.simulate_response.assert_called_once()
    
    def test_generate_spatial_distribution(self):
        """Test generation of spatial gas distribution."""
        # Mock the diffusion model
        self.generator.diffusion_model = MagicMock()
        self.generator.diffusion_model.simulate_diffusion.return_value = np.ones((20, 20, 10))
        
        # Generate spatial distribution
        gas_type = 'methane'
        source_points = [{'position': (10, 10, 5), 'strength': 1.0}]
        duration = 60.0
        
        distribution = self.generator.generate_spatial_distribution(gas_type, source_points, duration)
        
        # Check result
        self.assertEqual(distribution.shape, (20, 20, 10))
        self.generator.diffusion_model.simulate_diffusion.assert_called_once_with(
            gas_type=gas_type,
            source_points=source_points,
            duration=duration
        )
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        # Create test data
        timestamps = [datetime.now(), datetime.now() + timedelta(seconds=1)]
        gas_data = {
            'methane': {
                'true_concentrations': np.array([1.0, 2.0]),
                'measured_concentrations': np.array([1.1, 2.1]),
                'timestamps': timestamps
            },
            'propane': {
                'true_concentrations': np.array([0.5, 1.0]),
                'measured_concentrations': np.array([0.6, 1.1]),
                'timestamps': timestamps
            }
        }
        
        data = {
            'gas_data': gas_data,
            'timestamps': timestamps,
            'metadata': {'gas_types': ['methane', 'propane']}
        }
        
        # Convert to DataFrame
        df = self.generator.to_dataframe(data)
        
        # Check DataFrame
        self.assertEqual(len(df), 2)
        self.assertIn('timestamp', df.columns)
        self.assertIn('methane_concentration', df.columns)
        self.assertIn('methane_true_concentration', df.columns)
        self.assertIn('propane_concentration', df.columns)
        self.assertIn('propane_true_concentration', df.columns)
    
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.dump')
    @patch('numpy.savetxt')
    @patch('pandas.DataFrame.to_csv')
    def test_save(self, mock_to_csv, mock_savetxt, mock_json_dump, mock_open, mock_makedirs):
        """Test saving data to files."""
        # Create test data
        timestamps = [datetime.now(), datetime.now() + timedelta(seconds=1)]
        gas_data = {
            'methane': {
                'true_concentrations': np.array([1.0, 2.0]),
                'measured_concentrations': np.array([1.1, 2.1]),
                'timestamps': timestamps,
                'metadata': {'gas_type': 'methane'}
            }
        }
        
        data = {
            'gas_data': gas_data,
            'timestamps': timestamps,
            'metadata': {'gas_types': ['methane'], 'start_time': datetime.now()}
        }
        
        # Save data
        filepath = '/test/path/data'
        self.generator.save(data, filepath)
        
        # Check that directories were created
        mock_makedirs.assert_called_once()
        
        # Check that files were opened and written
        self.assertEqual(mock_open.call_count, 3)  # metadata, timestamps, gas metadata
        self.assertEqual(mock_json_dump.call_count, 3)
        self.assertEqual(mock_savetxt.call_count, 2)  # true and measured concentrations
        self.assertEqual(mock_to_csv.call_count, 1)  # DataFrame


if __name__ == '__main__':
    unittest.main()