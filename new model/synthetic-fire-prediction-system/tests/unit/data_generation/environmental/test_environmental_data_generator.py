"""
Unit tests for the EnvironmentalDataGenerator class.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data_generation.environmental.environmental_data_generator import EnvironmentalDataGenerator
from src.data_generation.environmental.voc_pattern_generator import VOCPatternGenerator
from src.data_generation.environmental.correlation_engine import CorrelationEngine
from src.data_generation.environmental.environmental_variation_model import EnvironmentalVariationModel


class TestEnvironmentalDataGenerator(unittest.TestCase):
    """Test cases for the EnvironmentalDataGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
            'parameter_ranges': {
                'temperature': {'min': 15.0, 'max': 35.0, 'unit': '°C'},
                'humidity': {'min': 20.0, 'max': 80.0, 'unit': '%'},
                'pressure': {'min': 990.0, 'max': 1030.0, 'unit': 'hPa'},
                'voc': {'min': 0.0, 'max': 2000.0, 'unit': 'ppb'}
            },
            'aws_integration': False
        }
        self.generator = EnvironmentalDataGenerator(self.config)

    def test_initialization(self):
        """Test initialization of the generator."""
        self.assertIsInstance(self.generator.voc_generator, VOCPatternGenerator)
        self.assertIsInstance(self.generator.correlation_engine, CorrelationEngine)
        self.assertIsInstance(self.generator.variation_model, EnvironmentalVariationModel)
        self.assertIsNone(self.generator.s3_service)

    def test_validate_config_default_parameters(self):
        """Test validation of configuration with default parameters."""
        # Create generator with empty config
        generator = EnvironmentalDataGenerator({})
        
        # Check that default parameters were set
        self.assertIn('parameters', generator.config)
        self.assertIn('parameter_ranges', generator.config)
        self.assertEqual(generator.config['parameters'], ['temperature', 'humidity', 'pressure', 'voc'])

    def test_validate_config_invalid_parameters(self):
        """Test validation of configuration with invalid parameters."""
        # Test with invalid parameters format
        with self.assertRaises(ValueError):
            EnvironmentalDataGenerator({'parameters': 'not_a_list'})

    def test_generate(self):
        """Test generation of environmental data."""
        # Set up test parameters
        timestamp = datetime.now()
        duration_seconds = 60
        sample_rate_hz = 1.0
        seed = 42
        
        # Generate data
        data = self.generator.generate(timestamp, duration_seconds, sample_rate_hz, seed)
        
        # Check structure of returned data
        self.assertIn('environmental_data', data)
        self.assertIn('timestamps', data)
        self.assertIn('metadata', data)
        
        # Check that all parameters are present
        env_data = data['environmental_data']
        for param in self.config['parameters']:
            self.assertIn(param, env_data)
            self.assertIn('values', env_data[param])
            self.assertIn('unit', env_data[param])
        
        # Check number of samples
        expected_samples = int(duration_seconds * sample_rate_hz)
        self.assertEqual(len(data['timestamps']), expected_samples)
        for param in self.config['parameters']:
            self.assertEqual(len(env_data[param]['values']), expected_samples)
        
        # Check metadata
        metadata = data['metadata']
        self.assertEqual(metadata['parameters'], self.config['parameters'])
        self.assertEqual(metadata['duration'], duration_seconds)
        self.assertEqual(metadata['sample_rate'], sample_rate_hz)
        self.assertEqual(metadata['num_samples'], expected_samples)
        self.assertEqual(metadata['seed'], seed)

    def test_generate_environmental_reading(self):
        """Test generation of a single environmental reading."""
        # Set up test parameters
        timestamp = datetime.now()
        parameter = 'temperature'
        baseline = 25.0
        daily_variation = 0.1
        noise_level = 0.05
        
        # Generate reading
        reading = self.generator.generate_environmental_reading(
            timestamp, parameter, baseline, daily_variation, noise_level
        )
        
        # Check that reading is a float
        self.assertIsInstance(reading, float)
        
        # Check that reading is within reasonable bounds
        # (baseline +/- daily_variation * baseline + some noise)
        max_variation = baseline * daily_variation + baseline * noise_level * 3  # 3 sigma
        self.assertLess(abs(reading - baseline), max_variation)

    def test_generate_fire_scenario(self):
        """Test generation of a fire scenario."""
        # Set up test parameters
        start_time = datetime.now()
        duration = 300
        sample_rate = 1.0
        fire_type = 'flaming'
        room_params = {
            'room_volume': 50.0,
            'ventilation_rate': 0.5,
            'initial_temperature': 20.0,
            'initial_humidity': 50.0
        }
        seed = 42
        
        # Generate scenario
        scenario = self.generator.generate_fire_scenario(
            start_time, duration, sample_rate, fire_type, room_params, seed
        )
        
        # Check structure of returned data
        self.assertIn('environmental_data', scenario)
        self.assertIn('timestamps', scenario)
        self.assertIn('metadata', scenario)
        
        # Check that fire type and room params are in metadata
        metadata = scenario['metadata']
        self.assertEqual(metadata['fire_type'], fire_type)
        self.assertEqual(metadata['room_params'], room_params)
        
        # Check that temperature increases for a fire scenario
        env_data = scenario['environmental_data']
        temp_values = env_data['temperature']['values']
        self.assertGreater(max(temp_values), room_params['initial_temperature'])

    def test_to_dataframe(self):
        """Test conversion of generated data to DataFrame."""
        # Generate some data
        timestamp = datetime.now()
        duration_seconds = 10
        sample_rate_hz = 1.0
        data = self.generator.generate(timestamp, duration_seconds, sample_rate_hz)
        
        # Convert to DataFrame
        df = self.generator.to_dataframe(data)
        
        # Check that DataFrame has expected columns
        self.assertIn('timestamp', df.columns)
        for param in self.config['parameters']:
            self.assertIn(param, df.columns)
            self.assertIn(f'{param}_unit', df.columns)
        
        # Check number of rows
        expected_rows = int(duration_seconds * sample_rate_hz)
        self.assertEqual(len(df), expected_rows)

    @patch('os.makedirs')
    @patch('json.dump')
    @patch('numpy.savetxt')
    @patch('pandas.DataFrame.to_csv')
    def test_save(self, mock_to_csv, mock_savetxt, mock_json_dump, mock_makedirs):
        """Test saving of generated data to files."""
        # Generate some data
        timestamp = datetime.now()
        duration_seconds = 10
        sample_rate_hz = 1.0
        data = self.generator.generate(timestamp, duration_seconds, sample_rate_hz)
        
        # Save data
        filepath = '/test/path/data'
        self.generator.save(data, filepath)
        
        # Check that directory was created
        mock_makedirs.assert_called()
        
        # Check that files were saved
        # - Metadata JSON
        # - Timestamps JSON
        # - Parameter CSV files
        # - DataFrame CSV
        self.assertTrue(mock_json_dump.called)
        self.assertTrue(mock_savetxt.called)
        self.assertTrue(mock_to_csv.called)

    @patch('src.aws.s3.service.S3ServiceImpl')
    def test_aws_integration(self, mock_s3_service):
        """Test AWS integration."""
        # Create config with AWS integration
        config = self.config.copy()
        config['aws_integration'] = True
        config['aws_config'] = {'default_bucket': 'test-bucket'}
        
        # Create generator with AWS integration
        generator = EnvironmentalDataGenerator(config)
        
        # Check that S3 service was initialized
        self.assertIsNotNone(generator.s3_service)
        
        # Mock S3 service upload_file method
        mock_upload = MagicMock()
        generator.s3_service.upload_file = mock_upload
        
        # Generate and save data
        timestamp = datetime.now()
        duration_seconds = 10
        sample_rate_hz = 1.0
        data = generator.generate(timestamp, duration_seconds, sample_rate_hz)
        
        # Save data with S3 upload
        filepath = '/test/path/data'
        generator._upload_to_s3(filepath, data)
        
        # Check that upload_file was called
        self.assertTrue(mock_upload.called)

    def test_generate_and_save_dataset(self):
        """Test generation and saving of a dataset."""
        # Mock save method
        self.generator.save = MagicMock()
        
        # Set up test parameters
        output_dir = '/test/output'
        num_sequences = 2
        sequence_duration = 10
        sample_rate = 1.0
        fire_types = ['normal', 'flaming']
        room_params = [{'room_volume': 50.0, 'ventilation_rate': 0.5}]
        seed = 42
        
        # Generate dataset
        dataset = self.generator.generate_and_save_dataset(
            output_dir, num_sequences, sequence_duration, sample_rate,
            fire_types, room_params, False, seed
        )
        
        # Check that save was called for each sequence
        self.assertEqual(self.generator.save.call_count, num_sequences)
        
        # Check dataset metadata
        self.assertEqual(dataset['dataset_name'], 'environmental_data_dataset')
        self.assertEqual(dataset['num_sequences'], num_sequences)
        self.assertEqual(dataset['sequence_duration'], sequence_duration)
        self.assertEqual(dataset['sample_rate'], sample_rate)
        self.assertEqual(dataset['parameters'], self.config['parameters'])
        self.assertEqual(dataset['fire_types'], fire_types)
        self.assertEqual(len(dataset['sequences']), num_sequences)


if __name__ == '__main__':
    unittest.main()