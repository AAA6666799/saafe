"""
Unit tests for the ThermalImageGenerator class.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import shutil
import tempfile
import json
from unittest.mock import MagicMock, patch

from src.data_generation.thermal.thermal_image_generator import ThermalImageGenerator
from src.data_generation.thermal.hotspot_simulator import HotspotSimulator
from src.data_generation.thermal.temporal_evolution_model import TemporalEvolutionModel
from src.data_generation.thermal.noise_injector import NoiseInjector


class TestThermalImageGenerator(unittest.TestCase):
    """Test cases for the ThermalImageGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.config = {
            'resolution': (288, 384),
            'min_temperature': 20.0,  # 20°C
            'max_temperature': 500.0,  # 500°C
            'output_formats': ['numpy', 'png'],
            'hotspot_config': {
                'min_temperature': 20.0,
                'max_temperature': 500.0,
                'temperature_unit': 'celsius',
                'default_shape': 'circular',
                'default_growth': 'exponential'
            },
            'temporal_config': {
                'default_fire_type': 'standard',
                'default_duration': 300,  # 5 minutes
                'default_frame_rate': 1.0  # 1 frame per second
            },
            'noise_config': {
                'noise_types': ['gaussian'],
                'noise_params': {
                    'gaussian': {
                        'mean': 0,
                        'std': 2.0
                    }
                }
            },
            'aws_integration': False
        }
        
        # Create thermal image generator
        self.generator = ThermalImageGenerator(self.config)
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of ThermalImageGenerator."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.config, self.config)
        self.assertIsInstance(self.generator.hotspot_simulator, HotspotSimulator)
        self.assertIsInstance(self.generator.temporal_model, TemporalEvolutionModel)
        self.assertIsInstance(self.generator.noise_injector, NoiseInjector)
        self.assertIsNone(self.generator.s3_service)
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        # Should not raise an exception
        self.generator.validate_config()
    
    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        # Invalid resolution
        invalid_config = {
            'resolution': 'invalid_resolution'
        }
        
        with self.assertRaises(ValueError):
            ThermalImageGenerator(invalid_config)
        
        # Invalid temperature range
        invalid_config = {
            'min_temperature': 500.0,
            'max_temperature': 20.0  # Max less than min
        }
        
        with self.assertRaises(ValueError):
            ThermalImageGenerator(invalid_config)
        
        # Invalid output format
        invalid_config = {
            'output_formats': ['invalid_format']
        }
        
        with self.assertRaises(ValueError):
            ThermalImageGenerator(invalid_config)
        
        # AWS integration without config
        invalid_config = {
            'aws_integration': True
        }
        
        with self.assertRaises(ValueError):
            ThermalImageGenerator(invalid_config)
    
    def test_generate(self):
        """Test generating a sequence of thermal images."""
        # Generate a short sequence for testing
        start_time = datetime.now()
        duration = 5  # 5 seconds
        sample_rate = 1.0  # 1 frame per second
        
        sequence = self.generator.generate(
            timestamp=start_time,
            duration_seconds=duration,
            sample_rate_hz=sample_rate,
            seed=42
        )
        
        # Check that the sequence has the expected structure
        self.assertIn('frames', sequence)
        self.assertIn('timestamps', sequence)
        self.assertIn('metadata', sequence)
        
        # Check that the number of frames is correct
        self.assertEqual(len(sequence['frames']), duration * sample_rate)
        self.assertEqual(len(sequence['timestamps']), duration * sample_rate)
        
        # Check that the frames have the correct shape
        for frame in sequence['frames']:
            self.assertEqual(frame.shape, self.config['resolution'])
        
        # Check that the timestamps are correct
        for i, timestamp in enumerate(sequence['timestamps']):
            expected_timestamp = start_time + timedelta(seconds=i / sample_rate)
            self.assertEqual(timestamp, expected_timestamp)
    
    def test_generate_frame(self):
        """Test generating a single thermal image frame."""
        # Generate a frame with default parameters
        timestamp = datetime.now()
        frame = self.generator.generate_frame(timestamp)
        
        # Check that the frame has the correct shape
        self.assertEqual(frame.shape, self.config['resolution'])
        
        # Check that the frame has values in the correct temperature range
        self.assertGreaterEqual(np.min(frame), self.config['min_temperature'])
        self.assertLessEqual(np.max(frame), self.config['max_temperature'])
        
        # Generate a frame with custom hotspots
        hotspots = [
            {
                'center': (100, 150),
                'radius': 30,
                'intensity': 0.8,
                'shape': 'circular'
            }
        ]
        
        frame = self.generator.generate_frame(timestamp, hotspots)
        
        # Check that the frame has the correct shape
        self.assertEqual(frame.shape, self.config['resolution'])
        
        # Check that the frame has values in the correct temperature range
        self.assertGreaterEqual(np.min(frame), self.config['min_temperature'])
        self.assertLessEqual(np.max(frame), self.config['max_temperature'])
        
        # Check that the hotspot is present (peak at the specified center)
        center_value = frame[hotspots[0]['center'][0], hotspots[0]['center'][1]]
        edge_value = frame[0, 0]
        self.assertGreater(center_value, edge_value)
    
    def test_to_dataframe(self):
        """Test converting generated data to a DataFrame."""
        # Generate a short sequence for testing
        start_time = datetime.now()
        duration = 3  # 3 seconds
        sample_rate = 1.0  # 1 frame per second
        
        sequence = self.generator.generate(
            timestamp=start_time,
            duration_seconds=duration,
            sample_rate_hz=sample_rate,
            seed=42
        )
        
        # Convert to DataFrame
        df = self.generator.to_dataframe(sequence)
        
        # Check that the DataFrame has the expected structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), duration * sample_rate)
        
        # Check that the DataFrame has the expected columns
        expected_columns = [
            'timestamp', 'min_temperature', 'max_temperature', 'mean_temperature',
            'std_temperature', 'hotspot_area_ratio', 'frame_index'
        ]
        
        for column in expected_columns:
            self.assertIn(column, df.columns)
        
        # Check that the timestamps match
        for i, timestamp in enumerate(sequence['timestamps']):
            self.assertEqual(df.iloc[i]['timestamp'], timestamp)
    
    def test_save(self):
        """Test saving generated data to files."""
        # Generate a short sequence for testing
        start_time = datetime.now()
        duration = 2  # 2 seconds
        sample_rate = 1.0  # 1 frame per second
        
        sequence = self.generator.generate(
            timestamp=start_time,
            duration_seconds=duration,
            sample_rate_hz=sample_rate,
            seed=42
        )
        
        # Save the sequence
        filepath = os.path.join(self.temp_dir, 'test_sequence')
        self.generator.save(sequence, filepath)
        
        # Check that the files were created
        self.assertTrue(os.path.exists(f"{filepath}_metadata.json"))
        self.assertTrue(os.path.exists(f"{filepath}_timestamps.json"))
        self.assertTrue(os.path.exists(f"{filepath}_stats.csv"))
        
        for i in range(duration * int(sample_rate)):
            self.assertTrue(os.path.exists(f"{filepath}_frame_{i:04d}.npy"))
            self.assertTrue(os.path.exists(f"{filepath}_frame_{i:04d}.png"))
        
        # Check that the metadata file contains the expected data
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
            
            self.assertIn('fire_type', metadata)
            self.assertIn('duration', metadata)
            self.assertIn('frame_rate', metadata)
            self.assertIn('num_frames', metadata)
            self.assertIn('image_shape', metadata)
        
        # Check that the timestamps file contains the expected data
        with open(f"{filepath}_timestamps.json", 'r') as f:
            timestamps = json.load(f)
            
            self.assertEqual(len(timestamps), duration * sample_rate)
    
    def test_normalize_for_visualization(self):
        """Test normalizing a thermal frame for visualization."""
        # Create a test frame
        frame = np.linspace(
            self.config['min_temperature'],
            self.config['max_temperature'],
            self.config['resolution'][0] * self.config['resolution'][1]
        ).reshape(self.config['resolution'])
        
        # Normalize the frame
        normalized = self.generator._normalize_for_visualization(frame)
        
        # Check that the normalized frame has the correct shape
        self.assertEqual(normalized.shape[:2], self.config['resolution'])
        
        # Check that the normalized frame has values in the range [0, 255]
        self.assertGreaterEqual(np.min(normalized), 0)
        self.assertLessEqual(np.max(normalized), 255)
    
    @patch('src.aws.s3.service.S3ServiceImpl')
    def test_aws_integration(self, mock_s3_service):
        """Test AWS S3 integration."""
        # Create a configuration with AWS integration
        aws_config = self.config.copy()
        aws_config['aws_integration'] = True
        aws_config['aws_config'] = {
            'default_bucket': 'test-bucket',
            'region_name': 'us-west-2'
        }
        
        # Create a mock S3 service
        mock_s3_instance = MagicMock()
        mock_s3_service.return_value = mock_s3_instance
        
        # Create a generator with AWS integration
        generator = ThermalImageGenerator(aws_config)
        
        # Check that the S3 service was initialized
        self.assertIsNotNone(generator.s3_service)
        
        # Generate a short sequence for testing
        start_time = datetime.now()
        duration = 1  # 1 second
        sample_rate = 1.0  # 1 frame per second
        
        sequence = generator.generate(
            timestamp=start_time,
            duration_seconds=duration,
            sample_rate_hz=sample_rate,
            seed=42
        )
        
        # Save the sequence
        filepath = os.path.join(self.temp_dir, 'test_sequence')
        
        # Mock the upload_file method
        mock_s3_instance.upload_file.return_value = True
        
        # This should call _upload_to_s3
        generator.save(sequence, filepath)
        
        # Check that upload_file was called for each file
        expected_calls = 4  # metadata, timestamps, stats, and 1 frame in 2 formats
        self.assertEqual(mock_s3_instance.upload_file.call_count, expected_calls)
    
    def test_generate_and_save_dataset(self):
        """Test generating and saving a dataset."""
        # Generate a small dataset for testing
        num_sequences = 2
        output_dir = os.path.join(self.temp_dir, 'thermal_dataset')
        
        dataset_metadata = self.generator.generate_and_save_dataset(
            output_dir=output_dir,
            num_sequences=num_sequences,
            sequence_duration=2,
            sample_rate=1.0,
            fire_types=['standard', 'smoldering'],
            upload_to_s3=False,
            seed=42
        )
        
        # Check that the dataset has the expected structure
        self.assertIn('dataset_name', dataset_metadata)
        self.assertIn('num_sequences', dataset_metadata)
        self.assertIn('sequences', dataset_metadata)
        
        # Check that the number of sequences is correct
        self.assertEqual(dataset_metadata['num_sequences'], num_sequences)
        self.assertEqual(len(dataset_metadata['sequences']), num_sequences)
        
        # Check that the dataset files were created
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'dataset_metadata.json')))
        
        for i in range(num_sequences):
            sequence_path = os.path.join(output_dir, f'sequence_{i:04d}')
            self.assertTrue(os.path.exists(f"{sequence_path}_metadata.json"))
            self.assertTrue(os.path.exists(f"{sequence_path}_timestamps.json"))
            self.assertTrue(os.path.exists(f"{sequence_path}_stats.csv"))
            
            # Check that frames were saved
            self.assertTrue(os.path.exists(f"{sequence_path}_frame_0000.npy"))
            self.assertTrue(os.path.exists(f"{sequence_path}_frame_0000.png"))
    
    def test_export_to_standard_format(self):
        """Test exporting a thermal image to standard formats."""
        # Create a test frame
        frame = np.linspace(
            self.config['min_temperature'],
            self.config['max_temperature'],
            self.config['resolution'][0] * self.config['resolution'][1]
        ).reshape(self.config['resolution'])
        
        # Test radiometric JPEG export
        jpg_path = os.path.join(self.temp_dir, 'test_radiometric.jpg')
        self.generator.export_to_standard_format(frame, 'radiometric_jpg', jpg_path)
        self.assertTrue(os.path.exists(jpg_path))
        
        # Test FLIR SEQ export
        seq_path = os.path.join(self.temp_dir, 'test_flir.seq')
        self.generator.export_to_standard_format(frame, 'flir_seq', seq_path)
        self.assertTrue(os.path.exists(seq_path))
        self.assertTrue(os.path.exists(seq_path + '.meta.json'))
        
        # Test 16-bit TIFF export
        tiff_path = os.path.join(self.temp_dir, 'test_thermal.tiff')
        self.generator.export_to_standard_format(frame, 'tiff16', tiff_path)
        self.assertTrue(os.path.exists(tiff_path))
        self.assertTrue(os.path.exists(tiff_path + '.calibration.json'))
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.generator.export_to_standard_format(frame, 'invalid_format', 'invalid.file')


if __name__ == '__main__':
    unittest.main()