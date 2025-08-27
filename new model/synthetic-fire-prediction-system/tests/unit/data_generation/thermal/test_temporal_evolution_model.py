"""
Unit tests for the TemporalEvolutionModel class.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import tempfile

from src.data_generation.thermal.temporal_evolution_model import (
    TemporalEvolutionModel, FireType, FireStage
)
from src.data_generation.thermal.hotspot_simulator import HotspotSimulator


class TestTemporalEvolutionModel(unittest.TestCase):
    """Test cases for the TemporalEvolutionModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.hotspot_config = {
            'min_temperature': 20.0,  # 20°C
            'max_temperature': 500.0,  # 500°C
            'temperature_unit': 'celsius',
            'default_shape': 'circular',
            'default_growth': 'exponential'
        }
        
        self.temporal_config = {
            'default_fire_type': 'standard',
            'default_duration': 300,  # 5 minutes
            'default_frame_rate': 1.0  # 1 frame per second
        }
        
        # Create a test image shape
        self.image_shape = (288, 384)
        
        # Create hotspot simulator
        self.hotspot_simulator = HotspotSimulator(self.hotspot_config)
        
        # Create temporal evolution model
        self.temporal_model = TemporalEvolutionModel(self.temporal_config, self.hotspot_simulator)
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of TemporalEvolutionModel."""
        self.assertIsNotNone(self.temporal_model)
        self.assertEqual(self.temporal_model.config, self.temporal_config)
        self.assertEqual(self.temporal_model.hotspot_simulator, self.hotspot_simulator)
        self.assertEqual(self.temporal_model.default_fire_type, FireType.STANDARD)
        self.assertEqual(self.temporal_model.default_duration, 300)
        self.assertEqual(self.temporal_model.default_frame_rate, 1.0)
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        # Should not raise an exception
        self.temporal_model.validate_config()
    
    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        # Invalid fire type
        invalid_config = {
            'default_fire_type': 'invalid_fire_type'
        }
        
        with self.assertRaises(ValueError):
            TemporalEvolutionModel(invalid_config, self.hotspot_simulator)
        
        # Invalid duration
        invalid_config = {
            'default_duration': -10  # Negative duration
        }
        
        with self.assertRaises(ValueError):
            TemporalEvolutionModel(invalid_config, self.hotspot_simulator)
        
        # Invalid frame rate
        invalid_config = {
            'default_frame_rate': 0  # Zero frame rate
        }
        
        with self.assertRaises(ValueError):
            TemporalEvolutionModel(invalid_config, self.hotspot_simulator)
    
    def test_fire_parameters(self):
        """Test that fire parameters are initialized correctly."""
        # Check that parameters exist for all fire types
        for fire_type in FireType:
            self.assertIn(fire_type, self.temporal_model.fire_params)
            
            # Check that each fire type has the required parameters
            params = self.temporal_model.fire_params[fire_type]
            self.assertIn('growth_rate', params)
            self.assertIn('max_temperature_factor', params)
            self.assertIn('temperature_variance', params)
            self.assertIn('hotspot_count', params)
            self.assertIn('hotspot_size_factor', params)
            self.assertIn('preferred_shapes', params)
            self.assertIn('preferred_growth', params)
            self.assertIn('stage_durations', params)
            
            # Check that stage durations sum to 1.0
            stage_durations = params['stage_durations']
            self.assertAlmostEqual(sum(stage_durations.values()), 1.0)
    
    def test_determine_fire_stage(self):
        """Test determining fire stage based on elapsed time."""
        # Test for standard fire type
        fire_type = FireType.STANDARD
        
        # Get stage durations
        stage_durations = self.temporal_model.fire_params[fire_type]['stage_durations']
        
        # Test each stage
        total_duration = 100.0
        cumulative = 0.0
        
        for stage, duration in stage_durations.items():
            # Test at the middle of each stage
            cumulative += duration / 2
            elapsed_time = cumulative * total_duration
            
            determined_stage = self.temporal_model.determine_fire_stage(
                elapsed_time=elapsed_time,
                total_duration=total_duration,
                fire_type=fire_type
            )
            
            self.assertEqual(determined_stage, stage)
            
            # Move to the end of the stage
            cumulative += duration / 2
        
        # Test beyond total duration (should be decay stage)
        determined_stage = self.temporal_model.determine_fire_stage(
            elapsed_time=total_duration * 2,
            total_duration=total_duration,
            fire_type=fire_type
        )
        
        self.assertEqual(determined_stage, FireStage.DECAY)
    
    def test_generate_fire_sequence(self):
        """Test generating a fire sequence."""
        # Generate a short sequence for testing
        start_time = datetime.now()
        duration = 10  # 10 seconds
        frame_rate = 1.0  # 1 frame per second
        
        sequence = self.temporal_model.generate_fire_sequence(
            image_shape=self.image_shape,
            start_time=start_time,
            duration=duration,
            frame_rate=frame_rate,
            fire_type=FireType.STANDARD,
            seed=42
        )
        
        # Check that the sequence has the expected structure
        self.assertIn('frames', sequence)
        self.assertIn('timestamps', sequence)
        self.assertIn('metadata', sequence)
        
        # Check that the number of frames is correct
        self.assertEqual(len(sequence['frames']), duration * frame_rate)
        self.assertEqual(len(sequence['timestamps']), duration * frame_rate)
        
        # Check that the frames have the correct shape
        for frame in sequence['frames']:
            self.assertEqual(frame.shape, self.image_shape)
        
        # Check that the timestamps are correct
        for i, timestamp in enumerate(sequence['timestamps']):
            expected_timestamp = start_time + timedelta(seconds=i / frame_rate)
            self.assertEqual(timestamp, expected_timestamp)
        
        # Check that the metadata is correct
        metadata = sequence['metadata']
        self.assertEqual(metadata['fire_type'], FireType.STANDARD.value)
        self.assertEqual(metadata['duration'], duration)
        self.assertEqual(metadata['frame_rate'], frame_rate)
        self.assertEqual(metadata['num_frames'], duration * frame_rate)
        self.assertEqual(metadata['image_shape'], self.image_shape)
        self.assertEqual(metadata['seed'], 42)
    
    def test_generate_fire_sequence_different_types(self):
        """Test generating fire sequences with different fire types."""
        # Generate a short sequence for each fire type
        start_time = datetime.now()
        duration = 5  # 5 seconds
        frame_rate = 1.0  # 1 frame per second
        
        for fire_type in FireType:
            sequence = self.temporal_model.generate_fire_sequence(
                image_shape=self.image_shape,
                start_time=start_time,
                duration=duration,
                frame_rate=frame_rate,
                fire_type=fire_type,
                seed=42
            )
            
            # Check that the sequence has the expected structure
            self.assertIn('frames', sequence)
            self.assertIn('timestamps', sequence)
            self.assertIn('metadata', sequence)
            
            # Check that the metadata has the correct fire type
            metadata = sequence['metadata']
            self.assertEqual(metadata['fire_type'], fire_type.value)
    
    def test_fire_evolution(self):
        """Test that fire evolves over time."""
        # Generate a sequence
        start_time = datetime.now()
        duration = 10  # 10 seconds
        frame_rate = 1.0  # 1 frame per second
        
        sequence = self.temporal_model.generate_fire_sequence(
            image_shape=self.image_shape,
            start_time=start_time,
            duration=duration,
            frame_rate=frame_rate,
            fire_type=FireType.STANDARD,
            seed=42
        )
        
        frames = sequence['frames']
        
        # Check that the fire evolves (temperature increases over time)
        first_frame_max = np.max(frames[0])
        last_frame_max = np.max(frames[-1])
        
        self.assertLess(first_frame_max, last_frame_max)
        
        # Check that the fire area increases over time
        threshold = self.hotspot_config['min_temperature'] + 50  # 50°C above ambient
        first_frame_area = np.sum(frames[0] > threshold)
        last_frame_area = np.sum(frames[-1] > threshold)
        
        self.assertLess(first_frame_area, last_frame_area)
    
    def test_fire_evolution_dataset(self):
        """Test generating a fire evolution dataset."""
        # Generate a small dataset for testing
        num_sequences = 2
        output_dir = os.path.join(self.temp_dir, 'fire_dataset')
        
        metadata = self.temporal_model.generate_fire_evolution_dataset(
            image_shape=self.image_shape,
            num_sequences=num_sequences,
            output_dir=output_dir,
            fire_types=[FireType.STANDARD, FireType.SMOLDERING],
            duration_range=(5, 10),
            frame_rate=1.0,
            seed=42
        )
        
        # Check that the dataset has the expected structure
        self.assertIn('dataset_name', metadata)
        self.assertIn('num_sequences', metadata)
        self.assertIn('sequences', metadata)
        
        # Check that the number of sequences is correct
        self.assertEqual(metadata['num_sequences'], num_sequences)
        self.assertEqual(len(metadata['sequences']), num_sequences)
        
        # Check that the dataset files were created
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'dataset_metadata.json')))
        
        for i in range(num_sequences):
            sequence_dir = os.path.join(output_dir, f'sequence_{i:04d}')
            self.assertTrue(os.path.exists(sequence_dir))
            self.assertTrue(os.path.exists(os.path.join(sequence_dir, 'metadata.json')))
            self.assertTrue(os.path.exists(os.path.join(sequence_dir, 'timestamps.json')))
            self.assertTrue(os.path.exists(os.path.join(sequence_dir, 'frames')))
            
            # Check that frames were saved
            frames_dir = os.path.join(sequence_dir, 'frames')
            frame_files = os.listdir(frames_dir)
            self.assertGreater(len(frame_files), 0)
            
            # Check that the frame files are numpy arrays
            for frame_file in frame_files:
                if frame_file.endswith('.npy'):
                    frame_path = os.path.join(frames_dir, frame_file)
                    frame = np.load(frame_path)
                    self.assertEqual(frame.shape, self.image_shape)


if __name__ == '__main__':
    unittest.main()