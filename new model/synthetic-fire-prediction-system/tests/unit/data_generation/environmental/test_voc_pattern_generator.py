"""
Unit tests for the VOCPatternGenerator class.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.environmental.voc_pattern_generator import VOCPatternGenerator


class TestVOCPatternGenerator(unittest.TestCase):
    """Test cases for the VOCPatternGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'baseline_voc': 50.0,
            'max_voc': 2000.0,
            'default_material': 'mixed',
            'temperature_baseline': 20.0
        }
        self.generator = VOCPatternGenerator(self.config)

    def test_initialization(self):
        """Test initialization of the generator."""
        self.assertEqual(self.generator.config['baseline_voc'], 50.0)
        self.assertEqual(self.generator.config['max_voc'], 2000.0)
        self.assertEqual(self.generator.config['default_material'], 'mixed')
        self.assertEqual(self.generator.config['temperature_baseline'], 20.0)
        
        # Check that material profiles are initialized
        self.assertIn('wood', self.generator.material_profiles)
        self.assertIn('plastic', self.generator.material_profiles)
        self.assertIn('fabric', self.generator.material_profiles)
        self.assertIn('electronics', self.generator.material_profiles)
        self.assertIn('mixed', self.generator.material_profiles)

    def test_validate_config_default_values(self):
        """Test validation of configuration with default values."""
        # Create generator with empty config
        generator = VOCPatternGenerator({})
        
        # Check that default values were set
        self.assertIn('baseline_voc', generator.config)
        self.assertIn('max_voc', generator.config)
        self.assertIn('default_material', generator.config)
        self.assertIn('temperature_baseline', generator.config)

    def test_generate_voc_time_series(self):
        """Test generation of VOC time series."""
        # Set up test parameters
        start_time = datetime.now()
        num_samples = 60
        timestamps = [start_time + timedelta(seconds=i) for i in range(num_samples)]
        baseline = 50.0
        material_type = 'wood'
        seed = 42
        
        # Generate VOC time series
        voc_values = self.generator.generate_voc_time_series(
            timestamps, baseline, material_type, seed=seed
        )
        
        # Check that the correct number of values was generated
        self.assertEqual(len(voc_values), num_samples)
        
        # Check that values are within reasonable bounds
        self.assertTrue(all(v >= 0 for v in voc_values))
        self.assertTrue(all(v <= self.config['max_voc'] for v in voc_values))
        
        # Check that values are not all the same (variation is applied)
        self.assertGreater(np.std(voc_values), 0)

    def test_generate_voc_time_series_with_temperature(self):
        """Test generation of VOC time series with temperature effects."""
        # Set up test parameters
        start_time = datetime.now()
        num_samples = 60
        timestamps = [start_time + timedelta(seconds=i) for i in range(num_samples)]
        baseline = 50.0
        material_type = 'wood'
        
        # Create temperature values with increasing trend
        temp_baseline = self.config['temperature_baseline']
        temperature_values = [temp_baseline + i * 0.5 for i in range(num_samples)]
        
        # Generate VOC time series with temperature values
        voc_values = self.generator.generate_voc_time_series(
            timestamps, baseline, material_type, temperature_values
        )
        
        # Check that VOC values increase with temperature
        # (Compare first half average with second half average)
        first_half_avg = np.mean(voc_values[:num_samples//2])
        second_half_avg = np.mean(voc_values[num_samples//2:])
        self.assertLess(first_half_avg, second_half_avg)

    def test_generate_voc_time_series_with_fire_event(self):
        """Test generation of VOC time series with fire event."""
        # Set up test parameters
        start_time = datetime.now()
        num_samples = 100
        timestamps = [start_time + timedelta(seconds=i) for i in range(num_samples)]
        baseline = 50.0
        material_type = 'wood'
        
        # Define fire event
        fire_event = {
            'start_idx': 20,
            'duration_idx': 40,
            'peak_multiplier': 10.0
        }
        
        # Generate VOC time series with fire event
        voc_values = self.generator.generate_voc_time_series(
            timestamps, baseline, material_type, fire_event=fire_event
        )
        
        # Check that VOC values increase during fire event
        before_fire_avg = np.mean(voc_values[:fire_event['start_idx']])
        during_fire_avg = np.mean(voc_values[fire_event['start_idx']:fire_event['start_idx'] + fire_event['duration_idx']])
        after_fire_avg = np.mean(voc_values[fire_event['start_idx'] + fire_event['duration_idx']:])
        
        self.assertLess(before_fire_avg, during_fire_avg)
        self.assertLess(after_fire_avg, during_fire_avg)

    def test_generate_compound_specific_vocs(self):
        """Test generation of compound-specific VOC concentrations."""
        # Set up test parameters
        start_time = datetime.now()
        num_samples = 60
        timestamps = [start_time + timedelta(seconds=i) for i in range(num_samples)]
        total_voc = [100.0] * num_samples
        material_type = 'wood'
        
        # Generate compound-specific VOCs
        compound_vocs = self.generator.generate_compound_specific_vocs(
            timestamps, total_voc, material_type
        )
        
        # Check that all expected compounds are present
        expected_compounds = self.generator.material_profiles[material_type]['primary_compounds']
        for compound in expected_compounds:
            self.assertIn(compound, compound_vocs)
            
        # Check that compound values sum approximately to total VOC
        # (with some variation allowed)
        for i in range(num_samples):
            compound_sum = sum(compound_vocs[compound][i] for compound in compound_vocs)
            self.assertAlmostEqual(compound_sum, total_voc[i], delta=total_voc[i] * 0.2)

    def test_simulate_material_specific_emission(self):
        """Test simulation of material-specific VOC emissions."""
        # Set up test parameters
        material_type = 'plastic'
        temperature = 30.0
        duration_hours = 2.0
        sample_rate_hz = 0.1
        
        # Simulate emissions
        result = self.generator.simulate_material_specific_emission(
            material_type, temperature, duration_hours, sample_rate_hz
        )
        
        # Check structure of returned data
        self.assertIn('material_type', result)
        self.assertIn('temperature', result)
        self.assertIn('duration_hours', result)
        self.assertIn('sample_rate_hz', result)
        self.assertIn('timestamps', result)
        self.assertIn('total_voc', result)
        self.assertIn('compound_vocs', result)
        self.assertIn('material_profile', result)
        
        # Check that the correct number of samples was generated
        expected_samples = int(duration_hours * 3600 * sample_rate_hz)
        self.assertEqual(len(result['timestamps']), expected_samples)
        self.assertEqual(len(result['total_voc']), expected_samples)
        
        # Check that emissions decrease over time due to decay
        first_quarter_avg = np.mean(result['total_voc'][:expected_samples//4])
        last_quarter_avg = np.mean(result['total_voc'][-expected_samples//4:])
        self.assertGreater(first_quarter_avg, last_quarter_avg)

    def test_simulate_fire_voc_profile(self):
        """Test simulation of VOC emissions during a fire event."""
        # Set up test parameters
        fire_type = 'flaming'
        material_type = 'wood'
        duration_minutes = 10.0
        sample_rate_hz = 0.5
        
        # Simulate fire VOC profile
        result = self.generator.simulate_fire_voc_profile(
            fire_type, material_type, duration_minutes, sample_rate_hz
        )
        
        # Check structure of returned data
        self.assertIn('fire_type', result)
        self.assertIn('material_type', result)
        self.assertIn('duration_minutes', result)
        self.assertIn('sample_rate_hz', result)
        self.assertIn('timestamps', result)
        self.assertIn('temperature_profile', result)
        self.assertIn('total_voc', result)
        self.assertIn('compound_vocs', result)
        
        # Check that the correct number of samples was generated
        expected_samples = int(duration_minutes * 60 * sample_rate_hz)
        self.assertEqual(len(result['timestamps']), expected_samples)
        self.assertEqual(len(result['temperature_profile']), expected_samples)
        self.assertEqual(len(result['total_voc']), expected_samples)
        
        # Check that VOC levels follow expected pattern for a fire
        # (increase, peak, decrease)
        first_third = result['total_voc'][:expected_samples//3]
        middle_third = result['total_voc'][expected_samples//3:2*expected_samples//3]
        last_third = result['total_voc'][2*expected_samples//3:]
        
        first_third_avg = np.mean(first_third)
        middle_third_avg = np.mean(middle_third)
        last_third_avg = np.mean(last_third)
        
        # Middle should be higher than beginning
        self.assertLess(first_third_avg, middle_third_avg)
        
        # Check compound VOCs
        for compound in result['compound_vocs']:
            self.assertEqual(len(result['compound_vocs'][compound]), expected_samples)


if __name__ == '__main__':
    unittest.main()