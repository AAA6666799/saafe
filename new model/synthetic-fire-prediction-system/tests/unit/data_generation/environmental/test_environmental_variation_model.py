"""
Unit tests for the EnvironmentalVariationModel class.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.environmental.environmental_variation_model import EnvironmentalVariationModel


class TestEnvironmentalVariationModel(unittest.TestCase):
    """Test cases for the EnvironmentalVariationModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'sensor_noise_factor': 1.0,
            'sensor_drift_factor': 1.0,
            'daily_variation_factor': 1.0,
            'seasonal_variation_factor': 1.0,
            'environment_type': 'indoor'
        }
        self.model = EnvironmentalVariationModel(self.config)

    def test_initialization(self):
        """Test initialization of the variation model."""
        self.assertEqual(self.model.config['sensor_noise_factor'], 1.0)
        self.assertEqual(self.model.config['sensor_drift_factor'], 1.0)
        self.assertEqual(self.model.config['daily_variation_factor'], 1.0)
        self.assertEqual(self.model.config['seasonal_variation_factor'], 1.0)
        self.assertEqual(self.model.config['environment_type'], 'indoor')
        
        # Check that parameter variations are initialized
        self.assertIn('temperature', self.model.parameter_variations)
        self.assertIn('humidity', self.model.parameter_variations)
        self.assertIn('pressure', self.model.parameter_variations)
        self.assertIn('voc', self.model.parameter_variations)

    def test_validate_config_default_values(self):
        """Test validation of configuration with default values."""
        # Create model with empty config
        model = EnvironmentalVariationModel({})
        
        # Check that default values were set
        self.assertIn('sensor_noise_factor', model.config)
        self.assertIn('sensor_drift_factor', model.config)
        self.assertIn('daily_variation_factor', model.config)
        self.assertIn('seasonal_variation_factor', model.config)
        self.assertIn('environment_type', model.config)
        self.assertEqual(model.config['environment_type'], 'indoor')

    def test_generate_time_series(self):
        """Test generation of time series with variations."""
        # Set up test parameters
        parameter = 'temperature'
        baseline = 20.0
        start_time = datetime.now()
        num_samples = 24 * 7  # One week with hourly samples
        timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
        seed = 42
        
        # Generate time series
        values = self.model.generate_time_series(parameter, baseline, timestamps, seed)
        
        # Check that the correct number of values was generated
        self.assertEqual(len(values), num_samples)
        
        # Check that values have variation
        self.assertGreater(np.std(values), 0)
        
        # Check that values are centered around baseline (within reasonable bounds)
        mean_value = np.mean(values)
        self.assertAlmostEqual(mean_value, baseline, delta=5.0)

    def test_generate_time_series_different_parameters(self):
        """Test generation of time series for different parameters."""
        # Set up test parameters
        start_time = datetime.now()
        num_samples = 24  # One day with hourly samples
        timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
        
        # Test for each parameter
        parameters = ['temperature', 'humidity', 'pressure', 'voc']
        baselines = [20.0, 50.0, 1010.0, 100.0]
        
        for parameter, baseline in zip(parameters, baselines):
            # Generate time series
            values = self.model.generate_time_series(parameter, baseline, timestamps)
            
            # Check that the correct number of values was generated
            self.assertEqual(len(values), num_samples)
            
            # Check that values have variation
            self.assertGreater(np.std(values), 0)
            
            # Check that values are centered around baseline (within reasonable bounds)
            mean_value = np.mean(values)
            self.assertAlmostEqual(mean_value, baseline, delta=baseline * 0.25)

    def test_generate_time_series_environment_types(self):
        """Test generation of time series for different environment types."""
        # Set up test parameters
        parameter = 'temperature'
        baseline = 20.0
        start_time = datetime.now()
        num_samples = 24  # One day with hourly samples
        timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
        
        # Generate time series for indoor environment
        indoor_config = self.config.copy()
        indoor_config['environment_type'] = 'indoor'
        indoor_model = EnvironmentalVariationModel(indoor_config)
        indoor_values = indoor_model.generate_time_series(parameter, baseline, timestamps)
        
        # Generate time series for outdoor environment
        outdoor_config = self.config.copy()
        outdoor_config['environment_type'] = 'outdoor'
        outdoor_model = EnvironmentalVariationModel(outdoor_config)
        outdoor_values = outdoor_model.generate_time_series(parameter, baseline, timestamps)
        
        # Check that outdoor environment has more variation than indoor
        indoor_std = np.std(indoor_values)
        outdoor_std = np.std(outdoor_values)
        self.assertGreater(outdoor_std, indoor_std)

    def test_generate_single_reading(self):
        """Test generation of a single environmental reading."""
        # Set up test parameters
        parameter = 'temperature'
        timestamp = datetime.now()
        baseline = 20.0
        daily_variation = 0.1
        noise_level = 0.05
        
        # Generate reading
        reading = self.model.generate_single_reading(
            parameter, timestamp, baseline, daily_variation, noise_level
        )
        
        # Check that reading is a float
        self.assertIsInstance(reading, float)
        
        # Check that reading is within reasonable bounds
        # (baseline +/- daily_variation * baseline + some noise)
        max_variation = baseline * daily_variation * 3 + baseline * noise_level * 3  # 3 sigma
        self.assertLess(abs(reading - baseline), max_variation)

    def test_apply_sensor_characteristics(self):
        """Test application of sensor characteristics to a time series."""
        # Set up test parameters
        num_samples = 100
        values = np.ones(num_samples) * 20.0  # Constant temperature
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i) for i in range(num_samples)]
        noise_level = 0.2
        drift_rate = 0.01
        response_time = 60.0
        
        # Apply sensor characteristics
        sensor_values = self.model._apply_sensor_characteristics(
            values, timestamps, noise_level, drift_rate, response_time
        )
        
        # Check that the correct number of values was generated
        self.assertEqual(len(sensor_values), num_samples)
        
        # Check that values have been modified
        self.assertTrue(np.any(values != sensor_values))
        
        # Check that noise has been added
        self.assertGreater(np.std(sensor_values), 0)

    def test_generate_long_term_variation(self):
        """Test generation of long-term environmental parameter variations."""
        # Set up test parameters
        parameter = 'temperature'
        baseline = 20.0
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        sample_interval_minutes = 1440  # Daily samples
        seed = 42
        
        # Generate long-term variation
        result = self.model.generate_long_term_variation(
            parameter, baseline, start_date, end_date, sample_interval_minutes, seed
        )
        
        # Check structure of returned data
        self.assertIn('parameter', result)
        self.assertIn('baseline', result)
        self.assertIn('start_date', result)
        self.assertIn('end_date', result)
        self.assertIn('sample_interval_minutes', result)
        self.assertIn('timestamps', result)
        self.assertIn('values', result)
        self.assertIn('num_samples', result)
        
        # Check that the correct parameter and baseline were used
        self.assertEqual(result['parameter'], parameter)
        self.assertEqual(result['baseline'], baseline)
        
        # Check that the correct number of samples was generated
        expected_samples = 365  # Daily samples for a year
        self.assertEqual(result['num_samples'], expected_samples)
        self.assertEqual(len(result['timestamps']), expected_samples)
        self.assertEqual(len(result['values']), expected_samples)
        
        # Check that values have seasonal variation
        # (summer months should be warmer than winter months)
        summer_indices = [i for i, ts in enumerate(result['timestamps']) if 6 <= ts.month <= 8]
        winter_indices = [i for i, ts in enumerate(result['timestamps']) if ts.month <= 2 or ts.month >= 12]
        
        summer_avg = np.mean([result['values'][i] for i in summer_indices])
        winter_avg = np.mean([result['values'][i] for i in winter_indices])
        
        self.assertGreater(summer_avg, winter_avg)

    def test_simulate_sensor_aging(self):
        """Test simulation of sensor aging effects."""
        # Set up test parameters
        parameter = 'temperature'
        num_samples = 100
        values = np.linspace(15.0, 25.0, num_samples)  # Linear temperature increase
        start_time = datetime.now()
        timestamps = [start_time + timedelta(minutes=i) for i in range(num_samples)]
        
        # Simulate aging for different sensor ages
        new_sensor_values = self.model.simulate_sensor_aging(parameter, values, timestamps, 0.0)
        old_sensor_values = self.model.simulate_sensor_aging(parameter, values, timestamps, 365.0)
        
        # Check that the correct number of values was generated
        self.assertEqual(len(new_sensor_values), num_samples)
        self.assertEqual(len(old_sensor_values), num_samples)
        
        # Check that aging affects the values
        # (older sensor should have more noise and drift)
        new_sensor_std = np.std(np.array(new_sensor_values) - values)
        old_sensor_std = np.std(np.array(old_sensor_values) - values)
        
        self.assertLess(new_sensor_std, old_sensor_std)

    def test_generate_seasonal_profile(self):
        """Test generation of seasonal profile."""
        # Set up test parameters
        parameter = 'temperature'
        baseline = 20.0
        year = 2023
        samples_per_month = 30
        seed = 42
        
        # Generate seasonal profile
        result = self.model.generate_seasonal_profile(parameter, baseline, year, samples_per_month, seed)
        
        # Check structure of returned data
        self.assertIn('parameter', result)
        self.assertIn('baseline', result)
        self.assertIn('year', result)
        self.assertIn('timestamps', result)
        self.assertIn('values', result)
        self.assertIn('monthly_averages', result)
        self.assertIn('num_samples', result)
        
        # Check that the correct parameter and baseline were used
        self.assertEqual(result['parameter'], parameter)
        self.assertEqual(result['baseline'], baseline)
        self.assertEqual(result['year'], year)
        
        # Check that the correct number of samples was generated
        expected_samples = samples_per_month * 12
        self.assertEqual(result['num_samples'], expected_samples)
        self.assertEqual(len(result['timestamps']), expected_samples)
        self.assertEqual(len(result['values']), expected_samples)
        
        # Check that monthly averages were generated
        self.assertEqual(len(result['monthly_averages']), 12)
        
        # Check that summer months are warmer than winter months
        summer_avg = np.mean(result['monthly_averages'][5:8])  # June, July, August
        winter_avg = np.mean([result['monthly_averages'][0], result['monthly_averages'][1], result['monthly_averages'][11]])  # Jan, Feb, Dec
        
        self.assertGreater(summer_avg, winter_avg)


if __name__ == '__main__':
    unittest.main()