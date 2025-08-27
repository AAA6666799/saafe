"""
Unit tests for the CorrelationEngine class.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.environmental.correlation_engine import CorrelationEngine


class TestCorrelationEngine(unittest.TestCase):
    """Test cases for the CorrelationEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'correlation_strengths': {
                'temperature_humidity': 0.7,
                'temperature_pressure': 0.3,
                'humidity_pressure': 0.4,
                'temperature_voc': 0.6
            },
            'scenario_type': 'indoor'
        }
        self.engine = CorrelationEngine(self.config)

    def test_initialization(self):
        """Test initialization of the correlation engine."""
        self.assertEqual(self.engine.config['correlation_strengths']['temperature_humidity'], 0.7)
        self.assertEqual(self.engine.config['correlation_strengths']['temperature_pressure'], 0.3)
        self.assertEqual(self.engine.config['correlation_strengths']['humidity_pressure'], 0.4)
        self.assertEqual(self.engine.config['correlation_strengths']['temperature_voc'], 0.6)
        self.assertEqual(self.engine.config['scenario_type'], 'indoor')
        
        # Check that correlation models are initialized
        self.assertIn('temperature_humidity', self.engine.correlation_models)
        self.assertIn('temperature_pressure', self.engine.correlation_models)
        self.assertIn('humidity_pressure', self.engine.correlation_models)
        self.assertIn('temperature_voc', self.engine.correlation_models)
        
        # Check that parameter dependencies are initialized
        self.assertIn('temperature', self.engine.parameter_dependencies)
        self.assertIn('humidity', self.engine.parameter_dependencies)
        self.assertIn('pressure', self.engine.parameter_dependencies)
        self.assertIn('voc', self.engine.parameter_dependencies)

    def test_validate_config_default_values(self):
        """Test validation of configuration with default values."""
        # Create engine with empty config
        engine = CorrelationEngine({})
        
        # Check that default values were set
        self.assertIn('correlation_strengths', engine.config)
        self.assertIn('scenario_type', engine.config)
        self.assertEqual(engine.config['scenario_type'], 'indoor')

    def create_test_data(self):
        """Create test environmental data for correlation tests."""
        # Create timestamps
        start_time = datetime.now()
        num_samples = 24  # One day with hourly samples
        timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
        
        # Create environmental data
        env_data = {
            'temperature': {
                'values': [20.0 + 5.0 * np.sin(2 * np.pi * i / num_samples) for i in range(num_samples)],
                'unit': '°C',
                'min': 15.0,
                'max': 25.0
            },
            'humidity': {
                'values': [50.0 - 10.0 * np.sin(2 * np.pi * i / num_samples) for i in range(num_samples)],
                'unit': '%',
                'min': 40.0,
                'max': 60.0
            },
            'pressure': {
                'values': [1010.0 + 2.0 * np.sin(2 * np.pi * i / num_samples + np.pi/2) for i in range(num_samples)],
                'unit': 'hPa',
                'min': 1008.0,
                'max': 1012.0
            },
            'voc': {
                'values': [100.0 + 50.0 * np.sin(2 * np.pi * i / num_samples + np.pi/4) for i in range(num_samples)],
                'unit': 'ppb',
                'min': 50.0,
                'max': 150.0
            }
        }
        
        return env_data, timestamps

    def test_apply_correlations(self):
        """Test application of correlations to environmental data."""
        # Create test data
        env_data, timestamps = self.create_test_data()
        
        # Store original values
        original_values = {
            param: env_data[param]['values'].copy() for param in env_data
        }
        
        # Apply correlations
        correlated_data = self.engine.apply_correlations(env_data, timestamps)
        
        # Check that all parameters are present
        for param in original_values:
            self.assertIn(param, correlated_data)
            self.assertIn('values', correlated_data[param])
        
        # Check that values have been modified by correlations
        for param in original_values:
            if param != 'temperature':  # Temperature is the primary parameter that affects others
                original = np.array(original_values[param])
                correlated = np.array(correlated_data[param]['values'])
                
                # Check that values have changed
                self.assertTrue(np.any(original != correlated))

    def test_temperature_humidity_model(self):
        """Test the temperature-humidity correlation model."""
        # Create test data
        env_data, timestamps = self.create_test_data()
        
        # Store original humidity values
        original_humidity = env_data['humidity']['values'].copy()
        
        # Apply temperature-humidity correlation
        updated_data = self.engine._temperature_humidity_model(
            env_data, 'temperature', 'humidity', timestamps, 'indoor'
        )
        
        # Check that humidity values have been modified
        updated_humidity = updated_data['humidity']['values']
        self.assertTrue(np.any(original_humidity != updated_humidity))
        
        # Check inverse relationship: when temperature is high, humidity should be lower
        temp_values = np.array(env_data['temperature']['values'])
        humid_values = np.array(updated_humidity)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(temp_values, humid_values)[0, 1]
        
        # Should be negative correlation
        self.assertLess(correlation, 0)

    def test_temperature_pressure_model(self):
        """Test the temperature-pressure correlation model."""
        # Create test data
        env_data, timestamps = self.create_test_data()
        
        # Store original pressure values
        original_pressure = env_data['pressure']['values'].copy()
        
        # Apply temperature-pressure correlation
        updated_data = self.engine._temperature_pressure_model(
            env_data, 'temperature', 'pressure', timestamps, 'indoor'
        )
        
        # Check that pressure values have been modified
        updated_pressure = updated_data['pressure']['values']
        self.assertTrue(np.any(original_pressure != updated_pressure))

    def test_humidity_pressure_model(self):
        """Test the humidity-pressure correlation model."""
        # Create test data
        env_data, timestamps = self.create_test_data()
        
        # Store original pressure values
        original_pressure = env_data['pressure']['values'].copy()
        
        # Apply humidity-pressure correlation
        updated_data = self.engine._humidity_pressure_model(
            env_data, 'humidity', 'pressure', timestamps, 'outdoor'
        )
        
        # Check that pressure values have been modified
        updated_pressure = updated_data['pressure']['values']
        self.assertTrue(np.any(original_pressure != updated_pressure))

    def test_temperature_voc_model(self):
        """Test the temperature-VOC correlation model."""
        # Create test data
        env_data, timestamps = self.create_test_data()
        
        # Store original VOC values
        original_voc = env_data['voc']['values'].copy()
        
        # Apply temperature-VOC correlation
        updated_data = self.engine._temperature_voc_model(
            env_data, 'temperature', 'voc', timestamps, 'indoor'
        )
        
        # Check that VOC values have been modified
        updated_voc = updated_data['voc']['values']
        self.assertTrue(np.any(original_voc != updated_voc))
        
        # Check positive relationship: when temperature is high, VOC should be higher
        temp_values = np.array(env_data['temperature']['values'])
        voc_values = np.array(updated_voc)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(temp_values, voc_values)[0, 1]
        
        # Should be positive correlation
        self.assertGreater(correlation, 0)

    def test_calculate_dew_point(self):
        """Test calculation of dew point from temperature and humidity."""
        # Test cases with known results
        test_cases = [
            # (temperature, humidity, expected_dew_point)
            (20.0, 50.0, 9.3),
            (25.0, 60.0, 16.7),
            (30.0, 80.0, 26.2),
            (15.0, 40.0, 1.9)
        ]
        
        for temp, humidity, expected_dew_point in test_cases:
            dew_point = self.engine.calculate_dew_point(temp, humidity)
            self.assertAlmostEqual(dew_point, expected_dew_point, delta=0.5)

    def test_add_derived_parameters(self):
        """Test addition of derived parameters."""
        # Create test data
        env_data, timestamps = self.create_test_data()
        
        # Add derived parameters
        updated_data = self.engine.add_derived_parameters(env_data, timestamps)
        
        # Check that dew point was added
        self.assertIn('dew_point', updated_data)
        self.assertIn('values', updated_data['dew_point'])
        self.assertIn('unit', updated_data['dew_point'])
        self.assertEqual(updated_data['dew_point']['unit'], '°C')
        
        # Check that heat index was added
        self.assertIn('heat_index', updated_data)
        self.assertIn('values', updated_data['heat_index'])
        self.assertIn('unit', updated_data['heat_index'])
        self.assertEqual(updated_data['heat_index']['unit'], '°C')
        
        # Check that the correct number of values was generated
        self.assertEqual(len(updated_data['dew_point']['values']), len(timestamps))
        self.assertEqual(len(updated_data['heat_index']['values']), len(timestamps))

    def test_calculate_heat_index(self):
        """Test calculation of heat index from temperature and humidity."""
        # Test cases with known results
        test_cases = [
            # (temperature, humidity, expected_heat_index)
            (30.0, 70.0, 34.5),
            (25.0, 50.0, 25.9),
            (35.0, 80.0, 49.5),
            (20.0, 40.0, 19.8)
        ]
        
        for temp, humidity, expected_heat_index in test_cases:
            heat_index = self.engine.calculate_heat_index(temp, humidity)
            self.assertAlmostEqual(heat_index, expected_heat_index, delta=1.0)


if __name__ == '__main__':
    unittest.main()