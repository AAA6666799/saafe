"""
Unit tests for the SensorResponseModel class.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.gas.sensor_response_model import SensorResponseModel, SensorType


class TestSensorResponseModel(unittest.TestCase):
    """Test cases for the SensorResponseModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.config = {
            'sensor_type': 'electrochemical',
            'noise_level': 0.02,
            'drift_rate': 0.5,
            'response_time': 30.0,
            'recovery_time': 60.0,
            'calibration_error': 1.0,
            'resolution': 1.0,
            'cross_sensitivity': {
                'methane': {
                    'hydrogen': 0.05,
                    'carbon_monoxide': 0.02
                }
            }
        }
        
        # Create the sensor model
        self.model = SensorResponseModel(self.config)
    
    def test_initialization(self):
        """Test initialization of the sensor model."""
        self.assertEqual(self.model.sensor_type, SensorType.ELECTROCHEMICAL)
        self.assertEqual(self.model.noise_level, 0.02)
        self.assertEqual(self.model.drift_rate, 0.5)
        self.assertEqual(self.model.response_time, 30.0)
        self.assertEqual(self.model.recovery_time, 60.0)
        self.assertEqual(self.model.calibration_error, 1.0)
        self.assertEqual(self.model.resolution, 1.0)
        self.assertIsNotNone(self.model.cross_sensitivity)
        self.assertEqual(self.model.last_reading, 0.0)
        self.assertIsNone(self.model.last_timestamp)
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Test with valid config
        self.model.validate_config()
        
        # Test with invalid sensor type
        config_invalid_type = self.config.copy()
        config_invalid_type['sensor_type'] = 'invalid_type'
        with self.assertRaises(ValueError):
            SensorResponseModel(config_invalid_type).validate_config()
        
        # Test with negative noise level
        config_negative_noise = self.config.copy()
        config_negative_noise['noise_level'] = -0.1
        with self.assertRaises(ValueError):
            SensorResponseModel(config_negative_noise).validate_config()
        
        # Test with negative drift rate
        config_negative_drift = self.config.copy()
        config_negative_drift['drift_rate'] = -0.5
        with self.assertRaises(ValueError):
            SensorResponseModel(config_negative_drift).validate_config()
        
        # Test with zero response time
        config_zero_response = self.config.copy()
        config_zero_response['response_time'] = 0.0
        with self.assertRaises(ValueError):
            SensorResponseModel(config_zero_response).validate_config()
        
        # Test with zero recovery time
        config_zero_recovery = self.config.copy()
        config_zero_recovery['recovery_time'] = 0.0
        with self.assertRaises(ValueError):
            SensorResponseModel(config_zero_recovery).validate_config()
        
        # Test with zero resolution
        config_zero_resolution = self.config.copy()
        config_zero_resolution['resolution'] = 0.0
        with self.assertRaises(ValueError):
            SensorResponseModel(config_zero_resolution).validate_config()
    
    def test_default_cross_sensitivity(self):
        """Test default cross-sensitivity values."""
        # Test electrochemical sensor
        model = SensorResponseModel({'sensor_type': 'electrochemical'})
        cross_sens = model._default_cross_sensitivity()
        self.assertIn('methane', cross_sens)
        self.assertIn('hydrogen', cross_sens)
        
        # Test catalytic sensor
        model = SensorResponseModel({'sensor_type': 'catalytic'})
        cross_sens = model._default_cross_sensitivity()
        self.assertIn('methane', cross_sens)
        self.assertIn('hydrogen', cross_sens)
        
        # Test infrared sensor
        model = SensorResponseModel({'sensor_type': 'infrared'})
        cross_sens = model._default_cross_sensitivity()
        self.assertIn('methane', cross_sens)
        self.assertIn('propane', cross_sens)
        
        # Test semiconductor sensor
        model = SensorResponseModel({'sensor_type': 'semiconductor'})
        cross_sens = model._default_cross_sensitivity()
        self.assertIn('methane', cross_sens)
        self.assertIn('hydrogen', cross_sens)
        
        # Test photoionization sensor
        model = SensorResponseModel({'sensor_type': 'photoionization'})
        cross_sens = model._default_cross_sensitivity()
        self.assertIn('benzene', cross_sens)
        self.assertIn('toluene', cross_sens)
    
    def test_simulate_response(self):
        """Test simulation of sensor response."""
        # Test first reading (no previous state)
        timestamp = datetime.now()
        true_concentration = 100.0
        gas_type = 'methane'
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        reading = self.model.simulate_response(true_concentration, gas_type, timestamp)
        
        # Check that reading is not exactly equal to true concentration
        self.assertNotEqual(reading, true_concentration)
        
        # Check that last_reading and last_timestamp were updated
        self.assertEqual(self.model.last_reading, reading)
        self.assertEqual(self.model.last_timestamp, timestamp)
        
        # Test second reading (with previous state)
        timestamp2 = timestamp + timedelta(seconds=10)
        true_concentration2 = 150.0
        
        reading2 = self.model.simulate_response(true_concentration2, gas_type, timestamp2)
        
        # Check that reading is influenced by previous reading (response dynamics)
        self.assertLess(reading2, true_concentration2)  # Should not reach full value yet
        
        # Test with cross-sensitivity
        other_gases = {'hydrogen': 50.0, 'carbon_monoxide': 20.0}
        
        reading3 = self.model.simulate_response(true_concentration2, gas_type, timestamp2, other_gases)
        
        # Check that cross-sensitivity affects the reading
        self.assertNotEqual(reading3, reading2)
    
    def test_simulate_batch_response(self):
        """Test simulation of batch sensor response."""
        # Create test data
        timestamps = [
            datetime.now(),
            datetime.now() + timedelta(seconds=10),
            datetime.now() + timedelta(seconds=20)
        ]
        true_concentrations = [100.0, 150.0, 120.0]
        gas_type = 'methane'
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Reset sensor state
        self.model.reset_state()
        
        # Simulate batch response
        readings = self.model.simulate_batch_response(true_concentrations, gas_type, timestamps)
        
        # Check results
        self.assertEqual(len(readings), len(true_concentrations))
        
        # Check that readings are not exactly equal to true concentrations
        for true_conc, reading in zip(true_concentrations, readings):
            self.assertNotEqual(reading, true_conc)
        
        # Check that readings show response dynamics
        # Second reading should be less than true concentration due to response time
        self.assertLess(readings[1], true_concentrations[1])
        
        # Test with other gases
        other_gases = [
            {'hydrogen': 50.0},
            {'hydrogen': 60.0},
            {'hydrogen': 40.0}
        ]
        
        # Reset sensor state
        self.model.reset_state()
        
        # Simulate batch response with other gases
        readings_with_others = self.model.simulate_batch_response(
            true_concentrations, gas_type, timestamps, other_gases
        )
        
        # Check that cross-sensitivity affects the readings
        for r1, r2 in zip(readings, readings_with_others):
            self.assertNotEqual(r1, r2)
    
    def test_simulate_temperature_effect(self):
        """Test simulation of temperature effect on sensor readings."""
        # Test at reference temperature
        reading = 100.0
        temperature = 25.0
        adjusted = self.model.simulate_temperature_effect(reading, temperature)
        self.assertEqual(adjusted, reading)  # No change at reference temperature
        
        # Test at higher temperature
        temperature_high = 35.0
        adjusted_high = self.model.simulate_temperature_effect(reading, temperature_high)
        self.assertGreater(adjusted_high, reading)  # Should increase with temperature
        
        # Test at lower temperature
        temperature_low = 15.0
        adjusted_low = self.model.simulate_temperature_effect(reading, temperature_low)
        self.assertLess(adjusted_low, reading)  # Should decrease with temperature
        
        # Test with different sensor types
        # Create models for different sensor types
        sensor_types = ['electrochemical', 'catalytic', 'infrared', 'semiconductor', 'photoionization']
        adjustments = []
        
        for sensor_type in sensor_types:
            model = SensorResponseModel({'sensor_type': sensor_type})
            adj = model.simulate_temperature_effect(reading, temperature_high)
            adjustments.append(adj)
        
        # Different sensor types should have different temperature coefficients
        self.assertNotEqual(len(set(adjustments)), 1)
    
    def test_simulate_humidity_effect(self):
        """Test simulation of humidity effect on sensor readings."""
        # Test at reference humidity
        reading = 100.0
        humidity = 50.0
        adjusted = self.model.simulate_humidity_effect(reading, humidity)
        self.assertEqual(adjusted, reading)  # No change at reference humidity
        
        # Test at higher humidity
        humidity_high = 70.0
        adjusted_high = self.model.simulate_humidity_effect(reading, humidity_high)
        self.assertNotEqual(adjusted_high, reading)  # Should change with humidity
        
        # Test at lower humidity
        humidity_low = 30.0
        adjusted_low = self.model.simulate_humidity_effect(reading, humidity_low)
        self.assertNotEqual(adjusted_low, reading)  # Should change with humidity
        
        # Test with different sensor types
        # Create models for different sensor types
        sensor_types = ['electrochemical', 'catalytic', 'infrared', 'semiconductor', 'photoionization']
        adjustments = []
        
        for sensor_type in sensor_types:
            model = SensorResponseModel({'sensor_type': sensor_type})
            adj = model.simulate_humidity_effect(reading, humidity_high)
            adjustments.append(adj)
        
        # Different sensor types should have different humidity coefficients
        self.assertNotEqual(len(set(adjustments)), 1)
    
    def test_simulate_aging_effect(self):
        """Test simulation of aging effect on sensor readings."""
        # Test with no aging
        reading = 100.0
        age_hours = 0.0
        adjusted = self.model.simulate_aging_effect(reading, age_hours)
        self.assertEqual(adjusted, reading)  # No change with no aging
        
        # Test with aging
        age_hours = 1000.0
        adjusted_aged = self.model.simulate_aging_effect(reading, age_hours)
        self.assertLess(adjusted_aged, reading)  # Should decrease with aging
        
        # Test with different sensor types
        # Create models for different sensor types
        sensor_types = ['electrochemical', 'catalytic', 'infrared', 'semiconductor', 'photoionization']
        adjustments = []
        
        for sensor_type in sensor_types:
            model = SensorResponseModel({'sensor_type': sensor_type})
            adj = model.simulate_aging_effect(reading, age_hours)
            adjustments.append(adj)
        
        # Different sensor types should have different aging coefficients
        self.assertNotEqual(len(set(adjustments)), 1)
    
    def test_get_detection_limit(self):
        """Test getting detection limit for different gases."""
        # Test with electrochemical sensor
        model = SensorResponseModel({'sensor_type': 'electrochemical'})
        
        # Detection limits should vary by gas type
        methane_limit = model.get_detection_limit('methane')
        hydrogen_limit = model.get_detection_limit('hydrogen')
        co_limit = model.get_detection_limit('carbon_monoxide')
        
        self.assertNotEqual(methane_limit, hydrogen_limit)
        self.assertNotEqual(methane_limit, co_limit)
        
        # Test with different sensor types
        catalytic_model = SensorResponseModel({'sensor_type': 'catalytic'})
        infrared_model = SensorResponseModel({'sensor_type': 'infrared'})
        
        # Different sensor types should have different detection limits for the same gas
        methane_limit_cat = catalytic_model.get_detection_limit('methane')
        methane_limit_ir = infrared_model.get_detection_limit('methane')
        
        self.assertNotEqual(methane_limit, methane_limit_cat)
        self.assertNotEqual(methane_limit, methane_limit_ir)
        self.assertNotEqual(methane_limit_cat, methane_limit_ir)


if __name__ == '__main__':
    unittest.main()