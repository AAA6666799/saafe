"""
Tests for Hardware Abstraction Layer.

This module provides comprehensive tests for the sensor management system,
including synthetic/real sensor integration and mode switching capabilities.
"""

import unittest
import tempfile
import os
import shutil
import time
from typing import Dict, Any
import numpy as np

import sys
sys.path.append('/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

try:
    from src.hardware.sensor_manager import SensorManager, SensorMode, create_sensor_manager
    from src.hardware.real_sensors import RealThermalSensor, RealGasSensor, RealEnvironmentalSensor
    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import hardware components: {e}")
    HARDWARE_AVAILABLE = False


class TestSensorManager(unittest.TestCase):
    """Test cases for the Sensor Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HARDWARE_AVAILABLE:
            self.skipTest("Hardware components not available")
        
        self.test_config = {
            'mode': SensorMode.SYNTHETIC,
            'auto_fallback': True,
            'buffer_size': 100,
            'collection_interval': 0.1,  # Fast for testing
            'max_retry_attempts': 2
        }
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_sensor_manager_initialization(self):
        """Test sensor manager initialization."""
        manager = SensorManager(self.test_config)
        self.assertEqual(manager.mode, SensorMode.SYNTHETIC)
        self.assertTrue(manager.auto_fallback)
        self.assertFalse(manager.is_collecting)
    
    def test_convenience_function(self):
        """Test convenience function for creating sensor manager."""
        manager = create_sensor_manager(self.test_config)
        self.assertIsInstance(manager, SensorManager)
        
        # Test with default config
        default_manager = create_sensor_manager()
        self.assertIsInstance(default_manager, SensorManager)
    
    def test_sensor_registration(self):
        """Test registering different types of sensors."""
        manager = SensorManager(self.test_config)
        
        # Create test sensors
        thermal_config = {'sensor_id': 'thermal_01', 'sensor_type': 'thermal'}
        thermal_sensor = RealThermalSensor(thermal_config)
        
        gas_config = {'sensor_id': 'gas_01', 'sensor_type': 'gas'}
        gas_sensor = RealGasSensor(gas_config)
        
        env_config = {'sensor_id': 'env_01', 'sensor_type': 'environmental'}
        env_sensor = RealEnvironmentalSensor(env_config)
        
        # Register sensors
        self.assertTrue(manager.register_thermal_sensor('thermal_01', thermal_sensor))
        self.assertTrue(manager.register_gas_sensor('gas_01', gas_sensor))
        self.assertTrue(manager.register_environmental_sensor('env_01', env_sensor))
        
        # Check registration
        self.assertIn('thermal_01', manager.thermal_sensors)
        self.assertIn('gas_01', manager.gas_sensors)
        self.assertIn('env_01', manager.environmental_sensors)
        self.assertEqual(len(manager.all_sensors), 3)
    
    def test_sensor_initialization(self):
        """Test sensor initialization process."""
        manager = SensorManager(self.test_config)
        
        # Register a test sensor
        thermal_config = {'sensor_id': 'thermal_test', 'sensor_type': 'thermal'}
        thermal_sensor = RealThermalSensor(thermal_config)
        manager.register_thermal_sensor('thermal_test', thermal_sensor)
        
        # Initialize sensors
        results = manager.initialize_sensors()
        self.assertIn('thermal_test', results)
        self.assertTrue(results['thermal_test'])
    
    def test_data_collection(self):
        """Test data collection functionality."""
        manager = SensorManager(self.test_config)
        
        # Register sensors
        thermal_config = {'sensor_id': 'thermal_test', 'sensor_type': 'thermal'}
        thermal_sensor = RealThermalSensor(thermal_config)
        manager.register_thermal_sensor('thermal_test', thermal_sensor)
        
        gas_config = {'sensor_id': 'gas_test', 'sensor_type': 'gas'}
        gas_sensor = RealGasSensor(gas_config)
        manager.register_gas_sensor('gas_test', gas_sensor)
        
        # Initialize and read data
        manager.initialize_sensors()
        sensor_data = manager.read_all_sensors()
        
        # Validate data structure
        self.assertIn('thermal', sensor_data)
        self.assertIn('gas', sensor_data)
        self.assertIn('environmental', sensor_data)
        self.assertIn('metadata', sensor_data)
        
        # Check thermal data
        if sensor_data['thermal']:
            thermal_data = list(sensor_data['thermal'].values())[0]
            self.assertIn('temperature_max', thermal_data)
            self.assertIn('timestamp', thermal_data)
    
    def test_continuous_collection(self):
        """Test continuous data collection in background."""
        manager = SensorManager(self.test_config)
        
        # Register a sensor
        thermal_config = {'sensor_id': 'thermal_continuous', 'sensor_type': 'thermal'}
        thermal_sensor = RealThermalSensor(thermal_config)
        manager.register_thermal_sensor('thermal_continuous', thermal_sensor)
        
        # Initialize and start collection
        manager.initialize_sensors()
        self.assertTrue(manager.start_data_collection())
        self.assertTrue(manager.is_collecting)
        
        # Wait for some data collection
        time.sleep(0.5)
        
        # Check buffer has data
        recent_data = manager.get_recent_data(5)
        self.assertGreater(len(recent_data), 0)
        
        # Stop collection
        manager.stop_data_collection()
        self.assertFalse(manager.is_collecting)
    
    def test_mode_switching(self):
        """Test switching between sensor modes."""
        manager = SensorManager(self.test_config)
        
        # Test mode switching
        self.assertTrue(manager.switch_mode(SensorMode.REAL))
        self.assertEqual(manager.mode, SensorMode.REAL)
        
        self.assertTrue(manager.switch_mode(SensorMode.HYBRID))
        self.assertEqual(manager.mode, SensorMode.HYBRID)
        
        self.assertTrue(manager.switch_mode(SensorMode.SYNTHETIC))
        self.assertEqual(manager.mode, SensorMode.SYNTHETIC)
        
        # Test invalid mode
        self.assertFalse(manager.switch_mode('invalid_mode'))
    
    def test_health_monitoring(self):
        """Test sensor health monitoring."""
        manager = SensorManager(self.test_config)
        
        # Register sensor
        thermal_config = {'sensor_id': 'thermal_health', 'sensor_type': 'thermal'}
        thermal_sensor = RealThermalSensor(thermal_config)
        manager.register_thermal_sensor('thermal_health', thermal_sensor)
        
        # Get initial health
        health = manager.get_sensor_health()
        self.assertIn('overall_status', health)
        self.assertIn('sensors', health)
        self.assertIn('thermal_health', health['sensors'])
        
        # Initialize and check health again
        manager.initialize_sensors()
        health = manager.get_sensor_health()
        self.assertEqual(health['sensors']['thermal_health']['status'], 'connected')
    
    def test_synthetic_fallback(self):
        """Test synthetic data fallback functionality."""
        config = self.test_config.copy()
        config['mode'] = SensorMode.REAL
        config['auto_fallback'] = True
        
        manager = SensorManager(config)
        
        # Don't register any real sensors - should use fallback
        sensor_data = manager.read_all_sensors()
        
        # Should still get data (synthetic fallback)
        self.assertIn('thermal', sensor_data)
        self.assertIn('gas', sensor_data)
        self.assertIn('environmental', sensor_data)
    
    def test_data_export(self):
        """Test data export functionality."""
        manager = SensorManager(self.test_config)
        
        # Add some data to buffer
        for _ in range(5):
            data = manager.read_all_sensors()
        
        # Test JSON export
        json_file = os.path.join(self.temp_dir, 'test_data.json')
        self.assertTrue(manager.export_data(json_file, 'json'))
        self.assertTrue(os.path.exists(json_file))
        
        # Test CSV export
        csv_file = os.path.join(self.temp_dir, 'test_data.csv')
        self.assertTrue(manager.export_data(csv_file, 'csv'))
        self.assertTrue(os.path.exists(csv_file))
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        manager = SensorManager(self.test_config)
        
        # Test with malformed sensor config
        try:
            bad_config = {}  # Missing required fields
            bad_sensor = RealThermalSensor(bad_config)
            # Should raise ValueError during validation
            self.fail("Should have raised ValueError for bad config")
        except ValueError:
            pass  # Expected
    
    def test_calibration(self):
        """Test sensor calibration functionality."""
        manager = SensorManager(self.test_config)
        
        # Register sensor
        thermal_config = {'sensor_id': 'thermal_cal', 'sensor_type': 'thermal'}
        thermal_sensor = RealThermalSensor(thermal_config)
        manager.register_thermal_sensor('thermal_cal', thermal_sensor)
        
        # Initialize and calibrate
        manager.initialize_sensors()
        results = manager.calibrate_sensors(['thermal_cal'])
        
        self.assertIn('thermal_cal', results)
        self.assertTrue(results['thermal_cal'])


class TestRealSensors(unittest.TestCase):
    """Test cases for real sensor implementations."""
    
    def test_real_thermal_sensor(self):
        """Test real thermal sensor implementation."""
        if not HARDWARE_AVAILABLE:
            self.skipTest("Hardware components not available")
        
        config = {
            'sensor_id': 'thermal_real_test',
            'sensor_type': 'thermal',
            'device_path': '/dev/thermal0',
            'resolution': (640, 480),
            'temperature_range': (-20.0, 150.0)
        }
        
        sensor = RealThermalSensor(config)
        
        # Test connection
        self.assertTrue(sensor.connect())
        self.assertTrue(sensor.connected)
        
        # Test reading
        data = sensor.read()
        self.assertIn('temperature_max', data)
        self.assertIn('temperature_avg', data)
        self.assertIn('thermal_image', data)
        
        # Test thermal-specific methods
        image = sensor.get_thermal_image()
        self.assertIsInstance(image, np.ndarray)
        
        resolution = sensor.get_resolution()
        self.assertEqual(resolution, (640, 480))
        
        temp_range = sensor.get_temperature_range()
        self.assertEqual(temp_range, (-20.0, 150.0))
        
        # Test disconnection
        self.assertTrue(sensor.disconnect())
        self.assertFalse(sensor.connected)
    
    def test_real_gas_sensor(self):
        """Test real gas sensor implementation."""
        if not HARDWARE_AVAILABLE:
            self.skipTest("Hardware components not available")
        
        config = {
            'sensor_id': 'gas_real_test',
            'sensor_type': 'gas',
            'device_address': '/dev/ttyUSB0',
            'supported_gases': ['co', 'co2', 'smoke', 'voc']
        }
        
        sensor = RealGasSensor(config)
        
        # Test connection
        self.assertTrue(sensor.connect())
        
        # Test reading
        data = sensor.read()
        self.assertIn('co_concentration', data)
        self.assertIn('timestamp', data)
        
        # Test gas-specific methods
        concentrations = sensor.get_gas_concentration()
        self.assertIsInstance(concentrations, dict)
        
        supported_gases = sensor.get_supported_gases()
        self.assertEqual(supported_gases, ['co', 'co2', 'smoke', 'voc'])
        
        # Test alarm threshold setting
        self.assertTrue(sensor.set_alarm_threshold('co', 50.0))
        self.assertFalse(sensor.set_alarm_threshold('invalid_gas', 50.0))
        
        sensor.disconnect()
    
    def test_real_environmental_sensor(self):
        """Test real environmental sensor implementation."""
        if not HARDWARE_AVAILABLE:
            self.skipTest("Hardware components not available")
        
        config = {
            'sensor_id': 'env_real_test',
            'sensor_type': 'environmental',
            'device_address': 0x77
        }
        
        sensor = RealEnvironmentalSensor(config)
        
        # Test connection
        self.assertTrue(sensor.connect())
        
        # Test reading
        data = sensor.read()
        self.assertIn('temperature', data)
        self.assertIn('humidity', data)
        self.assertIn('pressure', data)
        
        # Test environmental-specific methods
        temperature = sensor.get_temperature()
        self.assertIsInstance(temperature, float)
        
        humidity = sensor.get_humidity()
        self.assertIsInstance(humidity, float)
        
        sensor.disconnect()


class TestHardwareIntegration(unittest.TestCase):
    """Integration tests for hardware abstraction layer."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end sensor workflow."""
        if not HARDWARE_AVAILABLE:
            self.skipTest("Hardware components not available")
        
        # Create manager with hybrid mode
        config = {
            'mode': SensorMode.HYBRID,
            'auto_fallback': True,
            'collection_interval': 0.1
        }
        manager = create_sensor_manager(config)
        
        # Register sensors of each type
        thermal_config = {'sensor_id': 'thermal_e2e', 'sensor_type': 'thermal'}
        gas_config = {'sensor_id': 'gas_e2e', 'sensor_type': 'gas'}
        env_config = {'sensor_id': 'env_e2e', 'sensor_type': 'environmental'}
        
        thermal_sensor = RealThermalSensor(thermal_config)
        gas_sensor = RealGasSensor(gas_config)
        env_sensor = RealEnvironmentalSensor(env_config)
        
        manager.register_thermal_sensor('thermal_e2e', thermal_sensor)
        manager.register_gas_sensor('gas_e2e', gas_sensor)
        manager.register_environmental_sensor('env_e2e', env_sensor)
        
        # Initialize system
        init_results = manager.initialize_sensors()
        self.assertGreater(sum(init_results.values()), 0)  # At least one successful init
        
        # Start continuous collection
        self.assertTrue(manager.start_data_collection())
        
        # Let it collect for a short time
        time.sleep(0.5)
        
        # Check collected data
        recent_data = manager.get_recent_data(3)
        self.assertGreater(len(recent_data), 0)
        
        # Check health status
        health = manager.get_sensor_health()
        self.assertIn('overall_status', health)
        
        # Stop collection and shutdown
        manager.stop_data_collection()
        manager.shutdown()


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2, buffer=True)