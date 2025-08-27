"""
Unit tests for the system module.
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.system import SystemManager


class TestSystemManager(unittest.TestCase):
    """
    Test cases for the SystemManager class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a temporary directory for configuration files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Mock configuration
        self.mock_config = {
            'system': {
                'name': 'Test System',
                'version': '0.1.0',
                'log_level': 'DEBUG'
            },
            'hardware': {
                'thermal_sensors': {},
                'gas_sensors': {},
                'environmental_sensors': {}
            },
            'agents': {},
            'models': {}
        }
    
    def tearDown(self):
        """
        Tear down test fixtures.
        """
        self.temp_dir.cleanup()
    
    @patch('src.config.base.ConfigurationManager')
    def test_initialization(self, mock_config_manager):
        """
        Test system initialization.
        """
        # Set up mock
        mock_instance = mock_config_manager.return_value
        mock_instance.get.return_value = {}
        mock_instance.environment = 'test'
        
        # Create system manager
        system = SystemManager('config', 'test')
        
        # Check initialization
        self.assertIsNotNone(system)
        self.assertFalse(system.is_initialized)
        self.assertFalse(system.is_running)
    
    @patch('src.config.base.ConfigurationManager')
    @patch('src.hardware.base.HardwareAbstractionLayer')
    def test_hardware_initialization(self, mock_hal, mock_config_manager):
        """
        Test hardware initialization.
        """
        # Set up mocks
        mock_config_instance = mock_config_manager.return_value
        mock_config_instance.get.return_value = {}
        mock_config_instance.environment = 'test'
        
        mock_hal_instance = mock_hal.return_value
        mock_hal_instance.initialize.return_value = True
        
        # Create system manager
        system = SystemManager('config', 'test')
        
        # Initialize system
        result = system.initialize()
        
        # Check results
        self.assertTrue(result)
        mock_hal_instance.initialize.assert_called_once()
    
    @patch('src.config.base.ConfigurationManager')
    @patch('src.hardware.base.HardwareAbstractionLayer')
    def test_hardware_initialization_failure(self, mock_hal, mock_config_manager):
        """
        Test hardware initialization failure.
        """
        # Set up mocks
        mock_config_instance = mock_config_manager.return_value
        mock_config_instance.get.return_value = {}
        mock_config_instance.environment = 'test'
        
        mock_hal_instance = mock_hal.return_value
        mock_hal_instance.initialize.return_value = False
        
        # Create system manager
        system = SystemManager('config', 'test')
        
        # Initialize system
        result = system.initialize()
        
        # Check results
        self.assertFalse(result)
        mock_hal_instance.initialize.assert_called_once()
    
    @patch('src.config.base.ConfigurationManager')
    @patch('src.hardware.base.HardwareAbstractionLayer')
    def test_system_start_stop(self, mock_hal, mock_config_manager):
        """
        Test system start and stop.
        """
        # Set up mocks
        mock_config_instance = mock_config_manager.return_value
        mock_config_instance.get.return_value = {}
        mock_config_instance.environment = 'test'
        
        mock_hal_instance = mock_hal.return_value
        mock_hal_instance.initialize.return_value = True
        
        # Create system manager
        system = SystemManager('config', 'test')
        
        # Initialize system
        system.initialize()
        
        # Start system
        start_result = system.start()
        self.assertTrue(start_result)
        self.assertTrue(system.is_running)
        
        # Stop system
        stop_result = system.stop()
        self.assertTrue(stop_result)
        self.assertFalse(system.is_running)
    
    @patch('src.config.base.ConfigurationManager')
    def test_get_status(self, mock_config_manager):
        """
        Test getting system status.
        """
        # Set up mock
        mock_instance = mock_config_manager.return_value
        mock_instance.get.return_value = {}
        mock_instance.environment = 'test'
        
        # Create system manager
        system = SystemManager('config', 'test')
        
        # Get status
        status = system.get_status()
        
        # Check status
        self.assertIsInstance(status, dict)
        self.assertIn('timestamp', status)
        self.assertIn('is_initialized', status)
        self.assertIn('is_running', status)
        self.assertIn('environment', status)
        self.assertEqual(status['environment'], 'test')


if __name__ == '__main__':
    unittest.main()