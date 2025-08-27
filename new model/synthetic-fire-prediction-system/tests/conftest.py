"""
Pytest configuration and fixtures for the synthetic fire prediction system tests.
"""

import os
import sys
import pytest
import tempfile
import shutil
import yaml
import json
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.base import ConfigurationManager
from src.system import SystemManager


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    
    Yields:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_config_dir(temp_dir):
    """
    Create a temporary configuration directory with basic config files.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to the temporary configuration directory
    """
    config_dir = os.path.join(temp_dir, 'config')
    os.makedirs(os.path.join(config_dir, 'environments'), exist_ok=True)
    os.makedirs(os.path.join(config_dir, 'secrets'), exist_ok=True)
    
    # Create a basic test configuration
    test_config = {
        'system': {
            'name': 'Test System',
            'version': '0.1.0',
            'log_level': 'DEBUG',
            'data_dir': os.path.join(temp_dir, 'data'),
            'models_dir': os.path.join(temp_dir, 'models')
        },
        'hardware': {
            'thermal_sensors': {},
            'gas_sensors': {},
            'environmental_sensors': {}
        },
        'agents': {},
        'models': {}
    }
    
    # Write the test configuration to a file
    with open(os.path.join(config_dir, 'environments', 'test_config.yaml'), 'w') as f:
        yaml.dump(test_config, f)
    
    # Create a basic secrets file
    secrets = {
        'aws': {
            'access_key_id': 'test_access_key',
            'secret_access_key': 'test_secret_key'
        }
    }
    
    # Write the secrets to a file
    with open(os.path.join(config_dir, 'secrets', 'test_secrets.json'), 'w') as f:
        json.dump(secrets, f)
    
    return config_dir


@pytest.fixture
def mock_config_manager(temp_config_dir):
    """
    Create a configuration manager with test configuration.
    
    Args:
        temp_config_dir: Temporary configuration directory fixture
        
    Returns:
        ConfigurationManager instance
    """
    return ConfigurationManager(temp_config_dir, 'test')


@pytest.fixture
def mock_system_manager(monkeypatch, mock_config_manager):
    """
    Create a system manager with mocked dependencies.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
        mock_config_manager: Mock configuration manager fixture
        
    Returns:
        SystemManager instance
    """
    # Mock the initialize_config function to return our mock config manager
    def mock_initialize_config(*args, **kwargs):
        return mock_config_manager
    
    monkeypatch.setattr('src.system.initialize_config', mock_initialize_config)
    
    # Create a system manager
    system = SystemManager('config', 'test')
    
    return system


@pytest.fixture
def sample_thermal_data():
    """
    Generate sample thermal data for testing.
    
    Returns:
        Dictionary containing sample thermal data
    """
    import numpy as np
    from datetime import datetime
    
    # Create a sample thermal image (384x288)
    thermal_image = np.ones((288, 384)) * 25.0  # Ambient temperature
    
    # Add a hotspot
    center_y, center_x = 144, 192
    radius = 20
    for i in range(288):
        for j in range(384):
            distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            if distance <= radius:
                falloff = np.exp(-(distance / radius) ** 2)
                thermal_image[i, j] += 100.0 * falloff  # Hotspot temperature
    
    return {
        'timestamp': datetime.now().isoformat(),
        'thermal_image': thermal_image,
        'mean_temperature': np.mean(thermal_image),
        'max_temperature': np.max(thermal_image),
        'min_temperature': np.min(thermal_image),
        'sensor_id': 'test_thermal_sensor',
        'sensor_type': 'thermal'
    }


@pytest.fixture
def sample_gas_data():
    """
    Generate sample gas data for testing.
    
    Returns:
        Dictionary containing sample gas data
    """
    from datetime import datetime
    
    return {
        'timestamp': datetime.now().isoformat(),
        'concentrations': {
            'methane': 10.0,
            'propane': 5.0,
            'hydrogen': 2.0
        },
        'alarms': {
            'methane': False,
            'propane': False,
            'hydrogen': False
        },
        'sensor_id': 'test_gas_sensor',
        'sensor_type': 'gas'
    }


@pytest.fixture
def sample_environmental_data():
    """
    Generate sample environmental data for testing.
    
    Returns:
        Dictionary containing sample environmental data
    """
    from datetime import datetime
    
    return {
        'timestamp': datetime.now().isoformat(),
        'temperature': 25.0,
        'humidity': 50.0,
        'pressure': 1013.25,
        'voc': {
            'benzene': 0.1,
            'formaldehyde': 0.2,
            'toluene': 0.15
        },
        'dew_point': 13.9,  # Calculated from temperature and humidity
        'sensor_id': 'test_environmental_sensor',
        'sensor_type': 'environmental'
    }