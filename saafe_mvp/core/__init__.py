"""
Core module for Saafe MVP
Contains main business logic, scenario management, and data processing.
"""

from .data_models import SensorReading, ScenarioConfig, SensorLimits
from .scenario_generator import BaseScenarioGenerator
from .normal_scenario import NormalScenarioGenerator
from .cooking_scenario import CookingScenarioGenerator
from .fire_scenario import FireScenarioGenerator
from .scenario_manager import ScenarioManager, ScenarioType
from .data_stream import DataStreamManager, get_data_stream_manager

__all__ = [
    'SensorReading',
    'ScenarioConfig', 
    'SensorLimits',
    'BaseScenarioGenerator',
    'NormalScenarioGenerator',
    'CookingScenarioGenerator',
    'FireScenarioGenerator',
    'ScenarioManager',
    'ScenarioType',
    'DataStreamManager',
    'get_data_stream_manager'
]