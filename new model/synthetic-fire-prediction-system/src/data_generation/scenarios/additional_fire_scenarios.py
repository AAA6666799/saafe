"""
Additional fire scenario generators for synthetic fire data.

This module provides implementations of scenario generators for additional
fire types to augment the existing synthetic data generation capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
import copy

from .scenario_generator import ScenarioGenerator
from .specific_generators import SpecificScenarioGenerator
from ..thermal.thermal_image_generator import ThermalImageGenerator
from ..gas.gas_concentration_generator import GasConcentrationGenerator
from ..environmental.environmental_data_generator import EnvironmentalDataGenerator


class ChemicalFireScenarioGenerator(SpecificScenarioGenerator):
    """
    Generator for chemical fire scenarios.
    
    This class generates scenarios with chemical fire characteristics,
    including specific gas emissions and thermal signatures.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the chemical fire scenario generator.
        
        Args:
            thermal_generator: Thermal data generator instance
            gas_generator: Gas data generator instance
            environmental_generator: Environmental data generator instance
            config: Configuration parameters
        """
        super().__init__(thermal_generator, gas_generator, environmental_generator, config)
        
        # Set default parameters for chemical fire scenarios
        self.default_scenario_params.update({
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 5.0,  # kW
                'growth_rate': 0.5,  # kW/s
                'max_size': 50.0  # kW
            },
            'thermal_params': {
                'max_temperature': 1200.0,  # °C
                'hotspot_count': 3
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide', 'methane', 'hydrogen'],
                'release_rates': {
                    'carbon_monoxide': 0.1,  # g/s
                    'methane': 0.05,  # g/s
                    'hydrogen': 0.02  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 25.0, 'max': 100.0},
                    'humidity': {'min': 20.0, 'max': 80.0},
                    'pressure': {'min': 980.0, 'max': 1020.0},
                    'voc': {'min': 200.0, 'max': 1000.0}
                }
            },
            'metadata': {
                'description': 'Chemical fire scenario with specific gas emissions',
                'classification': 'fire',
                'fire_type': 'chemical'
            }
        })
    
    def get_scenario_type(self) -> str:
        """
        Get the type of scenario this generator produces.
        
        Returns:
            Scenario type string
        """
        return 'chemical_fire'


class SmolderingFireScenarioGenerator(SpecificScenarioGenerator):
    """
    Generator for smoldering fire scenarios.
    
    This class generates scenarios with smoldering fire characteristics,
    including slow temperature rise and specific gas emissions.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the smoldering fire scenario generator.
        
        Args:
            thermal_generator: Thermal data generator instance
            gas_generator: Gas data generator instance
            environmental_generator: Environmental data generator instance
            config: Configuration parameters
        """
        super().__init__(thermal_generator, gas_generator, environmental_generator, config)
        
        # Set default parameters for smoldering fire scenarios
        self.default_scenario_params.update({
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 1.0,  # kW
                'growth_rate': 0.05,  # kW/s
                'max_size': 10.0  # kW
            },
            'thermal_params': {
                'max_temperature': 400.0,  # °C
                'hotspot_count': 1
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide', 'methane'],
                'release_rates': {
                    'carbon_monoxide': 0.05,  # g/s
                    'methane': 0.01  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 22.0, 'max': 50.0},
                    'humidity': {'min': 30.0, 'max': 90.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 100.0, 'max': 800.0}
                }
            },
            'metadata': {
                'description': 'Smoldering fire scenario with slow temperature rise',
                'classification': 'fire',
                'fire_type': 'smoldering'
            }
        })
    
    def get_scenario_type(self) -> str:
        """
        Get the type of scenario this generator produces.
        
        Returns:
            Scenario type string
        """
        return 'smoldering_fire'


class RapidCombustionScenarioGenerator(SpecificScenarioGenerator):
    """
    Generator for rapid combustion fire scenarios.
    
    This class generates scenarios with rapid combustion characteristics,
    including fast temperature rise and high gas emissions.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the rapid combustion scenario generator.
        
        Args:
            thermal_generator: Thermal data generator instance
            gas_generator: Gas data generator instance
            environmental_generator: Environmental data generator instance
            config: Configuration parameters
        """
        super().__init__(thermal_generator, gas_generator, environmental_generator, config)
        
        # Set default parameters for rapid combustion scenarios
        self.default_scenario_params.update({
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 10.0,  # kW
                'growth_rate': 2.0,  # kW/s
                'max_size': 100.0  # kW
            },
            'thermal_params': {
                'max_temperature': 1500.0,  # °C
                'hotspot_count': 5
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide', 'methane', 'hydrogen'],
                'release_rates': {
                    'carbon_monoxide': 0.2,  # g/s
                    'methane': 0.1,  # g/s
                    'hydrogen': 0.05  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 30.0, 'max': 200.0},
                    'humidity': {'min': 10.0, 'max': 70.0},
                    'pressure': {'min': 970.0, 'max': 1030.0},
                    'voc': {'min': 300.0, 'max': 2000.0}
                }
            },
            'metadata': {
                'description': 'Rapid combustion fire scenario with fast temperature rise',
                'classification': 'fire',
                'fire_type': 'rapid_combustion'
            }
        })
    
    def get_scenario_type(self) -> str:
        """
        Get the type of scenario this generator produces.
        
        Returns:
            Scenario type string
        """
        return 'rapid_combustion'