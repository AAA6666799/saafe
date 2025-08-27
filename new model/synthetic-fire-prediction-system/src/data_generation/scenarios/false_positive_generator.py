"""
False positive scenario generator for synthetic fire data.

This module provides functionality for generating false positive scenarios
that mimic fire signatures but are not actual fires, such as cooking scenarios,
heating system scenarios, and other non-fire events.
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


class FalsePositiveGenerator(SpecificScenarioGenerator):
    """
    Generator for false positive scenarios.
    
    This class generates scenarios that mimic fire signatures but are not actual fires,
    such as cooking scenarios, heating system scenarios, and other non-fire events.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the false positive generator.
        
        Args:
            thermal_generator: Thermal data generator instance
            gas_generator: Gas data generator instance
            environmental_generator: Environmental data generator instance
            config: Configuration parameters
        """
        super().__init__(thermal_generator, gas_generator, environmental_generator, config)
        
        # Default false positive parameters
        self.default_scenario_params.update({
            'metadata': {
                'description': 'False positive scenario that mimics fire signatures',
                'classification': 'false_positive'
            }
        })
        
        # Initialize specific false positive types
        self.false_positive_types = {
            'cooking': self._get_cooking_params(),
            'heating': self._get_heating_params(),
            'steam': self._get_steam_params(),
            'sunlight': self._get_sunlight_params(),
            'electronic_equipment': self._get_electronic_equipment_params()
        }
    
    def _get_cooking_params(self) -> Dict[str, Any]:
        """
        Get parameters for cooking false positive scenarios.
        
        Returns:
            Dictionary of cooking scenario parameters
        """
        return {
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 0.0,  # No actual fire
                'growth_rate': 0.0,
                'max_size': 0.0
            },
            'thermal_params': {
                'max_temperature': 120.0,  # °C
                'hotspot_count': 1
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide', 'methane'],
                'release_rates': {
                    'carbon_monoxide': 0.005,  # g/s
                    'methane': 0.002  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 22.0, 'max': 30.0},
                    'humidity': {'min': 40.0, 'max': 70.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 100.0, 'max': 500.0}
                }
            },
            'metadata': {
                'description': 'Cooking scenario that mimics fire signatures',
                'classification': 'false_positive',
                'false_positive_type': 'cooking'
            }
        }
    
    def _get_heating_params(self) -> Dict[str, Any]:
        """
        Get parameters for heating system false positive scenarios.
        
        Returns:
            Dictionary of heating system scenario parameters
        """
        return {
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 0.0,  # No actual fire
                'growth_rate': 0.0,
                'max_size': 0.0
            },
            'thermal_params': {
                'max_temperature': 80.0,  # °C
                'hotspot_count': 1
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide'],
                'release_rates': {
                    'carbon_monoxide': 0.001  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 22.0, 'max': 28.0},
                    'humidity': {'min': 30.0, 'max': 50.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 50.0, 'max': 200.0}
                }
            },
            'metadata': {
                'description': 'Heating system scenario that mimics fire signatures',
                'classification': 'false_positive',
                'false_positive_type': 'heating'
            }
        }
    
    def _get_steam_params(self) -> Dict[str, Any]:
        """
        Get parameters for steam false positive scenarios.
        
        Returns:
            Dictionary of steam scenario parameters
        """
        return {
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 0.0,  # No actual fire
                'growth_rate': 0.0,
                'max_size': 0.0
            },
            'thermal_params': {
                'max_temperature': 60.0,  # °C
                'hotspot_count': 1
            },
            'gas_params': {
                'gas_types': [],
                'release_rates': {}
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 22.0, 'max': 30.0},
                    'humidity': {'min': 70.0, 'max': 90.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 0.0, 'max': 100.0}
                }
            },
            'metadata': {
                'description': 'Steam scenario that mimics fire signatures',
                'classification': 'false_positive',
                'false_positive_type': 'steam'
            }
        }
    
    def _get_sunlight_params(self) -> Dict[str, Any]:
        """
        Get parameters for sunlight false positive scenarios.
        
        Returns:
            Dictionary of sunlight scenario parameters
        """
        return {
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 0.0,  # No actual fire
                'growth_rate': 0.0,
                'max_size': 0.0
            },
            'thermal_params': {
                'max_temperature': 50.0,  # °C
                'hotspot_count': 1
            },
            'gas_params': {
                'gas_types': [],
                'release_rates': {}
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 22.0, 'max': 35.0},
                    'humidity': {'min': 30.0, 'max': 60.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 0.0, 'max': 50.0}
                }
            },
            'metadata': {
                'description': 'Sunlight scenario that mimics fire signatures',
                'classification': 'false_positive',
                'false_positive_type': 'sunlight'
            }
        }
    
    def _get_electronic_equipment_params(self) -> Dict[str, Any]:
        """
        Get parameters for electronic equipment false positive scenarios.
        
        Returns:
            Dictionary of electronic equipment scenario parameters
        """
        return {
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 0.0,  # No actual fire
                'growth_rate': 0.0,
                'max_size': 0.0
            },
            'thermal_params': {
                'max_temperature': 70.0,  # °C
                'hotspot_count': 1
            },
            'gas_params': {
                'gas_types': [],
                'release_rates': {}
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 22.0, 'max': 28.0},
                    'humidity': {'min': 30.0, 'max': 60.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 50.0, 'max': 200.0}
                }
            },
            'metadata': {
                'description': 'Electronic equipment scenario that mimics fire signatures',
                'classification': 'false_positive',
                'false_positive_type': 'electronic_equipment'
            }
        }
    
    def get_scenario_type(self) -> str:
        """
        Get the type of scenario this generator produces.
        
        Returns:
            Scenario type string
        """
        return 'false_positive'
    
    def generate_false_positive(self,
                               false_positive_type: str,
                               start_time: datetime,
                               duration_seconds: int,
                               sample_rate_hz: float,
                               scenario_params: Optional[Dict[str, Any]] = None,
                               seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a specific false positive scenario.
        
        Args:
            false_positive_type: Type of false positive scenario
            start_time: Start timestamp for the scenario
            duration_seconds: Duration of the scenario in seconds
            sample_rate_hz: Sample rate in Hz
            scenario_params: Optional parameters to override defaults
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated scenario data
        """
        if false_positive_type not in self.false_positive_types:
            raise ValueError(f"Unknown false positive type: {false_positive_type}")
        
        # Get default parameters for the specified false positive type
        fp_params = copy.deepcopy(self.false_positive_types[false_positive_type])
        
        # Combine with default scenario parameters
        combined_params = copy.deepcopy(self.default_scenario_params)
        self._deep_update(combined_params, fp_params)
        
        # Override with provided parameters if any
        if scenario_params:
            self._deep_update(combined_params, scenario_params)
        
        # Generate the scenario
        scenario_data = self.generate_scenario(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_type='false_positive',
            scenario_params=combined_params,
            seed=seed
        )
        
        # Add false positive type to metadata
        scenario_data['metadata']['false_positive_type'] = false_positive_type
        
        return scenario_data
    
    def generate_cooking_scenario(self,
                                 start_time: datetime,
                                 duration_seconds: int,
                                 sample_rate_hz: float,
                                 cooking_type: str = 'general',
                                 scenario_params: Optional[Dict[str, Any]] = None,
                                 seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a cooking false positive scenario.
        
        Args:
            start_time: Start timestamp for the scenario
            duration_seconds: Duration of the scenario in seconds
            sample_rate_hz: Sample rate in Hz
            cooking_type: Type of cooking (e.g., 'frying', 'baking')
            scenario_params: Optional parameters to override defaults
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated scenario data
        """
        # Get base cooking parameters
        cooking_params = copy.deepcopy(self.false_positive_types['cooking'])
        
        # Adjust parameters based on cooking type
        if cooking_type == 'frying':
            cooking_params['thermal_params']['max_temperature'] = 150.0
            cooking_params['gas_params']['release_rates']['carbon_monoxide'] = 0.008
            cooking_params['environmental_params']['parameter_ranges']['voc']['max'] = 800.0
        elif cooking_type == 'baking':
            cooking_params['thermal_params']['max_temperature'] = 100.0
            cooking_params['environmental_params']['parameter_ranges']['temperature']['max'] = 35.0
        elif cooking_type == 'boiling':
            cooking_params['thermal_params']['max_temperature'] = 110.0
            cooking_params['environmental_params']['parameter_ranges']['humidity']['max'] = 85.0
        
        # Update metadata
        cooking_params['metadata']['description'] = f'{cooking_type.capitalize()} cooking scenario that mimics fire signatures'
        cooking_params['metadata']['cooking_type'] = cooking_type
        
        # Combine with provided parameters if any
        if scenario_params:
            self._deep_update(cooking_params, scenario_params)
        
        # Generate the false positive scenario
        return self.generate_false_positive(
            false_positive_type='cooking',
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_params=cooking_params,
            seed=seed
        )
    
    def generate_heating_scenario(self,
                                 start_time: datetime,
                                 duration_seconds: int,
                                 sample_rate_hz: float,
                                 heating_type: str = 'general',
                                 scenario_params: Optional[Dict[str, Any]] = None,
                                 seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a heating system false positive scenario.
        
        Args:
            start_time: Start timestamp for the scenario
            duration_seconds: Duration of the scenario in seconds
            sample_rate_hz: Sample rate in Hz
            heating_type: Type of heating system (e.g., 'furnace', 'radiator')
            scenario_params: Optional parameters to override defaults
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated scenario data
        """
        # Get base heating parameters
        heating_params = copy.deepcopy(self.false_positive_types['heating'])
        
        # Adjust parameters based on heating type
        if heating_type == 'furnace':
            heating_params['thermal_params']['max_temperature'] = 90.0
            heating_params['gas_params']['release_rates']['carbon_monoxide'] = 0.002
        elif heating_type == 'radiator':
            heating_params['thermal_params']['max_temperature'] = 70.0
            heating_params['environmental_params']['parameter_ranges']['humidity']['min'] = 20.0
        elif heating_type == 'space_heater':
            heating_params['thermal_params']['max_temperature'] = 85.0
            heating_params['environmental_params']['parameter_ranges']['temperature']['max'] = 32.0
        
        # Update metadata
        heating_params['metadata']['description'] = f'{heating_type.capitalize()} heating system scenario that mimics fire signatures'
        heating_params['metadata']['heating_type'] = heating_type
        
        # Combine with provided parameters if any
        if scenario_params:
            self._deep_update(heating_params, scenario_params)
        
        # Generate the false positive scenario
        return self.generate_false_positive(
            false_positive_type='heating',
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_params=heating_params,
            seed=seed
        )
    
    def generate_random_false_positive(self,
                                      start_time: datetime,
                                      duration_seconds: int,
                                      sample_rate_hz: float,
                                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a random false positive scenario.
        
        Args:
            start_time: Start timestamp for the scenario
            duration_seconds: Duration of the scenario in seconds
            sample_rate_hz: Sample rate in Hz
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated scenario data
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Select a random false positive type
        fp_type = np.random.choice(list(self.false_positive_types.keys()))
        
        # Generate random variations
        variations = {}
        if fp_type == 'cooking':
            cooking_types = ['general', 'frying', 'baking', 'boiling']
            cooking_type = np.random.choice(cooking_types)
            return self.generate_cooking_scenario(
                start_time=start_time,
                duration_seconds=duration_seconds,
                sample_rate_hz=sample_rate_hz,
                cooking_type=cooking_type,
                seed=seed
            )
        elif fp_type == 'heating':
            heating_types = ['general', 'furnace', 'radiator', 'space_heater']
            heating_type = np.random.choice(heating_types)
            return self.generate_heating_scenario(
                start_time=start_time,
                duration_seconds=duration_seconds,
                sample_rate_hz=sample_rate_hz,
                heating_type=heating_type,
                seed=seed
            )
        else:
            # For other types, add some random variations
            base_params = copy.deepcopy(self.false_positive_types[fp_type])
            
            # Add random variations to thermal parameters
            if 'thermal_params' in base_params:
                max_temp = base_params['thermal_params']['max_temperature']
                base_params['thermal_params']['max_temperature'] = max_temp * np.random.uniform(0.8, 1.2)
            
            # Add random variations to environmental parameters
            if 'environmental_params' in base_params and 'parameter_ranges' in base_params['environmental_params']:
                for param, ranges in base_params['environmental_params']['parameter_ranges'].items():
                    if 'min' in ranges and 'max' in ranges:
                        min_val = ranges['min']
                        max_val = ranges['max']
                        ranges['min'] = min_val * np.random.uniform(0.9, 1.1)
                        ranges['max'] = max_val * np.random.uniform(0.9, 1.1)
            
            return self.generate_false_positive(
                false_positive_type=fp_type,
                start_time=start_time,
                duration_seconds=duration_seconds,
                sample_rate_hz=sample_rate_hz,
                scenario_params=base_params,
                seed=seed
            )