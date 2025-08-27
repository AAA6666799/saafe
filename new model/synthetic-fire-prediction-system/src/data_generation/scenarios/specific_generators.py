"""
Specific scenario generators for different fire types.

This module provides implementations of scenario generators for specific
fire types, including normal (non-fire), electrical fire, chemical fire,
smoldering fire, and rapid combustion scenarios.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
import copy

from .scenario_generator import ScenarioGenerator
from ..thermal.thermal_image_generator import ThermalImageGenerator
from ..gas.gas_concentration_generator import GasConcentrationGenerator
from ..environmental.environmental_data_generator import EnvironmentalDataGenerator


class SpecificScenarioGenerator(ScenarioGenerator):
    """
    Base class for specific scenario generators.
    
    This class extends the ScenarioGenerator with functionality specific to
    particular fire types.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the specific scenario generator.
        
        Args:
            thermal_generator: Thermal data generator instance
            gas_generator: Gas data generator instance
            environmental_generator: Environmental data generator instance
            config: Configuration parameters
        """
        super().__init__(thermal_generator, gas_generator, environmental_generator, config)
        
        # Default scenario parameters
        self.default_scenario_params = {
            'room_params': {
                'room_volume': 50.0,  # m³
                'ventilation_rate': 0.5,  # air changes per hour
                'initial_temperature': 20.0,  # °C
                'initial_humidity': 50.0,  # %
                'fuel_load': 10.0  # kg
            }
        }
    
    def generate_specific_scenario(self,
                                  start_time: datetime,
                                  duration_seconds: int,
                                  sample_rate_hz: float,
                                  scenario_params: Optional[Dict[str, Any]] = None,
                                  seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a specific scenario type.
        
        Args:
            start_time: Start timestamp for the scenario
            duration_seconds: Duration of the scenario in seconds
            sample_rate_hz: Sample rate in Hz
            scenario_params: Optional parameters to override defaults
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated scenario data
        """
        # Combine default parameters with provided parameters
        combined_params = copy.deepcopy(self.default_scenario_params)
        if scenario_params:
            self._deep_update(combined_params, scenario_params)
        
        # Generate the scenario using the base class method
        return self.generate_scenario(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_type=self.get_scenario_type(),
            scenario_params=combined_params,
            seed=seed
        )
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively update a dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def get_scenario_type(self) -> str:
        """
        Get the type of scenario this generator produces.
        
        Returns:
            Scenario type string
        """
        raise NotImplementedError("Subclasses must implement get_scenario_type()")


class NormalScenarioGenerator(SpecificScenarioGenerator):
    """
    Generator for normal (non-fire) scenarios.
    
    This class generates scenarios with normal environmental conditions
    without any fire or anomalies.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the normal scenario generator.
        
        Args:
            thermal_generator: Thermal data generator instance
            gas_generator: Gas data generator instance
            environmental_generator: Environmental data generator instance
            config: Configuration parameters
        """
        super().__init__(thermal_generator, gas_generator, environmental_generator, config)
        
        # Set default parameters for normal scenarios
        self.default_scenario_params.update({
            'thermal_params': {
                'max_temperature': 30.0,  # °C
                'hotspot_count': 0
            },
            'gas_params': {
                'gas_types': ['methane', 'carbon_monoxide'],
                'release_rates': {
                    'methane': 0.0,
                    'carbon_monoxide': 0.0
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 18.0, 'max': 25.0},
                    'humidity': {'min': 40.0, 'max': 60.0},
                    'pressure': {'min': 990.0, 'max': 1020.0},
                    'voc': {'min': 0.0, 'max': 100.0}
                }
            },
            'metadata': {
                'description': 'Normal scenario with no fire or anomalies',
                'classification': 'normal'
            }
        })
    
    def get_scenario_type(self) -> str:
        """
        Get the type of scenario this generator produces.
        
        Returns:
            Scenario type string
        """
        return 'normal'


class ElectricalFireScenarioGenerator(SpecificScenarioGenerator):
    """
    Generator for electrical fire scenarios.
    
    This class generates scenarios with electrical fire characteristics,
    including localized hotspots and specific gas emissions.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the electrical fire scenario generator.
        
        Args:
            thermal_generator: Thermal data generator instance
            gas_generator: Gas data generator instance
            environmental_generator: Environmental data generator instance
            config: Configuration parameters
        """
        super().__init__(thermal_generator, gas_generator, environmental_generator, config)
        
        # Set default parameters for electrical fire scenarios
        self.default_scenario_params.update({
            'fire_params': {
                'fire_location': [192, 144],  # Center of the image
                'fire_size': 5.0,  # kW
                'growth_rate': 0.05,  # kW/s
                'max_size': 50.0  # kW
            },
            'thermal_params': {
                'max_temperature': 200.0,  # °C
                'hotspot_count': 1
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide', 'hydrogen', 'methane'],
                'release_rates': {
                    'carbon_monoxide': 0.01,  # g/s
                    'hydrogen': 0.005,  # g/s
                    'methane': 0.002  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 20.0, 'max': 35.0},
                    'humidity': {'min': 30.0, 'max': 50.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 100.0, 'max': 500.0}
                }
            },
            'metadata': {
                'description': 'Electrical fire scenario with localized hotspots and specific gas emissions',
                'classification': 'fire',
                'fire_type': 'electrical'
            }
        })
    
    def get_scenario_type(self) -> str:
        """
        Get the type of scenario this generator produces.
        
        Returns:
            Scenario type string
        """
        return 'electrical_fire'


class ChemicalFireScenarioGenerator(SpecificScenarioGenerator):
    """
    Generator for chemical fire scenarios.
    
    This class generates scenarios with chemical fire characteristics,
    including high temperatures and specific chemical gas emissions.
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
                'fire_size': 10.0,  # kW
                'growth_rate': 0.2,  # kW/s
                'max_size': 100.0  # kW
            },
            'thermal_params': {
                'max_temperature': 300.0,  # °C
                'hotspot_count': 2
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide', 'hydrogen', 'methane', 'propane'],
                'release_rates': {
                    'carbon_monoxide': 0.05,  # g/s
                    'hydrogen': 0.02,  # g/s
                    'methane': 0.01,  # g/s
                    'propane': 0.03  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 20.0, 'max': 50.0},
                    'humidity': {'min': 20.0, 'max': 40.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 500.0, 'max': 2000.0}
                }
            },
            'metadata': {
                'description': 'Chemical fire scenario with high temperatures and specific chemical gas emissions',
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
                'fire_size': 2.0,  # kW
                'growth_rate': 0.01,  # kW/s
                'max_size': 20.0  # kW
            },
            'thermal_params': {
                'max_temperature': 150.0,  # °C
                'hotspot_count': 1
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide', 'methane'],
                'release_rates': {
                    'carbon_monoxide': 0.03,  # g/s
                    'methane': 0.005  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 20.0, 'max': 30.0},
                    'humidity': {'min': 30.0, 'max': 50.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 200.0, 'max': 800.0}
                }
            },
            'metadata': {
                'description': 'Smoldering fire scenario with slow temperature rise and specific gas emissions',
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
    including fast temperature rise and intense gas emissions.
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
                'fire_size': 20.0,  # kW
                'growth_rate': 0.5,  # kW/s
                'max_size': 200.0  # kW
            },
            'thermal_params': {
                'max_temperature': 500.0,  # °C
                'hotspot_count': 3
            },
            'gas_params': {
                'gas_types': ['carbon_monoxide', 'hydrogen', 'methane', 'propane'],
                'release_rates': {
                    'carbon_monoxide': 0.1,  # g/s
                    'hydrogen': 0.05,  # g/s
                    'methane': 0.03,  # g/s
                    'propane': 0.08  # g/s
                }
            },
            'environmental_params': {
                'parameters': ['temperature', 'humidity', 'pressure', 'voc'],
                'parameter_ranges': {
                    'temperature': {'min': 20.0, 'max': 80.0},
                    'humidity': {'min': 10.0, 'max': 30.0},
                    'pressure': {'min': 990.0, 'max': 1010.0},
                    'voc': {'min': 1000.0, 'max': 2000.0}
                }
            },
            'metadata': {
                'description': 'Rapid combustion scenario with fast temperature rise and intense gas emissions',
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