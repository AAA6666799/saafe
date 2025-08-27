"""
Base interfaces for synthetic data generation components.

This module defines the core interfaces and abstract classes for all data generation
components in the synthetic fire prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime


class DataGenerator(ABC):
    """
    Base abstract class for all data generators in the system.
    
    This class defines the common interface that all data generators must implement,
    regardless of the type of data they generate (thermal, gas, environmental).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data generator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for the generator
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def generate(self, 
                 timestamp: datetime, 
                 duration_seconds: int, 
                 sample_rate_hz: float,
                 seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate synthetic data for the specified duration.
        
        Args:
            timestamp: Start timestamp for the generated data
            duration_seconds: Duration of data to generate in seconds
            sample_rate_hz: Sample rate in Hz
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated data
        """
        pass
    
    @abstractmethod
    def to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert generated data to a pandas DataFrame.
        
        Args:
            data: Generated data from the generate method
            
        Returns:
            DataFrame containing the data in a structured format
        """
        pass
    
    @abstractmethod
    def save(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Save generated data to a file.
        
        Args:
            data: Generated data from the generate method
            filepath: Path to save the data
        """
        pass


class ThermalDataGenerator(DataGenerator):
    """
    Abstract base class for thermal data generators.
    
    This class extends the base DataGenerator with thermal-specific methods.
    """
    
    @abstractmethod
    def generate_frame(self, 
                      timestamp: datetime,
                      hotspots: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        Generate a single thermal image frame.
        
        Args:
            timestamp: Timestamp for the frame
            hotspots: Optional list of hotspot parameters
            
        Returns:
            2D numpy array representing the thermal image
        """
        pass


class GasDataGenerator(DataGenerator):
    """
    Abstract base class for gas data generators.
    
    This class extends the base DataGenerator with gas-specific methods.
    """
    
    @abstractmethod
    def generate_concentration(self, 
                              timestamp: datetime,
                              gas_type: str,
                              baseline: float,
                              anomaly_factor: float = 1.0) -> float:
        """
        Generate a gas concentration reading.
        
        Args:
            timestamp: Timestamp for the reading
            gas_type: Type of gas (e.g., 'methane', 'propane')
            baseline: Baseline concentration in PPM
            anomaly_factor: Factor to multiply baseline for anomalies
            
        Returns:
            Gas concentration in PPM
        """
        pass


class EnvironmentalDataGenerator(DataGenerator):
    """
    Abstract base class for environmental data generators.
    
    This class extends the base DataGenerator with environmental-specific methods.
    """
    
    @abstractmethod
    def generate_environmental_reading(self,
                                      timestamp: datetime,
                                      parameter: str,
                                      baseline: float,
                                      daily_variation: float = 0.1,
                                      noise_level: float = 0.05) -> float:
        """
        Generate an environmental parameter reading.
        
        Args:
            timestamp: Timestamp for the reading
            parameter: Environmental parameter (e.g., 'temperature', 'humidity')
            baseline: Baseline value
            daily_variation: Magnitude of daily variation as fraction of baseline
            noise_level: Magnitude of random noise as fraction of baseline
            
        Returns:
            Environmental parameter reading
        """
        pass


class ScenarioGenerator(ABC):
    """
    Abstract base class for scenario generators.
    
    This class defines the interface for generating complete fire scenarios
    that combine thermal, gas, and environmental data.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalDataGenerator,
                 gas_generator: GasDataGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the scenario generator.
        
        Args:
            thermal_generator: Thermal data generator instance
            gas_generator: Gas data generator instance
            environmental_generator: Environmental data generator instance
            config: Configuration parameters
        """
        self.thermal_generator = thermal_generator
        self.gas_generator = gas_generator
        self.environmental_generator = environmental_generator
        self.config = config
    
    @abstractmethod
    def generate_scenario(self,
                         start_time: datetime,
                         duration_seconds: int,
                         sample_rate_hz: float,
                         scenario_type: str,
                         scenario_params: Dict[str, Any],
                         seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a complete scenario with synchronized data from all sensors.
        
        Args:
            start_time: Start timestamp for the scenario
            duration_seconds: Duration of the scenario in seconds
            sample_rate_hz: Sample rate in Hz
            scenario_type: Type of scenario (e.g., 'normal', 'electrical_fire')
            scenario_params: Parameters specific to the scenario type
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated scenario data
        """
        pass
    
    @abstractmethod
    def save_scenario(self, scenario_data: Dict[str, Any], directory: str) -> None:
        """
        Save a generated scenario to files.
        
        Args:
            scenario_data: Generated scenario data
            directory: Directory to save the scenario files
        """
        pass