"""
Gas sensor response model for synthetic fire data.

This module provides functionality for simulating realistic gas sensor characteristics
including noise, drift, response time, and cross-sensitivity between gases.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import math


class SensorType(Enum):
    """Enumeration of supported gas sensor types."""
    ELECTROCHEMICAL = "electrochemical"
    CATALYTIC = "catalytic"
    INFRARED = "infrared"
    SEMICONDUCTOR = "semiconductor"
    PHOTOIONIZATION = "photoionization"


class SensorResponseModel:
    """
    Class for simulating realistic gas sensor characteristics.
    
    This class provides methods for modeling sensor noise, drift, response time,
    and cross-sensitivity between gases for different sensor types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sensor response model with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
                - sensor_type: Type of gas sensor
                - noise_level: Standard deviation of sensor noise as fraction of reading
                - drift_rate: Sensor drift rate in % per hour
                - response_time: T90 response time in seconds
                - recovery_time: T10 recovery time in seconds
                - cross_sensitivity: Dictionary of cross-sensitivity factors
                - calibration_error: Systematic error in calibration as % of reading
                - resolution: Sensor resolution in ppm
        """
        self.config = config
        self.validate_config()
        
        # Set default values
        self.sensor_type = SensorType(self.config.get('sensor_type', 'electrochemical'))
        self.noise_level = self.config.get('noise_level', 0.02)  # 2% of reading
        self.drift_rate = self.config.get('drift_rate', 0.5)  # 0.5% per hour
        self.response_time = self.config.get('response_time', 30.0)  # 30 seconds T90
        self.recovery_time = self.config.get('recovery_time', 60.0)  # 60 seconds T10
        self.calibration_error = self.config.get('calibration_error', 0.0)  # % of reading
        self.resolution = self.config.get('resolution', 1.0)  # 1 ppm
        
        # Cross-sensitivity between gases (default values based on sensor type)
        self.cross_sensitivity = self.config.get('cross_sensitivity', self._default_cross_sensitivity())
        
        # Initialize sensor state
        self.last_reading = 0.0
        self.last_timestamp = None
        self.drift_offset = 0.0
        self.drift_start_time = datetime.now()
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if 'sensor_type' in self.config:
            try:
                SensorType(self.config['sensor_type'])
            except ValueError:
                valid_types = [st.value for st in SensorType]
                raise ValueError(f"Invalid sensor type. Must be one of: {valid_types}")
        
        if 'noise_level' in self.config and self.config['noise_level'] < 0:
            raise ValueError("noise_level must be non-negative")
        
        if 'drift_rate' in self.config and self.config['drift_rate'] < 0:
            raise ValueError("drift_rate must be non-negative")
        
        if 'response_time' in self.config and self.config['response_time'] <= 0:
            raise ValueError("response_time must be positive")
        
        if 'recovery_time' in self.config and self.config['recovery_time'] <= 0:
            raise ValueError("recovery_time must be positive")
        
        if 'resolution' in self.config and self.config['resolution'] <= 0:
            raise ValueError("resolution must be positive")
    
    def _default_cross_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """
        Get default cross-sensitivity factors based on sensor type.
        
        Returns:
            Dictionary mapping target gas to dictionary of interferent gases and factors
        """
        if self.sensor_type == SensorType.ELECTROCHEMICAL:
            return {
                'methane': {
                    'hydrogen': 0.05,
                    'carbon_monoxide': 0.02,
                    'propane': 0.01
                },
                'hydrogen': {
                    'methane': 0.01,
                    'carbon_monoxide': 0.2,
                    'propane': 0.01
                },
                'propane': {
                    'methane': 0.3,
                    'hydrogen': 0.05,
                    'carbon_monoxide': 0.01
                },
                'carbon_monoxide': {
                    'hydrogen': 0.4,
                    'methane': 0.01,
                    'propane': 0.01
                }
            }
        elif self.sensor_type == SensorType.CATALYTIC:
            return {
                'methane': {
                    'hydrogen': 0.7,
                    'carbon_monoxide': 0.1,
                    'propane': 0.6
                },
                'hydrogen': {
                    'methane': 0.3,
                    'carbon_monoxide': 0.1,
                    'propane': 0.2
                },
                'propane': {
                    'methane': 0.5,
                    'hydrogen': 0.3,
                    'carbon_monoxide': 0.1
                },
                'carbon_monoxide': {
                    'hydrogen': 0.2,
                    'methane': 0.1,
                    'propane': 0.1
                }
            }
        elif self.sensor_type == SensorType.INFRARED:
            return {
                'methane': {
                    'propane': 0.1,
                    'carbon_dioxide': 0.05,
                    'water_vapor': 0.02
                },
                'propane': {
                    'methane': 0.1,
                    'carbon_dioxide': 0.05,
                    'water_vapor': 0.02
                },
                'carbon_dioxide': {
                    'methane': 0.01,
                    'propane': 0.01,
                    'water_vapor': 0.1
                }
            }
        elif self.sensor_type == SensorType.SEMICONDUCTOR:
            return {
                'methane': {
                    'hydrogen': 0.8,
                    'carbon_monoxide': 0.4,
                    'propane': 0.7,
                    'ethanol': 0.6
                },
                'hydrogen': {
                    'methane': 0.5,
                    'carbon_monoxide': 0.3,
                    'propane': 0.4,
                    'ethanol': 0.5
                },
                'propane': {
                    'methane': 0.6,
                    'hydrogen': 0.5,
                    'carbon_monoxide': 0.3,
                    'ethanol': 0.5
                },
                'carbon_monoxide': {
                    'hydrogen': 0.6,
                    'methane': 0.3,
                    'propane': 0.3,
                    'ethanol': 0.7
                }
            }
        elif self.sensor_type == SensorType.PHOTOIONIZATION:
            return {
                'benzene': {
                    'toluene': 0.5,
                    'xylene': 0.4,
                    'isobutylene': 0.8
                },
                'toluene': {
                    'benzene': 0.6,
                    'xylene': 0.7,
                    'isobutylene': 0.5
                },
                'xylene': {
                    'benzene': 0.4,
                    'toluene': 0.6,
                    'isobutylene': 0.4
                }
            }
        else:
            return {}
    
    def simulate_response(self, 
                         true_concentration: float, 
                         gas_type: str, 
                         timestamp: datetime,
                         other_gases: Optional[Dict[str, float]] = None) -> float:
        """
        Simulate sensor response to a given gas concentration.
        
        Args:
            true_concentration: True gas concentration in ppm
            gas_type: Type of gas being measured
            timestamp: Current timestamp
            other_gases: Optional dictionary of other gas concentrations for cross-sensitivity
            
        Returns:
            Simulated sensor reading in ppm
        """
        # Initialize last_timestamp if this is the first reading
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.drift_start_time = timestamp
        
        # Calculate time since last reading
        time_delta = (timestamp - self.last_timestamp).total_seconds()
        
        # Apply response dynamics (first-order system)
        if time_delta > 0:
            # Calculate response rate based on whether concentration is increasing or decreasing
            if true_concentration > self.last_reading:
                # Rising concentration - use response time
                tau = self.response_time / 2.3  # T90 corresponds to 2.3 time constants
            else:
                # Falling concentration - use recovery time
                tau = self.recovery_time / 2.3  # T10 corresponds to 2.3 time constants
            
            # First-order response equation
            response_factor = 1 - np.exp(-time_delta / tau)
            
            # Update reading based on response dynamics
            reading = self.last_reading + (true_concentration - self.last_reading) * response_factor
        else:
            reading = self.last_reading
        
        # Apply cross-sensitivity effects
        if other_gases and gas_type in self.cross_sensitivity:
            cross_effect = 0.0
            for other_gas, concentration in other_gases.items():
                if other_gas in self.cross_sensitivity[gas_type]:
                    sensitivity = self.cross_sensitivity[gas_type][other_gas]
                    cross_effect += concentration * sensitivity
            
            reading += cross_effect
        
        # Apply calibration error (systematic bias)
        reading *= (1.0 + self.calibration_error / 100.0)
        
        # Apply sensor drift
        hours_since_start = (timestamp - self.drift_start_time).total_seconds() / 3600.0
        self.drift_offset = hours_since_start * (self.drift_rate / 100.0) * true_concentration
        reading += self.drift_offset
        
        # Apply sensor noise (random)
        noise = np.random.normal(0, self.noise_level * abs(reading))
        reading += noise
        
        # Apply sensor resolution (quantization)
        if self.resolution > 0:
            reading = round(reading / self.resolution) * self.resolution
        
        # Ensure non-negative reading
        reading = max(0.0, reading)
        
        # Update state
        self.last_reading = reading
        self.last_timestamp = timestamp
        
        return reading
    
    def simulate_batch_response(self, 
                              true_concentrations: List[float], 
                              gas_type: str,
                              timestamps: List[datetime],
                              other_gases: Optional[List[Dict[str, float]]] = None) -> List[float]:
        """
        Simulate sensor response to a batch of gas concentration readings.
        
        Args:
            true_concentrations: List of true gas concentrations in ppm
            gas_type: Type of gas being measured
            timestamps: List of timestamps for each concentration
            other_gases: Optional list of dictionaries of other gas concentrations
            
        Returns:
            List of simulated sensor readings in ppm
        """
        readings = []
        
        # Reset sensor state
        self.last_reading = 0.0
        self.last_timestamp = None
        self.drift_offset = 0.0
        self.drift_start_time = timestamps[0] if timestamps else datetime.now()
        
        for i, conc in enumerate(true_concentrations):
            timestamp = timestamps[i]
            other = other_gases[i] if other_gases else None
            
            reading = self.simulate_response(conc, gas_type, timestamp, other)
            readings.append(reading)
        
        return readings
    
    def simulate_temperature_effect(self, 
                                  reading: float, 
                                  temperature: float, 
                                  reference_temp: float = 25.0) -> float:
        """
        Simulate the effect of temperature on sensor readings.
        
        Args:
            reading: Current sensor reading in ppm
            temperature: Current temperature in Celsius
            reference_temp: Reference temperature in Celsius (calibration temperature)
            
        Returns:
            Adjusted sensor reading in ppm
        """
        # Temperature coefficient depends on sensor type
        if self.sensor_type == SensorType.ELECTROCHEMICAL:
            temp_coeff = 0.02  # 2% per 10°C
        elif self.sensor_type == SensorType.CATALYTIC:
            temp_coeff = 0.05  # 5% per 10°C
        elif self.sensor_type == SensorType.INFRARED:
            temp_coeff = 0.01  # 1% per 10°C
        elif self.sensor_type == SensorType.SEMICONDUCTOR:
            temp_coeff = 0.07  # 7% per 10°C
        elif self.sensor_type == SensorType.PHOTOIONIZATION:
            temp_coeff = 0.03  # 3% per 10°C
        else:
            temp_coeff = 0.02  # Default
        
        # Calculate temperature difference
        temp_diff = temperature - reference_temp
        
        # Calculate temperature effect (non-linear for some sensors)
        if self.sensor_type == SensorType.SEMICONDUCTOR:
            # Semiconductor sensors have exponential temperature dependence
            temp_factor = np.exp(temp_coeff * temp_diff / 10.0) - 1.0
        else:
            # Linear temperature dependence for other sensors
            temp_factor = temp_coeff * temp_diff / 10.0
        
        # Apply temperature effect
        adjusted_reading = reading * (1.0 + temp_factor)
        
        return adjusted_reading
    
    def simulate_humidity_effect(self, 
                               reading: float, 
                               humidity: float, 
                               reference_humidity: float = 50.0) -> float:
        """
        Simulate the effect of humidity on sensor readings.
        
        Args:
            reading: Current sensor reading in ppm
            humidity: Current relative humidity in %
            reference_humidity: Reference humidity in % (calibration humidity)
            
        Returns:
            Adjusted sensor reading in ppm
        """
        # Humidity coefficient depends on sensor type
        if self.sensor_type == SensorType.ELECTROCHEMICAL:
            humidity_coeff = 0.01  # 1% per 10% RH
        elif self.sensor_type == SensorType.CATALYTIC:
            humidity_coeff = 0.005  # 0.5% per 10% RH
        elif self.sensor_type == SensorType.INFRARED:
            humidity_coeff = 0.002  # 0.2% per 10% RH
        elif self.sensor_type == SensorType.SEMICONDUCTOR:
            humidity_coeff = 0.03  # 3% per 10% RH
        elif self.sensor_type == SensorType.PHOTOIONIZATION:
            humidity_coeff = 0.01  # 1% per 10% RH
        else:
            humidity_coeff = 0.01  # Default
        
        # Calculate humidity difference
        humidity_diff = humidity - reference_humidity
        
        # Calculate humidity effect
        humidity_factor = humidity_coeff * humidity_diff / 10.0
        
        # Apply humidity effect
        adjusted_reading = reading * (1.0 + humidity_factor)
        
        return adjusted_reading
    
    def simulate_aging_effect(self, 
                            reading: float, 
                            age_hours: float) -> float:
        """
        Simulate the effect of sensor aging on readings.
        
        Args:
            reading: Current sensor reading in ppm
            age_hours: Sensor age in hours
            
        Returns:
            Adjusted sensor reading in ppm
        """
        # Aging coefficient depends on sensor type
        if self.sensor_type == SensorType.ELECTROCHEMICAL:
            aging_coeff = 0.0001  # 0.01% per hour
        elif self.sensor_type == SensorType.CATALYTIC:
            aging_coeff = 0.0002  # 0.02% per hour
        elif self.sensor_type == SensorType.INFRARED:
            aging_coeff = 0.00005  # 0.005% per hour
        elif self.sensor_type == SensorType.SEMICONDUCTOR:
            aging_coeff = 0.0003  # 0.03% per hour
        elif self.sensor_type == SensorType.PHOTOIONIZATION:
            aging_coeff = 0.0002  # 0.02% per hour
        else:
            aging_coeff = 0.0001  # Default
        
        # Calculate aging effect (typically reduces sensitivity)
        aging_factor = -aging_coeff * age_hours
        
        # Apply aging effect
        adjusted_reading = reading * (1.0 + aging_factor)
        
        # Ensure non-negative reading
        adjusted_reading = max(0.0, adjusted_reading)
        
        return adjusted_reading
    
    def simulate_poisoning_effect(self, 
                                reading: float, 
                                poison_exposure: Dict[str, float]) -> float:
        """
        Simulate the effect of sensor poisoning on readings.
        
        Args:
            reading: Current sensor reading in ppm
            poison_exposure: Dictionary mapping poison gas to cumulative exposure in ppm-hours
            
        Returns:
            Adjusted sensor reading in ppm
        """
        # Poisoning effect depends on sensor type and poison gas
        poisoning_factor = 0.0
        
        if self.sensor_type == SensorType.ELECTROCHEMICAL:
            # Electrochemical sensors are sensitive to H2S, SO2, etc.
            if 'hydrogen_sulfide' in poison_exposure:
                poisoning_factor += 0.0001 * poison_exposure['hydrogen_sulfide']
            if 'sulfur_dioxide' in poison_exposure:
                poisoning_factor += 0.00005 * poison_exposure['sulfur_dioxide']
        
        elif self.sensor_type == SensorType.CATALYTIC:
            # Catalytic sensors are poisoned by silicones, lead compounds, etc.
            if 'silicone' in poison_exposure:
                poisoning_factor += 0.0005 * poison_exposure['silicone']
            if 'lead_compound' in poison_exposure:
                poisoning_factor += 0.001 * poison_exposure['lead_compound']
        
        elif self.sensor_type == SensorType.SEMICONDUCTOR:
            # Semiconductor sensors are affected by many compounds
            if 'hydrogen_sulfide' in poison_exposure:
                poisoning_factor += 0.0002 * poison_exposure['hydrogen_sulfide']
            if 'silicone' in poison_exposure:
                poisoning_factor += 0.0001 * poison_exposure['silicone']
        
        # Apply poisoning effect (typically reduces sensitivity)
        adjusted_reading = reading * (1.0 - poisoning_factor)
        
        # Ensure non-negative reading
        adjusted_reading = max(0.0, adjusted_reading)
        
        return adjusted_reading
    
    def reset_state(self) -> None:
        """
        Reset the sensor state.
        """
        self.last_reading = 0.0
        self.last_timestamp = None
        self.drift_offset = 0.0
        self.drift_start_time = datetime.now()
    
    def get_detection_limit(self, gas_type: str) -> float:
        """
        Get the lower detection limit for a specific gas.
        
        Args:
            gas_type: Type of gas
            
        Returns:
            Lower detection limit in ppm
        """
        # Detection limits depend on sensor type and gas
        if self.sensor_type == SensorType.ELECTROCHEMICAL:
            limits = {
                'hydrogen': 0.5,
                'carbon_monoxide': 0.1,
                'methane': 500,  # Not ideal for methane
                'propane': 500   # Not ideal for propane
            }
        elif self.sensor_type == SensorType.CATALYTIC:
            limits = {
                'methane': 500,
                'propane': 100,
                'hydrogen': 100,
                'carbon_monoxide': 50
            }
        elif self.sensor_type == SensorType.INFRARED:
            limits = {
                'methane': 50,
                'propane': 20,
                'carbon_dioxide': 10,
                'hydrogen': 10000  # Not detectable by IR
            }
        elif self.sensor_type == SensorType.SEMICONDUCTOR:
            limits = {
                'methane': 100,
                'propane': 50,
                'hydrogen': 5,
                'carbon_monoxide': 1
            }
        elif self.sensor_type == SensorType.PHOTOIONIZATION:
            limits = {
                'benzene': 0.001,
                'toluene': 0.01,
                'xylene': 0.01,
                'methane': 10000  # Not detectable by PID
            }
        else:
            limits = {}
        
        return limits.get(gas_type, self.resolution)