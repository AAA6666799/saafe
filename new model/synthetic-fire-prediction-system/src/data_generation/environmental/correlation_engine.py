"""
Correlation engine for synthetic environmental data.

This module provides functionality for modeling inter-parameter correlations
between environmental parameters to ensure physical consistency.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime


class CorrelationEngine:
    """
    Class for modeling inter-parameter correlations in environmental data.
    
    This class provides methods for ensuring physical consistency between
    environmental parameters based on known physical relationships.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the correlation engine with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.validate_config()
        
        # Define correlation models between parameters
        self.correlation_models = {
            'temperature_humidity': self._temperature_humidity_model,
            'temperature_pressure': self._temperature_pressure_model,
            'humidity_pressure': self._humidity_pressure_model,
            'temperature_voc': self._temperature_voc_model
        }
        
        # Define parameter dependencies
        self.parameter_dependencies = {
            'temperature': ['humidity', 'pressure', 'voc'],
            'humidity': ['temperature', 'pressure'],
            'pressure': ['temperature', 'humidity'],
            'voc': ['temperature']
        }
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Set default correlation strengths if not provided
        if 'correlation_strengths' not in self.config:
            self.config['correlation_strengths'] = {
                'temperature_humidity': 0.7,  # Strong negative correlation
                'temperature_pressure': 0.3,  # Moderate correlation
                'humidity_pressure': 0.4,     # Moderate correlation
                'temperature_voc': 0.6        # Strong positive correlation
            }
        
        # Set default scenario type if not provided
        if 'scenario_type' not in self.config:
            self.config['scenario_type'] = 'indoor'  # Default to indoor scenario
    
    def apply_correlations(self, 
                         env_data: Dict[str, Dict[str, Any]], 
                         timestamps: List[datetime]) -> Dict[str, Dict[str, Any]]:
        """
        Apply correlations to environmental data.
        
        Args:
            env_data: Dictionary mapping parameter names to data dictionaries
            timestamps: List of timestamps
            
        Returns:
            Updated environmental data with correlations applied
        """
        # Create a copy of the input data to avoid modifying the original
        correlated_data = {param: data.copy() for param, data in env_data.items()}
        
        # Get scenario type
        scenario_type = self.config.get('scenario_type', 'indoor')
        
        # Apply correlations for each parameter
        for param in correlated_data:
            if param in self.parameter_dependencies:
                for dependent_param in self.parameter_dependencies[param]:
                    # Check if both parameters exist in the data
                    if param in correlated_data and dependent_param in correlated_data:
                        # Get correlation key (alphabetical order)
                        corr_key = f"{param}_{dependent_param}" if param < dependent_param else f"{dependent_param}_{param}"
                        
                        # Check if we have a correlation model for these parameters
                        if corr_key in self.correlation_models:
                            # Apply correlation model
                            correlated_data = self.correlation_models[corr_key](
                                correlated_data, 
                                param, 
                                dependent_param, 
                                timestamps, 
                                scenario_type
                            )
        
        return correlated_data
    
    def _temperature_humidity_model(self, 
                                  data: Dict[str, Dict[str, Any]], 
                                  param1: str, 
                                  param2: str, 
                                  timestamps: List[datetime],
                                  scenario_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Model the correlation between temperature and humidity.
        
        Args:
            data: Environmental data dictionary
            param1: First parameter name
            param2: Second parameter name
            timestamps: List of timestamps
            scenario_type: Type of scenario (indoor, outdoor)
            
        Returns:
            Updated environmental data
        """
        # Ensure param1 is 'temperature' and param2 is 'humidity'
        if param1 == 'humidity':
            param1, param2 = param2, param1
        
        # Get correlation strength
        strength = self.config['correlation_strengths'].get('temperature_humidity', 0.7)
        
        # Get temperature and humidity values
        temp_values = data[param1]['values']
        humid_values = data[param2]['values']
        
        # Get parameter ranges
        temp_min = data[param1].get('min', 15.0)
        temp_max = data[param1].get('max', 35.0)
        humid_min = data[param2].get('min', 20.0)
        humid_max = data[param2].get('max', 80.0)
        
        # Calculate temperature normalized to 0-1 range
        # Add safety check for division by zero
        temp_range = temp_max - temp_min
        if temp_range <= 0:
            # If temperature range is zero or negative, use default values
            temp_norm = [0.5 for _ in temp_values]  # Default to middle value
        else:
            temp_norm = [(t - temp_min) / temp_range for t in temp_values]
        
        # Apply negative correlation: as temperature increases, humidity decreases
        for i in range(len(humid_values)):
            # Calculate base humidity based on temperature (inverse relationship)
            base_humid = humid_max - (humid_max - humid_min) * temp_norm[i]
            
            # Apply correlation with strength factor
            # Keep some of the original humidity value to maintain variation
            humid_values[i] = (1 - strength) * humid_values[i] + strength * base_humid
            
            # Apply scenario-specific adjustments
            if scenario_type == 'indoor':
                # Indoor environments have more stable humidity
                humid_values[i] = 0.8 * humid_values[i] + 0.2 * (humid_min + humid_max) / 2
            elif scenario_type == 'outdoor':
                # Outdoor environments have more variable humidity
                # Add some time-of-day variation
                hour = timestamps[i].hour
                # Higher humidity in early morning and evening
                if hour < 6 or hour > 18:
                    humid_values[i] *= 1.1
                # Lower humidity in middle of day
                elif 10 <= hour <= 16:
                    humid_values[i] *= 0.9
        
        # Ensure values are within range
        humid_values = [max(humid_min, min(h, humid_max)) for h in humid_values]
        
        # Update humidity values
        data[param2]['values'] = humid_values
        
        return data
    
    def _temperature_pressure_model(self, 
                                  data: Dict[str, Dict[str, Any]], 
                                  param1: str, 
                                  param2: str, 
                                  timestamps: List[datetime],
                                  scenario_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Model the correlation between temperature and pressure.
        
        Args:
            data: Environmental data dictionary
            param1: First parameter name
            param2: Second parameter name
            timestamps: List of timestamps
            scenario_type: Type of scenario (indoor, outdoor)
            
        Returns:
            Updated environmental data
        """
        # Ensure param1 is 'temperature' and param2 is 'pressure'
        if param1 == 'pressure':
            param1, param2 = param2, param1
        
        # Get correlation strength
        strength = self.config['correlation_strengths'].get('temperature_pressure', 0.3)
        
        # Get temperature and pressure values
        temp_values = data[param1]['values']
        press_values = data[param2]['values']
        
        # Get parameter ranges
        temp_min = data[param1].get('min', 15.0)
        temp_max = data[param1].get('max', 35.0)
        press_min = data[param2].get('min', 990.0)
        press_max = data[param2].get('max', 1030.0)
        
        # Calculate temperature normalized to 0-1 range
        # Add safety check for division by zero
        temp_range = temp_max - temp_min
        if temp_range <= 0:
            # If temperature range is zero or negative, use default values
            temp_norm = [0.5 for _ in temp_values]  # Default to middle value
        else:
            temp_norm = [(t - temp_min) / temp_range for t in temp_values]
        
        # Apply correlation: as temperature increases, pressure typically decreases
        for i in range(len(press_values)):
            # Calculate base pressure based on temperature (inverse relationship)
            base_press = press_max - (press_max - press_min) * temp_norm[i]
            
            # Apply correlation with strength factor
            # Keep some of the original pressure value to maintain variation
            press_values[i] = (1 - strength) * press_values[i] + strength * base_press
            
            # Apply scenario-specific adjustments
            if scenario_type == 'indoor':
                # Indoor environments have more stable pressure
                press_values[i] = 0.9 * press_values[i] + 0.1 * (press_min + press_max) / 2
            elif scenario_type == 'outdoor':
                # Outdoor environments have more variable pressure
                # Add some time-based variation (e.g., weather patterns)
                day_progress = (timestamps[i].hour + timestamps[i].minute / 60) / 24
                press_values[i] += 2.0 * np.sin(2 * np.pi * day_progress)
        
        # Ensure values are within range
        press_values = [max(press_min, min(p, press_max)) for p in press_values]
        
        # Update pressure values
        data[param2]['values'] = press_values
        
        return data
    
    def _humidity_pressure_model(self, 
                               data: Dict[str, Dict[str, Any]], 
                               param1: str, 
                               param2: str, 
                               timestamps: List[datetime],
                               scenario_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Model the correlation between humidity and pressure.
        
        Args:
            data: Environmental data dictionary
            param1: First parameter name
            param2: Second parameter name
            timestamps: List of timestamps
            scenario_type: Type of scenario (indoor, outdoor)
            
        Returns:
            Updated environmental data
        """
        # Ensure param1 is 'humidity' and param2 is 'pressure'
        if param1 == 'pressure':
            param1, param2 = param2, param1
        
        # Get correlation strength
        strength = self.config['correlation_strengths'].get('humidity_pressure', 0.4)
        
        # Get humidity and pressure values
        humid_values = data[param1]['values']
        press_values = data[param2]['values']
        
        # Get parameter ranges
        humid_min = data[param1].get('min', 20.0)
        humid_max = data[param1].get('max', 80.0)
        press_min = data[param2].get('min', 990.0)
        press_max = data[param2].get('max', 1030.0)
        
        # Calculate humidity normalized to 0-1 range
        # Add safety check for division by zero
        humid_range = humid_max - humid_min
        if humid_range <= 0:
            # If humidity range is zero or negative, use default values
            humid_norm = [0.5 for _ in humid_values]  # Default to middle value
        else:
            humid_norm = [(h - humid_min) / humid_range for h in humid_values]
        
        # Apply correlation: higher humidity often correlates with lower pressure
        for i in range(len(press_values)):
            # Calculate base pressure based on humidity (inverse relationship)
            base_press = press_max - (press_max - press_min) * humid_norm[i] * 0.5
            
            # Apply correlation with strength factor
            # Keep some of the original pressure value to maintain variation
            press_values[i] = (1 - strength) * press_values[i] + strength * base_press
            
            # Apply scenario-specific adjustments
            if scenario_type == 'outdoor':
                # In outdoor environments, this relationship is stronger during certain weather conditions
                # Add some time-based variation (e.g., weather patterns)
                day_progress = (timestamps[i].hour + timestamps[i].minute / 60) / 24
                weather_factor = np.sin(2 * np.pi * day_progress + np.pi/4)
                
                # Strengthen the correlation during "stormy" periods (negative weather factor)
                if weather_factor < 0:
                    press_values[i] = 0.7 * press_values[i] + 0.3 * (press_min + (press_max - press_min) * 0.3)
        
        # Ensure values are within range
        press_values = [max(press_min, min(p, press_max)) for p in press_values]
        
        # Update pressure values
        data[param2]['values'] = press_values
        
        return data
    
    def _temperature_voc_model(self, 
                             data: Dict[str, Dict[str, Any]], 
                             param1: str, 
                             param2: str, 
                             timestamps: List[datetime],
                             scenario_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Model the correlation between temperature and VOC levels.
        
        Args:
            data: Environmental data dictionary
            param1: First parameter name
            param2: Second parameter name
            timestamps: List of timestamps
            scenario_type: Type of scenario (indoor, outdoor)
            
        Returns:
            Updated environmental data
        """
        # Ensure param1 is 'temperature' and param2 is 'voc'
        if param1 == 'voc':
            param1, param2 = param2, param1
        
        # Get correlation strength
        strength = self.config['correlation_strengths'].get('temperature_voc', 0.6)
        
        # Get temperature and VOC values
        temp_values = data[param1]['values']
        voc_values = data[param2]['values']
        
        # Get parameter ranges
        temp_min = data[param1].get('min', 15.0)
        temp_max = data[param1].get('max', 35.0)
        voc_min = data[param2].get('min', 0.0)
        voc_max = data[param2].get('max', 2000.0)
        
        # Calculate temperature normalized to 0-1 range
        # Add safety check for division by zero
        temp_range = temp_max - temp_min
        if temp_range <= 0:
            # If temperature range is zero or negative, use default values
            temp_norm = [0.5 for _ in temp_values]  # Default to middle value
        else:
            temp_norm = [(t - temp_min) / temp_range for t in temp_values]
        
        # Apply correlation: as temperature increases, VOC emissions typically increase
        for i in range(len(voc_values)):
            # Calculate base VOC based on temperature (direct relationship)
            # Use exponential relationship to model accelerated VOC release at higher temperatures
            temp_factor = np.exp(2 * temp_norm[i]) / np.exp(2)
            base_voc = voc_min + (voc_max - voc_min) * temp_factor
            
            # Apply correlation with strength factor
            # Keep some of the original VOC value to maintain variation
            voc_values[i] = (1 - strength) * voc_values[i] + strength * base_voc
            
            # Apply scenario-specific adjustments
            if scenario_type == 'indoor':
                # Indoor environments have more concentrated VOCs
                # Add ventilation factor based on time of day
                hour = timestamps[i].hour
                # Better ventilation during working hours
                if 8 <= hour <= 18:
                    voc_values[i] *= 0.8  # Reduced VOC due to ventilation
                else:
                    voc_values[i] *= 1.1  # Higher VOC when ventilation is reduced
            elif scenario_type == 'outdoor':
                # Outdoor environments have more dispersed VOCs
                voc_values[i] *= 0.7
                
                # Add some weather-based variation
                day_progress = (timestamps[i].hour + timestamps[i].minute / 60) / 24
                weather_factor = np.sin(2 * np.pi * day_progress)
                
                # Wind effect (simplified)
                if weather_factor > 0.5:  # Windy conditions
                    voc_values[i] *= 0.8  # More dispersion
                elif weather_factor < -0.5:  # Still air
                    voc_values[i] *= 1.2  # Less dispersion
        
        # Ensure values are within range
        voc_values = [max(voc_min, min(v, voc_max)) for v in voc_values]
        
        # Update VOC values
        data[param2]['values'] = voc_values
        
        return data
    
    def calculate_dew_point(self, temperature: float, relative_humidity: float) -> float:
        """
        Calculate dew point from temperature and relative humidity.
        
        Args:
            temperature: Temperature in °C
            relative_humidity: Relative humidity in %
            
        Returns:
            Dew point temperature in °C
        """
        # Constants for Magnus formula
        a = 17.27
        b = 237.7  # °C
        
        # Calculate gamma parameter
        gamma = (a * temperature) / (b + temperature) + np.log(relative_humidity / 100.0)
        
        # Calculate dew point
        dew_point = (b * gamma) / (a - gamma)
        
        return dew_point
    
    def add_derived_parameters(self, 
                             env_data: Dict[str, Dict[str, Any]], 
                             timestamps: List[datetime]) -> Dict[str, Dict[str, Any]]:
        """
        Add derived environmental parameters based on existing parameters.
        
        Args:
            env_data: Dictionary mapping parameter names to data dictionaries
            timestamps: List of timestamps
            
        Returns:
            Updated environmental data with derived parameters
        """
        # Create a copy of the input data
        updated_data = {param: data.copy() for param, data in env_data.items()}
        
        # Check if we have temperature and humidity to calculate dew point
        if 'temperature' in updated_data and 'humidity' in updated_data:
            temp_values = updated_data['temperature']['values']
            humid_values = updated_data['humidity']['values']
            
            # Calculate dew point for each time step
            dew_point_values = []
            for i in range(len(temp_values)):
                dew_point = self.calculate_dew_point(temp_values[i], humid_values[i])
                dew_point_values.append(dew_point)
            
            # Add dew point to the data
            updated_data['dew_point'] = {
                'values': dew_point_values,
                'unit': '°C',
                'min': min(dew_point_values),
                'max': max(dew_point_values)
            }
        
        # Calculate heat index if we have temperature and humidity
        if 'temperature' in updated_data and 'humidity' in updated_data:
            temp_values = updated_data['temperature']['values']
            humid_values = updated_data['humidity']['values']
            
            # Calculate heat index for each time step
            heat_index_values = []
            for i in range(len(temp_values)):
                heat_index = self.calculate_heat_index(temp_values[i], humid_values[i])
                heat_index_values.append(heat_index)
            
            # Add heat index to the data
            updated_data['heat_index'] = {
                'values': heat_index_values,
                'unit': '°C',
                'min': min(heat_index_values),
                'max': max(heat_index_values)
            }
        
        return updated_data
    
    def calculate_heat_index(self, temperature: float, relative_humidity: float) -> float:
        """
        Calculate heat index from temperature and relative humidity.
        
        Args:
            temperature: Temperature in °C
            relative_humidity: Relative humidity in %
            
        Returns:
            Heat index in °C
        """
        # Convert temperature to Fahrenheit for the standard heat index formula
        temp_f = temperature * 9/5 + 32
        
        # Simple formula for heat index in Fahrenheit
        hi_f = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (relative_humidity * 0.094))
        
        # Use more complex formula if conditions warrant
        if temp_f >= 80 and relative_humidity >= 40:
            hi_f = -42.379 + 2.04901523 * temp_f + 10.14333127 * relative_humidity
            hi_f -= 0.22475541 * temp_f * relative_humidity
            hi_f -= 0.00683783 * temp_f * temp_f
            hi_f -= 0.05481717 * relative_humidity * relative_humidity
            hi_f += 0.00122874 * temp_f * temp_f * relative_humidity
            hi_f += 0.00085282 * temp_f * relative_humidity * relative_humidity
            hi_f -= 0.00000199 * temp_f * temp_f * relative_humidity * relative_humidity
        
        # Convert back to Celsius
        hi_c = (hi_f - 32) * 5/9
        
        return hi_c