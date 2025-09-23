"""
Seasonal temperature variation patterns for synthetic fire data.

This module provides functionality for generating realistic seasonal temperature
variations that affect both thermal and gas sensor readings.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
import math


class SeasonalTemperatureVariationGenerator:
    """
    Generator for seasonal temperature variations.
    
    This class generates realistic seasonal temperature variations that can
    affect both thermal and gas sensor readings in fire detection scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the seasonal temperature variation generator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.location_latitude = self.config.get('location_latitude', 40.7128)  # NYC latitude as default
        self.base_temperature = self.config.get('base_temperature', 20.0)  # °C
        self.seasonal_amplitude = self.config.get('seasonal_amplitude', 15.0)  # °C
        self.daily_amplitude = self.config.get('daily_amplitude', 10.0)  # °C
        self.noise_level = self.config.get('noise_level', 1.0)  # °C
        
        # Seasonal parameters (based on Northern Hemisphere)
        self.seasonal_phase = self.config.get('seasonal_phase', 0.0)  # Phase shift in radians
    
    def generate_seasonal_temperature(self, 
                                   timestamp: datetime,
                                   duration_seconds: int,
                                   sample_rate_hz: float) -> np.ndarray:
        """
        Generate seasonal temperature variations for a time period.
        
        Args:
            timestamp: Start timestamp
            duration_seconds: Duration in seconds
            sample_rate_hz: Sample rate in Hz
            
        Returns:
            Array of temperature values with seasonal variations
        """
        # Calculate number of samples
        num_samples = int(duration_seconds * sample_rate_hz)
        
        # Create time array
        time_seconds = np.arange(num_samples) / sample_rate_hz
        
        # Convert timestamp to day of year
        day_of_year = timestamp.timetuple().tm_yday
        
        # Calculate seasonal component (annual cycle)
        seasonal_component = self.seasonal_amplitude * np.sin(
            2 * np.pi * (day_of_year + time_seconds / 86400) / 365.25 + self.seasonal_phase
        )
        
        # Calculate daily component (diurnal cycle)
        daily_component = self.daily_amplitude * np.sin(
            2 * np.pi * (time_seconds / 86400) + self.seasonal_phase
        )
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, num_samples)
        
        # Combine components
        temperature_variations = self.base_temperature + seasonal_component + daily_component + noise
        
        return temperature_variations
    
    def apply_to_thermal_data(self, 
                            thermal_data: Dict[str, Any],
                            timestamp: datetime) -> Dict[str, Any]:
        """
        Apply seasonal temperature variations to thermal data.
        
        Args:
            thermal_data: Dictionary containing thermal data
            timestamp: Timestamp for the data
            
        Returns:
            Updated thermal data with seasonal variations
        """
        # Get temperature variations for the data duration
        duration = len(thermal_data.get('t_mean', [])) / thermal_data.get('sample_rate_hz', 9.0)
        sample_rate = thermal_data.get('sample_rate_hz', 9.0)
        
        temperature_variations = self.generate_seasonal_temperature(
            timestamp, int(duration), sample_rate
        )
        
        # Apply variations to thermal features
        updated_thermal_data = thermal_data.copy()
        
        # Apply to mean temperature
        if 't_mean' in updated_thermal_data:
            updated_thermal_data['t_mean'] = list(np.array(updated_thermal_data['t_mean']) + temperature_variations[:len(updated_thermal_data['t_mean'])])
        
        # Apply to max temperature
        if 't_max' in updated_thermal_data:
            # Max temperature should have a stronger seasonal effect
            updated_thermal_data['t_max'] = list(np.array(updated_thermal_data['t_max']) + 1.2 * temperature_variations[:len(updated_thermal_data['t_max'])])
        
        # Apply to other temperature features with appropriate scaling
        temp_features = ['t_p95', 'tproxy_val']
        for feature in temp_features:
            if feature in updated_thermal_data:
                updated_thermal_data[feature] = list(np.array(updated_thermal_data[feature]) + 0.8 * temperature_variations[:len(updated_thermal_data[feature])])
        
        return updated_thermal_data
    
    def apply_to_gas_data(self,
                         gas_data: Dict[str, Any],
                         timestamp: datetime) -> Dict[str, Any]:
        """
        Apply seasonal temperature variations to gas data.
        
        Args:
            gas_data: Dictionary containing gas data
            timestamp: Timestamp for the data
            
        Returns:
            Updated gas data with seasonal variations
        """
        # Temperature affects gas sensor readings, especially baseline
        duration = len(gas_data.get('gas_val', [])) / gas_data.get('sample_rate_hz', 0.2)
        sample_rate = gas_data.get('sample_rate_hz', 0.2)
        
        temperature_variations = self.generate_seasonal_temperature(
            timestamp, int(duration), sample_rate
        )
        
        # Apply variations to gas data
        updated_gas_data = gas_data.copy()
        
        # Temperature affects CO2 sensor baseline and response
        if 'gas_val' in updated_gas_data:
            gas_values = np.array(updated_gas_data['gas_val'])
            # Seasonal temperature affects baseline CO2 levels
            baseline_shift = 5.0 * temperature_variations[:len(gas_values)]  # ppm per °C
            updated_gas_data['gas_val'] = list(gas_values + baseline_shift)
            
            # Temperature also affects gas sensor response time
            # This is a simplified model - higher temperatures generally lead to faster response
            response_factor = 1.0 + 0.02 * temperature_variations[:len(gas_values)]  # 2% per °C
            if 'gas_vel' in updated_gas_data:
                updated_gas_data['gas_vel'] = list(np.array(updated_gas_data['gas_vel']) * response_factor)
        
        return updated_gas_data


class HVACSimulationGenerator:
    """
    Generator for HVAC effect simulation on gas distribution.
    
    This class simulates how HVAC systems affect gas distribution in indoor environments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HVAC simulation generator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.ventilation_rate = self.config.get('ventilation_rate', 0.5)  # air changes per hour
        self.hvac_cycle_duration = self.config.get('hvac_cycle_duration', 1800)  # seconds (30 minutes)
        self.hvac_on_duration = self.config.get('hvac_on_duration', 900)  # seconds (15 minutes)
        self.air_mixing_efficiency = self.config.get('air_mixing_efficiency', 0.8)  # 0-1
        self.temperature_setpoint = self.config.get('temperature_setpoint', 22.0)  # °C
    
    def simulate_hvac_effect(self,
                           gas_data: Dict[str, Any],
                           thermal_data: Dict[str, Any],
                           timestamp: datetime) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simulate HVAC effects on gas and thermal data.
        
        Args:
            gas_data: Dictionary containing gas data
            thermal_data: Dictionary containing thermal data
            timestamp: Timestamp for the data
            
        Returns:
            Tuple of (updated_gas_data, updated_thermal_data)
        """
        # Copy data to avoid modifying originals
        updated_gas_data = gas_data.copy()
        updated_thermal_data = thermal_data.copy()
        
        # Get time parameters
        duration = len(gas_data.get('gas_val', [])) / gas_data.get('sample_rate_hz', 0.2)
        sample_rate = gas_data.get('sample_rate_hz', 0.2)
        num_samples = int(duration * sample_rate)
        
        # Create time array
        time_seconds = np.arange(num_samples) / sample_rate
        
        # Determine HVAC on/off cycles
        cycle_period = self.hvac_cycle_duration
        on_period = self.hvac_on_duration
        cycle_phase = (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) % cycle_period
        
        # Create HVAC state array (1 = on, 0 = off)
        hvac_state = np.zeros(num_samples)
        for i in range(num_samples):
            time_in_cycle = (cycle_phase + time_seconds[i]) % cycle_period
            if time_in_cycle < on_period:
                hvac_state[i] = 1
        
        # Apply HVAC effects to gas data
        if 'gas_val' in updated_gas_data:
            gas_values = np.array(updated_gas_data['gas_val'])
            
            # HVAC dilutes gas concentrations when on
            dilution_factor = 1.0 - (0.3 * hvac_state * self.air_mixing_efficiency)  # Up to 30% dilution
            updated_gas_data['gas_val'] = list(gas_values * dilution_factor)
            
            # HVAC affects gas delta (rate of change)
            if 'gas_delta' in updated_gas_data:
                delta_values = np.array(updated_gas_data['gas_delta'])
                # HVAC reduces rate of change when on
                hvac_effect = 0.7 * hvac_state * self.air_mixing_efficiency
                updated_gas_data['gas_delta'] = list(delta_values * (1.0 - hvac_effect))
        
        # Apply HVAC effects to thermal data
        if 't_mean' in updated_thermal_data:
            temp_values = np.array(updated_thermal_data['t_mean'])
            
            # HVAC tries to maintain setpoint temperature
            temp_deviation = temp_values - self.temperature_setpoint
            hvac_correction = hvac_state * 0.1 * temp_deviation  # 10% correction per cycle
            updated_thermal_data['t_mean'] = list(temp_values - hvac_correction)
            
            # Apply to max temperature with stronger effect
            if 't_max' in updated_thermal_data:
                max_temp_values = np.array(updated_thermal_data['t_max'])
                updated_thermal_data['t_max'] = list(max_temp_values - 1.2 * hvac_correction)
        
        return updated_gas_data, updated_thermal_data


class SunlightHeatingGenerator:
    """
    Generator for sunlight heating patterns on different surfaces.
    
    This class simulates how sunlight affects thermal readings on different surfaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sunlight heating generator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.surface_types = self.config.get('surface_types', ['wall', 'floor', 'ceiling', 'window'])
        self.sunrise_time = self.config.get('sunrise_time', 6.0)  # hours
        self.sunset_time = self.config.get('sunset_time', 18.0)  # hours
        self.max_solar_irradiance = self.config.get('max_solar_irradiance', 1000.0)  # W/m²
        self.surface_absorptivity = self.config.get('surface_absorptivity', {
            'wall': 0.7,
            'floor': 0.6,
            'ceiling': 0.5,
            'window': 0.9
        })
    
    def simulate_sunlight_heating(self,
                                thermal_data: Dict[str, Any],
                                timestamp: datetime,
                                surface_type: str = 'wall') -> Dict[str, Any]:
        """
        Simulate sunlight heating effects on thermal data.
        
        Args:
            thermal_data: Dictionary containing thermal data
            timestamp: Timestamp for the data
            surface_type: Type of surface being heated
            
        Returns:
            Updated thermal data with sunlight heating effects
        """
        # Copy data to avoid modifying originals
        updated_thermal_data = thermal_data.copy()
        
        # Get time parameters
        duration = len(thermal_data.get('t_mean', [])) / thermal_data.get('sample_rate_hz', 9.0)
        sample_rate = thermal_data.get('sample_rate_hz', 9.0)
        num_samples = int(duration * sample_rate)
        
        # Create time array in hours
        start_hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
        time_hours = start_hour + np.arange(num_samples) / (sample_rate * 3600.0)
        
        # Calculate solar irradiance based on time of day
        # Simple model: sinusoidal irradiance between sunrise and sunset
        solar_irradiance = np.zeros(num_samples)
        daylight_mask = (time_hours % 24 >= self.sunrise_time) & (time_hours % 24 <= self.sunset_time)
        
        if np.any(daylight_mask):
            daylight_hours = time_hours[daylight_mask] % 24
            # Normalize to 0-1 range between sunrise and sunset
            normalized_time = (daylight_hours - self.sunrise_time) / (self.sunset_time - self.sunrise_time)
            # Sinusoidal irradiance peaking at noon
            solar_irradiance[daylight_mask] = self.max_solar_irradiance * np.sin(np.pi * normalized_time)
        
        # Get surface absorptivity
        absorptivity = self.surface_absorptivity.get(surface_type, 0.7)
        
        # Calculate temperature increase due to solar heating
        # Simplified model: 0.1°C per 100 W/m²
        temp_increase = 0.1 * solar_irradiance * absorptivity / 100.0
        
        # Apply heating effects to thermal data
        if 't_mean' in updated_thermal_data:
            temp_values = np.array(updated_thermal_data['t_mean'])
            updated_thermal_data['t_mean'] = list(temp_values + temp_increase[:len(temp_values)])
        
        if 't_max' in updated_thermal_data:
            max_temp_values = np.array(updated_thermal_data['t_max'])
            # Max temperature gets stronger heating effect
            updated_thermal_data['t_max'] = list(max_temp_values + 1.5 * temp_increase[:len(max_temp_values)])
        
        # Add metadata about sunlight heating
        if 'metadata' not in updated_thermal_data:
            updated_thermal_data['metadata'] = {}
        updated_thermal_data['metadata']['sunlight_heating_applied'] = True
        updated_thermal_data['metadata']['surface_type'] = surface_type
        updated_thermal_data['metadata']['max_temp_increase'] = float(np.max(temp_increase)) if len(temp_increase) > 0 else 0.0
        
        return updated_thermal_data


class FlirOcclusionGenerator:
    """
    Generator for FLIR occlusion scenarios.
    
    This class simulates various occlusion scenarios that can affect FLIR camera readings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FLIR occlusion generator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.occlusion_types = self.config.get('occlusion_types', ['dust', 'steam', 'blockage'])
        self.occlusion_severities = self.config.get('occlusion_severities', ['light', 'moderate', 'heavy'])
        self.occlusion_probability = self.config.get('occlusion_probability', 0.05)  # 5% chance per sample
    
    def simulate_flir_occlusion(self,
                              thermal_data: Dict[str, Any],
                              occlusion_type: str = 'dust',
                              severity: str = 'moderate') -> Dict[str, Any]:
        """
        Simulate FLIR occlusion effects on thermal data.
        
        Args:
            thermal_data: Dictionary containing thermal data
            occlusion_type: Type of occlusion ('dust', 'steam', 'blockage')
            severity: Severity level ('light', 'moderate', 'heavy')
            
        Returns:
            Updated thermal data with occlusion effects
        """
        # Copy data to avoid modifying originals
        updated_thermal_data = thermal_data.copy()
        
        # Determine severity factors
        severity_factors = {
            'light': 0.3,
            'moderate': 0.6,
            'heavy': 0.9
        }
        severity_factor = severity_factors.get(severity, 0.6)
        
        # Apply occlusion effects based on type
        if occlusion_type == 'dust':
            # Dust reduces overall temperature readings
            temp_reduction = 2.0 * severity_factor  # °C reduction
            
            if 't_mean' in updated_thermal_data:
                temp_values = np.array(updated_thermal_data['t_mean'])
                # Add some randomness to make it more realistic
                noise = np.random.normal(0, 0.5 * severity_factor, len(temp_values))
                updated_thermal_data['t_mean'] = list(temp_values - temp_reduction + noise)
            
            if 't_max' in updated_thermal_data:
                max_temp_values = np.array(updated_thermal_data['t_max'])
                updated_thermal_data['t_max'] = list(max_temp_values - 1.5 * temp_reduction + noise[:len(max_temp_values)])
        
        elif occlusion_type == 'steam':
            # Steam creates localized cooling and noise
            if 't_mean' in updated_thermal_data:
                temp_values = np.array(updated_thermal_data['t_mean'])
                # Steam creates random cooling spots
                steam_mask = np.random.random(len(temp_values)) < self.occlusion_probability
                steam_effect = np.where(steam_mask, -3.0 * severity_factor, 0)
                noise = np.random.normal(0, 1.0 * severity_factor, len(temp_values))
                updated_thermal_data['t_mean'] = list(temp_values + steam_effect + noise)
            
            if 't_max' in updated_thermal_data:
                max_temp_values = np.array(updated_thermal_data['t_max'])
                steam_mask = np.random.random(len(max_temp_values)) < self.occlusion_probability
                steam_effect = np.where(steam_mask, -4.0 * severity_factor, 0)
                noise = np.random.normal(0, 1.2 * severity_factor, len(max_temp_values))
                updated_thermal_data['t_max'] = list(max_temp_values + steam_effect + noise[:len(max_temp_values)])
        
        elif occlusion_type == 'blockage':
            # Blockage creates complete or partial loss of signal
            if 't_mean' in updated_thermal_data:
                temp_values = np.array(updated_thermal_data['t_mean'])
                # Blockage creates gaps in data
                blockage_mask = np.random.random(len(temp_values)) < self.occlusion_probability * severity_factor
                # Replace blocked readings with baseline values plus noise
                baseline_temp = np.mean(temp_values) if len(temp_values) > 0 else 20.0
                blocked_values = np.where(blockage_mask, baseline_temp + np.random.normal(0, 2.0, len(temp_values)), temp_values)
                updated_thermal_data['t_mean'] = list(blocked_values)
        
        # Add metadata about occlusion
        if 'metadata' not in updated_thermal_data:
            updated_thermal_data['metadata'] = {}
        updated_thermal_data['metadata']['occlusion_applied'] = True
        updated_thermal_data['metadata']['occlusion_type'] = occlusion_type
        updated_thermal_data['metadata']['occlusion_severity'] = severity
        
        return updated_thermal_data