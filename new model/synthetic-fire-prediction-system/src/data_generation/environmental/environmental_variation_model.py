"""
Environmental variation model for synthetic fire data.

This module provides functionality for modeling daily and seasonal variations
in environmental parameters, as well as sensor noise and drift.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
import math


class EnvironmentalVariationModel:
    """
    Class for modeling variations in environmental parameters.
    
    This class provides methods for generating realistic environmental parameter
    variations including daily cycles, seasonal patterns, and sensor characteristics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environmental variation model with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.validate_config()
        
        # Define parameter-specific variation characteristics
        self.parameter_variations = {
            'temperature': {
                'daily_amplitude': 5.0,  # °C variation throughout the day
                'seasonal_amplitude': 10.0,  # °C variation throughout the year
                'noise_level': 0.2,  # °C random noise
                'drift_rate': 0.01,  # °C per hour drift
                'daily_phase': 0.0,  # Peak at noon (0 phase)
                'response_time': 60.0  # Seconds for sensor to respond to changes
            },
            'humidity': {
                'daily_amplitude': 15.0,  # % variation throughout the day
                'seasonal_amplitude': 20.0,  # % variation throughout the year
                'noise_level': 1.0,  # % random noise
                'drift_rate': 0.05,  # % per hour drift
                'daily_phase': np.pi,  # Peak at midnight (π phase shift from temperature)
                'response_time': 120.0  # Seconds for sensor to respond to changes
            },
            'pressure': {
                'daily_amplitude': 2.0,  # hPa variation throughout the day
                'seasonal_amplitude': 5.0,  # hPa variation throughout the year
                'noise_level': 0.1,  # hPa random noise
                'drift_rate': 0.02,  # hPa per hour drift
                'daily_phase': np.pi/2,  # Peak in the evening (π/2 phase shift)
                'response_time': 30.0  # Seconds for sensor to respond to changes
            },
            'voc': {
                'daily_amplitude': 100.0,  # ppb variation throughout the day
                'seasonal_amplitude': 200.0,  # ppb variation throughout the year
                'noise_level': 10.0,  # ppb random noise
                'drift_rate': 0.5,  # ppb per hour drift
                'daily_phase': np.pi/4,  # Peak in the afternoon (π/4 phase shift)
                'response_time': 90.0  # Seconds for sensor to respond to changes
            }
        }
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Set default values if not provided
        if 'sensor_noise_factor' not in self.config:
            self.config['sensor_noise_factor'] = 1.0  # Default noise factor
        
        if 'sensor_drift_factor' not in self.config:
            self.config['sensor_drift_factor'] = 1.0  # Default drift factor
        
        if 'daily_variation_factor' not in self.config:
            self.config['daily_variation_factor'] = 1.0  # Default daily variation factor
        
        if 'seasonal_variation_factor' not in self.config:
            self.config['seasonal_variation_factor'] = 1.0  # Default seasonal variation factor
        
        if 'environment_type' not in self.config:
            self.config['environment_type'] = 'indoor'  # Default to indoor environment
    
    def generate_time_series(self,
                           parameter: str,
                           baseline: float,
                           timestamps: List[datetime],
                           seed: Optional[int] = None) -> List[float]:
        """
        Generate a time series for an environmental parameter with realistic variations.
        
        Args:
            parameter: Environmental parameter name
            baseline: Baseline value for the parameter
            timestamps: List of timestamps
            seed: Optional random seed for reproducibility
            
        Returns:
            List of parameter values with variations applied
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get parameter variation characteristics
        if parameter not in self.parameter_variations:
            # Use temperature as default if parameter not found
            param_var = self.parameter_variations['temperature']
        else:
            param_var = self.parameter_variations[parameter]
        
        # Get configuration factors
        noise_factor = self.config['sensor_noise_factor']
        drift_factor = self.config['sensor_drift_factor']
        daily_factor = self.config['daily_variation_factor']
        seasonal_factor = self.config['seasonal_variation_factor']
        
        # Adjust variation amplitudes based on environment type
        if self.config['environment_type'] == 'indoor':
            # Indoor environments have reduced variations
            daily_factor *= 0.5
            seasonal_factor *= 0.3
        
        # Initialize values with baseline
        values = np.ones(len(timestamps)) * baseline
        
        # Apply daily and seasonal variations
        for i, timestamp in enumerate(timestamps):
            # Calculate daily variation
            day_progress = (timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600) / 24
            daily_variation = param_var['daily_amplitude'] * daily_factor * \
                              np.sin(2 * np.pi * day_progress + param_var['daily_phase'])
            
            # Calculate seasonal variation
            day_of_year = timestamp.timetuple().tm_yday
            year_progress = day_of_year / 365.25
            seasonal_variation = param_var['seasonal_amplitude'] * seasonal_factor * \
                                np.sin(2 * np.pi * year_progress)
            
            # Apply variations to baseline
            values[i] += daily_variation + seasonal_variation
        
        # Apply sensor characteristics (noise and drift)
        values = self._apply_sensor_characteristics(
            values, 
            timestamps, 
            param_var['noise_level'] * noise_factor,
            param_var['drift_rate'] * drift_factor,
            param_var['response_time']
        )
        
        return values.tolist()
    
    def generate_single_reading(self,
                              parameter: str,
                              timestamp: datetime,
                              baseline: float,
                              daily_variation: float = 0.1,
                              noise_level: float = 0.05) -> float:
        """
        Generate a single environmental parameter reading.
        
        Args:
            parameter: Environmental parameter name
            timestamp: Timestamp for the reading
            baseline: Baseline value for the parameter
            daily_variation: Magnitude of daily variation as fraction of baseline
            noise_level: Magnitude of random noise as fraction of baseline
            
        Returns:
            Parameter reading with variations applied
        """
        # Get parameter variation characteristics
        if parameter not in self.parameter_variations:
            # Use temperature as default if parameter not found
            param_var = self.parameter_variations['temperature']
        else:
            param_var = self.parameter_variations[parameter]
        
        # Calculate daily variation
        day_progress = (timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600) / 24
        daily_amplitude = baseline * daily_variation
        daily_variation = daily_amplitude * np.sin(2 * np.pi * day_progress + param_var['daily_phase'])
        
        # Calculate seasonal variation (simplified)
        day_of_year = timestamp.timetuple().tm_yday
        year_progress = day_of_year / 365.25
        seasonal_amplitude = baseline * daily_variation * 2  # Seasonal variation is typically larger
        seasonal_variation = seasonal_amplitude * np.sin(2 * np.pi * year_progress)
        
        # Apply variations to baseline
        value = baseline + daily_variation + seasonal_variation
        
        # Add random noise
        noise = np.random.normal(0, baseline * noise_level)
        value += noise
        
        return value
    
    def _apply_sensor_characteristics(self,
                                    values: np.ndarray,
                                    timestamps: List[datetime],
                                    noise_level: float,
                                    drift_rate: float,
                                    response_time: float) -> np.ndarray:
        """
        Apply sensor characteristics to a time series.
        
        Args:
            values: Array of parameter values
            timestamps: List of timestamps
            noise_level: Magnitude of random noise
            drift_rate: Rate of sensor drift per hour
            response_time: Sensor response time in seconds
            
        Returns:
            Array of values with sensor characteristics applied
        """
        # Create a copy of the input values
        sensor_values = values.copy()
        
        # Add random noise
        noise = np.random.normal(0, noise_level, len(values))
        sensor_values += noise
        
        # Apply sensor drift
        for i in range(len(values)):
            # Calculate hours since start
            hours_elapsed = (timestamps[i] - timestamps[0]).total_seconds() / 3600
            
            # Apply drift (can be positive or negative)
            drift_direction = 1 if np.random.random() > 0.5 else -1
            drift = drift_direction * drift_rate * hours_elapsed
            sensor_values[i] += drift
        
        # Apply sensor response time (low-pass filter)
        if len(values) > 1:
            # Calculate time differences between samples
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                         for i in range(1, len(timestamps))]
            time_diffs = np.array([0] + time_diffs)  # Add 0 for the first sample
            
            # Calculate filter coefficient based on response time
            alpha = 1 - np.exp(-time_diffs / response_time)
            
            # Apply low-pass filter
            filtered_values = np.zeros_like(sensor_values)
            filtered_values[0] = sensor_values[0]
            
            for i in range(1, len(sensor_values)):
                filtered_values[i] = alpha[i] * sensor_values[i] + (1 - alpha[i]) * filtered_values[i-1]
            
            sensor_values = filtered_values
        
        return sensor_values
    
    def generate_long_term_variation(self,
                                   parameter: str,
                                   baseline: float,
                                   start_date: datetime,
                                   end_date: datetime,
                                   sample_interval_minutes: int = 60,
                                   seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate long-term environmental parameter variations.
        
        Args:
            parameter: Environmental parameter name
            baseline: Baseline value for the parameter
            start_date: Start date for the time series
            end_date: End date for the time series
            sample_interval_minutes: Sampling interval in minutes
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated time series and metadata
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate number of samples
        duration = (end_date - start_date).total_seconds()
        sample_interval_seconds = sample_interval_minutes * 60
        num_samples = int(duration / sample_interval_seconds) + 1
        
        # Generate timestamps
        timestamps = [start_date + timedelta(seconds=i * sample_interval_seconds) 
                     for i in range(num_samples)]
        
        # Generate parameter values
        values = self.generate_time_series(parameter, baseline, timestamps, seed)
        
        # Create result dictionary
        result = {
            'parameter': parameter,
            'baseline': baseline,
            'start_date': start_date,
            'end_date': end_date,
            'sample_interval_minutes': sample_interval_minutes,
            'timestamps': timestamps,
            'values': values,
            'num_samples': num_samples
        }
        
        return result
    
    def simulate_sensor_aging(self,
                            parameter: str,
                            values: List[float],
                            timestamps: List[datetime],
                            sensor_age_days: float = 0.0) -> List[float]:
        """
        Simulate the effects of sensor aging on measurements.
        
        Args:
            parameter: Environmental parameter name
            values: List of true parameter values
            timestamps: List of timestamps
            sensor_age_days: Age of the sensor in days
            
        Returns:
            List of values with sensor aging effects applied
        """
        # Get parameter variation characteristics
        if parameter not in self.parameter_variations:
            # Use temperature as default if parameter not found
            param_var = self.parameter_variations['temperature']
        else:
            param_var = self.parameter_variations[parameter]
        
        # Calculate aging factors
        # Sensitivity reduction (percentage per year, converted to days)
        sensitivity_reduction = 0.1 * sensor_age_days / 365.0  # 10% per year
        
        # Noise increase (percentage per year, converted to days)
        noise_increase = 0.2 * sensor_age_days / 365.0  # 20% per year
        
        # Drift rate increase (percentage per year, converted to days)
        drift_increase = 0.3 * sensor_age_days / 365.0  # 30% per year
        
        # Apply aging effects
        aged_values = np.array(values).copy()
        
        # Reduce sensitivity (compress the range around the mean)
        mean_value = np.mean(aged_values)
        aged_values = mean_value + (aged_values - mean_value) * (1.0 - sensitivity_reduction)
        
        # Increase noise
        increased_noise_level = param_var['noise_level'] * (1.0 + noise_increase)
        noise = np.random.normal(0, increased_noise_level, len(values))
        aged_values += noise
        
        # Increase drift
        increased_drift_rate = param_var['drift_rate'] * (1.0 + drift_increase)
        for i in range(len(values)):
            # Calculate hours since start
            hours_elapsed = (timestamps[i] - timestamps[0]).total_seconds() / 3600
            
            # Apply increased drift
            drift_direction = 1 if sensor_age_days % 2 == 0 else -1  # Consistent drift direction based on age
            drift = drift_direction * increased_drift_rate * hours_elapsed
            aged_values[i] += drift
        
        return aged_values.tolist()
    
    def generate_seasonal_profile(self,
                                parameter: str,
                                baseline: float,
                                year: int = 2023,
                                samples_per_month: int = 30,
                                seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a seasonal profile for an environmental parameter.
        
        Args:
            parameter: Environmental parameter name
            baseline: Baseline value for the parameter
            year: Year for the profile
            samples_per_month: Number of samples per month
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated seasonal profile
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get parameter variation characteristics
        if parameter not in self.parameter_variations:
            # Use temperature as default if parameter not found
            param_var = self.parameter_variations['temperature']
        else:
            param_var = self.parameter_variations[parameter]
        
        # Calculate total number of samples
        num_samples = samples_per_month * 12
        
        # Generate timestamps (evenly distributed throughout the year)
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        duration = (end_date - start_date).total_seconds()
        sample_interval = duration / (num_samples - 1)
        
        timestamps = [start_date + timedelta(seconds=i * sample_interval) 
                     for i in range(num_samples)]
        
        # Generate monthly averages
        monthly_averages = []
        for month in range(1, 13):
            # Calculate seasonal effect
            month_progress = (month - 1) / 12
            seasonal_variation = param_var['seasonal_amplitude'] * \
                               np.sin(2 * np.pi * month_progress)
            
            # Apply to baseline
            monthly_avg = baseline + seasonal_variation
            monthly_averages.append(monthly_avg)
        
        # Generate detailed profile
        values = []
        for i, timestamp in enumerate(timestamps):
            # Get month and interpolate between monthly averages
            month = timestamp.month
            next_month = month % 12 + 1
            
            # Calculate progress through the month
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month[1] = 29  # Leap year
            
            month_progress = (timestamp.day - 1) / days_in_month[month - 1]
            
            # Interpolate between monthly averages
            current_avg = monthly_averages[month - 1]
            next_avg = monthly_averages[next_month - 1]
            interpolated = current_avg + month_progress * (next_avg - current_avg)
            
            # Add daily variation
            day_progress = (timestamp.hour + timestamp.minute / 60) / 24
            daily_variation = param_var['daily_amplitude'] * \
                            np.sin(2 * np.pi * day_progress + param_var['daily_phase'])
            
            # Add random noise
            noise = np.random.normal(0, param_var['noise_level'] * 0.5)
            
            # Combine all effects
            value = interpolated + daily_variation + noise
            values.append(value)
        
        # Create result dictionary
        result = {
            'parameter': parameter,
            'baseline': baseline,
            'year': year,
            'timestamps': timestamps,
            'values': values,
            'monthly_averages': monthly_averages,
            'num_samples': num_samples
        }
        
        return result