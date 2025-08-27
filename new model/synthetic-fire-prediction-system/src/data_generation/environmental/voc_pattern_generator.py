"""
VOC pattern generator for synthetic fire data.

This module provides functionality for generating realistic volatile organic compound (VOC)
patterns based on different materials and fire scenarios.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
import math


class VOCPatternGenerator:
    """
    Class for generating synthetic VOC patterns.
    
    This class provides methods for generating realistic VOC concentration patterns
    based on different materials and environmental conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VOC pattern generator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.validate_config()
        
        # Define material-specific VOC emission characteristics
        self.material_profiles = {
            'wood': {
                'emission_rate': 1.0,
                'temp_sensitivity': 0.15,  # VOC emission increase per °C above baseline
                'primary_compounds': ['formaldehyde', 'acetic_acid', 'methanol'],
                'decay_rate': 0.05
            },
            'plastic': {
                'emission_rate': 2.5,
                'temp_sensitivity': 0.25,
                'primary_compounds': ['styrene', 'benzene', 'toluene', 'formaldehyde'],
                'decay_rate': 0.02
            },
            'fabric': {
                'emission_rate': 0.8,
                'temp_sensitivity': 0.1,
                'primary_compounds': ['formaldehyde', 'acetaldehyde', 'benzene'],
                'decay_rate': 0.08
            },
            'electronics': {
                'emission_rate': 1.8,
                'temp_sensitivity': 0.2,
                'primary_compounds': ['brominated_compounds', 'benzene', 'toluene', 'styrene'],
                'decay_rate': 0.03
            },
            'mixed': {
                'emission_rate': 1.5,
                'temp_sensitivity': 0.18,
                'primary_compounds': ['formaldehyde', 'benzene', 'toluene', 'styrene', 'acetic_acid'],
                'decay_rate': 0.04
            }
        }
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Set default values if not provided
        if 'baseline_voc' not in self.config:
            self.config['baseline_voc'] = 50.0  # ppb
        
        if 'max_voc' not in self.config:
            self.config['max_voc'] = 2000.0  # ppb
        
        if 'default_material' not in self.config:
            self.config['default_material'] = 'mixed'
        
        if 'temperature_baseline' not in self.config:
            self.config['temperature_baseline'] = 20.0  # °C
    
    def generate_voc_time_series(self,
                               timestamps: List[datetime],
                               baseline: float,
                               material_type: str = 'mixed',
                               temperature_values: Optional[List[float]] = None,
                               fire_event: Optional[Dict[str, Any]] = None,
                               seed: Optional[int] = None) -> List[float]:
        """
        Generate a time series of VOC concentrations.
        
        Args:
            timestamps: List of timestamps
            baseline: Baseline VOC concentration in ppb
            material_type: Type of material emitting VOCs
            temperature_values: Optional list of temperature values
            fire_event: Optional fire event parameters
            seed: Optional random seed for reproducibility
            
        Returns:
            List of VOC concentration values
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get material profile
        if material_type not in self.material_profiles:
            material_type = 'mixed'
        
        material = self.material_profiles[material_type]
        
        # Initialize VOC values with baseline
        num_samples = len(timestamps)
        voc_values = np.ones(num_samples) * baseline
        
        # Apply temperature effects if temperature values are provided
        if temperature_values is not None:
            temp_baseline = self.config['temperature_baseline']
            
            for i in range(num_samples):
                # Calculate temperature difference from baseline
                temp_diff = max(0, temperature_values[i] - temp_baseline)
                
                # Apply temperature effect on VOC emissions
                temp_effect = temp_diff * material['temp_sensitivity'] * baseline
                voc_values[i] += temp_effect
        
        # Apply fire event if provided
        if fire_event is not None:
            # Extract fire event parameters
            start_idx = fire_event.get('start_idx', 0)
            duration_idx = fire_event.get('duration_idx', num_samples // 4)
            peak_multiplier = fire_event.get('peak_multiplier', 10.0)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, num_samples - 1))
            duration_idx = max(1, min(duration_idx, num_samples - start_idx))
            end_idx = start_idx + duration_idx
            
            # Generate fire event VOC pattern
            for i in range(start_idx, min(end_idx, num_samples)):
                # Calculate progress through fire event (0 to 1)
                progress = (i - start_idx) / duration_idx
                
                # Apply different phases of fire VOC emissions
                if progress < 0.2:  # Initial phase
                    # Exponential increase
                    factor = peak_multiplier * (math.exp(5 * progress) - 1) / (math.exp(1) - 1)
                elif progress < 0.7:  # Peak phase
                    # Sustained high levels with fluctuations
                    factor = peak_multiplier * (0.9 + 0.2 * np.random.random())
                else:  # Decay phase
                    # Gradual decay
                    decay_progress = (progress - 0.7) / 0.3
                    factor = peak_multiplier * (1.0 - 0.8 * decay_progress)
                
                # Apply material-specific emission rate
                factor *= material['emission_rate']
                
                # Update VOC value
                voc_values[i] = baseline + (voc_values[i] - baseline) * factor
        
        # Add random variations
        noise_level = 0.05  # 5% noise
        for i in range(num_samples):
            noise = np.random.normal(0, noise_level * voc_values[i])
            voc_values[i] += noise
        
        # Ensure values are positive and within reasonable limits
        max_voc = self.config['max_voc']
        voc_values = np.clip(voc_values, 0, max_voc)
        
        return voc_values.tolist()
    
    def generate_compound_specific_vocs(self,
                                      timestamps: List[datetime],
                                      total_voc: List[float],
                                      material_type: str = 'mixed') -> Dict[str, List[float]]:
        """
        Generate compound-specific VOC concentrations from total VOC values.
        
        Args:
            timestamps: List of timestamps
            total_voc: List of total VOC concentration values
            material_type: Type of material emitting VOCs
            
        Returns:
            Dictionary mapping compound names to concentration lists
        """
        # Get material profile
        if material_type not in self.material_profiles:
            material_type = 'mixed'
        
        material = self.material_profiles[material_type]
        compounds = material['primary_compounds']
        
        # Define compound ratios based on material type
        if material_type == 'wood':
            ratios = {'formaldehyde': 0.4, 'acetic_acid': 0.35, 'methanol': 0.25}
        elif material_type == 'plastic':
            ratios = {'styrene': 0.3, 'benzene': 0.25, 'toluene': 0.25, 'formaldehyde': 0.2}
        elif material_type == 'fabric':
            ratios = {'formaldehyde': 0.45, 'acetaldehyde': 0.3, 'benzene': 0.25}
        elif material_type == 'electronics':
            ratios = {'brominated_compounds': 0.35, 'benzene': 0.25, 'toluene': 0.2, 'styrene': 0.2}
        else:  # mixed
            ratios = {'formaldehyde': 0.25, 'benzene': 0.2, 'toluene': 0.2, 'styrene': 0.15, 'acetic_acid': 0.2}
        
        # Generate compound-specific VOC values
        compound_vocs = {}
        for compound in compounds:
            ratio = ratios.get(compound, 1.0 / len(compounds))
            
            # Add some variation to the ratio over time
            compound_values = []
            for i, voc in enumerate(total_voc):
                # Vary the ratio slightly over time
                time_factor = 1.0 + 0.1 * np.sin(i / len(total_voc) * 2 * np.pi)
                compound_values.append(voc * ratio * time_factor)
            
            compound_vocs[compound] = compound_values
        
        return compound_vocs
    
    def simulate_material_specific_emission(self,
                                          material_type: str,
                                          temperature: float,
                                          duration_hours: float,
                                          sample_rate_hz: float) -> Dict[str, Any]:
        """
        Simulate material-specific VOC emissions over time.
        
        Args:
            material_type: Type of material emitting VOCs
            temperature: Temperature in °C
            duration_hours: Duration of simulation in hours
            sample_rate_hz: Sample rate in Hz
            
        Returns:
            Dictionary containing simulation results
        """
        # Get material profile
        if material_type not in self.material_profiles:
            material_type = 'mixed'
        
        material = self.material_profiles[material_type]
        
        # Calculate number of samples
        num_samples = int(duration_hours * 3600 * sample_rate_hz)
        
        # Generate timestamps
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i/sample_rate_hz) for i in range(num_samples)]
        
        # Calculate temperature effect
        temp_baseline = self.config['temperature_baseline']
        temp_diff = max(0, temperature - temp_baseline)
        temp_effect = temp_diff * material['emission_rate'] * material['temp_sensitivity']
        
        # Calculate baseline emission
        baseline_voc = self.config['baseline_voc']
        base_emission = baseline_voc * material['emission_rate']
        
        # Generate emission profile over time
        total_voc = []
        for i in range(num_samples):
            # Calculate time in hours
            time_hours = i / (3600 * sample_rate_hz)
            
            # Apply emission model with decay
            emission = base_emission * (1.0 + temp_effect) * math.exp(-material['decay_rate'] * time_hours)
            
            # Add random variations
            noise = np.random.normal(0, 0.05 * emission)
            emission += noise
            
            total_voc.append(max(0, emission))
        
        # Generate compound-specific VOCs
        compound_vocs = self.generate_compound_specific_vocs(timestamps, total_voc, material_type)
        
        # Create result dictionary
        result = {
            'material_type': material_type,
            'temperature': temperature,
            'duration_hours': duration_hours,
            'sample_rate_hz': sample_rate_hz,
            'timestamps': timestamps,
            'total_voc': total_voc,
            'compound_vocs': compound_vocs,
            'material_profile': material
        }
        
        return result
    
    def simulate_fire_voc_profile(self,
                                fire_type: str,
                                material_type: str,
                                duration_minutes: float,
                                sample_rate_hz: float,
                                temperature_profile: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Simulate VOC emissions during a fire event.
        
        Args:
            fire_type: Type of fire ('smoldering', 'flaming', 'electrical', 'chemical')
            material_type: Type of material burning
            duration_minutes: Duration of simulation in minutes
            sample_rate_hz: Sample rate in Hz
            temperature_profile: Optional temperature profile over time
            
        Returns:
            Dictionary containing simulation results
        """
        # Calculate number of samples
        num_samples = int(duration_minutes * 60 * sample_rate_hz)
        
        # Generate timestamps
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i/sample_rate_hz) for i in range(num_samples)]
        
        # Generate temperature profile if not provided
        if temperature_profile is None:
            temp_baseline = self.config['temperature_baseline']
            
            if fire_type == 'smoldering':
                max_temp = temp_baseline + 50.0
                rise_rate = 0.05  # Slower temperature rise
            elif fire_type == 'flaming':
                max_temp = temp_baseline + 200.0
                rise_rate = 0.2   # Rapid temperature rise
            elif fire_type == 'electrical':
                max_temp = temp_baseline + 80.0
                rise_rate = 0.1   # Moderate temperature rise
            elif fire_type == 'chemical':
                max_temp = temp_baseline + 150.0
                rise_rate = 0.15  # Variable temperature rise
            else:
                max_temp = temp_baseline + 100.0
                rise_rate = 0.1
            
            temperature_profile = []
            for i in range(num_samples):
                # Calculate time in minutes
                time_minutes = i / (60 * sample_rate_hz)
                
                # Apply temperature model
                if time_minutes < duration_minutes * 0.2:
                    # Initial phase - exponential rise
                    progress = time_minutes / (duration_minutes * 0.2)
                    temp = temp_baseline + (max_temp - temp_baseline) * (1 - math.exp(-rise_rate * progress * 10))
                elif time_minutes < duration_minutes * 0.7:
                    # Peak phase - sustained high temperature with fluctuations
                    temp = max_temp * (0.95 + 0.1 * np.random.random())
                else:
                    # Cooling phase - gradual decay
                    cooling_time = time_minutes - duration_minutes * 0.7
                    cooling_duration = duration_minutes * 0.3
                    cooling_progress = cooling_time / cooling_duration
                    temp = max_temp - (max_temp - temp_baseline) * (1 - math.exp(-cooling_progress * 2))
                
                temperature_profile.append(temp)
        
        # Create fire event parameters
        fire_event = {
            'start_idx': 0,
            'duration_idx': num_samples,
            'peak_multiplier': 1.0  # Will be adjusted based on fire type
        }
        
        # Adjust peak multiplier based on fire type
        if fire_type == 'smoldering':
            fire_event['peak_multiplier'] = 5.0
        elif fire_type == 'flaming':
            fire_event['peak_multiplier'] = 15.0
        elif fire_type == 'electrical':
            fire_event['peak_multiplier'] = 8.0
        elif fire_type == 'chemical':
            fire_event['peak_multiplier'] = 12.0
        
        # Generate VOC time series
        baseline_voc = self.config['baseline_voc']
        total_voc = self.generate_voc_time_series(
            timestamps=timestamps,
            baseline=baseline_voc,
            material_type=material_type,
            temperature_values=temperature_profile,
            fire_event=fire_event
        )
        
        # Generate compound-specific VOCs
        compound_vocs = self.generate_compound_specific_vocs(timestamps, total_voc, material_type)
        
        # Create result dictionary
        result = {
            'fire_type': fire_type,
            'material_type': material_type,
            'duration_minutes': duration_minutes,
            'sample_rate_hz': sample_rate_hz,
            'timestamps': timestamps,
            'temperature_profile': temperature_profile,
            'total_voc': total_voc,
            'compound_vocs': compound_vocs
        }
        
        return result