"""
Gas temporal evolution model for synthetic fire data.

This module provides functionality for creating time-series of gas concentration changes,
modeling different gas release patterns, and simulating gas behavior over time.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import math


class ReleasePattern(Enum):
    """Enumeration of supported gas release patterns."""
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INTERMITTENT = "intermittent"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SINUSOIDAL = "sinusoidal"
    CUSTOM = "custom"


class FireStage(Enum):
    """Enumeration of fire development stages."""
    INCIPIENT = "incipient"
    GROWTH = "growth"
    FULLY_DEVELOPED = "fully_developed"
    DECAY = "decay"


class GasTemporalEvolution:
    """
    Class for creating time-series of gas concentration changes.
    
    This class provides methods for modeling different gas release patterns
    and simulating gas behavior over time in different scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gas temporal evolution model with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
                - default_release_pattern: Default gas release pattern
                - default_duration: Default duration in seconds
                - default_sample_rate: Default sample rate in Hz
                - gas_properties: Dictionary of gas properties
                - fire_stage_durations: Dictionary mapping fire stages to durations
        """
        self.config = config
        self.validate_config()
        
        # Set default values
        self.default_release_pattern = ReleasePattern(self.config.get('default_release_pattern', 'gradual'))
        self.default_duration = self.config.get('default_duration', 600)  # 10 minutes
        self.default_sample_rate = self.config.get('default_sample_rate', 1.0)  # 1 Hz
        
        # Gas properties (production rates in fire scenarios)
        self.gas_properties = self.config.get('gas_properties', {
            'methane': {
                'production_rate': 0.01,  # g/s per kg of fuel
                'molecular_weight': 16.04,  # g/mol
                'flammability_limits': (5.0, 15.0),  # % by volume in air
                'ignition_temp': 580  # °C
            },
            'propane': {
                'production_rate': 0.005,  # g/s per kg of fuel
                'molecular_weight': 44.1,  # g/mol
                'flammability_limits': (2.1, 9.5),  # % by volume in air
                'ignition_temp': 470  # °C
            },
            'hydrogen': {
                'production_rate': 0.002,  # g/s per kg of fuel
                'molecular_weight': 2.02,  # g/mol
                'flammability_limits': (4.0, 75.0),  # % by volume in air
                'ignition_temp': 500  # °C
            },
            'carbon_monoxide': {
                'production_rate': 0.03,  # g/s per kg of fuel
                'molecular_weight': 28.01,  # g/mol
                'flammability_limits': (12.5, 74.0),  # % by volume in air
                'ignition_temp': 609  # °C
            }
        })
        
        # Fire stage durations as fractions of total duration
        self.fire_stage_durations = self.config.get('fire_stage_durations', {
            FireStage.INCIPIENT: 0.2,
            FireStage.GROWTH: 0.3,
            FireStage.FULLY_DEVELOPED: 0.4,
            FireStage.DECAY: 0.1
        })
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if 'default_release_pattern' in self.config:
            try:
                ReleasePattern(self.config['default_release_pattern'])
            except ValueError:
                valid_patterns = [rp.value for rp in ReleasePattern]
                raise ValueError(f"Invalid release pattern. Must be one of: {valid_patterns}")
        
        if 'default_duration' in self.config and self.config['default_duration'] <= 0:
            raise ValueError("default_duration must be positive")
        
        if 'default_sample_rate' in self.config and self.config['default_sample_rate'] <= 0:
            raise ValueError("default_sample_rate must be positive")
        
        if 'fire_stage_durations' in self.config:
            total = sum(self.config['fire_stage_durations'].values())
            if abs(total - 1.0) > 0.01:  # Allow small rounding errors
                raise ValueError("Fire stage durations must sum to 1.0")
    
    def generate_concentration_time_series(self, 
                                         gas_type: str,
                                         start_time: datetime,
                                         duration: Optional[int] = None,
                                         sample_rate: Optional[float] = None,
                                         release_pattern: Optional[Union[str, ReleasePattern]] = None,
                                         params: Optional[Dict[str, Any]] = None,
                                         seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a time series of gas concentrations.
        
        Args:
            gas_type: Type of gas
            start_time: Start time of the time series
            duration: Duration in seconds (default: from config)
            sample_rate: Sample rate in Hz (default: from config)
            release_pattern: Gas release pattern (default: from config)
            params: Additional parameters for the release pattern
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing the generated time series:
            - concentrations: List of gas concentrations in ppm
            - timestamps: List of timestamps
            - metadata: Dictionary with metadata about the time series
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set default values if not provided
        if duration is None:
            duration = self.default_duration
        
        if sample_rate is None:
            sample_rate = self.default_sample_rate
        
        if release_pattern is None:
            pattern_enum = self.default_release_pattern
        elif isinstance(release_pattern, str):
            pattern_enum = ReleasePattern(release_pattern)
        else:
            pattern_enum = release_pattern
        
        # Default parameters if not provided
        if params is None:
            params = {}
        
        # Set default parameters based on release pattern
        if pattern_enum == ReleasePattern.SUDDEN:
            default_params = {
                'peak_concentration': 1000.0,  # ppm
                'rise_time': 10.0,  # seconds
                'decay_rate': 0.01,  # concentration units per second
                'background': 0.0   # ppm
            }
        elif pattern_enum == ReleasePattern.GRADUAL:
            default_params = {
                'max_concentration': 500.0,  # ppm
                'rise_rate': 2.0,   # ppm per second
                'plateau_duration': duration * 0.5,  # seconds
                'decay_rate': 1.0,  # ppm per second
                'background': 0.0   # ppm
            }
        elif pattern_enum == ReleasePattern.INTERMITTENT:
            default_params = {
                'peak_concentration': 800.0,  # ppm
                'frequency': 0.01,  # Hz (cycles per second)
                'duty_cycle': 0.3,  # fraction of cycle that is "on"
                'rise_time': 5.0,   # seconds
                'decay_time': 15.0, # seconds
                'background': 0.0   # ppm
            }
        elif pattern_enum == ReleasePattern.EXPONENTIAL:
            default_params = {
                'initial_concentration': 10.0,  # ppm
                'growth_rate': 0.005,  # exponential growth rate
                'max_concentration': 1000.0,  # ppm
                'background': 0.0   # ppm
            }
        elif pattern_enum == ReleasePattern.LOGARITHMIC:
            default_params = {
                'max_concentration': 1000.0,  # ppm
                'growth_factor': 100.0,  # controls the steepness of the curve
                'background': 0.0   # ppm
            }
        elif pattern_enum == ReleasePattern.SINUSOIDAL:
            default_params = {
                'mean_concentration': 500.0,  # ppm
                'amplitude': 300.0,  # ppm
                'period': 120.0,  # seconds
                'phase_shift': 0.0,  # radians
                'background': 0.0   # ppm
            }
        else:  # CUSTOM or any other pattern
            default_params = {
                'max_concentration': 500.0,  # ppm
                'background': 0.0   # ppm
            }
        
        # Merge default parameters with provided parameters
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        
        # Calculate number of samples
        num_samples = int(duration * sample_rate)
        
        # Generate time points
        time_points = np.linspace(0, duration, num_samples)
        
        # Generate timestamps
        timestamps = [start_time + timedelta(seconds=t) for t in time_points]
        
        # Generate concentrations based on release pattern
        if pattern_enum == ReleasePattern.SUDDEN:
            concentrations = self._generate_sudden_release(time_points, params)
        elif pattern_enum == ReleasePattern.GRADUAL:
            concentrations = self._generate_gradual_release(time_points, params)
        elif pattern_enum == ReleasePattern.INTERMITTENT:
            concentrations = self._generate_intermittent_release(time_points, params)
        elif pattern_enum == ReleasePattern.EXPONENTIAL:
            concentrations = self._generate_exponential_release(time_points, params)
        elif pattern_enum == ReleasePattern.LOGARITHMIC:
            concentrations = self._generate_logarithmic_release(time_points, params)
        elif pattern_enum == ReleasePattern.SINUSOIDAL:
            concentrations = self._generate_sinusoidal_release(time_points, params)
        elif pattern_enum == ReleasePattern.CUSTOM:
            concentrations = self._generate_custom_release(time_points, params)
        else:
            # Default to gradual release
            concentrations = self._generate_gradual_release(time_points, params)
        
        # Create metadata
        metadata = {
            'gas_type': gas_type,
            'release_pattern': pattern_enum.value,
            'duration': duration,
            'sample_rate': sample_rate,
            'num_samples': num_samples,
            'start_time': start_time.isoformat(),
            'end_time': (start_time + timedelta(seconds=duration)).isoformat(),
            'parameters': params,
            'seed': seed
        }
        
        return {
            'concentrations': concentrations,
            'timestamps': timestamps,
            'metadata': metadata
        }
    
    def _generate_sudden_release(self, time_points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Generate a sudden release pattern.
        
        Args:
            time_points: Array of time points in seconds
            params: Parameters for the release pattern
            
        Returns:
            Array of gas concentrations
        """
        peak = params['peak_concentration']
        rise_time = params['rise_time']
        decay_rate = params['decay_rate']
        background = params['background']
        
        concentrations = np.zeros_like(time_points) + background
        
        # Find indices for rise phase
        rise_indices = time_points <= rise_time
        
        # Linear rise to peak
        if np.any(rise_indices):
            concentrations[rise_indices] = background + (peak - background) * (time_points[rise_indices] / rise_time)
        
        # Exponential decay after peak
        decay_indices = time_points > rise_time
        if np.any(decay_indices):
            decay_times = time_points[decay_indices] - rise_time
            concentrations[decay_indices] = peak * np.exp(-decay_rate * decay_times) + background
        
        return concentrations
    
    def _generate_gradual_release(self, time_points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Generate a gradual release pattern.
        
        Args:
            time_points: Array of time points in seconds
            params: Parameters for the release pattern
            
        Returns:
            Array of gas concentrations
        """
        max_conc = params['max_concentration']
        rise_rate = params['rise_rate']
        plateau_duration = params['plateau_duration']
        decay_rate = params['decay_rate']
        background = params['background']
        
        # Calculate time to reach maximum concentration
        rise_time = max_conc / rise_rate if rise_rate > 0 else 0
        
        # Calculate time when decay starts
        decay_start = rise_time + plateau_duration
        
        # Calculate time when concentration returns to background
        decay_time = max_conc / decay_rate if decay_rate > 0 else 0
        total_time = decay_start + decay_time
        
        concentrations = np.zeros_like(time_points) + background
        
        # Rise phase
        rise_indices = (time_points <= rise_time) & (time_points >= 0)
        if np.any(rise_indices):
            concentrations[rise_indices] = background + rise_rate * time_points[rise_indices]
        
        # Plateau phase
        plateau_indices = (time_points > rise_time) & (time_points <= decay_start)
        if np.any(plateau_indices):
            concentrations[plateau_indices] = max_conc + background
        
        # Decay phase
        decay_indices = (time_points > decay_start) & (time_points <= total_time)
        if np.any(decay_indices):
            decay_times = time_points[decay_indices] - decay_start
            concentrations[decay_indices] = max_conc - decay_rate * decay_times + background
            concentrations[decay_indices] = np.maximum(concentrations[decay_indices], background)
        
        # After total time, concentration is at background
        after_indices = time_points > total_time
        if np.any(after_indices):
            concentrations[after_indices] = background
        
        return concentrations
    
    def _generate_intermittent_release(self, time_points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Generate an intermittent release pattern.
        
        Args:
            time_points: Array of time points in seconds
            params: Parameters for the release pattern
            
        Returns:
            Array of gas concentrations
        """
        peak = params['peak_concentration']
        frequency = params['frequency']
        duty_cycle = params['duty_cycle']
        rise_time = params['rise_time']
        decay_time = params['decay_time']
        background = params['background']
        
        # Calculate period
        period = 1.0 / frequency if frequency > 0 else float('inf')
        
        # Calculate on and off times
        on_time = period * duty_cycle
        
        concentrations = np.zeros_like(time_points) + background
        
        # Process each cycle
        for i in range(int(np.max(time_points) / period) + 1):
            cycle_start = i * period
            cycle_end = (i + 1) * period
            
            # On phase start and end
            on_start = cycle_start
            on_end = cycle_start + on_time
            
            # Rise phase
            rise_end = on_start + rise_time
            rise_indices = (time_points >= on_start) & (time_points < rise_end) & (time_points < cycle_end)
            if np.any(rise_indices):
                rise_times = time_points[rise_indices] - on_start
                concentrations[rise_indices] = background + (peak - background) * (rise_times / rise_time)
            
            # Plateau phase
            plateau_end = on_end - decay_time
            plateau_indices = (time_points >= rise_end) & (time_points < plateau_end) & (time_points < cycle_end)
            if np.any(plateau_indices):
                concentrations[plateau_indices] = peak
            
            # Decay phase
            decay_indices = (time_points >= plateau_end) & (time_points < on_end) & (time_points < cycle_end)
            if np.any(decay_indices):
                decay_times = time_points[decay_indices] - plateau_end
                concentrations[decay_indices] = peak * (1 - decay_times / decay_time)
                concentrations[decay_indices] = np.maximum(concentrations[decay_indices], background)
        
        return concentrations
    
    def _generate_exponential_release(self, time_points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Generate an exponential release pattern.
        
        Args:
            time_points: Array of time points in seconds
            params: Parameters for the release pattern
            
        Returns:
            Array of gas concentrations
        """
        initial = params['initial_concentration']
        growth_rate = params['growth_rate']
        max_conc = params['max_concentration']
        background = params['background']
        
        # Exponential growth with cap at max_concentration
        concentrations = initial * np.exp(growth_rate * time_points) + background
        concentrations = np.minimum(concentrations, max_conc + background)
        
        return concentrations
    
    def _generate_logarithmic_release(self, time_points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Generate a logarithmic release pattern.
        
        Args:
            time_points: Array of time points in seconds
            params: Parameters for the release pattern
            
        Returns:
            Array of gas concentrations
        """
        max_conc = params['max_concentration']
        growth_factor = params['growth_factor']
        background = params['background']
        
        # Avoid log(0)
        adjusted_times = time_points + 1.0
        
        # Logarithmic growth
        concentrations = max_conc * np.log(adjusted_times * growth_factor) / np.log(growth_factor * np.max(adjusted_times)) + background
        
        # Ensure non-negative concentrations
        concentrations = np.maximum(concentrations, background)
        
        return concentrations
    
    def _generate_sinusoidal_release(self, time_points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Generate a sinusoidal release pattern.
        
        Args:
            time_points: Array of time points in seconds
            params: Parameters for the release pattern
            
        Returns:
            Array of gas concentrations
        """
        mean = params['mean_concentration']
        amplitude = params['amplitude']
        period = params['period']
        phase_shift = params['phase_shift']
        background = params['background']
        
        # Calculate frequency in radians per second
        omega = 2 * np.pi / period
        
        # Generate sinusoidal pattern
        concentrations = mean + amplitude * np.sin(omega * time_points + phase_shift) + background
        
        # Ensure non-negative concentrations
        concentrations = np.maximum(concentrations, background)
        
        return concentrations
    
    def _generate_custom_release(self, time_points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Generate a custom release pattern.
        
        This is a placeholder for custom patterns defined by the user.
        
        Args:
            time_points: Array of time points in seconds
            params: Parameters for the release pattern
            
        Returns:
            Array of gas concentrations
        """
        max_conc = params['max_concentration']
        background = params['background']
        
        # Example: combination of patterns
        # This can be customized based on specific requirements
        
        # First half: gradual increase
        half_time = np.max(time_points) / 2
        first_half = time_points <= half_time
        second_half = time_points > half_time
        
        concentrations = np.zeros_like(time_points) + background
        
        if np.any(first_half):
            # Linear increase in first half
            concentrations[first_half] = background + (max_conc - background) * (time_points[first_half] / half_time)
        
        if np.any(second_half):
            # Exponential decay in second half
            decay_rate = 2.0 / half_time  # Decay to ~13% of max in half time
            decay_times = time_points[second_half] - half_time
            concentrations[second_half] = max_conc * np.exp(-decay_rate * decay_times) + background
        
        return concentrations
    
    def generate_fire_scenario_concentrations(self, 
                                            gas_types: List[str],
                                            start_time: datetime,
                                            duration: int,
                                            sample_rate: float,
                                            fire_type: str,
                                            fuel_load: float,
                                            room_volume: float,
                                            ventilation_rate: float,
                                            seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate gas concentrations for a fire scenario.
        
        Args:
            gas_types: List of gas types to simulate
            start_time: Start time of the scenario
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            fire_type: Type of fire (e.g., 'smoldering', 'flaming')
            fuel_load: Fuel load in kg
            room_volume: Room volume in m³
            ventilation_rate: Ventilation rate in air changes per hour
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing the generated concentrations for each gas type
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate number of samples
        num_samples = int(duration * sample_rate)
        
        # Generate time points
        time_points = np.linspace(0, duration, num_samples)
        
        # Generate timestamps
        timestamps = [start_time + timedelta(seconds=t) for t in time_points]
        
        # Determine fire stages based on time
        fire_stages = self._determine_fire_stages(time_points, duration)
        
        # Generate concentrations for each gas type
        gas_data = {}
        
        for gas_type in gas_types:
            # Check if gas properties are available
            if gas_type not in self.gas_properties:
                continue
            
            # Get gas properties
            gas_props = self.gas_properties[gas_type]
            
            # Calculate production rate based on fire type
            base_rate = gas_props['production_rate']
            
            if fire_type == 'smoldering':
                # Smoldering fires produce more CO and less methane
                if gas_type == 'carbon_monoxide':
                    production_rate = base_rate * 3.0
                elif gas_type == 'methane':
                    production_rate = base_rate * 0.5
                else:
                    production_rate = base_rate
            elif fire_type == 'electrical':
                # Electrical fires might produce different gases
                if gas_type == 'hydrogen':
                    production_rate = base_rate * 2.0
                else:
                    production_rate = base_rate
            elif fire_type == 'chemical':
                # Chemical fires might produce more varied gases
                production_rate = base_rate * 1.5
            else:  # Default flaming fire
                production_rate = base_rate
            
            # Calculate concentrations based on fire stages
            concentrations = np.zeros_like(time_points)
            
            for i, (stage, t) in enumerate(zip(fire_stages, time_points)):
                # Calculate stage-specific production rate
                if stage == FireStage.INCIPIENT:
                    stage_factor = 0.2
                elif stage == FireStage.GROWTH:
                    stage_factor = 0.6 + 0.4 * (t / (duration * self.fire_stage_durations[FireStage.GROWTH]))
                elif stage == FireStage.FULLY_DEVELOPED:
                    stage_factor = 1.0
                elif stage == FireStage.DECAY:
                    decay_start = duration * (1.0 - self.fire_stage_durations[FireStage.DECAY])
                    decay_progress = (t - decay_start) / (duration - decay_start)
                    stage_factor = 1.0 - 0.8 * decay_progress
                else:
                    stage_factor = 0.0
                
                # Calculate gas production at this time point (g/s)
                gas_production = production_rate * fuel_load * stage_factor
                
                # Convert to concentration increase (ppm)
                # ppm = (mass * 24.45) / (molecular_weight * volume)
                # where 24.45 is the molar volume at 25°C and 1 atm in L/mol
                
                # Calculate cumulative mass produced up to this point (g)
                if i > 0:
                    time_step = time_points[i] - time_points[i-1]
                    mass_produced = gas_production * time_step
                    
                    # Convert to moles
                    moles_produced = mass_produced / gas_props['molecular_weight']
                    
                    # Convert to volume at STP (L)
                    volume_produced = moles_produced * 24.45
                    
                    # Convert to concentration (ppm)
                    # 1 ppm = 1 mL/m³
                    conc_increase = (volume_produced * 1000) / room_volume  # mL/m³ = ppm
                    
                    # Account for ventilation (simplified)
                    ventilation_factor = np.exp(-ventilation_rate * time_points[i] / 3600)
                    
                    # Add to concentration
                    concentrations[i] = concentrations[i-1] + conc_increase
                    
                    # Apply ventilation effect
                    concentrations[i] *= ventilation_factor
            
            # Add some random variations
            noise = np.random.normal(0, 0.05 * np.mean(concentrations), num_samples)
            concentrations += noise
            
            # Ensure non-negative concentrations
            concentrations = np.maximum(concentrations, 0)
            
            # Store results
            gas_data[gas_type] = {
                'concentrations': concentrations,
                'timestamps': timestamps,
                'metadata': {
                    'gas_type': gas_type,
                    'fire_type': fire_type,
                    'fuel_load': fuel_load,
                    'room_volume': room_volume,
                    'ventilation_rate': ventilation_rate,
                    'production_rate': production_rate,
                    'molecular_weight': gas_props['molecular_weight'],
                    'flammability_limits': gas_props['flammability_limits'],
                    'ignition_temp': gas_props['ignition_temp']
                }
            }
        
        return {
            'gas_data': gas_data,
            'timestamps': timestamps,
            'fire_stages': [stage.value for stage in fire_stages],
            'metadata': {
                'start_time': start_time.isoformat(),
                'duration': duration,
                'sample_rate': sample_rate,
                'num_samples': num_samples,
                'fire_type': fire_type,
                'fuel_load': fuel_load,
                'room_volume': room_volume,
                'ventilation_rate': ventilation_rate,
                'seed': seed
            }
        }
    
    def _determine_fire_stages(self, time_points: np.ndarray, duration: float) -> List[FireStage]:
        """
        Determine the fire stage for each time point.
        
        Args:
            time_points: Array of time points in seconds
            duration: Total duration in seconds
            
        Returns:
            List of fire stages for each time point
        """
        stages = []
        
        # Calculate stage boundaries
        incipient_end = duration * self.fire_stage_durations[FireStage.INCIPIENT]
        growth_end = incipient_end + duration * self.fire_stage_durations[FireStage.GROWTH]
        fully_developed_end = growth_end + duration * self.fire_stage_durations[FireStage.FULLY_DEVELOPED]
        
        for t in time_points:
            if t <= incipient_end:
                stages.append(FireStage.INCIPIENT)
            elif t <= growth_end:
                stages.append(FireStage.GROWTH)
            elif t <= fully_developed_end:
                stages.append(FireStage.FULLY_DEVELOPED)
            else:
                stages.append(FireStage.DECAY)
        
        return stages
    
    def coordinate_with_thermal(self, 
                              gas_data: Dict[str, Any], 
                              thermal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate gas concentration evolution with thermal data.
        
        Args:
            gas_data: Gas concentration data
            thermal_data: Thermal image data
            
        Returns:
            Updated gas concentration data coordinated with thermal data
        """
        # Extract thermal metadata
        thermal_metadata = thermal_data.get('metadata', {})
        thermal_frames = thermal_data.get('frames', [])
        thermal_timestamps = thermal_data.get('timestamps', [])
        
        if not thermal_frames or not thermal_timestamps:
            return gas_data
        
        # Extract gas data
        gas_types = list(gas_data.get('gas_data', {}).keys())
        gas_timestamps = gas_data.get('timestamps', [])
        
        if not gas_types or not gas_timestamps:
            return gas_data
        
        # Align timestamps if needed
        if len(gas_timestamps) != len(thermal_timestamps):
            # Resample gas data to match thermal timestamps
            for gas_type in gas_types:
                old_concentrations = gas_data['gas_data'][gas_type]['concentrations']
                new_concentrations = np.interp(
                    [ts.timestamp() for ts in thermal_timestamps],
                    [ts.timestamp() for ts in gas_timestamps],
                    old_concentrations
                )
                gas_data['gas_data'][gas_type]['concentrations'] = new_concentrations
                gas_data['gas_data'][gas_type]['timestamps'] = thermal_timestamps
            
            gas_data['timestamps'] = thermal_timestamps
        
        # Adjust gas concentrations based on thermal data
        for i, (thermal_frame, timestamp) in enumerate(zip(thermal_frames, thermal_timestamps)):
            # Extract temperature statistics from thermal frame
            max_temp = np.max(thermal_frame)
            mean_temp = np.mean(thermal_frame)
            
            # Calculate temperature factor (higher temperatures produce more gas)
            temp_factor = 1.0 + 0.01 * (max_temp - 25.0)  # 1% increase per degree above 25°C
            
            # Apply temperature factor to gas concentrations
            for gas_type in gas_types:
                # Different gases respond differently to temperature
                if gas_type == 'methane':
                    gas_temp_factor = temp_factor ** 1.2
                elif gas_type == 'carbon_monoxide':
                    gas_temp_factor = temp_factor ** 1.5
                elif gas_type == 'hydrogen':
                    gas_temp_factor = temp_factor ** 0.8
                else:
                    gas_temp_factor = temp_factor
                
                # Apply factor if within bounds
                if i < len(gas_data['gas_data'][gas_type]['concentrations']):
                    gas_data['gas_data'][gas_type]['concentrations'][i] *= gas_temp_factor
        
        # Update metadata to indicate coordination with thermal data
        gas_data['metadata']['coordinated_with_thermal'] = True
        gas_data['metadata']['thermal_data_reference'] = thermal_metadata.get('start_time', '')
        
        return gas_data