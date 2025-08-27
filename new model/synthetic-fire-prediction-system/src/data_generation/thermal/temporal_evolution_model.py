"""
Temporal evolution model for fire progression simulation.

This module provides functionality for simulating realistic fire progression
over time, including different fire growth patterns and behaviors.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import math

from .hotspot_simulator import HotspotSimulator, HotspotShape, GrowthPattern


class FireType(Enum):
    """Enumeration of supported fire types."""
    SMOLDERING = "smoldering"
    RAPID_COMBUSTION = "rapid_combustion"
    ELECTRICAL = "electrical"
    CHEMICAL = "chemical"
    STANDARD = "standard"


class FireStage(Enum):
    """Enumeration of fire development stages."""
    INCIPIENT = "incipient"
    GROWTH = "growth"
    FULLY_DEVELOPED = "fully_developed"
    DECAY = "decay"


class TemporalEvolutionModel:
    """
    Class for simulating realistic fire progression over time.
    
    This class provides methods for generating time-series of thermal images
    showing fire development with different growth patterns and behaviors.
    """
    
    def __init__(self, config: Dict[str, Any], hotspot_simulator: HotspotSimulator):
        """
        Initialize the temporal evolution model with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
            hotspot_simulator: HotspotSimulator instance for generating hotspots
        """
        self.config = config
        self.hotspot_simulator = hotspot_simulator
        self.validate_config()
        
        # Set default values
        self.default_fire_type = FireType(self.config.get('default_fire_type', 'standard'))
        self.default_duration = self.config.get('default_duration', 600)  # 10 minutes
        self.default_frame_rate = self.config.get('default_frame_rate', 1.0)  # 1 frame per second
        
        # Fire behavior parameters
        self.fire_params = self._initialize_fire_parameters()
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check fire type
        if 'default_fire_type' in self.config:
            try:
                FireType(self.config['default_fire_type'])
            except ValueError:
                valid_types = [fire_type.value for fire_type in FireType]
                raise ValueError(f"Invalid default fire type. Must be one of: {valid_types}")
        
        # Check duration
        if 'default_duration' in self.config and self.config['default_duration'] <= 0:
            raise ValueError("default_duration must be positive")
        
        # Check frame rate
        if 'default_frame_rate' in self.config and self.config['default_frame_rate'] <= 0:
            raise ValueError("default_frame_rate must be positive")
    
    def _initialize_fire_parameters(self) -> Dict[FireType, Dict[str, Any]]:
        """
        Initialize parameters for different fire types.
        
        Returns:
            Dictionary mapping fire types to their parameters
        """
        params = {}
        
        # Smoldering fire (slow, steady growth, lower temperatures)
        params[FireType.SMOLDERING] = {
            'growth_rate': 0.3,  # Slow growth
            'max_temperature_factor': 0.6,  # Lower max temperature
            'temperature_variance': 0.1,  # Low variance
            'hotspot_count': lambda t: 1 + int(t * 0.5),  # Slow increase in hotspots
            'hotspot_size_factor': 1.2,  # Larger hotspots
            'preferred_shapes': [HotspotShape.CIRCULAR, HotspotShape.IRREGULAR],
            'preferred_growth': GrowthPattern.LOGARITHMIC,
            'stage_durations': {  # As fractions of total duration
                FireStage.INCIPIENT: 0.3,
                FireStage.GROWTH: 0.3,
                FireStage.FULLY_DEVELOPED: 0.3,
                FireStage.DECAY: 0.1
            }
        }
        
        # Rapid combustion fire (fast growth, high temperatures)
        params[FireType.RAPID_COMBUSTION] = {
            'growth_rate': 1.5,  # Fast growth
            'max_temperature_factor': 1.0,  # Full temperature range
            'temperature_variance': 0.2,  # Higher variance
            'hotspot_count': lambda t: 1 + int(t * 2),  # Rapid increase in hotspots
            'hotspot_size_factor': 0.8,  # Smaller, more intense hotspots
            'preferred_shapes': [HotspotShape.CIRCULAR, HotspotShape.ELLIPTICAL],
            'preferred_growth': GrowthPattern.EXPONENTIAL,
            'stage_durations': {
                FireStage.INCIPIENT: 0.1,
                FireStage.GROWTH: 0.4,
                FireStage.FULLY_DEVELOPED: 0.4,
                FireStage.DECAY: 0.1
            }
        }
        
        # Electrical fire (focused hotspots, medium growth)
        params[FireType.ELECTRICAL] = {
            'growth_rate': 0.8,  # Medium growth
            'max_temperature_factor': 0.9,  # High temperature
            'temperature_variance': 0.15,  # Medium variance
            'hotspot_count': lambda t: 1 + int(t * 0.7),  # Medium increase in hotspots
            'hotspot_size_factor': 0.6,  # Smaller hotspots (focused heat)
            'preferred_shapes': [HotspotShape.POINT, HotspotShape.CIRCULAR],
            'preferred_growth': GrowthPattern.SIGMOID,
            'stage_durations': {
                FireStage.INCIPIENT: 0.2,
                FireStage.GROWTH: 0.3,
                FireStage.FULLY_DEVELOPED: 0.4,
                FireStage.DECAY: 0.1
            }
        }
        
        # Chemical fire (irregular shapes, rapid growth)
        params[FireType.CHEMICAL] = {
            'growth_rate': 1.2,  # Fast growth
            'max_temperature_factor': 1.0,  # Full temperature range
            'temperature_variance': 0.25,  # High variance
            'hotspot_count': lambda t: 1 + int(t * 1.5),  # Fast increase in hotspots
            'hotspot_size_factor': 1.0,  # Medium-sized hotspots
            'preferred_shapes': [HotspotShape.IRREGULAR, HotspotShape.ELLIPTICAL],
            'preferred_growth': GrowthPattern.EXPONENTIAL,
            'stage_durations': {
                FireStage.INCIPIENT: 0.1,
                FireStage.GROWTH: 0.3,
                FireStage.FULLY_DEVELOPED: 0.5,
                FireStage.DECAY: 0.1
            }
        }
        
        # Standard fire (balanced parameters)
        params[FireType.STANDARD] = {
            'growth_rate': 1.0,  # Standard growth
            'max_temperature_factor': 0.8,  # Standard temperature
            'temperature_variance': 0.15,  # Standard variance
            'hotspot_count': lambda t: 1 + int(t),  # Linear increase in hotspots
            'hotspot_size_factor': 1.0,  # Standard-sized hotspots
            'preferred_shapes': [HotspotShape.CIRCULAR, HotspotShape.ELLIPTICAL, HotspotShape.IRREGULAR],
            'preferred_growth': GrowthPattern.SIGMOID,
            'stage_durations': {
                FireStage.INCIPIENT: 0.2,
                FireStage.GROWTH: 0.3,
                FireStage.FULLY_DEVELOPED: 0.4,
                FireStage.DECAY: 0.1
            }
        }
        
        return params
    
    def determine_fire_stage(self, 
                           elapsed_time: float, 
                           total_duration: float,
                           fire_type: FireType) -> FireStage:
        """
        Determine the current fire stage based on elapsed time.
        
        Args:
            elapsed_time: Time elapsed since start in seconds
            total_duration: Total duration of the fire in seconds
            fire_type: Type of fire
            
        Returns:
            Current fire stage
        """
        # Get stage durations for the fire type
        stage_durations = self.fire_params[fire_type]['stage_durations']
        
        # Calculate normalized time
        t_norm = elapsed_time / total_duration
        
        # Determine stage based on cumulative durations
        cumulative = 0.0
        for stage, duration in stage_durations.items():
            cumulative += duration
            if t_norm <= cumulative:
                return stage
        
        # Default to decay stage if we've exceeded the total duration
        return FireStage.DECAY
    
    def generate_fire_sequence(self, 
                             image_shape: Tuple[int, int],
                             start_time: datetime,
                             duration: Optional[int] = None,
                             frame_rate: Optional[float] = None,
                             fire_type: Optional[Union[str, FireType]] = None,
                             seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a sequence of thermal images showing fire progression.
        
        Args:
            image_shape: Shape of the thermal images (height, width)
            start_time: Start time of the fire
            duration: Duration of the fire in seconds (default: from config)
            frame_rate: Frame rate in frames per second (default: from config)
            fire_type: Type of fire (default: from config)
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated sequence:
            - frames: List of thermal images as numpy arrays
            - timestamps: List of timestamps for each frame
            - metadata: Dictionary with metadata about the sequence
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set default values if not provided
        if duration is None:
            duration = self.default_duration
        
        if frame_rate is None:
            frame_rate = self.default_frame_rate
        
        if fire_type is None:
            fire_type_enum = self.default_fire_type
        elif isinstance(fire_type, str):
            fire_type_enum = FireType(fire_type)
        else:
            fire_type_enum = fire_type
        
        # Get fire parameters
        fire_params = self.fire_params[fire_type_enum]
        
        # Calculate number of frames
        num_frames = int(duration * frame_rate)
        
        # Initialize result containers
        frames = []
        timestamps = []
        
        # Generate initial fire configuration
        fire_config = self._generate_fire_configuration(
            image_shape=image_shape,
            fire_type=fire_type_enum,
            duration=duration
        )
        
        # Generate frames
        for i in range(num_frames):
            # Calculate current time
            elapsed_seconds = i / frame_rate
            current_time = start_time + timedelta(seconds=elapsed_seconds)
            
            # Determine fire stage
            stage = self.determine_fire_stage(elapsed_seconds, duration, fire_type_enum)
            
            # Generate frame
            frame = self._generate_fire_frame(
                image_shape=image_shape,
                fire_config=fire_config,
                elapsed_time=elapsed_seconds,
                total_duration=duration,
                fire_type=fire_type_enum,
                stage=stage
            )
            
            frames.append(frame)
            timestamps.append(current_time)
        
        # Create metadata
        metadata = {
            'fire_type': fire_type_enum.value,
            'duration': duration,
            'frame_rate': frame_rate,
            'num_frames': num_frames,
            'image_shape': image_shape,
            'start_time': start_time.isoformat(),
            'end_time': (start_time + timedelta(seconds=duration)).isoformat(),
            'seed': seed
        }
        
        return {
            'frames': frames,
            'timestamps': timestamps,
            'metadata': metadata
        }
    
    def _generate_fire_configuration(self, 
                                   image_shape: Tuple[int, int],
                                   fire_type: FireType,
                                   duration: int) -> Dict[str, Any]:
        """
        Generate initial fire configuration.
        
        Args:
            image_shape: Shape of the thermal images (height, width)
            fire_type: Type of fire
            duration: Duration of the fire in seconds
            
        Returns:
            Dictionary with fire configuration parameters
        """
        h, w = image_shape
        fire_params = self.fire_params[fire_type]
        
        # Calculate maximum number of hotspots
        max_hotspot_count = fire_params['hotspot_count'](1.0)  # At t=1.0 (end of sequence)
        
        # Generate initial hotspot locations
        hotspots = []
        
        # Start with a primary hotspot near the center
        center_y = h // 2 + np.random.randint(-h // 6, h // 6)
        center_x = w // 2 + np.random.randint(-w // 6, w // 6)
        
        primary_hotspot = {
            'center': (center_y, center_x),
            'max_radius': int(min(h, w) // 8 * fire_params['hotspot_size_factor']),
            'max_intensity': 1.0,
            'shape': np.random.choice(fire_params['preferred_shapes']).value,
            'growth_pattern': fire_params['preferred_growth'].value,
            'start_offset': 0,
            'duration': duration
        }
        hotspots.append(primary_hotspot)
        
        # Generate secondary hotspots
        for i in range(1, max_hotspot_count):
            # Calculate when this hotspot appears (as a fraction of total duration)
            appearance_time = i / max_hotspot_count
            
            # Calculate offset from primary hotspot
            distance = np.random.uniform(0.1, 0.5) * min(h, w) // 2
            angle = np.random.uniform(0, 2 * np.pi)
            offset_y = int(distance * np.sin(angle))
            offset_x = int(distance * np.cos(angle))
            
            # Calculate center coordinates
            sec_center_y = center_y + offset_y
            sec_center_x = center_x + offset_x
            
            # Ensure coordinates are within image bounds
            sec_center_y = max(0, min(sec_center_y, h - 1))
            sec_center_x = max(0, min(sec_center_x, w - 1))
            
            # Create secondary hotspot
            secondary_hotspot = {
                'center': (sec_center_y, sec_center_x),
                'max_radius': int(min(h, w) // 12 * fire_params['hotspot_size_factor']),
                'max_intensity': np.random.uniform(0.6, 0.9),
                'shape': np.random.choice(fire_params['preferred_shapes']).value,
                'growth_pattern': fire_params['preferred_growth'].value,
                'start_offset': int(appearance_time * duration * 0.7),  # Appear in first 70% of duration
                'duration': int(duration * (1.0 - appearance_time * 0.7))  # Duration until end
            }
            hotspots.append(secondary_hotspot)
        
        return {
            'hotspots': hotspots,
            'fire_type': fire_type,
            'growth_rate': fire_params['growth_rate'],
            'max_temperature_factor': fire_params['max_temperature_factor'],
            'temperature_variance': fire_params['temperature_variance']
        }
    
    def _generate_fire_frame(self, 
                           image_shape: Tuple[int, int],
                           fire_config: Dict[str, Any],
                           elapsed_time: float,
                           total_duration: float,
                           fire_type: FireType,
                           stage: FireStage) -> np.ndarray:
        """
        Generate a single frame of the fire sequence.
        
        Args:
            image_shape: Shape of the thermal image (height, width)
            fire_config: Fire configuration parameters
            elapsed_time: Time elapsed since start in seconds
            total_duration: Total duration of the fire in seconds
            fire_type: Type of fire
            stage: Current fire stage
            
        Returns:
            Thermal image as a 2D numpy array
        """
        # Get hotspots from fire configuration
        hotspots = fire_config['hotspots']
        
        # Adjust hotspots based on fire stage
        adjusted_hotspots = self._adjust_hotspots_for_stage(
            hotspots=hotspots,
            stage=stage,
            elapsed_time=elapsed_time,
            total_duration=total_duration,
            fire_type=fire_type
        )
        
        # Calculate current time
        start_time = datetime.now()  # Arbitrary start time
        current_time = start_time + timedelta(seconds=elapsed_time)
        
        # Generate frame with multiple evolving hotspots
        frame = self.hotspot_simulator.generate_evolving_multiple_hotspots(
            image_shape=image_shape,
            start_time=start_time,
            current_time=current_time,
            hotspots=adjusted_hotspots
        )
        
        return frame
    
    def _adjust_hotspots_for_stage(self, 
                                 hotspots: List[Dict[str, Any]],
                                 stage: FireStage,
                                 elapsed_time: float,
                                 total_duration: float,
                                 fire_type: FireType) -> List[Dict[str, Any]]:
        """
        Adjust hotspot parameters based on the current fire stage.
        
        Args:
            hotspots: List of hotspot parameters
            stage: Current fire stage
            elapsed_time: Time elapsed since start in seconds
            total_duration: Total duration of the fire in seconds
            fire_type: Type of fire
            
        Returns:
            Adjusted list of hotspot parameters
        """
        adjusted_hotspots = []
        fire_params = self.fire_params[fire_type]
        
        for hotspot in hotspots:
            # Skip hotspots that haven't appeared yet
            if elapsed_time < hotspot['start_offset']:
                continue
            
            # Create a copy of the hotspot parameters
            adjusted = hotspot.copy()
            
            # Apply stage-specific adjustments
            if stage == FireStage.INCIPIENT:
                # Incipient stage: Small, low-intensity hotspots
                adjusted['max_radius'] = int(hotspot['max_radius'] * 0.6)
                adjusted['max_intensity'] = hotspot['max_intensity'] * 0.5
            elif stage == FireStage.GROWTH:
                # Growth stage: Expanding hotspots with increasing intensity
                adjusted['max_radius'] = int(hotspot['max_radius'] * 0.8)
                adjusted['max_intensity'] = hotspot['max_intensity'] * 0.8
            elif stage == FireStage.FULLY_DEVELOPED:
                # Fully developed stage: Maximum size and intensity
                adjusted['max_radius'] = hotspot['max_radius']
                adjusted['max_intensity'] = hotspot['max_intensity']
            elif stage == FireStage.DECAY:
                # Decay stage: Maintaining size but decreasing intensity
                adjusted['max_radius'] = hotspot['max_radius']
                adjusted['max_intensity'] = hotspot['max_intensity'] * 0.7
                
                # Add some randomness to simulate uneven cooling
                if np.random.random() < 0.3:  # 30% chance
                    adjusted['max_intensity'] *= np.random.uniform(0.7, 1.0)
            
            adjusted_hotspots.append(adjusted)
        
        return adjusted_hotspots
    
    def generate_fire_evolution_dataset(self, 
                                      image_shape: Tuple[int, int],
                                      num_sequences: int,
                                      output_dir: str,
                                      fire_types: Optional[List[Union[str, FireType]]] = None,
                                      duration_range: Optional[Tuple[int, int]] = None,
                                      frame_rate: Optional[float] = None,
                                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a dataset of fire evolution sequences.
        
        Args:
            image_shape: Shape of the thermal images (height, width)
            num_sequences: Number of sequences to generate
            output_dir: Directory to save the sequences
            fire_types: List of fire types to include (default: all types)
            duration_range: Range of durations in seconds (default: 300-900)
            frame_rate: Frame rate in frames per second (default: from config)
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary with dataset metadata
        """
        import os
        import json
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set default values if not provided
        if fire_types is None:
            fire_types = [fire_type for fire_type in FireType]
        else:
            # Convert string fire types to enum
            fire_types = [
                FireType(ft) if isinstance(ft, str) else ft
                for ft in fire_types
            ]
        
        if duration_range is None:
            duration_range = (300, 900)  # 5-15 minutes
        
        if frame_rate is None:
            frame_rate = self.default_frame_rate
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sequences
        metadata = {
            'dataset_name': 'fire_evolution_dataset',
            'num_sequences': num_sequences,
            'image_shape': image_shape,
            'frame_rate': frame_rate,
            'duration_range': duration_range,
            'fire_types': [ft.value for ft in fire_types],
            'sequences': []
        }
        
        for i in range(num_sequences):
            # Select random fire type
            fire_type = np.random.choice(fire_types)
            
            # Select random duration
            duration = np.random.randint(duration_range[0], duration_range[1] + 1)
            
            # Generate start time (random time in the past week)
            start_time = datetime.now() - timedelta(days=np.random.randint(0, 7))
            
            # Generate sequence
            sequence = self.generate_fire_sequence(
                image_shape=image_shape,
                start_time=start_time,
                duration=duration,
                frame_rate=frame_rate,
                fire_type=fire_type,
                seed=seed + i if seed is not None else None
            )
            
            # Save sequence
            sequence_dir = os.path.join(output_dir, f'sequence_{i:04d}')
            os.makedirs(sequence_dir, exist_ok=True)
            
            # Save frames as numpy arrays
            frames_dir = os.path.join(sequence_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            for j, frame in enumerate(sequence['frames']):
                frame_path = os.path.join(frames_dir, f'frame_{j:04d}.npy')
                np.save(frame_path, frame)
            
            # Save timestamps
            timestamps = [ts.isoformat() for ts in sequence['timestamps']]
            timestamps_path = os.path.join(sequence_dir, 'timestamps.json')
            with open(timestamps_path, 'w') as f:
                json.dump(timestamps, f)
            
            # Save metadata
            sequence_metadata = sequence['metadata']
            sequence_metadata['sequence_id'] = i
            sequence_metadata['sequence_dir'] = sequence_dir
            
            metadata_path = os.path.join(sequence_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(sequence_metadata, f)
            
            # Add to dataset metadata
            metadata['sequences'].append(sequence_metadata)
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return metadata