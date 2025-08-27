"""
Scenario mixer for synthetic fire data.

This module provides functionality for combining different scenario types,
creating variations of scenarios with parameter adjustments, and generating
scenario datasets with controlled distributions.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import copy
import random

from .scenario_generator import ScenarioGenerator
from .specific_generators import (
    SpecificScenarioGenerator,
    NormalScenarioGenerator,
    ElectricalFireScenarioGenerator,
    ChemicalFireScenarioGenerator,
    SmolderingFireScenarioGenerator,
    RapidCombustionScenarioGenerator
)
from .false_positive_generator import FalsePositiveGenerator
from ..thermal.thermal_image_generator import ThermalImageGenerator
from ..gas.gas_concentration_generator import GasConcentrationGenerator
from ..environmental.environmental_data_generator import EnvironmentalDataGenerator


class ScenarioMixer:
    """
    Class for mixing and combining different scenario types.
    
    This class provides functionality for combining different scenario types,
    creating variations of scenarios with parameter adjustments, and generating
    scenario datasets with controlled distributions.
    """
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
                 environmental_generator: EnvironmentalDataGenerator,
                 config: Dict[str, Any]):
        """
        Initialize the scenario mixer.
        
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
        
        # Initialize scenario generators
        self._initialize_generators()
    
    def _initialize_generators(self) -> None:
        """
        Initialize all scenario generators.
        """
        # Base scenario generator
        self.scenario_generator = ScenarioGenerator(
            self.thermal_generator,
            self.gas_generator,
            self.environmental_generator,
            self.config
        )
        
        # Specific scenario generators
        self.normal_generator = NormalScenarioGenerator(
            self.thermal_generator,
            self.gas_generator,
            self.environmental_generator,
            self.config
        )
        
        self.electrical_fire_generator = ElectricalFireScenarioGenerator(
            self.thermal_generator,
            self.gas_generator,
            self.environmental_generator,
            self.config
        )
        
        self.chemical_fire_generator = ChemicalFireScenarioGenerator(
            self.thermal_generator,
            self.gas_generator,
            self.environmental_generator,
            self.config
        )
        
        self.smoldering_fire_generator = SmolderingFireScenarioGenerator(
            self.thermal_generator,
            self.gas_generator,
            self.environmental_generator,
            self.config
        )
        
        self.rapid_combustion_generator = RapidCombustionScenarioGenerator(
            self.thermal_generator,
            self.gas_generator,
            self.environmental_generator,
            self.config
        )
        
        # False positive generator
        self.false_positive_generator = FalsePositiveGenerator(
            self.thermal_generator,
            self.gas_generator,
            self.environmental_generator,
            self.config
        )
        
        # Map of scenario types to generators
        self.generator_map = {
            'normal': self.normal_generator,
            'electrical_fire': self.electrical_fire_generator,
            'chemical_fire': self.chemical_fire_generator,
            'smoldering_fire': self.smoldering_fire_generator,
            'rapid_combustion': self.rapid_combustion_generator,
            'false_positive': self.false_positive_generator
        }
    
    def create_scenario_variation(self,
                                 base_scenario: Dict[str, Any],
                                 variation_params: Dict[str, Any],
                                 seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a variation of a scenario by adjusting parameters.
        
        Args:
            base_scenario: Base scenario data
            variation_params: Parameters to adjust
            seed: Optional random seed
            
        Returns:
            Varied scenario data
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Extract scenario definition and metadata
        scenario_def = copy.deepcopy(base_scenario['scenario_definition'])
        metadata = copy.deepcopy(base_scenario['metadata'])
        
        # Apply variations to scenario definition
        self._apply_variations(scenario_def, variation_params)
        
        # Generate new scenario with varied parameters
        scenario_type = scenario_def['scenario_type']
        
        # Get the appropriate generator
        generator = self.generator_map.get(scenario_type, self.scenario_generator)
        
        # Generate new scenario
        start_time = datetime.fromisoformat(metadata['start_time']) if isinstance(metadata['start_time'], str) else metadata['start_time']
        
        # Create scenario parameters
        scenario_params = {k: v for k, v in scenario_def.items() if k not in ['scenario_type', 'duration', 'sample_rate']}
        
        # Generate the varied scenario
        varied_scenario = generator.generate_scenario(
            start_time=start_time,
            duration_seconds=scenario_def['duration'],
            sample_rate_hz=scenario_def['sample_rate'],
            scenario_type=scenario_type,
            scenario_params=scenario_params,
            seed=seed
        )
        
        # Add variation metadata
        varied_scenario['metadata']['variation_of'] = metadata.get('scenario_id', 'unknown')
        varied_scenario['metadata']['variation_params'] = variation_params
        
        return varied_scenario
    
    def _apply_variations(self, scenario_def: Dict[str, Any], variation_params: Dict[str, Any]) -> None:
        """
        Apply variations to a scenario definition.
        
        Args:
            scenario_def: Scenario definition to modify
            variation_params: Parameters to adjust
        """
        for param_path, variation in variation_params.items():
            # Parse parameter path (e.g., 'thermal_params.max_temperature')
            path_parts = param_path.split('.')
            
            # Navigate to the parameter
            current = scenario_def
            for i, part in enumerate(path_parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Apply variation to the parameter
            param_name = path_parts[-1]
            if param_name in current:
                if isinstance(variation, dict):
                    if 'factor' in variation:
                        # Multiplicative factor
                        current[param_name] *= variation['factor']
                    elif 'delta' in variation:
                        # Additive delta
                        current[param_name] += variation['delta']
                    elif 'value' in variation:
                        # Direct value
                        current[param_name] = variation['value']
                    elif 'min' in variation and 'max' in variation:
                        # Random value in range
                        current[param_name] = np.random.uniform(variation['min'], variation['max'])
                else:
                    # Direct value
                    current[param_name] = variation
            else:
                # Create new parameter
                current[param_name] = variation
    
    def combine_scenarios(self,
                         scenarios: List[Dict[str, Any]],
                         weights: Optional[List[float]] = None,
                         combination_method: str = 'weighted_average',
                         seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Combine multiple scenarios into a single scenario.
        
        Args:
            scenarios: List of scenarios to combine
            weights: Optional list of weights for each scenario
            combination_method: Method to use for combining ('weighted_average', 'max', 'sequential')
            seed: Optional random seed
            
        Returns:
            Combined scenario data
        """
        if not scenarios:
            raise ValueError("No scenarios provided to combine")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0 / len(scenarios)] * len(scenarios)
        elif len(weights) != len(scenarios):
            raise ValueError("Number of weights must match number of scenarios")
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        if combination_method == 'weighted_average':
            return self._combine_weighted_average(scenarios, weights, seed)
        elif combination_method == 'max':
            return self._combine_max(scenarios, seed)
        elif combination_method == 'sequential':
            return self._combine_sequential(scenarios, weights, seed)
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
    
    def _combine_weighted_average(self,
                                scenarios: List[Dict[str, Any]],
                                weights: List[float],
                                seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Combine scenarios using weighted average.
        
        Args:
            scenarios: List of scenarios to combine
            weights: List of weights for each scenario
            seed: Optional random seed
            
        Returns:
            Combined scenario data
        """
        # Extract base scenario for structure
        base_scenario = scenarios[0]
        
        # Create combined scenario definition
        combined_def = copy.deepcopy(base_scenario['scenario_definition'])
        combined_def['scenario_type'] = 'combined'
        
        # Combine thermal data
        thermal_frames = []
        for i, scenario in enumerate(scenarios):
            thermal_data = scenario['thermal_data']
            frames = thermal_data['frames']
            
            # Apply weight to each frame
            weighted_frames = [frame * weights[i] for frame in frames]
            thermal_frames.append(weighted_frames)
        
        # Average the frames
        combined_frames = []
        for frame_idx in range(len(thermal_frames[0])):
            combined_frame = sum(thermal_frames[i][frame_idx] for i in range(len(scenarios)))
            combined_frames.append(combined_frame)
        
        # Create combined thermal data
        combined_thermal = copy.deepcopy(base_scenario['thermal_data'])
        combined_thermal['frames'] = combined_frames
        
        # Combine gas data
        combined_gas = copy.deepcopy(base_scenario['gas_data'])
        for gas_type in combined_gas['gas_data']:
            if 'measured_concentrations' in combined_gas['gas_data'][gas_type]:
                combined_gas['gas_data'][gas_type]['measured_concentrations'] = np.zeros_like(
                    combined_gas['gas_data'][gas_type]['measured_concentrations']
                )
                
                for i, scenario in enumerate(scenarios):
                    if (gas_type in scenario['gas_data']['gas_data'] and
                        'measured_concentrations' in scenario['gas_data']['gas_data'][gas_type]):
                        combined_gas['gas_data'][gas_type]['measured_concentrations'] += (
                            scenario['gas_data']['gas_data'][gas_type]['measured_concentrations'] * weights[i]
                        )
        
        # Combine environmental data
        combined_env = copy.deepcopy(base_scenario['environmental_data'])
        for param in combined_env['environmental_data']:
            if 'values' in combined_env['environmental_data'][param]:
                combined_env['environmental_data'][param]['values'] = np.zeros_like(
                    combined_env['environmental_data'][param]['values']
                )
                
                for i, scenario in enumerate(scenarios):
                    if (param in scenario['environmental_data']['environmental_data'] and
                        'values' in scenario['environmental_data']['environmental_data'][param]):
                        combined_env['environmental_data'][param]['values'] += (
                            scenario['environmental_data']['environmental_data'][param]['values'] * weights[i]
                        )
        
        # Create combined metadata
        combined_metadata = copy.deepcopy(base_scenario['metadata'])
        combined_metadata['scenario_type'] = 'combined'
        combined_metadata['combination_method'] = 'weighted_average'
        combined_metadata['combined_from'] = [
            scenario['metadata'].get('scenario_id', f'scenario_{i}')
            for i, scenario in enumerate(scenarios)
        ]
        combined_metadata['weights'] = weights.tolist()
        combined_metadata['creation_timestamp'] = datetime.now().isoformat()
        
        # Create combined scenario
        combined_scenario = {
            'thermal_data': combined_thermal,
            'gas_data': combined_gas,
            'environmental_data': combined_env,
            'metadata': combined_metadata,
            'scenario_definition': combined_def
        }
        
        return combined_scenario
    
    def _combine_max(self,
                    scenarios: List[Dict[str, Any]],
                    seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Combine scenarios by taking the maximum values.
        
        Args:
            scenarios: List of scenarios to combine
            seed: Optional random seed
            
        Returns:
            Combined scenario data
        """
        # Extract base scenario for structure
        base_scenario = scenarios[0]
        
        # Create combined scenario definition
        combined_def = copy.deepcopy(base_scenario['scenario_definition'])
        combined_def['scenario_type'] = 'combined'
        
        # Combine thermal data
        thermal_frames = []
        for i, scenario in enumerate(scenarios):
            thermal_data = scenario['thermal_data']
            frames = thermal_data['frames']
            
            if i == 0:
                thermal_frames = copy.deepcopy(frames)
            else:
                for j, frame in enumerate(frames):
                    thermal_frames[j] = np.maximum(thermal_frames[j], frame)
        
        # Create combined thermal data
        combined_thermal = copy.deepcopy(base_scenario['thermal_data'])
        combined_thermal['frames'] = thermal_frames
        
        # Combine gas data
        combined_gas = copy.deepcopy(base_scenario['gas_data'])
        for gas_type in combined_gas['gas_data']:
            if 'measured_concentrations' in combined_gas['gas_data'][gas_type]:
                for i, scenario in enumerate(scenarios):
                    if i == 0:
                        continue
                    
                    if (gas_type in scenario['gas_data']['gas_data'] and
                        'measured_concentrations' in scenario['gas_data']['gas_data'][gas_type]):
                        combined_gas['gas_data'][gas_type]['measured_concentrations'] = np.maximum(
                            combined_gas['gas_data'][gas_type]['measured_concentrations'],
                            scenario['gas_data']['gas_data'][gas_type]['measured_concentrations']
                        )
        
        # Combine environmental data
        combined_env = copy.deepcopy(base_scenario['environmental_data'])
        for param in combined_env['environmental_data']:
            if 'values' in combined_env['environmental_data'][param]:
                for i, scenario in enumerate(scenarios):
                    if i == 0:
                        continue
                    
                    if (param in scenario['environmental_data']['environmental_data'] and
                        'values' in scenario['environmental_data']['environmental_data'][param]):
                        combined_env['environmental_data'][param]['values'] = np.maximum(
                            combined_env['environmental_data'][param]['values'],
                            scenario['environmental_data']['environmental_data'][param]['values']
                        )
        
        # Create combined metadata
        combined_metadata = copy.deepcopy(base_scenario['metadata'])
        combined_metadata['scenario_type'] = 'combined'
        combined_metadata['combination_method'] = 'max'
        combined_metadata['combined_from'] = [
            scenario['metadata'].get('scenario_id', f'scenario_{i}')
            for i, scenario in enumerate(scenarios)
        ]
        combined_metadata['creation_timestamp'] = datetime.now().isoformat()
        
        # Create combined scenario
        combined_scenario = {
            'thermal_data': combined_thermal,
            'gas_data': combined_gas,
            'environmental_data': combined_env,
            'metadata': combined_metadata,
            'scenario_definition': combined_def
        }
        
        return combined_scenario
    
    def _combine_sequential(self,
                          scenarios: List[Dict[str, Any]],
                          weights: List[float],
                          seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Combine scenarios sequentially, with transitions between them.
        
        Args:
            scenarios: List of scenarios to combine
            weights: List of weights for each scenario (determines duration)
            seed: Optional random seed
            
        Returns:
            Combined scenario data
        """
        # Extract base scenario for structure
        base_scenario = scenarios[0]
        
        # Calculate total duration and sample rate
        total_duration = sum(scenario['scenario_definition']['duration'] for scenario in scenarios)
        sample_rate = base_scenario['scenario_definition']['sample_rate']
        
        # Create combined scenario definition
        combined_def = copy.deepcopy(base_scenario['scenario_definition'])
        combined_def['scenario_type'] = 'combined'
        combined_def['duration'] = total_duration
        
        # Calculate number of frames for each scenario
        scenario_frames = []
        for scenario in scenarios:
            duration = scenario['scenario_definition']['duration']
            num_frames = int(duration * sample_rate)
            scenario_frames.append(num_frames)
        
        # Combine thermal data
        combined_frames = []
        combined_timestamps = []
        
        start_time = datetime.fromisoformat(base_scenario['metadata']['start_time']) if isinstance(base_scenario['metadata']['start_time'], str) else base_scenario['metadata']['start_time']
        current_time = start_time
        
        for i, scenario in enumerate(scenarios):
            thermal_data = scenario['thermal_data']
            frames = thermal_data['frames']
            
            # Take frames from this scenario
            num_frames = scenario_frames[i]
            if num_frames > len(frames):
                # Repeat frames if needed
                frames = frames * (num_frames // len(frames) + 1)
            
            combined_frames.extend(frames[:num_frames])
            
            # Generate timestamps
            for j in range(num_frames):
                combined_timestamps.append(current_time + timedelta(seconds=j/sample_rate))
            
            current_time += timedelta(seconds=num_frames/sample_rate)
        
        # Create combined thermal data
        combined_thermal = copy.deepcopy(base_scenario['thermal_data'])
        combined_thermal['frames'] = combined_frames
        combined_thermal['timestamps'] = combined_timestamps
        
        # Combine gas data (more complex, would need to interpolate between scenarios)
        # For simplicity, we'll just concatenate the gas data
        combined_gas = copy.deepcopy(base_scenario['gas_data'])
        for gas_type in combined_gas['gas_data']:
            if 'measured_concentrations' in combined_gas['gas_data'][gas_type]:
                combined_concentrations = []
                
                for i, scenario in enumerate(scenarios):
                    if (gas_type in scenario['gas_data']['gas_data'] and
                        'measured_concentrations' in scenario['gas_data']['gas_data'][gas_type]):
                        
                        concentrations = scenario['gas_data']['gas_data'][gas_type]['measured_concentrations']
                        num_frames = scenario_frames[i]
                        
                        if num_frames > len(concentrations):
                            # Repeat concentrations if needed
                            concentrations = np.tile(concentrations, (num_frames // len(concentrations) + 1))
                        
                        combined_concentrations.extend(concentrations[:num_frames])
                
                combined_gas['gas_data'][gas_type]['measured_concentrations'] = np.array(combined_concentrations)
        
        combined_gas['timestamps'] = combined_timestamps
        
        # Combine environmental data (similar to gas data)
        combined_env = copy.deepcopy(base_scenario['environmental_data'])
        for param in combined_env['environmental_data']:
            if 'values' in combined_env['environmental_data'][param]:
                combined_values = []
                
                for i, scenario in enumerate(scenarios):
                    if (param in scenario['environmental_data']['environmental_data'] and
                        'values' in scenario['environmental_data']['environmental_data'][param]):
                        
                        values = scenario['environmental_data']['environmental_data'][param]['values']
                        num_frames = scenario_frames[i]
                        
                        if num_frames > len(values):
                            # Repeat values if needed
                            values = np.tile(values, (num_frames // len(values) + 1))
                        
                        combined_values.extend(values[:num_frames])
                
                combined_env['environmental_data'][param]['values'] = np.array(combined_values)
        
        combined_env['timestamps'] = combined_timestamps
        
        # Create combined metadata
        combined_metadata = copy.deepcopy(base_scenario['metadata'])
        combined_metadata['scenario_type'] = 'combined'
        combined_metadata['combination_method'] = 'sequential'
        combined_metadata['combined_from'] = [
            scenario['metadata'].get('scenario_id', f'scenario_{i}')
            for i, scenario in enumerate(scenarios)
        ]
        combined_metadata['segment_durations'] = [scenario['scenario_definition']['duration'] for scenario in scenarios]
        combined_metadata['start_time'] = start_time.isoformat()
        combined_metadata['end_time'] = (start_time + timedelta(seconds=total_duration)).isoformat()
        combined_metadata['creation_timestamp'] = datetime.now().isoformat()
        
        # Create combined scenario
        combined_scenario = {
            'thermal_data': combined_thermal,
            'gas_data': combined_gas,
            'environmental_data': combined_env,
            'metadata': combined_metadata,
            'scenario_definition': combined_def
        }
        
        return combined_scenario
    
    def generate_dataset_with_distribution(self,
                                          output_dir: str,
                                          distribution: Dict[str, float],
                                          num_scenarios: int,
                                          duration_seconds: int,
                                          sample_rate_hz: float,
                                          variation_params: Optional[Dict[str, Dict[str, Any]]] = None,
                                          upload_to_s3: bool = False,
                                          seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a dataset with a controlled distribution of scenario types.
        
        Args:
            output_dir: Directory to save the dataset
            distribution: Dictionary mapping scenario types to their proportions
            num_scenarios: Total number of scenarios to generate
            duration_seconds: Duration of each scenario in seconds
            sample_rate_hz: Sample rate in Hz
            variation_params: Optional parameter variations for each scenario type
            upload_to_s3: Whether to upload the dataset to S3
            seed: Optional random seed
            
        Returns:
            Dictionary with dataset metadata
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate number of scenarios for each type
        scenario_counts = {}
        remaining = num_scenarios
        
        for scenario_type, proportion in distribution.items():
            count = int(num_scenarios * proportion)
            scenario_counts[scenario_type] = count
            remaining -= count
        
        # Distribute remaining scenarios
        for scenario_type in distribution.keys():
            if remaining <= 0:
                break
            scenario_counts[scenario_type] += 1
            remaining -= 1
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': 'mixed_scenario_dataset',
            'num_scenarios': num_scenarios,
            'distribution': distribution,
            'scenario_counts': scenario_counts,
            'duration_seconds': duration_seconds,
            'sample_rate_hz': sample_rate_hz,
            'creation_date': datetime.now().isoformat(),
            'scenarios': []
        }
        
        # Generate scenarios for each type
        scenario_count = 0
        for scenario_type, count in scenario_counts.items():
            for i in range(count):
                # Generate start time (random time in the past week)
                start_time = datetime.now() - timedelta(days=np.random.randint(0, 7))
                
                # Get the appropriate generator
                generator = self.generator_map.get(scenario_type, self.scenario_generator)
                
                # Apply variations if provided
                scenario_params = {}
                if variation_params and scenario_type in variation_params:
                    type_variations = variation_params[scenario_type]
                    
                    # Apply random variations within constraints
                    for param_path, variation_range in type_variations.items():
                        if isinstance(variation_range, dict) and 'min' in variation_range and 'max' in variation_range:
                            # Generate random value within range
                            value = np.random.uniform(variation_range['min'], variation_range['max'])
                            
                            # Parse parameter path
                            path_parts = param_path.split('.')
                            current = scenario_params
                            for j, part in enumerate(path_parts[:-1]):
                                if part not in current:
                                    current[part] = {}
                                current = current[part]
                            
                            # Set parameter value
                            current[path_parts[-1]] = value
                
                # Generate scenario
                if scenario_type == 'false_positive':
                    # For false positives, generate a random type
                    scenario_data = self.false_positive_generator.generate_random_false_positive(
                        start_time=start_time,
                        duration_seconds=duration_seconds,
                        sample_rate_hz=sample_rate_hz,
                        seed=seed + scenario_count if seed is not None else None
                    )
                else:
                    # For other types, use the specific generator
                    scenario_data = generator.generate_specific_scenario(
                        start_time=start_time,
                        duration_seconds=duration_seconds,
                        sample_rate_hz=sample_rate_hz,
                        scenario_params=scenario_params,
                        seed=seed + scenario_count if seed is not None else None
                    )
                
                # Add scenario ID
                scenario_data['metadata']['scenario_id'] = scenario_count
                
                # Save scenario
                scenario_dir = os.path.join(output_dir, f'scenario_{scenario_count:04d}')
                self.scenario_generator.save_scenario(scenario_data, scenario_dir)
                
                # Add to dataset metadata
                scenario_metadata = scenario_data['metadata'].copy()
                scenario_metadata['scenario_dir'] = scenario_dir
                dataset_metadata['scenarios'].append(scenario_metadata)
                
                scenario_count += 1
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload to S3 if requested
        if upload_to_s3 and self.scenario_generator.s3_service is not None:
            s3_dataset_key = f"scenarios/datasets/{os.path.basename(output_dir)}_metadata.json"
            self.scenario_generator.s3_service.upload_file(dataset_metadata_path, s3_dataset_key)
        
        return dataset_metadata
    
    def generate_transition_scenario(self,
                                    start_scenario_type: str,
                                    end_scenario_type: str,
                                    duration_seconds: int,
                                    sample_rate_hz: float,
                                    transition_point: float = 0.5,
                                    transition_duration: float = 0.2,
                                    seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a scenario that transitions from one type to another.
        
        Args:
            start_scenario_type: Starting scenario type
            end_scenario_type: Ending scenario type
            duration_seconds: Total duration in seconds
            sample_rate_hz: Sample rate in Hz
            transition_point: Point in the scenario where transition occurs (0-1)
            transition_duration: Duration of the transition as a fraction of total duration
            seed: Optional random seed
            
        Returns:
            Transition scenario data
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate durations
        start_duration = int(duration_seconds * transition_point)
        transition_duration_seconds = int(duration_seconds * transition_duration)
        end_duration = duration_seconds - start_duration - transition_duration_seconds
        
        # Generate start and end scenarios
        start_time = datetime.now() - timedelta(days=np.random.randint(0, 7))
        
        # Get generators
        start_generator = self.generator_map.get(start_scenario_type, self.scenario_generator)
        end_generator = self.generator_map.get(end_scenario_type, self.scenario_generator)
        
        # Generate start scenario
        start_scenario = start_generator.generate_specific_scenario(
            start_time=start_time,
            duration_seconds=start_duration + transition_duration_seconds,
            sample_rate_hz=sample_rate_hz,
            seed=seed if seed is not None else None
        )
        
        # Generate end scenario
        end_scenario = end_generator.generate_specific_scenario(
            start_time=start_time + timedelta(seconds=start_duration),
            duration_seconds=transition_duration_seconds + end_duration,
            sample_rate_hz=sample_rate_hz,
            seed=seed + 1 if seed is not None else None
        )
        
        # Calculate transition weights
        num_transition_frames = int(transition_duration_seconds * sample_rate_hz)
        transition_weights = np.linspace(1.0, 0.0, num_transition_frames)
        
        # Extract frames
        start_frames = start_scenario['thermal_data']['frames']
        end_frames = end_scenario['thermal_data']['frames']
        
        # Calculate frame indices
        start_only_frames = start_duration * sample_rate_hz
        transition_start_idx = int(start_only_frames)
        transition_end_idx = int(start_only_frames + num_transition_frames)
        
        # Create combined frames
        combined_frames = []
        
        # Add start-only frames
        combined_frames.extend(start_frames[:transition_start_idx])
        
        # Add transition frames
        for i in range(num_transition_frames):
            start_idx = transition_start_idx + i
            end_idx = i
            
            if start_idx < len(start_frames) and end_idx < len(end_frames):
                # Blend frames
                start_weight = transition_weights[i]
                end_weight = 1.0 - start_weight
                
                blended_frame = start_frames[start_idx] * start_weight + end_frames[end_idx] * end_weight
                combined_frames.append(blended_frame)
        
        # Add end-only frames
        combined_frames.extend(end_frames[num_transition_frames:])
        
        # Create timestamps
        combined_timestamps = []
        for i in range(len(combined_frames)):
            combined_timestamps.append(start_time + timedelta(seconds=i/sample_rate_hz))
        
        # Create combined thermal data
        combined_thermal = copy.deepcopy(start_scenario['thermal_data'])
        combined_thermal['frames'] = combined_frames
        combined_thermal['timestamps'] = combined_timestamps
        
        # Combine gas data
        combined_gas = copy.deepcopy(start_scenario['gas_data'])
        for gas_type in combined_gas['gas_data']:
            if 'measured_concentrations' in combined_gas['gas_data'][gas_type]:
                start_concentrations = start_scenario['gas_data']['gas_data'][gas_type]['measured_concentrations']
                
                if gas_type in end_scenario['gas_data']['gas_data'] and 'measured_concentrations' in end_scenario['gas_data']['gas_data'][gas_type]:
                    end_concentrations = end_scenario['gas_data']['gas_data'][gas_type]['measured_concentrations']
                    
                    # Create combined concentrations
                    combined_concentrations = []
                    
                    # Add start-only concentrations
                    combined_concentrations.extend(start_concentrations[:transition_start_idx])
                    
                    # Add transition concentrations
                    for i in range(num_transition_frames):
                        start_idx = transition_start_idx + i
                        end_idx = i
                        
                        if start_idx < len(start_concentrations) and end_idx < len(end_concentrations):
                            # Blend concentrations
                            start_weight = transition_weights[i]
                            end_weight = 1.0 - start_weight
                            
                            blended_concentration = start_concentrations[start_idx] * start_weight + end_concentrations[end_idx] * end_weight
                            combined_concentrations.append(blended_concentration)
                    
                    # Add end-only concentrations
                    combined_concentrations.extend(end_concentrations[num_transition_frames:])
                    
                    # Update combined gas data
                    combined_gas['gas_data'][gas_type]['measured_concentrations'] = np.array(combined_concentrations)
        
        combined_gas['timestamps'] = combined_timestamps
        
        # Combine environmental data
        combined_env = copy.deepcopy(start_scenario['environmental_data'])
        for param in combined_env['environmental_data']:
            if 'values' in combined_env['environmental_data'][param]:
                start_values = start_scenario['environmental_data']['environmental_data'][param]['values']
                
                if param in end_scenario['environmental_data']['environmental_data'] and 'values' in end_scenario['environmental_data']['environmental_data'][param]:
                    end_values = end_scenario['environmental_data']['environmental_data'][param]['values']
                    
                    # Create combined values
                    combined_values = []
                    
                    # Add start-only values
                    combined_values.extend(start_values[:transition_start_idx])
                    
                    # Add transition values
                    for i in range(num_transition_frames):
                        start_idx = transition_start_idx + i
                        end_idx = i
                        
                        if start_idx < len(start_values) and end_idx < len(end_values):
                            # Blend values
                            start_weight = transition_weights[i]
                            end_weight = 1.0 - start_weight
                            
                            blended_value = start_values[start_idx] * start_weight + end_values[end_idx] * end_weight
                            combined_values.append(blended_value)
                    
                    # Add end-only values
                    combined_values.extend(end_values[num_transition_frames:])
                    
                    # Update combined environmental data
                    combined_env['environmental_data'][param]['values'] = np.array(combined_values)
        
        combined_env['timestamps'] = combined_timestamps
        
        # Create combined metadata
        combined_metadata = {
            'scenario_type': 'transition',
            'start_scenario_type': start_scenario_type,
            'end_scenario_type': end_scenario_type,
            'duration': duration_seconds,
            'sample_rate': sample_rate_hz,
            'transition_point': transition_point,
            'transition_duration': transition_duration,
            'start_time': start_time.isoformat(),
            'end_time': (start_time + timedelta(seconds=duration_seconds)).isoformat(),
            'creation_timestamp': datetime.now().isoformat()
        }
        
        # Create combined scenario definition
        combined_def = {
            'scenario_type': 'transition',
            'duration': duration_seconds,
            'sample_rate': sample_rate_hz,
            'room_params': start_scenario['scenario_definition']['room_params'],
            'transition_params': {
                'start_type': start_scenario_type,
                'end_type': end_scenario_type,
                'transition_point': transition_point,
                'transition_duration': transition_duration
            }
        }
        
        # Create combined scenario
        combined_scenario = {
            'thermal_data': combined_thermal,
            'gas_data': combined_gas,
            'environmental_data': combined_env,
            'metadata': combined_metadata,
            'scenario_definition': combined_def
        }
        
        return combined_scenario