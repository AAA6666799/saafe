"""
Scenario generator for synthetic fire data.

This module provides functionality for generating complete fire scenarios
that combine thermal, gas, and environmental data.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import jsonschema
from abc import ABC, abstractmethod

from ..base import ScenarioGenerator as ScenarioGeneratorBase
from ..thermal.thermal_image_generator import ThermalImageGenerator
from ..gas.gas_concentration_generator import GasConcentrationGenerator
from ..environmental.environmental_data_generator import EnvironmentalDataGenerator

# AWS integration (optional)
try:
    from ...aws.s3.service import S3ServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None


class ScenarioGenerator(ScenarioGeneratorBase):
    """
    Class for generating complete fire scenarios.
    
    This class implements the ScenarioGenerator interface and provides
    methods for generating realistic fire scenarios that combine thermal,
    gas, and environmental data.
    """
    
    # Define the JSON schema for scenario definitions
    SCENARIO_SCHEMA = {
        "type": "object",
        "required": ["scenario_type", "duration", "sample_rate", "room_params"],
        "properties": {
            "scenario_type": {
                "type": "string",
                "enum": ["normal", "electrical_fire", "chemical_fire", "smoldering_fire", "rapid_combustion", 
                         "cooking", "heating", "false_positive"]
            },
            "duration": {
                "type": "integer",
                "minimum": 1,
                "description": "Duration in seconds"
            },
            "sample_rate": {
                "type": "number",
                "minimum": 0.1,
                "description": "Sample rate in Hz"
            },
            "room_params": {
                "type": "object",
                "required": ["room_volume", "ventilation_rate"],
                "properties": {
                    "room_volume": {
                        "type": "number",
                        "minimum": 1.0,
                        "description": "Room volume in cubic meters"
                    },
                    "ventilation_rate": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Ventilation rate in air changes per hour"
                    },
                    "initial_temperature": {
                        "type": "number",
                        "description": "Initial room temperature in Celsius"
                    },
                    "initial_humidity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 100.0,
                        "description": "Initial room humidity in percentage"
                    },
                    "fuel_load": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Fuel load in kg"
                    }
                }
            },
            "fire_params": {
                "type": "object",
                "properties": {
                    "fire_location": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 3,
                        "description": "Fire location coordinates [x, y] or [x, y, z]"
                    },
                    "fire_size": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Initial fire size in kW"
                    },
                    "growth_rate": {
                        "type": "number",
                        "description": "Fire growth rate in kW/s"
                    },
                    "max_size": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Maximum fire size in kW"
                    }
                }
            },
            "gas_params": {
                "type": "object",
                "properties": {
                    "gas_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gas types to include"
                    },
                    "release_rates": {
                        "type": "object",
                        "description": "Gas release rates in g/s for each gas type"
                    }
                }
            },
            "thermal_params": {
                "type": "object",
                "properties": {
                    "max_temperature": {
                        "type": "number",
                        "description": "Maximum temperature in Celsius"
                    },
                    "hotspot_count": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of hotspots to generate"
                    }
                }
            },
            "environmental_params": {
                "type": "object",
                "properties": {
                    "parameters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of environmental parameters to include"
                    }
                }
            },
            "variations": {
                "type": "object",
                "description": "Parameter variations for the scenario"
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata for the scenario"
            }
        }
    }
    
    def __init__(self, 
                 thermal_generator: ThermalImageGenerator,
                 gas_generator: GasConcentrationGenerator,
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
        super().__init__(thermal_generator, gas_generator, environmental_generator, config)
        
        # Initialize S3 service if AWS integration is enabled and available
        self.s3_service = None
        if self.config.get('aws_integration', False) and AWS_AVAILABLE:
            self.s3_service = S3ServiceImpl(self.config.get('aws_config', {}))
    
    def validate_scenario_definition(self, scenario_def: Dict[str, Any]) -> None:
        """
        Validate a scenario definition against the schema.
        
        Args:
            scenario_def: Scenario definition dictionary
            
        Raises:
            jsonschema.exceptions.ValidationError: If validation fails
        """
        jsonschema.validate(instance=scenario_def, schema=self.SCENARIO_SCHEMA)
    
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
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create a complete scenario definition
        scenario_def = {
            "scenario_type": scenario_type,
            "duration": duration_seconds,
            "sample_rate": sample_rate_hz,
            "room_params": scenario_params.get('room_params', {
                "room_volume": 50.0,
                "ventilation_rate": 0.5,
                "initial_temperature": 20.0,
                "initial_humidity": 50.0,
                "fuel_load": 10.0
            })
        }
        
        # Add fire parameters if provided
        if 'fire_params' in scenario_params:
            scenario_def['fire_params'] = scenario_params['fire_params']
        
        # Add gas parameters if provided
        if 'gas_params' in scenario_params:
            scenario_def['gas_params'] = scenario_params['gas_params']
        
        # Add thermal parameters if provided
        if 'thermal_params' in scenario_params:
            scenario_def['thermal_params'] = scenario_params['thermal_params']
        
        # Add environmental parameters if provided
        if 'environmental_params' in scenario_params:
            scenario_def['environmental_params'] = scenario_params['environmental_params']
        
        # Validate the scenario definition
        try:
            self.validate_scenario_definition(scenario_def)
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Invalid scenario definition: {e}")
        
        # Generate thermal data
        thermal_data = self._generate_thermal_data(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_type=scenario_type,
            scenario_def=scenario_def,
            seed=seed
        )
        
        # Generate gas data
        gas_data = self._generate_gas_data(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_type=scenario_type,
            scenario_def=scenario_def,
            seed=seed + 1 if seed is not None else None
        )
        
        # Generate environmental data
        environmental_data = self._generate_environmental_data(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_type=scenario_type,
            scenario_def=scenario_def,
            seed=seed + 2 if seed is not None else None
        )
        
        # Coordinate data between sensors
        gas_data = self.gas_generator.coordinate_with_thermal_data(gas_data, thermal_data)
        
        # Create scenario metadata
        scenario_metadata = {
            "scenario_type": scenario_type,
            "start_time": start_time.isoformat(),
            "end_time": (start_time + timedelta(seconds=duration_seconds)).isoformat(),
            "duration": duration_seconds,
            "sample_rate": sample_rate_hz,
            "num_samples": int(duration_seconds * sample_rate_hz),
            "seed": seed,
            "room_params": scenario_def["room_params"],
            "creation_timestamp": datetime.now().isoformat()
        }
        
        # Add additional metadata from scenario parameters
        if 'metadata' in scenario_params:
            scenario_metadata.update(scenario_params['metadata'])
        
        # Combine all data into a single scenario
        scenario_data = {
            "thermal_data": thermal_data,
            "gas_data": gas_data,
            "environmental_data": environmental_data,
            "metadata": scenario_metadata,
            "scenario_definition": scenario_def
        }
        
        return scenario_data
    
    def _generate_thermal_data(self,
                              start_time: datetime,
                              duration_seconds: int,
                              sample_rate_hz: float,
                              scenario_type: str,
                              scenario_def: Dict[str, Any],
                              seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate thermal data for a scenario.
        
        Args:
            start_time: Start timestamp
            duration_seconds: Duration in seconds
            sample_rate_hz: Sample rate in Hz
            scenario_type: Type of scenario
            scenario_def: Complete scenario definition
            seed: Optional random seed
            
        Returns:
            Dictionary containing thermal data
        """
        # Configure thermal generator based on scenario type
        thermal_config = self.thermal_generator.config.copy()
        
        # Get thermal parameters from scenario definition
        thermal_params = scenario_def.get('thermal_params', {})
        
        # Set fire type based on scenario type
        if scenario_type == 'normal':
            thermal_config['fire_type'] = 'none'
        elif scenario_type == 'electrical_fire':
            thermal_config['fire_type'] = 'electrical'
        elif scenario_type == 'chemical_fire':
            thermal_config['fire_type'] = 'chemical'
        elif scenario_type == 'smoldering_fire':
            thermal_config['fire_type'] = 'smoldering'
        elif scenario_type == 'rapid_combustion':
            thermal_config['fire_type'] = 'rapid_combustion'
        elif scenario_type in ['cooking', 'heating', 'false_positive']:
            thermal_config['fire_type'] = 'false_positive'
        else:
            thermal_config['fire_type'] = 'standard'
        
        # Set max temperature if provided
        if 'max_temperature' in thermal_params:
            thermal_config['max_temperature'] = thermal_params['max_temperature']
        
        # Update thermal generator config
        self.thermal_generator.config.update(thermal_config)
        
        # Generate thermal data
        thermal_data = self.thermal_generator.generate(
            timestamp=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            seed=seed
        )
        
        return thermal_data
    
    def _generate_gas_data(self,
                          start_time: datetime,
                          duration_seconds: int,
                          sample_rate_hz: float,
                          scenario_type: str,
                          scenario_def: Dict[str, Any],
                          seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate gas data for a scenario.
        
        Args:
            start_time: Start timestamp
            duration_seconds: Duration in seconds
            sample_rate_hz: Sample rate in Hz
            scenario_type: Type of scenario
            scenario_def: Complete scenario definition
            seed: Optional random seed
            
        Returns:
            Dictionary containing gas data
        """
        # Get gas parameters from scenario definition
        gas_params = scenario_def.get('gas_params', {})
        room_params = scenario_def['room_params']
        
        # Get gas types from parameters or use defaults
        gas_types = gas_params.get('gas_types', ['methane', 'propane', 'hydrogen', 'carbon_monoxide'])
        
        # Generate gas data based on scenario type
        if scenario_type == 'normal':
            # Normal scenario with baseline gas levels
            gas_data = self.gas_generator.generate(
                timestamp=start_time,
                duration_seconds=duration_seconds,
                sample_rate_hz=sample_rate_hz,
                seed=seed
            )
        else:
            # Fire scenario with specific gas patterns
            gas_data = self.gas_generator.generate_fire_scenario(
                gas_types=gas_types,
                start_time=start_time,
                duration=duration_seconds,
                sample_rate=sample_rate_hz,
                fire_type=scenario_type,
                room_params=room_params,
                seed=seed
            )
        
        return gas_data
    
    def _generate_environmental_data(self,
                                   start_time: datetime,
                                   duration_seconds: int,
                                   sample_rate_hz: float,
                                   scenario_type: str,
                                   scenario_def: Dict[str, Any],
                                   seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate environmental data for a scenario.
        
        Args:
            start_time: Start timestamp
            duration_seconds: Duration in seconds
            sample_rate_hz: Sample rate in Hz
            scenario_type: Type of scenario
            scenario_def: Complete scenario definition
            seed: Optional random seed
            
        Returns:
            Dictionary containing environmental data
        """
        # Get environmental parameters from scenario definition
        env_params = scenario_def.get('environmental_params', {})
        room_params = scenario_def['room_params']
        
        # Generate environmental data based on scenario type
        env_data = self.environmental_generator.generate_fire_scenario(
            start_time=start_time,
            duration=duration_seconds,
            sample_rate=sample_rate_hz,
            fire_type=scenario_type,
            room_params=room_params,
            seed=seed
        )
        
        return env_data
    
    def save_scenario(self, scenario_data: Dict[str, Any], directory: str) -> None:
        """
        Save a generated scenario to files.
        
        Args:
            scenario_data: Generated scenario data
            directory: Directory to save the scenario files
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Extract data components
        thermal_data = scenario_data['thermal_data']
        gas_data = scenario_data['gas_data']
        environmental_data = scenario_data['environmental_data']
        metadata = scenario_data['metadata']
        scenario_def = scenario_data['scenario_definition']
        
        # Save metadata
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save scenario definition
        scenario_def_path = os.path.join(directory, 'scenario_definition.json')
        with open(scenario_def_path, 'w') as f:
            json.dump(scenario_def, f, indent=2)
        
        # Save thermal data
        thermal_dir = os.path.join(directory, 'thermal')
        os.makedirs(thermal_dir, exist_ok=True)
        self.thermal_generator.save(thermal_data, os.path.join(thermal_dir, 'thermal_data'))
        
        # Save gas data
        gas_dir = os.path.join(directory, 'gas')
        os.makedirs(gas_dir, exist_ok=True)
        self.gas_generator.save(gas_data, os.path.join(gas_dir, 'gas_data'))
        
        # Save environmental data
        env_dir = os.path.join(directory, 'environmental')
        os.makedirs(env_dir, exist_ok=True)
        self.environmental_generator.save(environmental_data, os.path.join(env_dir, 'environmental_data'))
        
        # Create a combined DataFrame with synchronized data
        combined_df = self._create_combined_dataframe(scenario_data)
        combined_df_path = os.path.join(directory, 'combined_data.csv')
        combined_df.to_csv(combined_df_path, index=False)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            self._upload_to_s3(directory, scenario_data)
    
    def _create_combined_dataframe(self, scenario_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a combined DataFrame with synchronized data from all sensors.
        
        Args:
            scenario_data: Generated scenario data
            
        Returns:
            Combined DataFrame
        """
        # Extract timestamps
        timestamps = scenario_data['thermal_data']['timestamps']
        
        # Create base DataFrame with timestamps
        df = pd.DataFrame({'timestamp': timestamps})
        
        # Add thermal data statistics
        thermal_df = self.thermal_generator.to_dataframe(scenario_data['thermal_data'])
        thermal_cols = [col for col in thermal_df.columns if col != 'timestamp']
        df = pd.merge(df, thermal_df[['timestamp'] + thermal_cols], on='timestamp', how='left')
        
        # Add gas data
        gas_df = self.gas_generator.to_dataframe(scenario_data['gas_data'])
        gas_cols = [col for col in gas_df.columns if col != 'timestamp']
        df = pd.merge(df, gas_df[['timestamp'] + gas_cols], on='timestamp', how='left')
        
        # Add environmental data
        env_df = self.environmental_generator.to_dataframe(scenario_data['environmental_data'])
        env_cols = [col for col in env_df.columns if col != 'timestamp']
        df = pd.merge(df, env_df[['timestamp'] + env_cols], on='timestamp', how='left')
        
        # Add scenario type and other metadata
        df['scenario_type'] = scenario_data['metadata']['scenario_type']
        
        return df
    
    def _upload_to_s3(self, directory: str, scenario_data: Dict[str, Any]) -> None:
        """
        Upload scenario data to S3.
        
        Args:
            directory: Directory containing scenario files
            scenario_data: Generated scenario data
        """
        if self.s3_service is None:
            raise ValueError("S3 service is not initialized")
        
        # Get scenario type and timestamp for S3 key prefix
        scenario_type = scenario_data['metadata']['scenario_type']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_prefix = f"scenarios/{scenario_type}/{timestamp}"
        
        # Upload all files in the directory
        for root, _, files in os.walk(directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, directory)
                s3_key = f"{s3_prefix}/{relative_path}"
                self.s3_service.upload_file(local_path, s3_key)
    
    def generate_from_definition(self, 
                                scenario_def: Dict[str, Any],
                                start_time: Optional[datetime] = None,
                                seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a scenario from a scenario definition.
        
        Args:
            scenario_def: Scenario definition dictionary
            start_time: Optional start timestamp (default: current time)
            seed: Optional random seed
            
        Returns:
            Generated scenario data
        """
        # Validate the scenario definition
        self.validate_scenario_definition(scenario_def)
        
        # Set default start time if not provided
        if start_time is None:
            start_time = datetime.now()
        
        # Extract parameters from scenario definition
        scenario_type = scenario_def['scenario_type']
        duration_seconds = scenario_def['duration']
        sample_rate_hz = scenario_def['sample_rate']
        
        # Create scenario parameters
        scenario_params = {
            'room_params': scenario_def['room_params']
        }
        
        # Add optional parameters if present
        for param in ['fire_params', 'gas_params', 'thermal_params', 'environmental_params', 'metadata']:
            if param in scenario_def:
                scenario_params[param] = scenario_def[param]
        
        # Generate the scenario
        return self.generate_scenario(
            start_time=start_time,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            scenario_type=scenario_type,
            scenario_params=scenario_params,
            seed=seed
        )
    
    def generate_dataset(self,
                        output_dir: str,
                        scenario_definitions: List[Dict[str, Any]],
                        num_variations: int = 1,
                        upload_to_s3: bool = False,
                        seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a dataset of scenarios from definitions.
        
        Args:
            output_dir: Directory to save the dataset
            scenario_definitions: List of scenario definitions
            num_variations: Number of variations to generate for each definition
            upload_to_s3: Whether to upload the dataset to S3
            seed: Optional random seed
            
        Returns:
            Dictionary with dataset metadata
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': 'fire_scenario_dataset',
            'num_scenarios': len(scenario_definitions) * num_variations,
            'creation_date': datetime.now().isoformat(),
            'scenarios': []
        }
        
        # Generate scenarios
        scenario_count = 0
        for i, scenario_def in enumerate(scenario_definitions):
            for j in range(num_variations):
                # Generate variation seed
                variation_seed = seed + i * 100 + j if seed is not None else None
                
                # Generate start time (random time in the past week)
                start_time = datetime.now() - timedelta(days=np.random.randint(0, 7))
                
                # Generate scenario
                scenario_data = self.generate_from_definition(
                    scenario_def=scenario_def,
                    start_time=start_time,
                    seed=variation_seed
                )
                
                # Create scenario directory
                scenario_dir = os.path.join(output_dir, f'scenario_{scenario_count:04d}')
                
                # Save scenario
                self.save_scenario(scenario_data, scenario_dir)
                
                # Add to dataset metadata
                scenario_metadata = scenario_data['metadata'].copy()
                scenario_metadata['scenario_id'] = scenario_count
                scenario_metadata['scenario_dir'] = scenario_dir
                scenario_metadata['variation_id'] = j
                dataset_metadata['scenarios'].append(scenario_metadata)
                
                scenario_count += 1
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload to S3 if requested
        if upload_to_s3 and self.s3_service is not None:
            s3_dataset_key = f"scenarios/datasets/{os.path.basename(output_dir)}_metadata.json"
            self.s3_service.upload_file(dataset_metadata_path, s3_dataset_key)
        
        return dataset_metadata