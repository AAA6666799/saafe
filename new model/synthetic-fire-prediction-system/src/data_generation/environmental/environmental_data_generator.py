"""
Environmental data generator for synthetic fire data.

This module provides functionality for generating realistic environmental data
including temperature, humidity, and pressure based on fire scenarios.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import io
import json
import matplotlib.pyplot as plt

from ..base import EnvironmentalDataGenerator as EnvironmentalDataGeneratorBase
from .voc_pattern_generator import VOCPatternGenerator
from .correlation_engine import CorrelationEngine
from .environmental_variation_model import EnvironmentalVariationModel

# AWS integration (optional)
try:
    from ...aws.s3.service import S3ServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None


class EnvironmentalDataGenerator(EnvironmentalDataGeneratorBase):
    """
    Class for generating synthetic environmental data.
    
    This class implements the EnvironmentalDataGenerator interface and provides
    methods for generating realistic environmental parameter values (temperature,
    humidity, pressure) based on fire scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environmental data generator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        
        # Initialize components
        self.voc_generator = VOCPatternGenerator(self.config.get('voc_config', {}))
        self.correlation_engine = CorrelationEngine(self.config.get('correlation_config', {}))
        self.variation_model = EnvironmentalVariationModel(self.config.get('variation_config', {}))
        
        # Initialize S3 service if AWS integration is enabled and available
        self.s3_service = None
        if self.config.get('aws_integration', False) and AWS_AVAILABLE:
            self.s3_service = S3ServiceImpl(self.config.get('aws_config', {}))
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check environmental parameters
        if 'parameters' in self.config:
            if not isinstance(self.config['parameters'], list) or len(self.config['parameters']) == 0:
                raise ValueError("parameters must be a non-empty list")
        else:
            # Set default parameters if not provided
            self.config['parameters'] = ['temperature', 'humidity', 'pressure', 'voc']
        
        # Check parameter ranges
        if 'parameter_ranges' not in self.config:
            # Set default parameter ranges if not provided
            self.config['parameter_ranges'] = {
                'temperature': {'min': 15.0, 'max': 35.0, 'unit': '°C'},  # Celsius
                'humidity': {'min': 20.0, 'max': 80.0, 'unit': '%'},      # Percentage
                'pressure': {'min': 990.0, 'max': 1030.0, 'unit': 'hPa'}, # Hectopascals
                'voc': {'min': 0.0, 'max': 2000.0, 'unit': 'ppb'}         # Parts per billion
            }
        
        # Check AWS integration
        if self.config.get('aws_integration', False):
            if 'aws_config' not in self.config:
                raise ValueError("aws_config is required when aws_integration is enabled")
            
            if 'default_bucket' not in self.config['aws_config']:
                raise ValueError("default_bucket is required in aws_config")
    
    def generate(self, 
                timestamp: datetime, 
                duration_seconds: int, 
                sample_rate_hz: float,
                seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate synthetic environmental data for the specified duration.
        
        Args:
            timestamp: Start timestamp for the generated data
            duration_seconds: Duration of data to generate in seconds
            sample_rate_hz: Sample rate in Hz
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated data:
            - environmental_data: Dictionary mapping parameter types to value arrays
            - timestamps: List of timestamps for each data point
            - metadata: Dictionary with metadata about the sequence
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get parameters from config
        parameters = self.config['parameters']
        parameter_ranges = self.config['parameter_ranges']
        
        # Calculate number of samples
        num_samples = int(duration_seconds * sample_rate_hz)
        
        # Generate timestamps
        timestamps = [timestamp + timedelta(seconds=i/sample_rate_hz) for i in range(num_samples)]
        
        # Generate base environmental data with daily and seasonal variations
        env_data = {}
        for param in parameters:
            if param != 'voc':  # VOC will be handled separately
                param_range = parameter_ranges.get(param, {'min': 0.0, 'max': 100.0})
                baseline = (param_range['min'] + param_range['max']) / 2
                
                # Generate time series with variations
                # Ensure seed is within valid numpy range if provided
                param_seed = None
                if seed is not None:
                    param_seed = (seed + abs(hash(param))) % (2**32 - 1)
                
                values = self.variation_model.generate_time_series(
                    parameter=param,
                    baseline=baseline,
                    timestamps=timestamps,
                    seed=param_seed
                )
                
                env_data[param] = {
                    'values': values,
                    'unit': param_range.get('unit', ''),
                    'min': param_range['min'],
                    'max': param_range['max']
                }
        
        # Generate VOC data if required
        if 'voc' in parameters:
            voc_range = parameter_ranges.get('voc', {'min': 0.0, 'max': 2000.0})
            
            # Generate VOC patterns
            # Ensure seed is within valid numpy range if provided
            voc_seed = None
            if seed is not None:
                voc_seed = (seed + abs(hash('voc'))) % (2**32 - 1)
            
            voc_values = self.voc_generator.generate_voc_time_series(
                timestamps=timestamps,
                baseline=voc_range['min'],
                material_type='mixed',  # Default material type
                temperature_values=env_data.get('temperature', {}).get('values'),
                seed=voc_seed
            )
            
            env_data['voc'] = {
                'values': voc_values,
                'unit': voc_range.get('unit', 'ppb'),
                'min': voc_range['min'],
                'max': voc_range['max']
            }
        
        # Apply correlations between parameters
        env_data = self.correlation_engine.apply_correlations(env_data, timestamps)
        
        # Create metadata
        metadata = {
            'parameters': parameters,
            'duration': duration_seconds,
            'sample_rate': sample_rate_hz,
            'num_samples': num_samples,
            'start_time': timestamp.isoformat(),
            'end_time': (timestamp + timedelta(seconds=duration_seconds)).isoformat(),
            'seed': seed
        }
        
        return {
            'environmental_data': env_data,
            'timestamps': timestamps,
            'metadata': metadata
        }
    
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
        # Generate a single reading using the variation model
        reading = self.variation_model.generate_single_reading(
            parameter=parameter,
            timestamp=timestamp,
            baseline=baseline,
            daily_variation=daily_variation,
            noise_level=noise_level
        )
        
        return reading
    
    def generate_fire_scenario(self,
                             start_time: datetime,
                             duration: int,
                             sample_rate: float,
                             fire_type: str,
                             room_params: Dict[str, float],
                             seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate environmental data for a fire scenario.
        
        Args:
            start_time: Start time of the scenario
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            fire_type: Type of fire (e.g., 'smoldering', 'flaming')
            room_params: Dictionary with room parameters (volume, ventilation, etc.)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing the generated environmental data
        """
        # Extract room parameters
        room_volume = room_params.get('room_volume', 50.0)  # m³
        ventilation_rate = room_params.get('ventilation_rate', 0.5)  # air changes per hour
        initial_temperature = room_params.get('initial_temperature', 20.0)  # °C
        initial_humidity = room_params.get('initial_humidity', 50.0)  # %
        
        # Set parameter ranges based on fire type
        if fire_type == 'smoldering':
            temp_increase = 5.0  # Slower temperature rise
            humidity_decrease = 10.0  # Moderate humidity decrease
            voc_increase = 500.0  # Moderate VOC increase
        elif fire_type == 'flaming':
            temp_increase = 15.0  # Rapid temperature rise
            humidity_decrease = 30.0  # Significant humidity decrease
            voc_increase = 1500.0  # Significant VOC increase
        elif fire_type == 'electrical':
            temp_increase = 8.0  # Moderate temperature rise
            humidity_decrease = 5.0  # Minimal humidity decrease
            voc_increase = 300.0  # Specific VOC pattern for electrical fires
        elif fire_type == 'chemical':
            temp_increase = 12.0  # Variable temperature rise
            humidity_decrease = 20.0  # Moderate humidity decrease
            voc_increase = 2000.0  # High VOC levels with specific chemical signatures
        else:  # Default/normal
            temp_increase = 0.0
            humidity_decrease = 0.0
            voc_increase = 0.0
        
        # Adjust parameter ranges based on fire type
        parameter_ranges = {
            'temperature': {
                'min': initial_temperature,
                'max': initial_temperature + temp_increase,
                'unit': '°C'
            },
            'humidity': {
                'min': max(10.0, initial_humidity - humidity_decrease),
                'max': initial_humidity,
                'unit': '%'
            },
            'pressure': {
                'min': 990.0,
                'max': 1010.0,
                'unit': 'hPa'
            },
            'voc': {
                'min': 50.0,
                'max': 50.0 + voc_increase,
                'unit': 'ppb'
            }
        }
        
        # Update config with scenario-specific parameter ranges
        self.config['parameter_ranges'] = parameter_ranges
        
        # Generate environmental data
        env_data = self.generate(
            timestamp=start_time,
            duration_seconds=duration,
            sample_rate_hz=sample_rate,
            seed=seed
        )
        
        # Add fire scenario metadata
        env_data['metadata']['fire_type'] = fire_type
        env_data['metadata']['room_params'] = room_params
        
        return env_data
    
    def to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert generated data to a pandas DataFrame.
        
        Args:
            data: Generated data from the generate method
            
        Returns:
            DataFrame containing the data in a structured format
        """
        # Extract data
        env_data = data['environmental_data']
        timestamps = data['timestamps']
        parameters = list(env_data.keys())
        
        # Create DataFrame
        df_data = {'timestamp': timestamps}
        
        for param in parameters:
            if 'values' in env_data[param]:
                df_data[param] = env_data[param]['values']
                df_data[f'{param}_unit'] = [env_data[param]['unit']] * len(timestamps)
        
        return pd.DataFrame(df_data)
    
    def save(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Save generated data to files.
        
        Args:
            data: Generated data from the generate method
            filepath: Base path to save the data
        """
        # Extract data
        env_data = data['environmental_data']
        timestamps = data['timestamps']
        metadata = data['metadata']
        parameters = list(env_data.keys())
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save metadata
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'w') as f:
            # Convert timestamps to ISO format strings
            metadata_copy = metadata.copy()
            if 'start_time' in metadata_copy and isinstance(metadata_copy['start_time'], datetime):
                metadata_copy['start_time'] = metadata_copy['start_time'].isoformat()
            if 'end_time' in metadata_copy and isinstance(metadata_copy['end_time'], datetime):
                metadata_copy['end_time'] = metadata_copy['end_time'].isoformat()
            
            json.dump(metadata_copy, f, indent=2)
        
        # Save timestamps
        timestamps_path = f"{filepath}_timestamps.json"
        with open(timestamps_path, 'w') as f:
            json.dump([ts.isoformat() for ts in timestamps], f, indent=2)
        
        # Save environmental parameter data
        for param in parameters:
            param_path = f"{filepath}_{param}.csv"
            
            # Save values
            if 'values' in env_data[param]:
                np.savetxt(param_path, env_data[param]['values'], delimiter=',')
            
            # Save parameter metadata
            param_metadata_path = f"{filepath}_{param}_metadata.json"
            with open(param_metadata_path, 'w') as f:
                param_metadata = {
                    'unit': env_data[param].get('unit', ''),
                    'min': env_data[param].get('min', 0.0),
                    'max': env_data[param].get('max', 0.0)
                }
                json.dump(param_metadata, f, indent=2)
        
        # Save DataFrame with all data
        df = self.to_dataframe(data)
        df.to_csv(f"{filepath}_data.csv", index=False)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            self._upload_to_s3(filepath, data)
    
    def _upload_to_s3(self, filepath: str, data: Dict[str, Any]) -> None:
        """
        Upload generated data to S3.
        
        Args:
            filepath: Base path of the saved files
            data: Generated data from the generate method
        """
        if self.s3_service is None:
            raise ValueError("S3 service is not initialized")
        
        # Extract data
        env_data = data['environmental_data']
        parameters = list(env_data.keys())
        
        # Get base filename without directory
        base_filename = os.path.basename(filepath)
        
        # Upload metadata
        metadata_path = f"{filepath}_metadata.json"
        s3_metadata_key = f"environmental_data/{base_filename}_metadata.json"
        self.s3_service.upload_file(metadata_path, s3_metadata_key)
        
        # Upload timestamps
        timestamps_path = f"{filepath}_timestamps.json"
        s3_timestamps_key = f"environmental_data/{base_filename}_timestamps.json"
        self.s3_service.upload_file(timestamps_path, s3_timestamps_key)
        
        # Upload environmental parameter data
        for param in parameters:
            param_path = f"{filepath}_{param}.csv"
            if os.path.exists(param_path):
                s3_param_key = f"environmental_data/{base_filename}_{param}.csv"
                self.s3_service.upload_file(param_path, s3_param_key)
            
            # Upload parameter metadata
            param_metadata_path = f"{filepath}_{param}_metadata.json"
            if os.path.exists(param_metadata_path):
                s3_param_metadata_key = f"environmental_data/{base_filename}_{param}_metadata.json"
                self.s3_service.upload_file(param_metadata_path, s3_param_metadata_key)
        
        # Upload DataFrame with all data
        data_path = f"{filepath}_data.csv"
        s3_data_key = f"environmental_data/{base_filename}_data.csv"
        self.s3_service.upload_file(data_path, s3_data_key)
    
    def generate_and_save_dataset(self, 
                                output_dir: str,
                                num_sequences: int,
                                sequence_duration: int,
                                sample_rate: float,
                                fire_types: Optional[List[str]] = None,
                                room_params: Optional[List[Dict[str, float]]] = None,
                                upload_to_s3: bool = False,
                                seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate and save a dataset of environmental data sequences.
        
        Args:
            output_dir: Directory to save the dataset
            num_sequences: Number of sequences to generate
            sequence_duration: Duration of each sequence in seconds
            sample_rate: Sample rate in Hz
            fire_types: Optional list of fire types to include
            room_params: Optional list of room parameter dictionaries
            upload_to_s3: Whether to upload the dataset to S3
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary with dataset metadata
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set default fire types if not provided
        if fire_types is None:
            fire_types = ['normal', 'smoldering', 'flaming', 'electrical', 'chemical']
        
        # Set default room parameters if not provided
        if room_params is None:
            room_params = [
                {'room_volume': 50.0, 'ventilation_rate': 0.5, 'initial_temperature': 20.0, 'initial_humidity': 50.0},
                {'room_volume': 100.0, 'ventilation_rate': 1.0, 'initial_temperature': 22.0, 'initial_humidity': 45.0},
                {'room_volume': 30.0, 'ventilation_rate': 0.2, 'initial_temperature': 18.0, 'initial_humidity': 60.0}
            ]
        
        # Get parameters from config
        parameters = self.config['parameters']
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': 'environmental_data_dataset',
            'num_sequences': num_sequences,
            'sequence_duration': sequence_duration,
            'sample_rate': sample_rate,
            'parameters': parameters,
            'fire_types': fire_types,
            'creation_date': datetime.now().isoformat(),
            'sequences': []
        }
        
        # Generate sequences
        for i in range(num_sequences):
            # Select random fire type and room parameters
            fire_type = np.random.choice(fire_types)
            room_param = np.random.choice(room_params)
            
            # Generate start time (random time in the past week)
            start_time = datetime.now() - timedelta(days=np.random.randint(0, 7))
            
            # Generate sequence
            sequence = self.generate_fire_scenario(
                start_time=start_time,
                duration=sequence_duration,
                sample_rate=sample_rate,
                fire_type=fire_type,
                room_params=room_param,
                seed=seed + i if seed is not None else None
            )
            
            # Save sequence
            sequence_path = os.path.join(output_dir, f'sequence_{i:04d}')
            self.save(sequence, sequence_path)
            
            # Add to dataset metadata
            sequence_metadata = sequence['metadata'].copy()
            sequence_metadata['sequence_id'] = i
            sequence_metadata['sequence_path'] = sequence_path
            sequence_metadata['fire_type'] = fire_type
            sequence_metadata['room_params'] = room_param
            dataset_metadata['sequences'].append(sequence_metadata)
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload to S3 if requested
        if upload_to_s3 and self.s3_service is not None:
            s3_dataset_key = f"environmental_data/datasets/{os.path.basename(output_dir)}_metadata.json"
            self.s3_service.upload_file(dataset_metadata_path, s3_dataset_key)
        
        return dataset_metadata
    
    def visualize_environmental_data(self, 
                                   data: Dict[str, Any], 
                                   parameters: Optional[List[str]] = None,
                                   show: bool = True,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize environmental data time series.
        
        Args:
            data: Generated data from the generate method
            parameters: Optional list of parameters to visualize (default: all)
            show: Whether to display the plot
            save_path: Optional path to save the visualization
        """
        # Extract data
        env_data = data['environmental_data']
        timestamps = data['timestamps']
        all_parameters = list(env_data.keys())
        
        # Use all parameters if not specified
        if parameters is None:
            parameters = all_parameters
        
        # Filter to only include available parameters
        parameters = [p for p in parameters if p in all_parameters]
        
        if not parameters:
            print("No valid parameters to visualize")
            return
        
        # Create figure
        fig, axes = plt.subplots(len(parameters), 1, figsize=(12, 4 * len(parameters)), sharex=True)
        
        # Handle single parameter case
        if len(parameters) == 1:
            axes = [axes]
        
        # Convert timestamps to relative seconds
        start_time = timestamps[0]
        relative_times = [(ts - start_time).total_seconds() for ts in timestamps]
        
        # Plot each parameter
        for i, param in enumerate(parameters):
            ax = axes[i]
            
            if 'values' in env_data[param]:
                values = env_data[param]['values']
                unit = env_data[param].get('unit', '')
                
                ax.plot(relative_times, values, label=param)
                
                # Add labels and legend
                ax.set_ylabel(f'{param} ({unit})')
                ax.set_title(f'{param.capitalize()} over Time')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Add common x-axis label
        axes[-1].set_xlabel('Time (seconds)')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()