"""
Gas concentration generator for synthetic fire data.

This module provides functionality for generating realistic gas concentration values
based on fire scenarios for multiple gas types (methane, propane, hydrogen).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import io
import json
import matplotlib.pyplot as plt

from ..base import GasDataGenerator
from .diffusion_model import DiffusionModel
from .sensor_response_model import SensorResponseModel, SensorType
from .gas_temporal_evolution import GasTemporalEvolution, ReleasePattern

# AWS integration (optional)
try:
    from ...aws.s3.service import S3ServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None


class GasConcentrationGenerator(GasDataGenerator):
    """
    Class for generating synthetic gas concentration data.
    
    This class implements the GasDataGenerator interface and provides
    methods for generating realistic gas concentration values for multiple gas types
    based on fire scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gas concentration generator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        
        # Initialize components
        self.diffusion_model = DiffusionModel(self.config.get('diffusion_config', {}))
        self.temporal_model = GasTemporalEvolution(self.config.get('temporal_config', {}))
        
        # Initialize sensor models for different gas types
        self.sensor_models = {}
        sensor_configs = self.config.get('sensor_configs', {})
        
        for gas_type, sensor_config in sensor_configs.items():
            self.sensor_models[gas_type] = SensorResponseModel(sensor_config)
        
        # Default sensor model for gases without specific configuration
        default_sensor_config = self.config.get('default_sensor_config', {
            'sensor_type': 'electrochemical',
            'noise_level': 0.02,
            'drift_rate': 0.5,
            'response_time': 30.0,
            'recovery_time': 60.0
        })
        self.default_sensor_model = SensorResponseModel(default_sensor_config)
        
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
        # Check gas types
        if 'gas_types' in self.config:
            if not isinstance(self.config['gas_types'], list) or len(self.config['gas_types']) == 0:
                raise ValueError("gas_types must be a non-empty list")
        else:
            # Set default gas types if not provided
            self.config['gas_types'] = ['methane', 'propane', 'hydrogen', 'carbon_monoxide']
        
        # Check sensor configurations
        if 'sensor_configs' in self.config:
            if not isinstance(self.config['sensor_configs'], dict):
                raise ValueError("sensor_configs must be a dictionary")
        
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
        Generate synthetic gas concentration data for the specified duration.
        
        Args:
            timestamp: Start timestamp for the generated data
            duration_seconds: Duration of data to generate in seconds
            sample_rate_hz: Sample rate in Hz
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated data:
            - concentrations: Dictionary mapping gas types to concentration arrays
            - timestamps: List of timestamps for each data point
            - metadata: Dictionary with metadata about the sequence
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get gas types from config
        gas_types = self.config['gas_types']
        
        # Generate gas concentration time series for each gas type
        gas_data = {}
        
        for gas_type in gas_types:
            # Determine release pattern based on gas type
            if gas_type == 'methane':
                release_pattern = 'gradual'
            elif gas_type == 'propane':
                release_pattern = 'exponential'
            elif gas_type == 'hydrogen':
                release_pattern = 'sudden'
            elif gas_type == 'carbon_monoxide':
                release_pattern = 'logarithmic'
            else:
                release_pattern = 'gradual'
            
            # Generate time series
            # Ensure seed is within valid numpy range if provided
            gas_seed = None
            if seed is not None:
                gas_seed = (seed + abs(hash(gas_type))) % (2**32 - 1)
            
            time_series = self.temporal_model.generate_concentration_time_series(
                gas_type=gas_type,
                start_time=timestamp,
                duration=duration_seconds,
                sample_rate=sample_rate_hz,
                release_pattern=release_pattern,
                seed=gas_seed
            )
            
            # Apply sensor response model
            sensor_model = self.sensor_models.get(gas_type, self.default_sensor_model)
            
            # Get true concentrations and timestamps
            true_concentrations = time_series['concentrations']
            timestamps = time_series['timestamps']
            
            # Simulate sensor response
            measured_concentrations = sensor_model.simulate_batch_response(
                true_concentrations=true_concentrations,
                gas_type=gas_type,
                timestamps=timestamps
            )
            
            # Store both true and measured concentrations
            gas_data[gas_type] = {
                'true_concentrations': true_concentrations,
                'measured_concentrations': measured_concentrations,
                'timestamps': timestamps,
                'metadata': time_series['metadata']
            }
        
        # Create metadata
        metadata = {
            'gas_types': gas_types,
            'duration': duration_seconds,
            'sample_rate': sample_rate_hz,
            'num_samples': int(duration_seconds * sample_rate_hz),
            'start_time': timestamp.isoformat(),
            'end_time': (timestamp + timedelta(seconds=duration_seconds)).isoformat(),
            'seed': seed
        }
        
        return {
            'gas_data': gas_data,
            'timestamps': gas_data[gas_types[0]]['timestamps'] if gas_types else [],
            'metadata': metadata
        }
    
    def generate_concentration(self, 
                              timestamp: datetime,
                              gas_type: str,
                              baseline: float,
                              anomaly_factor: float = 1.0) -> float:
        """
        Generate a gas concentration reading.
        
        Args:
            timestamp: Timestamp for the reading
            gas_type: Type of gas (e.g., 'methane', 'propane')
            baseline: Baseline concentration in PPM
            anomaly_factor: Factor to multiply baseline for anomalies
            
        Returns:
            Gas concentration in PPM
        """
        # Apply anomaly factor to baseline
        true_concentration = baseline * anomaly_factor
        
        # Add some random variation
        variation = np.random.normal(0, 0.05 * true_concentration)
        true_concentration += variation
        
        # Apply sensor response model
        sensor_model = self.sensor_models.get(gas_type, self.default_sensor_model)
        measured_concentration = sensor_model.simulate_response(
            true_concentration=true_concentration,
            gas_type=gas_type,
            timestamp=timestamp
        )
        
        return measured_concentration
    
    def generate_spatial_distribution(self, 
                                    gas_type: str,
                                    source_points: List[Dict[str, Any]],
                                    duration: float,
                                    grid_resolution: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Generate a spatial distribution of gas concentrations.
        
        Args:
            gas_type: Type of gas
            source_points: List of source points with position, strength, etc.
            duration: Duration of diffusion simulation in seconds
            grid_resolution: Optional grid resolution (x, y, z)
            
        Returns:
            3D numpy array representing the gas concentration distribution
        """
        # Set grid resolution if provided
        if grid_resolution is not None:
            self.diffusion_model.grid_resolution = grid_resolution
        
        # Simulate diffusion
        concentration_grid = self.diffusion_model.simulate_diffusion(
            gas_type=gas_type,
            source_points=source_points,
            duration=duration
        )
        
        return concentration_grid
    
    def generate_fire_scenario(self, 
                             gas_types: List[str],
                             start_time: datetime,
                             duration: int,
                             sample_rate: float,
                             fire_type: str,
                             room_params: Dict[str, float],
                             seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate gas concentrations for a fire scenario.
        
        Args:
            gas_types: List of gas types to simulate
            start_time: Start time of the scenario
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            fire_type: Type of fire (e.g., 'smoldering', 'flaming')
            room_params: Dictionary with room parameters (volume, ventilation, etc.)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing the generated concentrations for each gas type
        """
        # Extract room parameters
        fuel_load = room_params.get('fuel_load', 10.0)  # kg
        room_volume = room_params.get('room_volume', 50.0)  # m³
        ventilation_rate = room_params.get('ventilation_rate', 0.5)  # air changes per hour
        
        # Generate fire scenario concentrations
        scenario_data = self.temporal_model.generate_fire_scenario_concentrations(
            gas_types=gas_types,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            fire_type=fire_type,
            fuel_load=fuel_load,
            room_volume=room_volume,
            ventilation_rate=ventilation_rate,
            seed=seed
        )
        
        # Apply sensor response models to each gas type
        for gas_type in gas_types:
            if gas_type in scenario_data['gas_data']:
                sensor_model = self.sensor_models.get(gas_type, self.default_sensor_model)
                
                # Get true concentrations and timestamps
                true_concentrations = scenario_data['gas_data'][gas_type]['concentrations']
                timestamps = scenario_data['timestamps']
                
                # Simulate sensor response
                measured_concentrations = sensor_model.simulate_batch_response(
                    true_concentrations=true_concentrations,
                    gas_type=gas_type,
                    timestamps=timestamps
                )
                
                # Store measured concentrations
                scenario_data['gas_data'][gas_type]['measured_concentrations'] = measured_concentrations
        
        return scenario_data
    
    def coordinate_with_thermal_data(self, 
                                   gas_data: Dict[str, Any],
                                   thermal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate gas concentration data with thermal data.
        
        Args:
            gas_data: Gas concentration data
            thermal_data: Thermal image data
            
        Returns:
            Updated gas concentration data coordinated with thermal data
        """
        return self.temporal_model.coordinate_with_thermal(gas_data, thermal_data)
    
    def to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert generated data to a pandas DataFrame.
        
        Args:
            data: Generated data from the generate method
            
        Returns:
            DataFrame containing the data in a structured format
        """
        # Extract data
        gas_data = data['gas_data']
        timestamps = data['timestamps']
        gas_types = list(gas_data.keys())
        
        # Create DataFrame
        df_data = {'timestamp': timestamps}
        
        for gas_type in gas_types:
            if 'measured_concentrations' in gas_data[gas_type]:
                df_data[f'{gas_type}_concentration'] = gas_data[gas_type]['measured_concentrations']
            
            if 'true_concentrations' in gas_data[gas_type]:
                df_data[f'{gas_type}_true_concentration'] = gas_data[gas_type]['true_concentrations']
        
        return pd.DataFrame(df_data)
    
    def save(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Save generated data to files.
        
        Args:
            data: Generated data from the generate method
            filepath: Base path to save the data
        """
        # Extract data
        gas_data = data['gas_data']
        timestamps = data['timestamps']
        metadata = data['metadata']
        gas_types = list(gas_data.keys())
        
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
        
        # Save gas concentration data
        for gas_type in gas_types:
            gas_path = f"{filepath}_{gas_type}"
            
            # Save measured concentrations
            if 'measured_concentrations' in gas_data[gas_type]:
                measured_path = f"{gas_path}_measured.csv"
                np.savetxt(measured_path, gas_data[gas_type]['measured_concentrations'], delimiter=',')
            
            # Save true concentrations
            if 'true_concentrations' in gas_data[gas_type]:
                true_path = f"{gas_path}_true.csv"
                np.savetxt(true_path, gas_data[gas_type]['true_concentrations'], delimiter=',')
            
            # Save gas metadata
            if 'metadata' in gas_data[gas_type]:
                gas_metadata_path = f"{gas_path}_metadata.json"
                with open(gas_metadata_path, 'w') as f:
                    json.dump(gas_data[gas_type]['metadata'], f, indent=2)
        
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
        gas_data = data['gas_data']
        gas_types = list(gas_data.keys())
        
        # Get base filename without directory
        base_filename = os.path.basename(filepath)
        
        # Upload metadata
        metadata_path = f"{filepath}_metadata.json"
        s3_metadata_key = f"gas_data/{base_filename}_metadata.json"
        self.s3_service.upload_file(metadata_path, s3_metadata_key)
        
        # Upload timestamps
        timestamps_path = f"{filepath}_timestamps.json"
        s3_timestamps_key = f"gas_data/{base_filename}_timestamps.json"
        self.s3_service.upload_file(timestamps_path, s3_timestamps_key)
        
        # Upload gas concentration data
        for gas_type in gas_types:
            gas_path = f"{filepath}_{gas_type}"
            
            # Upload measured concentrations
            measured_path = f"{gas_path}_measured.csv"
            if os.path.exists(measured_path):
                s3_measured_key = f"gas_data/{base_filename}_{gas_type}_measured.csv"
                self.s3_service.upload_file(measured_path, s3_measured_key)
            
            # Upload true concentrations
            true_path = f"{gas_path}_true.csv"
            if os.path.exists(true_path):
                s3_true_key = f"gas_data/{base_filename}_{gas_type}_true.csv"
                self.s3_service.upload_file(true_path, s3_true_key)
            
            # Upload gas metadata
            gas_metadata_path = f"{gas_path}_metadata.json"
            if os.path.exists(gas_metadata_path):
                s3_gas_metadata_key = f"gas_data/{base_filename}_{gas_type}_metadata.json"
                self.s3_service.upload_file(gas_metadata_path, s3_gas_metadata_key)
        
        # Upload DataFrame with all data
        data_path = f"{filepath}_data.csv"
        s3_data_key = f"gas_data/{base_filename}_data.csv"
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
        Generate and save a dataset of gas concentration sequences.
        
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
            fire_types = ['smoldering', 'flaming', 'electrical', 'chemical']
        
        # Set default room parameters if not provided
        if room_params is None:
            room_params = [
                {'fuel_load': 10.0, 'room_volume': 50.0, 'ventilation_rate': 0.5},
                {'fuel_load': 20.0, 'room_volume': 100.0, 'ventilation_rate': 1.0},
                {'fuel_load': 5.0, 'room_volume': 30.0, 'ventilation_rate': 0.2}
            ]
        
        # Get gas types from config
        gas_types = self.config['gas_types']
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': 'gas_concentration_dataset',
            'num_sequences': num_sequences,
            'sequence_duration': sequence_duration,
            'sample_rate': sample_rate,
            'gas_types': gas_types,
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
                gas_types=gas_types,
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
            s3_dataset_key = f"gas_data/datasets/{os.path.basename(output_dir)}_metadata.json"
            self.s3_service.upload_file(dataset_metadata_path, s3_dataset_key)
        
        return dataset_metadata
    
    def visualize_concentrations(self, 
                               data: Dict[str, Any], 
                               gas_types: Optional[List[str]] = None,
                               show_true: bool = True,
                               show_measured: bool = True,
                               show: bool = True,
                               save_path: Optional[str] = None) -> None:
        """
        Visualize gas concentration time series.
        
        Args:
            data: Generated data from the generate method
            gas_types: Optional list of gas types to visualize (default: all)
            show_true: Whether to show true concentrations
            show_measured: Whether to show measured concentrations
            show: Whether to display the plot
            save_path: Optional path to save the visualization
        """
        # Extract data
        gas_data = data['gas_data']
        timestamps = data['timestamps']
        all_gas_types = list(gas_data.keys())
        
        # Use all gas types if not specified
        if gas_types is None:
            gas_types = all_gas_types
        
        # Filter to only include available gas types
        gas_types = [g for g in gas_types if g in all_gas_types]
        
        if not gas_types:
            print("No valid gas types to visualize")
            return
        
        # Create figure
        fig, axes = plt.subplots(len(gas_types), 1, figsize=(12, 4 * len(gas_types)), sharex=True)
        
        # Handle single gas type case
        if len(gas_types) == 1:
            axes = [axes]
        
        # Convert timestamps to relative seconds
        start_time = timestamps[0]
        relative_times = [(ts - start_time).total_seconds() for ts in timestamps]
        
        # Plot each gas type
        for i, gas_type in enumerate(gas_types):
            ax = axes[i]
            
            # Plot true concentrations if available and requested
            if show_true and 'true_concentrations' in gas_data[gas_type]:
                ax.plot(relative_times, gas_data[gas_type]['true_concentrations'], 
                       label=f'True {gas_type}', color='blue', alpha=0.7)
            
            # Plot measured concentrations if available and requested
            if show_measured and 'measured_concentrations' in gas_data[gas_type]:
                ax.plot(relative_times, gas_data[gas_type]['measured_concentrations'], 
                       label=f'Measured {gas_type}', color='red', linestyle='--')
            
            # Add labels and legend
            ax.set_ylabel(f'{gas_type} (ppm)')
            ax.set_title(f'{gas_type} Concentration')
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
    
    def visualize_spatial_distribution(self, 
                                     concentration_grid: np.ndarray,
                                     slice_axis: str = 'z',
                                     slice_index: Optional[int] = None,
                                     show: bool = True,
                                     save_path: Optional[str] = None) -> None:
        """
        Visualize spatial distribution of gas concentrations.
        
        Args:
            concentration_grid: 3D numpy array of gas concentrations
            slice_axis: Axis along which to slice ('x', 'y', or 'z')
            slice_index: Index of the slice (default: middle of the axis)
            show: Whether to display the plot
            save_path: Optional path to save the visualization
        """
        # Get grid dimensions
        x_dim, y_dim, z_dim = concentration_grid.shape
        
        # Set default slice index to middle of the axis
        if slice_index is None:
            if slice_axis == 'x':
                slice_index = x_dim // 2
            elif slice_axis == 'y':
                slice_index = y_dim // 2
            else:  # z
                slice_index = z_dim // 2
        
        # Extract 2D slice
        if slice_axis == 'x':
            slice_data = concentration_grid[slice_index, :, :]
            x_label, y_label = 'Y', 'Z'
        elif slice_axis == 'y':
            slice_data = concentration_grid[:, slice_index, :]
            x_label, y_label = 'X', 'Z'
        else:  # z
            slice_data = concentration_grid[:, :, slice_index]
            x_label, y_label = 'X', 'Y'
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot concentration map
        im = plt.imshow(slice_data, cmap='hot', interpolation='bilinear', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Concentration (ppm)')
        
        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Gas Concentration Distribution ({slice_axis}={slice_index})')
        
        # Add grid
        plt.grid(False)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()