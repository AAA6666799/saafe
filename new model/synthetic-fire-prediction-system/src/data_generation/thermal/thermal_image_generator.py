"""
Thermal image generator for synthetic fire data.

This module provides functionality for generating synthetic thermal images
with realistic thermal gradients based on fire physics.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, BinaryIO
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import io
import json
import matplotlib.pyplot as plt
from matplotlib import cm
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
from PIL import Image

from ..base import ThermalDataGenerator
from .hotspot_simulator import HotspotSimulator, HotspotShape
from .temporal_evolution_model import TemporalEvolutionModel, FireType
from .noise_injector import NoiseInjector

# AWS integration (optional)
try:
    from ...aws.s3.service import S3ServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None


class ThermalImageGenerator(ThermalDataGenerator):
    """
    Class for generating synthetic thermal images.
    
    This class implements the ThermalDataGenerator interface and provides
    methods for generating realistic thermal images with configurable parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the thermal image generator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        
        # Initialize components
        self.hotspot_simulator = HotspotSimulator(self.config.get('hotspot_config', {}))
        self.temporal_model = TemporalEvolutionModel(
            self.config.get('temporal_config', {}),
            self.hotspot_simulator
        )
        self.noise_injector = NoiseInjector(self.config.get('noise_config', {}))
        
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
        # Check resolution
        if 'resolution' in self.config:
            resolution = self.config['resolution']
            if not isinstance(resolution, tuple) or len(resolution) != 2:
                raise ValueError("resolution must be a tuple of (height, width)")
            
            if resolution[0] <= 0 or resolution[1] <= 0:
                raise ValueError("resolution dimensions must be positive")
        else:
            # Set default resolution if not provided
            self.config['resolution'] = (288, 384)  # Standard thermal camera resolution
        
        # Check temperature range
        if 'min_temperature' in self.config and 'max_temperature' in self.config:
            if self.config['min_temperature'] >= self.config['max_temperature']:
                raise ValueError("min_temperature must be less than max_temperature")
        else:
            # Set default temperature range if not provided
            self.config['min_temperature'] = 20.0  # 20°C
            self.config['max_temperature'] = 500.0  # 500°C
        
        # Check output formats
        if 'output_formats' in self.config:
            valid_formats = ['numpy', 'png', 'jpg', 'tiff', 'csv']
            for fmt in self.config['output_formats']:
                if fmt not in valid_formats:
                    raise ValueError(f"Invalid output format: {fmt}. Must be one of: {valid_formats}")
        else:
            # Set default output formats if not provided
            self.config['output_formats'] = ['numpy', 'png']
        
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
        Generate a sequence of synthetic thermal images.
        
        Args:
            timestamp: Start timestamp for the generated data
            duration_seconds: Duration of data to generate in seconds
            sample_rate_hz: Sample rate in Hz
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary containing the generated data:
            - frames: List of thermal images as numpy arrays
            - timestamps: List of timestamps for each frame
            - metadata: Dictionary with metadata about the sequence
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get resolution from config
        resolution = self.config['resolution']
        
        # Generate fire sequence using temporal evolution model
        fire_type = self.config.get('fire_type', 'standard')
        
        sequence = self.temporal_model.generate_fire_sequence(
            image_shape=resolution,
            start_time=timestamp,
            duration=duration_seconds,
            frame_rate=sample_rate_hz,
            fire_type=fire_type,
            seed=seed
        )
        
        # Apply noise to each frame
        frames = sequence['frames']
        noisy_frames = []
        
        for frame in frames:
            # Apply noise
            noisy_frame = self.noise_injector.apply_noise(frame)
            
            # Apply camera characteristics
            processed_frame = self.noise_injector.simulate_camera_characteristics(noisy_frame)
            
            noisy_frames.append(processed_frame)
        
        # Update sequence with noisy frames
        sequence['frames'] = noisy_frames
        
        return sequence
    
    def generate_frame(self,
                      timestamp: datetime,
                      hotspots: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        Generate a single thermal image frame.
        
        Args:
            timestamp: Timestamp for the frame
            hotspots: Optional list of hotspot parameters
            
        Returns:
            2D numpy array representing the thermal image
        """
        # Get resolution from config
        resolution = self.config['resolution']
        
        if hotspots is None:
            # Generate a default hotspot if none provided
            hotspots = [{
                'center': (resolution[0] // 2, resolution[1] // 2),
                'radius': min(resolution) // 10,
                'intensity': 1.0,
                'shape': 'circular'
            }]
        
        # Generate frame with hotspots
        frame = self.hotspot_simulator.generate_multiple_hotspots(
            image_shape=resolution,
            hotspots=hotspots
        )
        
        # Apply noise
        noisy_frame = self.noise_injector.apply_noise(frame)
        
        # Apply camera characteristics
        processed_frame = self.noise_injector.simulate_camera_characteristics(noisy_frame)
        
        return processed_frame
    
    def to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert generated data to a pandas DataFrame.
        
        Args:
            data: Generated data from the generate method
            
        Returns:
            DataFrame containing the data in a structured format
        """
        frames = data['frames']
        timestamps = data['timestamps']
        
        # Extract key statistics from each frame
        stats = []
        
        for i, frame in enumerate(frames):
            # Calculate statistics
            min_temp = np.min(frame)
            max_temp = np.max(frame)
            mean_temp = np.mean(frame)
            std_temp = np.std(frame)
            
            # Calculate hotspot area (pixels above threshold)
            threshold = min_temp + 0.5 * (max_temp - min_temp)
            hotspot_area = np.sum(frame > threshold) / frame.size
            
            stats.append({
                'timestamp': timestamps[i],
                'min_temperature': min_temp,
                'max_temperature': max_temp,
                'mean_temperature': mean_temp,
                'std_temperature': std_temp,
                'hotspot_area_ratio': hotspot_area,
                'frame_index': i
            })
        
        return pd.DataFrame(stats)
    
    def save(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Save generated data to files.
        
        Args:
            data: Generated data from the generate method
            filepath: Base path to save the data
        """
        frames = data['frames']
        timestamps = data['timestamps']
        metadata = data['metadata']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save frames in specified formats
        output_formats = self.config['output_formats']
        
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
        
        # Save frames
        for i, frame in enumerate(frames):
            frame_path_base = f"{filepath}_frame_{i:04d}"
            
            # Save in each specified format
            for fmt in output_formats:
                if fmt == 'numpy':
                    np.save(f"{frame_path_base}.npy", frame)
                elif fmt == 'png' or fmt == 'jpg':
                    # Normalize to 0-255 for image formats
                    norm_frame = self._normalize_for_visualization(frame)
                    cv2.imwrite(f"{frame_path_base}.{fmt}", norm_frame)
                elif fmt == 'tiff':
                    # Save as 16-bit TIFF with original values
                    # Scale to 0-65535 range
                    min_val = np.min(frame)
                    max_val = np.max(frame)
                    scaled = (frame - min_val) / (max_val - min_val) * 65535
                    scaled_uint16 = scaled.astype(np.uint16)
                    cv2.imwrite(f"{frame_path_base}.tiff", scaled_uint16)
                elif fmt == 'csv':
                    # Save as CSV with original values
                    np.savetxt(f"{frame_path_base}.csv", frame, delimiter=',')
        
        # Save DataFrame with statistics
        df = self.to_dataframe(data)
        df.to_csv(f"{filepath}_stats.csv", index=False)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            self._upload_to_s3(filepath, data)
    
    def _normalize_for_visualization(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize a thermal frame for visualization.
        
        Args:
            frame: Thermal image as a 2D numpy array
            
        Returns:
            Normalized image as a uint8 numpy array
        """
        # Get temperature range
        min_temp = np.min(frame)
        max_temp = np.max(frame)
        
        # Normalize to 0-255
        normalized = (frame - min_temp) / (max_temp - min_temp) * 255
        
        # Apply colormap (inferno is good for thermal visualization)
        colormap = cm.get_cmap('inferno')
        colored = colormap(normalized / 255.0)
        
        # Convert to BGR for OpenCV
        colored_bgr = (colored[:, :, :3] * 255).astype(np.uint8)
        colored_bgr = cv2.cvtColor(colored_bgr, cv2.COLOR_RGB2BGR)
        
        return colored_bgr
    
    def _upload_to_s3(self, filepath: str, data: Dict[str, Any]) -> None:
        """
        Upload generated data to S3.
        
        Args:
            filepath: Base path of the saved files
            data: Generated data from the generate method
        """
        if self.s3_service is None:
            raise ValueError("S3 service is not initialized")
        
        # Get base filename without directory
        base_filename = os.path.basename(filepath)
        
        # Upload metadata
        metadata_path = f"{filepath}_metadata.json"
        s3_metadata_key = f"thermal_data/{base_filename}_metadata.json"
        self.s3_service.upload_file(metadata_path, s3_metadata_key)
        
        # Upload timestamps
        timestamps_path = f"{filepath}_timestamps.json"
        s3_timestamps_key = f"thermal_data/{base_filename}_timestamps.json"
        self.s3_service.upload_file(timestamps_path, s3_timestamps_key)
        
        # Upload frames
        frames = data['frames']
        output_formats = self.config['output_formats']
        
        for i, frame in enumerate(frames):
            frame_path_base = f"{filepath}_frame_{i:04d}"
            
            for fmt in output_formats:
                local_path = f"{frame_path_base}.{fmt if fmt != 'numpy' else 'npy'}"
                s3_key = f"thermal_data/{base_filename}_frame_{i:04d}.{fmt if fmt != 'numpy' else 'npy'}"
                self.s3_service.upload_file(local_path, s3_key)
        
        # Upload statistics
        stats_path = f"{filepath}_stats.csv"
        s3_stats_key = f"thermal_data/{base_filename}_stats.csv"
        self.s3_service.upload_file(stats_path, s3_stats_key)
    
    def generate_and_save_dataset(self, 
                                output_dir: str,
                                num_sequences: int,
                                sequence_duration: int,
                                sample_rate: float,
                                fire_types: Optional[List[str]] = None,
                                upload_to_s3: bool = False,
                                seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate and save a dataset of thermal image sequences.
        
        Args:
            output_dir: Directory to save the dataset
            num_sequences: Number of sequences to generate
            sequence_duration: Duration of each sequence in seconds
            sample_rate: Sample rate in Hz
            fire_types: Optional list of fire types to include
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
            fire_types = ['standard', 'smoldering', 'rapid_combustion', 'electrical', 'chemical']
        
        # Generate dataset metadata
        dataset_metadata = {
            'dataset_name': 'thermal_image_dataset',
            'num_sequences': num_sequences,
            'sequence_duration': sequence_duration,
            'sample_rate': sample_rate,
            'resolution': self.config['resolution'],
            'temperature_range': [self.config['min_temperature'], self.config['max_temperature']],
            'fire_types': fire_types,
            'creation_date': datetime.now().isoformat(),
            'sequences': []
        }
        
        # Generate sequences
        for i in range(num_sequences):
            # Select random fire type
            fire_type = np.random.choice(fire_types)
            
            # Set fire type in config
            self.config['fire_type'] = fire_type
            
            # Generate start time (random time in the past week)
            start_time = datetime.now() - timedelta(days=np.random.randint(0, 7))
            
            # Generate sequence
            sequence = self.generate(
                timestamp=start_time,
                duration_seconds=sequence_duration,
                sample_rate_hz=sample_rate,
                seed=seed + i if seed is not None else None
            )
            
            # Save sequence
            sequence_path = os.path.join(output_dir, f'sequence_{i:04d}')
            self.save(sequence, sequence_path)
            
            # Add to dataset metadata
            sequence_metadata = sequence['metadata'].copy()
            sequence_metadata['sequence_id'] = i
            sequence_metadata['sequence_path'] = sequence_path
            dataset_metadata['sequences'].append(sequence_metadata)
        
        # Save dataset metadata
        dataset_metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Upload to S3 if requested
        if upload_to_s3 and self.s3_service is not None:
            s3_dataset_key = f"thermal_data/datasets/{os.path.basename(output_dir)}_metadata.json"
            self.s3_service.upload_file(dataset_metadata_path, s3_dataset_key)
        
        return dataset_metadata
    
    def visualize_frame(self, 
                       frame: np.ndarray, 
                       show: bool = True, 
                       save_path: Optional[str] = None) -> None:
        """
        Visualize a thermal image frame.
        
        Args:
            frame: Thermal image as a 2D numpy array
            show: Whether to display the image
            save_path: Optional path to save the visualization
        """
        # Normalize and apply colormap
        colored = self._normalize_for_visualization(frame)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Display image
        plt.imshow(colored_rgb)
        
        # Add colorbar
        min_temp = np.min(frame)
        max_temp = np.max(frame)
        cbar = plt.colorbar()
        cbar.set_label('Temperature (°C)')
        
        # Set ticks on colorbar
        num_ticks = 5
        ticks = np.linspace(0, 1, num_ticks)
        tick_labels = np.linspace(min_temp, max_temp, num_ticks)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:.1f}' for t in tick_labels])
        
        # Add title with statistics
        mean_temp = np.mean(frame)
        std_temp = np.std(frame)
        plt.title(f'Thermal Image\nMin: {min_temp:.1f}°C, Max: {max_temp:.1f}°C\nMean: {mean_temp:.1f}°C, Std: {std_temp:.1f}°C')
        
        # Remove axis ticks
        plt.xticks([])
        plt.yticks([])
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def export_to_standard_format(self, 
                                frame: np.ndarray, 
                                format_type: str, 
                                output_path: str) -> None:
        """
        Export a thermal image to a standard thermal image format.
        
        Args:
            frame: Thermal image as a 2D numpy array
            format_type: Format type ('radiometric_jpg', 'flir_seq', 'tiff16')
            output_path: Path to save the exported file
        """
        if format_type == 'radiometric_jpg':
            # Create a radiometric JPEG (simplified version)
            # In a real implementation, this would include proper EXIF data
            # with temperature calibration parameters
            
            # Normalize and apply colormap
            colored = self._normalize_for_visualization(frame)
            
            # Add temperature scale
            h, w = colored.shape[:2]
            scale_width = 30
            scale_img = np.zeros((h, w + scale_width, 3), dtype=np.uint8)
            scale_img[:, :w] = colored
            
            # Create temperature scale on the right
            for i in range(h):
                scale_value = 255 - int(i * 255 / h)
                scale_img[i, w:] = [scale_value, scale_value, scale_value]
            
            # Add temperature labels
            min_temp = np.min(frame)
            max_temp = np.max(frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(scale_img, f'{max_temp:.1f}°C', (w + 5, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(scale_img, f'{min_temp:.1f}°C', (w + 5, h - 10), font, 0.5, (255, 255, 255), 1)
            
            # Save as JPEG
            cv2.imwrite(output_path, scale_img)
            
            # In a real implementation, we would also save temperature data in EXIF
            
        elif format_type == 'flir_seq':
            # FLIR SEQ format is proprietary and would require a specialized library
            # This is a simplified placeholder
            
            # Save metadata in a separate JSON file
            metadata = {
                'camera_model': 'FLIR A320',
                'resolution': frame.shape,
                'temperature_range': [np.min(frame), np.max(frame)],
                'emissivity': 0.95,
                'distance': 1.0,
                'relative_humidity': 50.0,
                'atmospheric_temperature': 20.0,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = output_path + '.meta.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save raw temperature data
            np.save(output_path, frame)
            
        elif format_type == 'tiff16':
            # Save as 16-bit TIFF with temperature data
            min_val = np.min(frame)
            max_val = np.max(frame)
            
            # Scale to 0-65535 range
            scaled = (frame - min_val) / (max_val - min_val) * 65535
            scaled_uint16 = scaled.astype(np.uint16)
            
            # Save as TIFF
            cv2.imwrite(output_path, scaled_uint16)
            
            # Save calibration data
            calibration = {
                'min_temperature': float(min_val),
                'max_temperature': float(max_val),
                'scale_factor': float((max_val - min_val) / 65535),
                'offset': float(min_val)
            }
            
            calibration_path = output_path + '.calibration.json'
            with open(calibration_path, 'w') as f:
                json.dump(calibration, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")