"""
Example usage of the feature extraction framework.

This module demonstrates how to use the feature extraction framework
to extract features from synthetic datasets.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from .framework import FeatureExtractionFramework
from .orchestrator import FeatureExtractionOrchestrator
from .job_manager import FeatureExtractionJobManager
from .storage import FeatureStorageSystem
from .versioning_fix import FeatureVersioningSystemFixed as FeatureVersioningSystem
from .aws_integration import AWSFeatureExtractionIntegration


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_example_config() -> Dict[str, Any]:
    """
    Create an example configuration for the feature extraction framework.
    
    Returns:
        Dictionary containing configuration parameters
    """
    # Base directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(base_dir, 'output', 'features')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration for feature extraction framework
    config = {
        'output_dir': output_dir,
        'log_level': 'INFO',
        'extractors': [
            {
                'type': 'thermal',
                'config': {
                    'max_temperature_threshold': 100.0,
                    'hotspot_threshold': 80.0,
                    'regions_of_interest': [
                        {'x': 0, 'y': 0, 'width': 100, 'height': 100},
                        {'x': 100, 'y': 100, 'width': 100, 'height': 100}
                    ]
                }
            },
            {
                'type': 'gas',
                'config': {
                    'concentration_thresholds': {
                        'methane': 500.0,
                        'propane': 100.0,
                        'hydrogen': 40.0
                    },
                    'window_sizes': [10, 30, 60]
                }
            },
            {
                'type': 'environmental',
                'config': {
                    'temperature_range': [-10.0, 50.0],
                    'humidity_range': [0.0, 100.0],
                    'pressure_range': [900.0, 1100.0],
                    'window_sizes': [10, 30, 60]
                }
            }
        ],
        'storage_config': {
            'storage_dir': os.path.join(output_dir, 'storage'),
            'cache_size_mb': 512,
            'use_compression': True,
            'compression_level': 6
        },
        'versioning_config': {
            'versioning_dir': os.path.join(output_dir, 'versions'),
            'max_versions': 5
        },
        'job_manager_config': {
            'max_concurrent_jobs': 4,
            'max_retries': 3,
            'retry_delay_seconds': 60,
            'job_timeout_seconds': 3600
        },
        'aws_integration': False,
        'aws_config': {
            'region_name': 'us-west-2',
            'default_bucket': 'fire-prediction-features',
            'use_glue': False,
            'use_dynamodb': False,
            'use_step_functions': False,
            'use_cloudwatch': False
        }
    }
    
    return config


def create_example_dataset() -> Dict[str, Any]:
    """
    Create an example synthetic dataset for feature extraction.
    
    Returns:
        Dictionary containing the dataset
    """
    # Create timestamp range
    timestamps = pd.date_range(
        start=datetime.now().replace(microsecond=0),
        periods=100,
        freq='1s'
    )
    
    # Create thermal data (simulated thermal images)
    thermal_data = {
        'timestamps': timestamps,
        'frames': []
    }
    
    for i in range(len(timestamps)):
        # Create a 32x32 thermal image with random temperatures
        # Higher temperatures in the center to simulate a hotspot
        frame = np.random.uniform(20.0, 30.0, size=(32, 32))
        
        # Add a hotspot if in the middle of the dataset
        if 30 <= i < 70:
            center_x, center_y = 16, 16
            radius = 5
            hotspot_temp = 50.0 + (i - 30) * 1.0  # Increasing temperature
            
            for x in range(32):
                for y in range(32):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        frame[y, x] = hotspot_temp * (1 - dist/radius)
        
        thermal_data['frames'].append(frame)
    
    # Create gas concentration data
    gas_data = pd.DataFrame({
        'timestamp': timestamps,
        'methane': np.random.uniform(10.0, 20.0, size=len(timestamps)),
        'propane': np.random.uniform(5.0, 10.0, size=len(timestamps)),
        'hydrogen': np.random.uniform(1.0, 5.0, size=len(timestamps))
    })
    
    # Add gas concentration spike if in the middle of the dataset
    for i in range(30, 70):
        gas_data.loc[i, 'methane'] += (i - 30) * 10.0
        gas_data.loc[i, 'propane'] += (i - 30) * 5.0
        gas_data.loc[i, 'hydrogen'] += (i - 30) * 2.0
    
    # Create environmental data
    environmental_data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.random.uniform(20.0, 25.0, size=len(timestamps)),
        'humidity': np.random.uniform(40.0, 60.0, size=len(timestamps)),
        'pressure': np.random.uniform(1000.0, 1010.0, size=len(timestamps)),
        'voc': np.random.uniform(0.1, 0.5, size=len(timestamps))
    })
    
    # Add temperature increase if in the middle of the dataset
    for i in range(30, 70):
        environmental_data.loc[i, 'temperature'] += (i - 30) * 0.2
        environmental_data.loc[i, 'humidity'] -= (i - 30) * 0.1
        environmental_data.loc[i, 'voc'] += (i - 30) * 0.05
    
    # Combine into a dataset
    dataset = {
        'metadata': {
            'dataset_id': 'example_dataset_001',
            'creation_time': datetime.now().isoformat(),
            'scenario_type': 'fire_scenario',
            'duration_seconds': len(timestamps)
        },
        'thermal_data': thermal_data,
        'gas_data': gas_data,
        'environmental_data': environmental_data
    }
    
    return dataset


def save_example_dataset(dataset: Dict[str, Any], output_dir: str) -> str:
    """
    Save an example dataset to disk.
    
    Args:
        dataset: Dictionary containing the dataset
        output_dir: Output directory
        
    Returns:
        Path to the saved dataset
    """
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, 'example_dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save metadata
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(dataset['metadata'], f, indent=2)
    
    # Save thermal data
    thermal_dir = os.path.join(dataset_dir, 'thermal')
    os.makedirs(thermal_dir, exist_ok=True)
    
    # Save thermal metadata
    thermal_metadata = {
        'timestamps': [ts.isoformat() for ts in dataset['thermal_data']['timestamps']],
        'frame_shape': dataset['thermal_data']['frames'][0].shape
    }
    thermal_metadata_path = os.path.join(thermal_dir, 'metadata.json')
    with open(thermal_metadata_path, 'w') as f:
        json.dump(thermal_metadata, f, indent=2)
    
    # Save thermal frames as numpy arrays
    for i, frame in enumerate(dataset['thermal_data']['frames']):
        frame_path = os.path.join(thermal_dir, f'frame_{i:04d}.npy')
        np.save(frame_path, frame)
    
    # Save gas data as CSV
    gas_path = os.path.join(dataset_dir, 'gas_data.csv')
    dataset['gas_data'].to_csv(gas_path, index=False)
    
    # Save environmental data as CSV
    env_path = os.path.join(dataset_dir, 'environmental_data.csv')
    dataset['environmental_data'].to_csv(env_path, index=False)
    
    # Save a combined dataset file for convenience
    thermal_data_json = {
        'timestamps': [ts.isoformat() for ts in dataset['thermal_data']['timestamps']],
        'frames': [frame.tolist() for frame in dataset['thermal_data']['frames']]
    }
    
    combined_data = {
        'metadata': dataset['metadata'],
        'thermal_data': thermal_data_json,
        'gas_data': dataset['gas_data'].to_dict(orient='records'),
        'environmental_data': dataset['environmental_data'].to_dict(orient='records')
    }
    
    combined_path = os.path.join(dataset_dir, 'dataset.json')
    with open(combined_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    return dataset_dir


def run_feature_extraction_example():
    """
    Run an example of the feature extraction framework.
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting feature extraction example")
    
    # Create configuration
    config = create_example_config()
    logger.info(f"Created configuration with output directory: {config['output_dir']}")
    
    # Create example dataset
    dataset = create_example_dataset()
    logger.info("Created example dataset")
    
    # Save dataset to disk
    dataset_dir = save_example_dataset(dataset, config['output_dir'])
    logger.info(f"Saved example dataset to: {dataset_dir}")
    
    # Initialize components
    storage_system = FeatureStorageSystem(config['storage_config'])
    versioning_system = FeatureVersioningSystem(config['versioning_config'], storage_system)
    framework = FeatureExtractionFramework(config)
    job_manager = FeatureExtractionJobManager(config['job_manager_config'])
    orchestrator = FeatureExtractionOrchestrator(config, framework)
    
    # Initialize AWS integration if enabled
    aws_integration = None
    if config['aws_integration']:
        aws_integration = AWSFeatureExtractionIntegration(config['aws_config'])
    
    logger.info("Initialized feature extraction components")
    
    # Extract features
    logger.info("Extracting features from example dataset")
    extraction_results = framework.extract_features(
        dataset_path=dataset_dir,
        output_path=os.path.join(config['output_dir'], 'example_features'),
        metadata=dataset['metadata']
    )
    
    logger.info(f"Extracted features: {extraction_results}")
    
    # Register feature version
    for extractor_type, features in extraction_results['features'].items():
        feature_id = f"{extractor_type}_{dataset['metadata']['dataset_id']}"
        
        # Store features
        storage_system.store_features(
            features=features,
            feature_type=extractor_type,
            metadata=dataset['metadata'],
            feature_id=feature_id
        )
        
        # Register version
        version_metadata = {
            'feature_type': extractor_type,
            'dataset_id': dataset['metadata']['dataset_id'],
            'scenario_type': dataset['metadata']['scenario_type'],
            'extractor_version': '1.0.0'
        }
        
        version_id = versioning_system.register_feature_version(
            feature_id=feature_id,
            metadata=version_metadata
        )
        
        logger.info(f"Registered {extractor_type} feature version: {version_id}")
    
    logger.info("Feature extraction example completed successfully")


if __name__ == "__main__":
    run_feature_extraction_example()