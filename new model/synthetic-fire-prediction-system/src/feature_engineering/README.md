# Feature Extraction Framework

This directory contains the implementation of the feature extraction framework for the synthetic fire prediction system. The framework is designed to process synthetic datasets and extract meaningful features for machine learning models.

## Architecture

The feature extraction framework consists of the following main components:

1. **FeatureExtractionFramework**: The main entry point for the feature extraction system. It manages the overall feature extraction pipeline, provides interfaces for different feature extractors, and handles configuration and parameter management.

2. **FeatureExtractionOrchestrator**: Coordinates the execution of feature extraction jobs, manages dependencies between feature extraction steps, handles parallel processing, and provides monitoring and logging capabilities.

3. **FeatureExtractionJobManager**: Creates and manages feature extraction jobs, handles job scheduling and resource allocation, provides job status tracking and error handling, and implements retry mechanisms for failed jobs.

4. **FeatureStorageSystem**: Stores extracted features efficiently, provides fast retrieval of features for model training, implements caching mechanisms for frequently used features, and handles feature compression and decompression.

5. **FeatureVersioningSystem**: Tracks versions of extracted features, maintains feature lineage information, provides feature comparison capabilities, and implements feature metadata management.

6. **AWSFeatureExtractionIntegration**: Provides integration with AWS services for feature extraction, including AWS Glue for feature extraction jobs, S3 and DynamoDB for feature storage, AWS Step Functions for orchestration, and AWS CloudWatch for monitoring.

## Directory Structure

```
feature_engineering/
├── __init__.py
├── base.py                  # Base interfaces and abstract classes
├── framework.py             # Main feature extraction framework
├── orchestrator.py          # Feature extraction orchestrator
├── job_manager.py           # Feature extraction job manager
├── storage.py               # Feature storage system
├── versioning.py            # Feature versioning system
├── versioning_fix.py        # Fixed implementation of versioning system
├── versioning_utils.py      # Utility functions for versioning
├── aws_integration.py       # AWS integration for feature extraction
├── example.py               # Example usage of the framework
├── extractors/              # Feature extractors for different data types
│   ├── __init__.py
│   ├── environmental/       # Environmental data feature extractors
│   ├── gas/                 # Gas data feature extractors
│   └── thermal/             # Thermal data feature extractors
└── fusion/                  # Feature fusion components
```

## Usage

### Basic Usage

```python
from feature_engineering.framework import FeatureExtractionFramework

# Create configuration
config = {
    'output_dir': 'output/features',
    'extractors': [
        {
            'type': 'thermal',
            'config': {
                'max_temperature_threshold': 100.0,
                'hotspot_threshold': 80.0
            }
        },
        {
            'type': 'gas',
            'config': {
                'concentration_thresholds': {
                    'methane': 500.0,
                    'propane': 100.0
                }
            }
        }
    ]
}

# Initialize framework
framework = FeatureExtractionFramework(config)

# Extract features
results = framework.extract_features(
    dataset_path='path/to/dataset',
    output_path='output/features/dataset_001'
)
```

### Advanced Usage with Orchestration

```python
from feature_engineering.framework import FeatureExtractionFramework
from feature_engineering.orchestrator import FeatureExtractionOrchestrator
from feature_engineering.job_manager import FeatureExtractionJobManager

# Initialize components
framework = FeatureExtractionFramework(config)
job_manager = FeatureExtractionJobManager(config['job_manager_config'])
orchestrator = FeatureExtractionOrchestrator(config, framework)

# Process dataset with orchestration
process_metadata = orchestrator.process_dataset(
    dataset_path='path/to/dataset',
    output_path='output/features/dataset_001',
    metadata={'dataset_id': 'dataset_001'}
)
```

### Feature Storage and Versioning

```python
from feature_engineering.storage import FeatureStorageSystem
from feature_engineering.versioning_fix import FeatureVersioningSystemFixed

# Initialize components
storage = FeatureStorageSystem(config['storage_config'])
versioning = FeatureVersioningSystemFixed(config['versioning_config'], storage)

# Store features
feature_id = storage.store_features(
    features=extracted_features,
    feature_type='thermal',
    metadata={'dataset_id': 'dataset_001'}
)

# Register feature version
version_id = versioning.register_feature_version(
    feature_id=feature_id,
    metadata={
        'feature_type': 'thermal',
        'dataset_id': 'dataset_001',
        'extractor_version': '1.0.0'
    }
)

# Retrieve features
features = storage.retrieve_features(feature_id)

# Compare versions
comparison = versioning.compare_versions(version_id1, version_id2)
```

### AWS Integration

```python
from feature_engineering.aws_integration import AWSFeatureExtractionIntegration

# Initialize AWS integration
aws = AWSFeatureExtractionIntegration(config['aws_config'])

# Create Glue job for feature extraction
job_name = aws.create_glue_job(
    job_name='feature-extraction-job',
    script_location='s3://bucket/scripts/feature_extraction.py',
    role_arn='arn:aws:iam::123456789012:role/GlueServiceRole'
)

# Start Glue job
job_run_id = aws.start_glue_job(
    job_name=job_name,
    arguments={
        '--dataset_path': 's3://bucket/datasets/dataset_001',
        '--output_path': 's3://bucket/features/dataset_001'
    }
)

# Store features in DynamoDB
aws.store_features_in_dynamodb(
    table_name='features',
    feature_id=feature_id,
    features=extracted_features,
    metadata={'dataset_id': 'dataset_001'}
)
```

## Example

See `example.py` for a complete example of using the feature extraction framework with synthetic data.

## AWS Integration

The framework provides integration with the following AWS services:

- **AWS Glue**: For running feature extraction jobs at scale
- **Amazon S3**: For storing features and metadata
- **Amazon DynamoDB**: For fast retrieval of features
- **AWS Step Functions**: For orchestrating complex feature extraction workflows
- **Amazon CloudWatch**: For monitoring feature extraction jobs

## Configuration

The framework is highly configurable through a configuration dictionary. Here's an example configuration:

```python
config = {
    'output_dir': 'output/features',
    'log_level': 'INFO',
    'extractors': [
        {
            'type': 'thermal',
            'config': {
                'max_temperature_threshold': 100.0,
                'hotspot_threshold': 80.0,
                'regions_of_interest': [
                    {'x': 0, 'y': 0, 'width': 100, 'height': 100}
                ]
            }
        },
        {
            'type': 'gas',
            'config': {
                'concentration_thresholds': {
                    'methane': 500.0,
                    'propane': 100.0
                },
                'window_sizes': [10, 30, 60]
            }
        }
    ],
    'storage_config': {
        'storage_dir': 'output/features/storage',
        'cache_size_mb': 512,
        'use_compression': True
    },
    'versioning_config': {
        'versioning_dir': 'output/features/versions',
        'max_versions': 5
    },
    'job_manager_config': {
        'max_concurrent_jobs': 4,
        'max_retries': 3
    },
    'aws_config': {
        'region_name': 'us-west-2',
        'default_bucket': 'fire-prediction-features'
    }
}
```

## Extending the Framework

### Adding a New Feature Extractor

1. Create a new class that extends the appropriate base class (`ThermalFeatureExtractor`, `GasFeatureExtractor`, or `EnvironmentalFeatureExtractor`).
2. Implement the required abstract methods.
3. Place the implementation in the appropriate subdirectory under `extractors/`.

Example:

```python
from feature_engineering.base import ThermalFeatureExtractor

class AdvancedThermalFeatureExtractor(ThermalFeatureExtractor):
    def validate_config(self) -> None:
        if 'hotspot_threshold' not in self.config:
            raise ValueError("Missing required config parameter: hotspot_threshold")
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation here
        return features
    
    # Implement other required methods
```

### Adding a New Feature Fusion Component

1. Create a new class that extends the `FeatureFusion` base class.
2. Implement the required abstract methods.
3. Place the implementation in the `fusion/` directory.

Example:

```python
from feature_engineering.base import FeatureFusion

class AdvancedFeatureFusion(FeatureFusion):
    def validate_config(self) -> None:
        if 'fusion_method' not in self.config:
            raise ValueError("Missing required config parameter: fusion_method")
    
    def fuse_features(self, thermal_features, gas_features, environmental_features):
        # Implementation here
        return fused_features
    
    # Implement other required methods