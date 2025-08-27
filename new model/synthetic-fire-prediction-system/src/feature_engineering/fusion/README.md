# Feature Fusion System

The Feature Fusion System is a comprehensive framework for combining features from different sources (thermal, gas, environmental, temporal) to create more powerful predictive features for fire detection.

## Overview

The system provides various fusion techniques organized into the following categories:

1. **Early Fusion**: Combines features at the data level or early in the feature extraction process
2. **Late Fusion**: Combines decisions or outputs from different models
3. **Hybrid Fusion**: Uses a combination of early and late fusion approaches
4. **Feature Selection**: Selects the most relevant features for the fire prediction task

## Architecture

The feature fusion system consists of the following main components:

- `FeatureFusionSystem`: Main class that manages the feature fusion workflow
- `FusionPipeline`: Combines feature extraction and fusion into a single pipeline
- Fusion techniques:
  - Early fusion: `DataLevelFusion`, `FeatureConcatenation`, `FeatureAveraging`, `WeightedFeatureCombination`
  - Late fusion: `DecisionLevelFusion`, `ProbabilityFusion`, `RankingFusion`, `VotingFusion`
  - Hybrid fusion: `HierarchicalFusion`, `CascadedFusion`, `AdaptiveFusion`, `MultiLevelFusion`
  - Feature selection: `CorrelationBasedSelection`, `MutualInformationSelection`, `PrincipalComponentAnalysis`, `RecursiveFeatureElimination`, `GeneticAlgorithmSelection`

## Usage

### Basic Usage

```python
from src.feature_engineering.fusion.feature_fusion_system import FeatureFusionSystem

# Create configuration
config = {
    'output_dir': 'output',
    'fusion_components': [
        {
            'type': 'early.FeatureConcatenation',
            'config': {
                'normalization': 'min_max',
                'feature_selection': 'none'
            }
        }
    ]
}

# Create fusion system
fusion_system = FeatureFusionSystem(config)

# Fuse features
result = fusion_system.fuse_features(
    thermal_features,
    gas_features,
    environmental_features
)

# Calculate risk score
for component_id, fused_features in result['fused_features'].items():
    component = fusion_system.fusion_components[component_id]
    risk_score = component.calculate_risk_score(fused_features)
    print(f"Risk score: {risk_score}")
```

### Using the Fusion Pipeline

```python
from src.feature_engineering.fusion.feature_fusion_system import FusionPipeline

# Create configuration
config = {
    'output_dir': 'output',
    'extraction_config': {
        'output_dir': 'output/extracted_features',
        'extractors': [
            {
                'type': 'thermal',
                'config': {'max_temperature_threshold': 100, 'hotspot_threshold': 80}
            },
            {
                'type': 'gas',
                'config': {'concentration_threshold': 50}
            },
            {
                'type': 'environmental',
                'config': {'temperature_threshold': 30}
            }
        ]
    },
    'fusion_config': {
        'output_dir': 'output/fused_features',
        'fusion_components': [
            {
                'type': 'early.FeatureConcatenation',
                'config': {
                    'normalization': 'min_max',
                    'feature_selection': 'none'
                }
            }
        ]
    }
}

# Create fusion pipeline
pipeline = FusionPipeline(config)

# Process a dataset
result = pipeline.process_dataset('path/to/dataset')

# Process streaming data
stream_result = pipeline.process_stream(
    thermal_data,
    gas_data,
    environmental_data
)
```

## Fusion Techniques

### Early Fusion

- **DataLevelFusion**: Combines raw data before feature extraction
- **FeatureConcatenation**: Concatenates features from different sources
- **FeatureAveraging**: Averages features from different sources
- **WeightedFeatureCombination**: Combines features with learned weights

### Late Fusion

- **DecisionLevelFusion**: Combines decisions from different models
- **ProbabilityFusion**: Combines probability outputs from different models
- **RankingFusion**: Combines rankings from different models
- **VotingFusion**: Uses voting mechanisms to combine model outputs

### Hybrid Fusion

- **HierarchicalFusion**: Uses a hierarchical approach to fusion
- **CascadedFusion**: Uses a cascaded approach to fusion
- **AdaptiveFusion**: Adapts fusion strategy based on data characteristics
- **MultiLevelFusion**: Combines features at multiple levels

### Feature Selection

- **CorrelationBasedSelection**: Selects features based on correlation
- **MutualInformationSelection**: Selects features based on mutual information
- **PrincipalComponentAnalysis**: Reduces dimensionality using PCA
- **RecursiveFeatureElimination**: Eliminates features recursively
- **GeneticAlgorithmSelection**: Uses genetic algorithms for feature selection

## Configuration

Each fusion component has its own configuration parameters. Here are some examples:

### FeatureConcatenation

```python
{
    'type': 'early.FeatureConcatenation',
    'config': {
        'normalization': 'min_max',  # 'min_max', 'z_score', or 'none'
        'feature_selection': 'none',  # 'all', 'top_k', 'threshold', or 'none'
        'top_k': 10,  # Number of top features to select
        'threshold': 0.5,  # Threshold for feature selection
        'include_metadata': True  # Whether to include metadata in the result
    }
}
```

### VotingFusion

```python
{
    'type': 'late.VotingFusion',
    'config': {
        'voting_method': 'weighted',  # 'hard', 'soft', 'weighted', or 'dynamic'
        'decision_threshold': 0.6,  # Threshold for decision
        'model_weights': {
            'thermal': 0.5,
            'gas': 0.3,
            'environmental': 0.2
        }
    }
}
```

### AdaptiveFusion

```python
{
    'type': 'hybrid.AdaptiveFusion',
    'config': {
        'adaptation_strategy': 'confidence',  # 'confidence', 'quality', 'context', or 'ensemble'
        'confidence_thresholds': {
            'high': 0.7,
            'low': 0.3
        },
        'component_mappings': {
            'all_high': 'early.FeatureConcatenation',
            'all_low': 'late.VotingFusion',
            'thermal_high': 'early.FeatureAveraging',
            'gas_high': 'late.ProbabilityFusion',
            'env_high': 'late.ProbabilityFusion',
            'mixed': 'hybrid.MultiLevelFusion'
        },
        'default_component': 'early.FeatureConcatenation'
    }
}
```

### PrincipalComponentAnalysis

```python
{
    'type': 'selection.PrincipalComponentAnalysis',
    'config': {
        'n_components': 3,  # Number of components to keep
        'standardize': True,  # Whether to standardize features
        'whiten': False,  # Whether to whiten features
        'svd_solver': 'auto'  # SVD solver to use
    }
}
```

## AWS Integration

The feature fusion system integrates with AWS services for storing and retrieving features. To enable AWS integration, include the following in your configuration:

```python
{
    'aws_integration': True,
    'aws_config': {
        'region_name': 'us-west-2',
        'bucket_name': 'my-feature-bucket',
        'prefix': 'features/'
    }
}
```

## Examples

See the `examples/feature_fusion_example.py` file for complete examples of using the feature fusion system.

## Testing

Unit tests are provided for all components of the feature fusion system. To run the tests:

```bash
python -m unittest discover -s tests/unit/feature_engineering/fusion