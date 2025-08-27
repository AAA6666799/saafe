"""
Example script demonstrating how to use the feature fusion system.

This script shows how to:
1. Configure the feature fusion system
2. Create sample feature data
3. Fuse features using different fusion techniques
4. Calculate risk scores from fused features
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Import the feature fusion system
from src.feature_engineering.fusion.feature_fusion_system import FeatureFusionSystem, FusionPipeline


def create_sample_thermal_features():
    """
    Create sample thermal features.
    
    Returns:
        Dictionary of thermal features
    """
    return {
        'max_temperature': 120.5,
        'mean_temperature': 85.2,
        'min_temperature': 25.3,
        'temperature_range': 95.2,
        'temperature_std': 15.7,
        'hotspot_features': {
            'hotspot_count': 3,
            'max_hotspot_count': 5,
            'hotspot_area_percentage': 12.5,
            'max_hotspot_area_percentage': 15.2,
            'max_hotspot_temperature': 150.3,
            'max_max_hotspot_temperature': 175.8
        },
        'entropy_features': {
            'temperature_entropy': 4.2,
            'mean_entropy': 3.8,
            'max_entropy': 4.5,
            'entropy_change': 0.7,
            'max_entropy_change': 1.2,
            'mean_entropy_change': 0.5
        },
        'motion_features': {
            'motion_detected': [True, False, True],
            'motion_area_percentage': [5.2, 0.0, 7.5],
            'max_motion_area_percentage': 7.5,
            'motion_intensity': [3.2, 0.0, 4.5],
            'max_motion_intensity': 4.5,
            'motion_frames_percentage': 66.7
        },
        'temperature_rise_features': {
            'max_temperature_slope': 2.5,
            'mean_temperature_slope': 1.8,
            'temperature_slopes': [1.2, 2.5, 1.7],
            'max_temperature_acceleration': 0.8,
            'temperature_accelerations': [0.5, 0.8, 0.3]
        }
    }


def create_sample_gas_features():
    """
    Create sample gas features.
    
    Returns:
        Dictionary of gas features
    """
    return {
        'max_concentration': 75.8,
        'mean_concentration': 45.2,
        'min_concentration': 15.3,
        'concentration_range': 60.5,
        'concentration_std': 12.3,
        'concentration_slopes': {
            'methane': {
                'window_10s': 2.5,
                'window_30s': 1.8,
                'window_60s': 1.2
            },
            'propane': {
                'window_10s': 1.5,
                'window_30s': 1.0,
                'window_60s': 0.7
            }
        },
        'peak_features': {
            'peak_count': 2,
            'max_peak_height': 85.3,
            'mean_peak_height': 80.1,
            'peak_interval': 45.2
        },
        'anomaly_scores': {
            'z_score': 2.8,
            'iqr_score': 1.9,
            'isolation_forest_score': 0.75
        }
    }


def create_sample_environmental_features():
    """
    Create sample environmental features.
    
    Returns:
        Dictionary of environmental features
    """
    return {
        'temperature': 32.5,
        'humidity': 45.2,
        'pressure': 1013.2,
        'temperature_rise': 2.3,
        'humidity_change': -5.2,
        'pressure_change': 1.5,
        'dew_point': 19.8,
        'heat_index': 35.2,
        'voc_levels': {
            'total_voc': 450.3,
            'voc_change_rate': 25.3,
            'voc_anomaly_score': 0.65
        },
        'correlation_features': {
            'temp_humidity_corr': -0.75,
            'temp_pressure_corr': 0.25,
            'humidity_pressure_corr': -0.15
        },
        'anomaly_scores': {
            'temperature_anomaly': 0.85,
            'humidity_anomaly': 0.35,
            'pressure_anomaly': 0.15,
            'combined_anomaly': 0.65
        }
    }


def example_early_fusion():
    """
    Example of early fusion techniques.
    """
    print("\n=== Early Fusion Example ===")
    
    # Create configuration for early fusion
    config = {
        'output_dir': 'examples/output',
        'fusion_components': [
            {
                'type': 'early.FeatureConcatenation',
                'config': {
                    'normalization': 'min_max',
                    'feature_selection': 'none',
                    'include_metadata': True
                }
            },
            {
                'type': 'early.FeatureAveraging',
                'config': {
                    'normalization': 'min_max',
                    'feature_mapping': {
                        'temperature': ['thermal_max_temperature', 'environmental_temperature'],
                        'concentration': ['gas_max_concentration'],
                        'anomaly': ['gas_anomaly_scores_z_score', 'environmental_anomaly_scores_combined_anomaly']
                    },
                    'default_strategy': 'mean',
                    'include_unmapped': False
                }
            }
        ],
        'log_level': 'INFO'
    }
    
    # Create the fusion system
    fusion_system = FeatureFusionSystem(config)
    
    # Create sample features
    thermal_features = create_sample_thermal_features()
    gas_features = create_sample_gas_features()
    environmental_features = create_sample_environmental_features()
    
    # Fuse features
    result = fusion_system.fuse_features(
        thermal_features,
        gas_features,
        environmental_features
    )
    
    # Print results
    print(f"Available fusion components: {fusion_system.get_available_components()}")
    print(f"Feature concatenation result: {len(result['fused_features']['early.FeatureConcatenation']['concatenated_features'])} features")
    print(f"Feature averaging result: {len(result['fused_features']['early.FeatureAveraging']['averaged_features'])} features")
    
    # Calculate risk scores
    for component_id, fused_features in result['fused_features'].items():
        component = fusion_system.fusion_components[component_id]
        risk_score = component.calculate_risk_score(fused_features)
        print(f"Risk score from {component_id}: {risk_score:.4f}")


def example_late_fusion():
    """
    Example of late fusion techniques.
    """
    print("\n=== Late Fusion Example ===")
    
    # Create configuration for late fusion
    config = {
        'output_dir': 'examples/output',
        'fusion_components': [
            {
                'type': 'late.VotingFusion',
                'config': {
                    'voting_method': 'weighted',
                    'decision_threshold': 0.6,
                    'model_weights': {
                        'thermal': 0.5,
                        'gas': 0.3,
                        'environmental': 0.2
                    }
                }
            },
            {
                'type': 'late.ProbabilityFusion',
                'config': {
                    'fusion_method': 'weighted_average',
                    'decision_threshold': 0.6,
                    'model_weights': {
                        'thermal': 0.5,
                        'gas': 0.3,
                        'environmental': 0.2
                    },
                    'calibration': False
                }
            }
        ],
        'log_level': 'INFO'
    }
    
    # Create the fusion system
    fusion_system = FeatureFusionSystem(config)
    
    # Create sample features with decisions and probabilities
    thermal_features = {
        'decision': True,
        'probability': 0.8,
        'confidence': 0.75,
        **create_sample_thermal_features()
    }
    
    gas_features = {
        'decision': True,
        'probability': 0.7,
        'confidence': 0.65,
        **create_sample_gas_features()
    }
    
    environmental_features = {
        'decision': False,
        'probability': 0.3,
        'confidence': 0.6,
        **create_sample_environmental_features()
    }
    
    # Fuse features
    result = fusion_system.fuse_features(
        thermal_features,
        gas_features,
        environmental_features
    )
    
    # Print results
    print(f"Available fusion components: {fusion_system.get_available_components()}")
    
    # Print voting fusion results
    voting_result = result['fused_features']['late.VotingFusion']
    print(f"Voting fusion decision: {voting_result['fused_decision']}")
    print(f"Voting fusion confidence: {voting_result['fused_confidence']:.4f}")
    
    # Print probability fusion results
    prob_result = result['fused_features']['late.ProbabilityFusion']
    print(f"Probability fusion decision: {prob_result['fused_decision']}")
    print(f"Probability fusion probability: {prob_result['fused_probability']:.4f}")
    
    # Calculate risk scores
    for component_id, fused_features in result['fused_features'].items():
        component = fusion_system.fusion_components[component_id]
        risk_score = component.calculate_risk_score(fused_features)
        print(f"Risk score from {component_id}: {risk_score:.4f}")


def example_hybrid_fusion():
    """
    Example of hybrid fusion techniques.
    """
    print("\n=== Hybrid Fusion Example ===")
    
    # Create configuration for hybrid fusion
    config = {
        'output_dir': 'examples/output',
        'fusion_components': [
            {
                'type': 'hybrid.AdaptiveFusion',
                'config': {
                    'adaptation_strategy': 'confidence',
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
                    'default_component': 'early.FeatureConcatenation',
                    'decision_threshold': 0.6
                }
            }
        ],
        'log_level': 'INFO'
    }
    
    # Create the fusion system
    fusion_system = FeatureFusionSystem(config)
    
    # Create sample features with confidence scores
    thermal_features = {
        'confidence': 0.8,
        **create_sample_thermal_features()
    }
    
    gas_features = {
        'confidence': 0.4,
        **create_sample_gas_features()
    }
    
    environmental_features = {
        'confidence': 0.6,
        **create_sample_environmental_features()
    }
    
    # Mock the fusion components (for demonstration purposes)
    fusion_system.fusion_components = {
        'early.FeatureConcatenation': MagicMock(),
        'early.FeatureAveraging': MagicMock(),
        'late.VotingFusion': MagicMock(),
        'late.ProbabilityFusion': MagicMock(),
        'hybrid.MultiLevelFusion': MagicMock(),
        'hybrid.AdaptiveFusion': fusion_system.fusion_components['hybrid.AdaptiveFusion']
    }
    
    # Set up mock return values
    for component_id, component in fusion_system.fusion_components.items():
        if component_id != 'hybrid.AdaptiveFusion':
            component.fuse_features.return_value = {
                'fused_features': {'feature1': 1.0, 'feature2': 2.0},
                'risk_score': 0.75
            }
    
    # Fuse features
    result = fusion_system.fuse_features(
        thermal_features,
        gas_features,
        environmental_features
    )
    
    # Print results
    print(f"Available fusion components: {fusion_system.get_available_components()}")
    print(f"Selected component: {result['fused_features']['hybrid.AdaptiveFusion']['selected_component']}")
    
    # Calculate risk score
    component = fusion_system.fusion_components['hybrid.AdaptiveFusion']
    risk_score = component.calculate_risk_score(result['fused_features']['hybrid.AdaptiveFusion'])
    print(f"Risk score: {risk_score:.4f}")


def example_feature_selection():
    """
    Example of feature selection techniques.
    """
    print("\n=== Feature Selection Example ===")
    
    # Create configuration for feature selection
    config = {
        'output_dir': 'examples/output',
        'fusion_components': [
            {
                'type': 'selection.CorrelationBasedSelection',
                'config': {
                    'correlation_threshold': 0.5,
                    'redundancy_threshold': 0.7,
                    'target_variable': 'risk_score',
                    'correlation_method': 'pearson'
                }
            },
            {
                'type': 'selection.PrincipalComponentAnalysis',
                'config': {
                    'n_components': 3,
                    'standardize': True,
                    'whiten': False,
                    'svd_solver': 'auto'
                }
            }
        ],
        'log_level': 'INFO'
    }
    
    # Create the fusion system
    fusion_system = FeatureFusionSystem(config)
    
    # Create sample features with a risk score
    thermal_features = {
        'risk_score': 0.8,
        **create_sample_thermal_features()
    }
    
    gas_features = {
        'risk_score': 0.6,
        **create_sample_gas_features()
    }
    
    environmental_features = {
        'risk_score': 0.4,
        **create_sample_environmental_features()
    }
    
    # Mock the fusion components (for demonstration purposes)
    fusion_system.fusion_components = {
        'selection.CorrelationBasedSelection': MagicMock(),
        'selection.PrincipalComponentAnalysis': MagicMock()
    }
    
    # Set up mock return values
    fusion_system.fusion_components['selection.CorrelationBasedSelection'].fuse_features.return_value = {
        'selected_features': ['thermal_max_temperature', 'gas_max_concentration', 'environmental_temperature'],
        'selected_feature_count': 3,
        'original_feature_count': 10
    }
    
    fusion_system.fusion_components['selection.PrincipalComponentAnalysis'].fuse_features.return_value = {
        'transformed_feature_count': 3,
        'original_feature_count': 10,
        'total_explained_variance': 0.85,
        'principal_components': {
            'PC1': {'explained_variance': 0.5},
            'PC2': {'explained_variance': 0.25},
            'PC3': {'explained_variance': 0.1}
        }
    }
    
    # Fuse features
    result = fusion_system.fuse_features(
        thermal_features,
        gas_features,
        environmental_features
    )
    
    # Print results
    print(f"Available fusion components: {fusion_system.get_available_components()}")
    
    # Print correlation-based selection results
    corr_result = result['fused_features']['selection.CorrelationBasedSelection']
    print(f"Correlation-based selection: {corr_result['selected_feature_count']} features selected out of {corr_result['original_feature_count']}")
    print(f"Selected features: {corr_result['selected_features']}")
    
    # Print PCA results
    pca_result = result['fused_features']['selection.PrincipalComponentAnalysis']
    print(f"PCA: {pca_result['transformed_feature_count']} components explain {pca_result['total_explained_variance']:.2f} of variance")
    for pc, info in pca_result['principal_components'].items():
        print(f"  {pc}: {info['explained_variance']:.2f} explained variance")


def example_fusion_pipeline():
    """
    Example of using the fusion pipeline.
    """
    print("\n=== Fusion Pipeline Example ===")
    
    # Create configuration for the fusion pipeline
    config = {
        'output_dir': 'examples/output',
        'extraction_config': {
            'output_dir': 'examples/output/extracted_features',
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
            'output_dir': 'examples/output/fused_features',
            'fusion_components': [
                {
                    'type': 'early.FeatureConcatenation',
                    'config': {
                        'normalization': 'min_max',
                        'feature_selection': 'none'
                    }
                },
                {
                    'type': 'late.VotingFusion',
                    'config': {
                        'voting_method': 'weighted',
                        'decision_threshold': 0.6
                    }
                }
            ]
        }
    }
    
    # Create the fusion pipeline
    pipeline = FusionPipeline(config)
    
    # Mock the extraction framework and fusion system
    pipeline.extraction_framework = MagicMock()
    pipeline.fusion_system = MagicMock()
    
    # Set up mock return values
    pipeline.extraction_framework.extract_features.return_value = {
        'metadata': {'dataset_path': 'examples/dataset'},
        'features': {
            'thermal': create_sample_thermal_features(),
            'gas': create_sample_gas_features(),
            'environmental': create_sample_environmental_features()
        }
    }
    
    pipeline.fusion_system.fuse_features.return_value = {
        'metadata': {'fusion_time': datetime.now().isoformat()},
        'fused_features': {
            'early.FeatureConcatenation': {
                'concatenated_features': {'feature1': 1.0, 'feature2': 2.0},
                'risk_score': 0.75
            },
            'late.VotingFusion': {
                'fused_decision': True,
                'fused_confidence': 0.8,
                'risk_score': 0.8
            }
        }
    }
    
    # Process a dataset
    result = pipeline.process_dataset('examples/dataset')
    
    # Print results
    print(f"Extraction completed: {len(result['extracted_features'])} feature sets extracted")
    print(f"Fusion completed: {len(result['fused_features'])} fusion results")
    
    # Print risk scores
    for component_id, fused_features in result['fused_features'].items():
        print(f"Risk score from {component_id}: {fused_features.get('risk_score', 'N/A')}")


if __name__ == '__main__':
    # Create output directory
    os.makedirs('examples/output', exist_ok=True)
    
    # Import mock for examples that use mocking
    from unittest.mock import MagicMock
    
    # Run examples
    example_early_fusion()
    example_late_fusion()
    example_hybrid_fusion()
    example_feature_selection()
    example_fusion_pipeline()