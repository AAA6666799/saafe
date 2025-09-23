#!/usr/bin/env python3
"""
Test script for Task 6: Validation Expansion.
This script implements dataset expansion, cross-validation framework, edge case scenarios, and performance benchmarking.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, Any, List, Tuple
import logging

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expand_synthetic_dataset():
    """Expand synthetic dataset by 50% with diverse scenarios."""
    logger.info("Expanding synthetic dataset by 50% with diverse scenarios...")
    
    try:
        from src.data_generation.flir_scd41_data_generator import create_flir_scd41_generator
        
        # Create generator with default configuration
        generator = create_flir_scd41_generator()
        
        # Generate expanded dataset (50% increase from default 10,000 samples)
        expanded_samples = 15000  # 50% increase
        fire_ratio = 0.15  # Maintain same fire ratio
        
        logger.info(f"Generating {expanded_samples} samples with {fire_ratio*100}% fire ratio...")
        expanded_dataset = generator.generate_training_dataset(expanded_samples, fire_ratio)
        
        logger.info(f"Expanded dataset generated with {len(expanded_dataset)} samples")
        logger.info(f"Fire samples: {sum(expanded_dataset['fire_detected'])}")
        logger.info(f"Normal samples: {len(expanded_dataset) - sum(expanded_dataset['fire_detected'])}")
        
        # Save expanded dataset
        output_path = os.path.join(os.path.dirname(__file__), 'data', 'expanded_flir_scd41_dataset.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        expanded_dataset.to_csv(output_path, index=False)
        logger.info(f"Expanded dataset saved to {output_path}")
        
        return expanded_dataset, output_path
        
    except Exception as e:
        logger.error(f"Error expanding synthetic dataset: {str(e)}")
        return None, None

def implement_cross_validation_framework(dataset: pd.DataFrame):
    """Implement cross-validation framework for robust testing."""
    logger.info("Implementing cross-validation framework...")
    
    try:
        # Extract features and labels
        feature_columns = [col for col in dataset.columns if col not in ['fire_detected', 'timestamp']]
        X = dataset[feature_columns]
        y = dataset['fire_detected']
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Feature columns: {len(feature_columns)}")
        
        # Implement different cross-validation strategies
        cv_strategies = {
            'k_fold': KFold(n_splits=5, shuffle=True, random_state=42),
            'stratified_k_fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        }
        
        cv_results = {}
        
        for name, cv_strategy in cv_strategies.items():
            logger.info(f"Running {name} cross-validation...")
            
            fold_scores = []
            fold_indices = []
            
            for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y)):
                train_samples = len(train_idx)
                val_samples = len(val_idx)
                train_fire_ratio = np.mean(y.iloc[train_idx])
                val_fire_ratio = np.mean(y.iloc[val_idx])
                
                fold_info = {
                    'fold': fold + 1,
                    'train_samples': train_samples,
                    'val_samples': val_samples,
                    'train_fire_ratio': train_fire_ratio,
                    'val_fire_ratio': val_fire_ratio
                }
                
                fold_indices.append(fold_info)
                fold_scores.append(val_fire_ratio)
                
                logger.info(f"  Fold {fold + 1}: Train={train_samples} ({train_fire_ratio:.2%} fire), "
                           f"Val={val_samples} ({val_fire_ratio:.2%} fire)")
            
            cv_results[name] = {
                'folds': fold_indices,
                'mean_fire_ratio': np.mean(fold_scores),
                'std_fire_ratio': np.std(fold_scores),
                'min_fire_ratio': np.min(fold_scores),
                'max_fire_ratio': np.max(fold_scores)
            }
            
            logger.info(f"  {name} results: Mean={cv_results[name]['mean_fire_ratio']:.2%}, "
                       f"Std={cv_results[name]['std_fire_ratio']:.4f}")
        
        return cv_results
        
    except Exception as e:
        logger.error(f"Error implementing cross-validation framework: {str(e)}")
        return None

def add_edge_case_scenarios():
    """Add edge case scenarios to validation set."""
    logger.info("Adding edge case scenarios to validation set...")
    
    try:
        edge_cases = [
            {
                'name': 'Low_temperature_high_co2',
                'description': 'Low temperature but high CO2 levels indicating smoldering fire',
                'thermal_features': {
                    't_mean': 18.0, 't_std': 1.5, 't_max': 22.0, 't_p95': 20.0,
                    't_hot_area_pct': 0.1, 't_hot_largest_blob_pct': 0.05,
                    't_grad_mean': 0.5, 't_grad_std': 0.2, 't_diff_mean': 0.05,
                    't_diff_std': 0.02, 'flow_mag_mean': 0.1, 'flow_mag_std': 0.05,
                    'tproxy_val': 21.0, 'tproxy_delta': 0.05, 'tproxy_vel': 0.02
                },
                'gas_features': {
                    'gas_val': 2200.0, 'gas_delta': 400.0, 'gas_vel': 200.0
                },
                'expected_fire': True
            },
            {
                'name': 'High_temperature_normal_co2',
                'description': 'High temperature but normal CO2 levels indicating electrical fire',
                'thermal_features': {
                    't_mean': 65.0, 't_std': 20.0, 't_max': 95.0, 't_p95': 85.0,
                    't_hot_area_pct': 30.0, 't_hot_largest_blob_pct': 20.0,
                    't_grad_mean': 12.0, 't_grad_std': 5.0, 't_diff_mean': 8.0,
                    't_diff_std': 3.0, 'flow_mag_mean': 6.0, 'flow_mag_std': 2.0,
                    'tproxy_val': 90.0, 'tproxy_delta': 35.0, 'tproxy_vel': 18.0
                },
                'gas_features': {
                    'gas_val': 480.0, 'gas_delta': 10.0, 'gas_vel': 5.0
                },
                'expected_fire': True
            },
            {
                'name': 'Sunlight_heating_effect',
                'description': 'Sunlight heating effect on thermal sensor (false positive)',
                'thermal_features': {
                    't_mean': 45.0, 't_std': 5.0, 't_max': 65.0, 't_p95': 55.0,
                    't_hot_area_pct': 15.0, 't_hot_largest_blob_pct': 8.0,
                    't_grad_mean': 3.0, 't_grad_std': 1.0, 't_diff_mean': 1.0,
                    't_diff_std': 0.5, 'flow_mag_mean': 0.5, 'flow_mag_std': 0.2,
                    'tproxy_val': 60.0, 'tproxy_delta': 5.0, 'tproxy_vel': 2.0
                },
                'gas_features': {
                    'gas_val': 420.0, 'gas_delta': 5.0, 'gas_vel': 2.0
                },
                'expected_fire': False
            },
            {
                'name': 'Steam_obscuration',
                'description': 'Steam obscuring thermal sensor (false positive risk)',
                'thermal_features': {
                    't_mean': 32.0, 't_std': 8.0, 't_max': 55.0, 't_p95': 45.0,
                    't_hot_area_pct': 25.0, 't_hot_largest_blob_pct': 12.0,
                    't_grad_mean': 4.0, 't_grad_std': 2.0, 't_diff_mean': 2.0,
                    't_diff_std': 1.0, 'flow_mag_mean': 1.5, 'flow_mag_std': 0.8,
                    'tproxy_val': 50.0, 'tproxy_delta': 8.0, 'tproxy_vel': 4.0
                },
                'gas_features': {
                    'gas_val': 550.0, 'gas_delta': 25.0, 'gas_vel': 12.0
                },
                'expected_fire': False
            }
        ]
        
        # Convert to DataFrame format
        edge_case_data = []
        for case in edge_cases:
            row_data = {}
            # Add thermal features
            row_data.update(case['thermal_features'])
            # Add gas features
            row_data.update(case['gas_features'])
            # Add label
            row_data['fire_detected'] = int(case['expected_fire'])
            # Add metadata
            row_data['scenario_name'] = case['name']
            row_data['description'] = case['description']
            
            edge_case_data.append(row_data)
        
        edge_case_df = pd.DataFrame(edge_case_data)
        logger.info(f"Created {len(edge_case_df)} edge case scenarios")
        
        # Save edge cases
        output_path = os.path.join(os.path.dirname(__file__), 'data', 'edge_case_scenarios.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        edge_case_df.to_csv(output_path, index=False)
        logger.info(f"Edge case scenarios saved to {output_path}")
        
        return edge_case_df, output_path
        
    except Exception as e:
        logger.error(f"Error adding edge case scenarios: {str(e)}")
        return None, None

def create_performance_benchmarking_suite():
    """Create performance benchmarking suite."""
    logger.info("Creating performance benchmarking suite...")
    
    try:
        # Define benchmark scenarios
        benchmark_scenarios = [
            {
                'name': 'Baseline_performance',
                'description': 'Standard fire detection scenario',
                'thermal_features': {
                    't_mean': 25.0, 't_std': 3.0, 't_max': 45.0, 't_p95': 38.0,
                    't_hot_area_pct': 8.0, 't_hot_largest_blob_pct': 5.0,
                    't_grad_mean': 2.0, 't_grad_std': 0.8, 't_diff_mean': 1.5,
                    't_diff_std': 0.6, 'flow_mag_mean': 1.2, 'flow_mag_std': 0.4,
                    'tproxy_val': 42.0, 'tproxy_delta': 8.0, 'tproxy_vel': 3.0
                },
                'gas_features': {
                    'gas_val': 485.0, 'gas_delta': 25.0, 'gas_vel': 25.0
                }
            },
            {
                'name': 'High_intensity_fire',
                'description': 'High-intensity fire scenario',
                'thermal_features': {
                    't_mean': 75.0, 't_std': 25.0, 't_max': 120.0, 't_p95': 105.0,
                    't_hot_area_pct': 45.0, 't_hot_largest_blob_pct': 30.0,
                    't_grad_mean': 15.0, 't_grad_std': 7.0, 't_diff_mean': 12.0,
                    't_diff_std': 5.0, 'flow_mag_mean': 8.0, 'flow_mag_std': 3.0,
                    'tproxy_val': 110.0, 'tproxy_delta': 45.0, 'tproxy_vel': 22.0
                },
                'gas_features': {
                    'gas_val': 3500.0, 'gas_delta': 800.0, 'gas_vel': 400.0
                }
            },
            {
                'name': 'Early_detection',
                'description': 'Early fire detection scenario',
                'thermal_features': {
                    't_mean': 30.0, 't_std': 5.0, 't_max': 50.0, 't_p95': 42.0,
                    't_hot_area_pct': 3.0, 't_hot_largest_blob_pct': 1.5,
                    't_grad_mean': 1.5, 't_grad_std': 0.6, 't_diff_mean': 0.8,
                    't_diff_std': 0.3, 'flow_mag_mean': 0.5, 'flow_mag_std': 0.2,
                    'tproxy_val': 48.0, 'tproxy_delta': 3.0, 'tproxy_vel': 1.5
                },
                'gas_features': {
                    'gas_val': 650.0, 'gas_delta': 40.0, 'gas_vel': 20.0
                }
            }
        ]
        
        # Create benchmark configuration
        benchmark_config = {
            'scenarios': benchmark_scenarios,
            'iterations_per_scenario': 100,
            'metrics': ['processing_time', 'memory_usage', 'accuracy', 'confidence'],
            'thresholds': {
                'max_processing_time_ms': 1000,
                'min_accuracy': 0.85,
                'min_confidence': 0.7
            }
        }
        
        # Save benchmark configuration
        import json
        output_path = os.path.join(os.path.dirname(__file__), 'config', 'benchmark_config.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(benchmark_config, f, indent=2)
        
        logger.info(f"Benchmark configuration saved to {output_path}")
        logger.info(f"Defined {len(benchmark_scenarios)} benchmark scenarios")
        
        return benchmark_config, output_path
        
    except Exception as e:
        logger.error(f"Error creating performance benchmarking suite: {str(e)}")
        return None, None

def main():
    """Main function to run all Task 6 validation expansion tests."""
    logger.info("Starting Task 6: Validation Expansion Tests")
    logger.info("=" * 50)
    
    results = {}
    
    # Test 1: Expand synthetic dataset
    logger.info("Test 1: Expanding Synthetic Dataset")
    dataset, dataset_path = expand_synthetic_dataset()
    results['dataset_expansion'] = {
        'success': dataset is not None,
        'dataset_path': dataset_path,
        'samples': len(dataset) if dataset is not None else 0
    }
    
    if dataset is None:
        logger.error("Dataset expansion failed, skipping dependent tests")
        results['cross_validation'] = {'success': False, 'reason': 'dataset_expansion_failed'}
        results['edge_cases'] = {'success': False, 'reason': 'dataset_expansion_failed'}
    else:
        # Test 2: Implement cross-validation framework
        logger.info("\nTest 2: Implementing Cross-Validation Framework")
        cv_results = implement_cross_validation_framework(dataset)
        results['cross_validation'] = {
            'success': cv_results is not None,
            'cv_strategies': list(cv_results.keys()) if cv_results else []
        }
        
        # Test 3: Add edge case scenarios
        logger.info("\nTest 3: Adding Edge Case Scenarios")
        edge_cases, edge_case_path = add_edge_case_scenarios()
        results['edge_cases'] = {
            'success': edge_cases is not None,
            'edge_case_path': edge_case_path,
            'scenarios': len(edge_cases) if edge_cases is not None else 0
        }
    
    # Test 4: Create performance benchmarking suite
    logger.info("\nTest 4: Creating Performance Benchmarking Suite")
    benchmark_config, benchmark_path = create_performance_benchmarking_suite()
    results['benchmarking'] = {
        'success': benchmark_config is not None,
        'benchmark_path': benchmark_path,
        'scenarios': len(benchmark_config.get('scenarios', [])) if benchmark_config else 0
    }
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Task 6 Validation Expansion Test Results:")
    
    success_count = sum(1 for test in results.values() if test.get('success', False))
    total_tests = len(results)
    
    for test_name, test_result in results.items():
        status = "✅ PASSED" if test_result.get('success', False) else "❌ FAILED"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        logger.info("🎉 Task 6 Validation Expansion COMPLETED SUCCESSFULLY")
        logger.info("✅ Dataset expanded by 50% with diverse scenarios")
        logger.info("✅ Cross-validation framework implemented")
        logger.info("✅ Edge case scenarios added to validation set")
        logger.info("✅ Performance benchmarking suite created")
        return 0
    else:
        logger.info("❌ Task 6 Validation Expansion had some failures")
        return 1

if __name__ == "__main__":
    sys.exit(main())