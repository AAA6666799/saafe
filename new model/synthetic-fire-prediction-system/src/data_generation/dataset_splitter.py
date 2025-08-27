"""
Dataset splitter for synthetic fire data.

This module provides functionality for splitting datasets into train, validation,
and test sets with balanced representation of different scenario types.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import os
import json
import logging
import shutil
from datetime import datetime
import random

from ..aws.s3.service import S3ServiceImpl


class DatasetSplitter:
    """
    Class for splitting datasets into train, validation, and test sets.
    
    This class provides methods for creating balanced splits of the generated data
    with configurable splitting ratios and strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset splitter with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.validate_config()
        
        # Initialize S3 service if AWS integration is enabled
        self.s3_service = None
        if self.config.get('aws_integration', False):
            self.s3_service = S3ServiceImpl(self.config.get('aws_config', {}))
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """
        Set up logging configuration.
        """
        log_level = self.config.get('log_level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required configuration sections
        required_sections = ['output_dir']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check AWS integration
        if self.config.get('aws_integration', False):
            if 'aws_config' not in self.config:
                raise ValueError("aws_config is required when aws_integration is enabled")
            
            if 'default_bucket' not in self.config['aws_config']:
                raise ValueError("default_bucket is required in aws_config")
        
        # Set default values if not provided
        if 'train_ratio' not in self.config:
            self.config['train_ratio'] = 0.7
        
        if 'val_ratio' not in self.config:
            self.config['val_ratio'] = 0.15
        
        if 'test_ratio' not in self.config:
            self.config['test_ratio'] = 0.15
        
        if 'stratify' not in self.config:
            self.config['stratify'] = True
        
        if 'random_seed' not in self.config:
            self.config['random_seed'] = 42
    
    def split_dataset(self, 
                     dataset_dir: str, 
                     output_dir: Optional[str] = None,
                     copy_files: bool = False) -> Dict[str, Any]:
        """
        Split a dataset into train, validation, and test sets.
        
        Args:
            dataset_dir: Directory containing the dataset
            output_dir: Directory to save the split dataset (default: dataset_dir/split)
            copy_files: Whether to copy files to output directory (default: False)
            
        Returns:
            Dictionary with metadata about the split dataset
        """
        self.logger.info(f"Splitting dataset in {dataset_dir}")
        
        # Set output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(dataset_dir, 'split')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset metadata
        metadata_path = os.path.join(dataset_dir, 'complete_dataset_metadata.json')
        if not os.path.exists(metadata_path):
            # Try to find any metadata file
            metadata_files = [f for f in os.listdir(dataset_dir) if f.endswith('_metadata.json')]
            if metadata_files:
                metadata_path = os.path.join(dataset_dir, metadata_files[0])
            else:
                raise FileNotFoundError(f"No metadata file found in {dataset_dir}")
        
        with open(metadata_path, 'r') as f:
            dataset_metadata = json.load(f)
        
        # Collect all scenarios from all components
        all_scenarios = []
        scenario_types = {}
        
        if 'components' in dataset_metadata:
            # Process complete dataset metadata
            for component in dataset_metadata['components']:
                component_type = component['type']
                component_metadata_path = component['metadata_path']
                
                if os.path.exists(component_metadata_path):
                    with open(component_metadata_path, 'r') as f:
                        component_metadata = json.load(f)
                        
                    if 'scenarios' in component_metadata:
                        for scenario in component_metadata['scenarios']:
                            scenario['scenario_type'] = component_type
                            all_scenarios.append(scenario)
                            
                            # Track scenario types
                            if component_type not in scenario_types:
                                scenario_types[component_type] = []
                            scenario_types[component_type].append(scenario)
        elif 'scenarios' in dataset_metadata:
            # Process single component metadata
            component_type = dataset_metadata.get('scenario_type', 'unknown')
            for scenario in dataset_metadata['scenarios']:
                scenario['scenario_type'] = component_type
                all_scenarios.append(scenario)
                
                # Track scenario types
                if component_type not in scenario_types:
                    scenario_types[component_type] = []
                scenario_types[component_type].append(scenario)
        
        self.logger.info(f"Found {len(all_scenarios)} scenarios of {len(scenario_types)} types")
        
        # Set random seed
        random_seed = self.config['random_seed']
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Split the dataset
        train_scenarios = []
        val_scenarios = []
        test_scenarios = []
        
        if self.config['stratify']:
            # Stratified split by scenario type
            for scenario_type, scenarios in scenario_types.items():
                # Shuffle scenarios
                random.shuffle(scenarios)
                
                # Calculate split indices
                n_scenarios = len(scenarios)
                n_train = int(n_scenarios * self.config['train_ratio'])
                n_val = int(n_scenarios * self.config['val_ratio'])
                
                # Split scenarios
                train_scenarios.extend(scenarios[:n_train])
                val_scenarios.extend(scenarios[n_train:n_train + n_val])
                test_scenarios.extend(scenarios[n_train + n_val:])
                
                self.logger.debug(f"Split {scenario_type}: {n_train} train, {n_val} val, {len(scenarios) - n_train - n_val} test")
        else:
            # Random split
            random.shuffle(all_scenarios)
            
            # Calculate split indices
            n_scenarios = len(all_scenarios)
            n_train = int(n_scenarios * self.config['train_ratio'])
            n_val = int(n_scenarios * self.config['val_ratio'])
            
            # Split scenarios
            train_scenarios = all_scenarios[:n_train]
            val_scenarios = all_scenarios[n_train:n_train + n_val]
            test_scenarios = all_scenarios[n_train + n_val:]
        
        self.logger.info(f"Split dataset: {len(train_scenarios)} train, {len(val_scenarios)} val, {len(test_scenarios)} test")
        
        # Create split metadata
        split_metadata = {
            'dataset_name': dataset_metadata.get('dataset_name', 'split_dataset'),
            'original_dataset_dir': dataset_dir,
            'split_date': datetime.now().isoformat(),
            'train_ratio': self.config['train_ratio'],
            'val_ratio': self.config['val_ratio'],
            'test_ratio': self.config['test_ratio'],
            'stratify': self.config['stratify'],
            'random_seed': random_seed,
            'train_scenarios': len(train_scenarios),
            'val_scenarios': len(val_scenarios),
            'test_scenarios': len(test_scenarios),
            'scenario_type_distribution': {
                'train': self._count_scenario_types(train_scenarios),
                'val': self._count_scenario_types(val_scenarios),
                'test': self._count_scenario_types(test_scenarios)
            },
            'splits': {
                'train': [s['scenario_id'] if 'scenario_id' in s else i for i, s in enumerate(train_scenarios)],
                'val': [s['scenario_id'] if 'scenario_id' in s else i for i, s in enumerate(val_scenarios)],
                'test': [s['scenario_id'] if 'scenario_id' in s else i for i, s in enumerate(test_scenarios)]
            }
        }
        
        # Save split metadata
        split_metadata_path = os.path.join(output_dir, 'split_metadata.json')
        with open(split_metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        # Create split directories
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Copy or link files if requested
        if copy_files:
            self._copy_scenarios(train_scenarios, train_dir)
            self._copy_scenarios(val_scenarios, val_dir)
            self._copy_scenarios(test_scenarios, test_dir)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"datasets/splits/{os.path.basename(output_dir)}_split_metadata.json"
            self.s3_service.upload_file(split_metadata_path, s3_key)
            self.logger.info(f"Uploaded split metadata to S3: {s3_key}")
        
        return split_metadata
    
    def _count_scenario_types(self, scenarios: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count the number of scenarios of each type.
        
        Args:
            scenarios: List of scenario metadata dictionaries
            
        Returns:
            Dictionary mapping scenario types to counts
        """
        type_counts = {}
        for scenario in scenarios:
            scenario_type = scenario.get('scenario_type', 'unknown')
            if scenario_type not in type_counts:
                type_counts[scenario_type] = 0
            type_counts[scenario_type] += 1
        return type_counts
    
    def _copy_scenarios(self, scenarios: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Copy scenario files to the output directory.
        
        Args:
            scenarios: List of scenario metadata dictionaries
            output_dir: Directory to copy files to
        """
        for i, scenario in enumerate(scenarios):
            if 'scenario_dir' in scenario:
                scenario_dir = scenario['scenario_dir']
                scenario_type = scenario.get('scenario_type', 'unknown')
                scenario_id = scenario.get('scenario_id', i)
                
                # Create output directory for this scenario
                scenario_output_dir = os.path.join(output_dir, f"{scenario_type}_{scenario_id}")
                os.makedirs(scenario_output_dir, exist_ok=True)
                
                # Copy files
                if os.path.exists(scenario_dir):
                    for filename in os.listdir(scenario_dir):
                        src_path = os.path.join(scenario_dir, filename)
                        dst_path = os.path.join(scenario_output_dir, filename)
                        
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dst_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    
    def create_data_manifest(self, split_dir: str) -> Dict[str, Any]:
        """
        Create a data manifest for the split dataset.
        
        Args:
            split_dir: Directory containing the split dataset
            
        Returns:
            Dictionary with the data manifest
        """
        self.logger.info(f"Creating data manifest for split dataset in {split_dir}")
        
        # Load split metadata
        split_metadata_path = os.path.join(split_dir, 'split_metadata.json')
        if not os.path.exists(split_metadata_path):
            raise FileNotFoundError(f"Split metadata file not found: {split_metadata_path}")
        
        with open(split_metadata_path, 'r') as f:
            split_metadata = json.load(f)
        
        # Create data manifest
        manifest = {
            'dataset_name': split_metadata.get('dataset_name', 'split_dataset'),
            'creation_date': datetime.now().isoformat(),
            'splits': {}
        }
        
        # Process each split
        for split_name in ['train', 'val', 'test']:
            split_dir_path = os.path.join(split_dir, split_name)
            if not os.path.exists(split_dir_path):
                continue
            
            # Get all scenario directories in this split
            scenario_dirs = [d for d in os.listdir(split_dir_path) 
                           if os.path.isdir(os.path.join(split_dir_path, d))]
            
            # Create manifest for this split
            split_manifest = []
            for scenario_dir in scenario_dirs:
                scenario_path = os.path.join(split_dir_path, scenario_dir)
                
                # Find metadata file
                metadata_files = [f for f in os.listdir(scenario_path) 
                                if f.endswith('_metadata.json')]
                
                if metadata_files:
                    metadata_path = os.path.join(scenario_path, metadata_files[0])
                    with open(metadata_path, 'r') as f:
                        scenario_metadata = json.load(f)
                    
                    # Find data files
                    data_files = []
                    for root, _, files in os.walk(scenario_path):
                        for file in files:
                            if file.endswith(('.csv', '.npy', '.json')):
                                rel_path = os.path.relpath(os.path.join(root, file), split_dir)
                                data_files.append(rel_path)
                    
                    # Add to manifest
                    split_manifest.append({
                        'scenario_id': scenario_dir,
                        'scenario_type': scenario_metadata.get('scenario_type', 'unknown'),
                        'metadata_path': os.path.relpath(metadata_path, split_dir),
                        'data_files': data_files
                    })
            
            manifest['splits'][split_name] = split_manifest
        
        # Save manifest
        manifest_path = os.path.join(split_dir, 'data_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"datasets/manifests/{os.path.basename(split_dir)}_manifest.json"
            self.s3_service.upload_file(manifest_path, s3_key)
            self.logger.info(f"Uploaded data manifest to S3: {s3_key}")
        
        return manifest
    
    def create_balanced_subset(self, 
                              split_dir: str, 
                              output_dir: str,
                              max_scenarios_per_type: Optional[int] = None,
                              scenario_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a balanced subset of the split dataset.
        
        Args:
            split_dir: Directory containing the split dataset
            output_dir: Directory to save the balanced subset
            max_scenarios_per_type: Maximum number of scenarios per type (default: use all)
            scenario_types: List of scenario types to include (default: all types)
            
        Returns:
            Dictionary with metadata about the balanced subset
        """
        self.logger.info(f"Creating balanced subset of split dataset in {split_dir}")
        
        # Load split metadata
        split_metadata_path = os.path.join(split_dir, 'split_metadata.json')
        if not os.path.exists(split_metadata_path):
            raise FileNotFoundError(f"Split metadata file not found: {split_metadata_path}")
        
        with open(split_metadata_path, 'r') as f:
            split_metadata = json.load(f)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each split
        balanced_metadata = {
            'dataset_name': f"{split_metadata.get('dataset_name', 'split_dataset')}_balanced",
            'original_split_dir': split_dir,
            'creation_date': datetime.now().isoformat(),
            'max_scenarios_per_type': max_scenarios_per_type,
            'scenario_types': scenario_types,
            'splits': {}
        }
        
        for split_name in ['train', 'val', 'test']:
            split_dir_path = os.path.join(split_dir, split_name)
            if not os.path.exists(split_dir_path):
                continue
            
            # Get all scenario directories in this split
            scenario_dirs = [d for d in os.listdir(split_dir_path) 
                           if os.path.isdir(os.path.join(split_dir_path, d))]
            
            # Group scenarios by type
            scenarios_by_type = {}
            for scenario_dir in scenario_dirs:
                # Extract scenario type from directory name
                parts = scenario_dir.split('_')
                if len(parts) > 1:
                    scenario_type = parts[0]
                else:
                    scenario_type = 'unknown'
                
                if scenario_types is None or scenario_type in scenario_types:
                    if scenario_type not in scenarios_by_type:
                        scenarios_by_type[scenario_type] = []
                    scenarios_by_type[scenario_type].append(scenario_dir)
            
            # Balance scenarios
            balanced_scenarios = {}
            for scenario_type, scenarios in scenarios_by_type.items():
                # Shuffle scenarios
                random.shuffle(scenarios)
                
                # Limit number of scenarios if requested
                if max_scenarios_per_type is not None:
                    scenarios = scenarios[:max_scenarios_per_type]
                
                balanced_scenarios[scenario_type] = scenarios
            
            # Create output directory for this split
            split_output_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_output_dir, exist_ok=True)
            
            # Copy selected scenarios
            for scenario_type, scenarios in balanced_scenarios.items():
                for scenario_dir in scenarios:
                    src_dir = os.path.join(split_dir_path, scenario_dir)
                    dst_dir = os.path.join(split_output_dir, scenario_dir)
                    
                    if os.path.exists(src_dir):
                        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            
            # Add to metadata
            balanced_metadata['splits'][split_name] = {
                'scenario_counts': {t: len(s) for t, s in balanced_scenarios.items()},
                'total_scenarios': sum(len(s) for s in balanced_scenarios.values())
            }
        
        # Save balanced metadata
        balanced_metadata_path = os.path.join(output_dir, 'balanced_metadata.json')
        with open(balanced_metadata_path, 'w') as f:
            json.dump(balanced_metadata, f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"datasets/balanced/{os.path.basename(output_dir)}_metadata.json"
            self.s3_service.upload_file(balanced_metadata_path, s3_key)
            self.logger.info(f"Uploaded balanced subset metadata to S3: {s3_key}")
        
        return balanced_metadata