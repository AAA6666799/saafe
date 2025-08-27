"""
Feature Extraction Framework for the synthetic fire prediction system.

This module provides the core framework for extracting features from synthetic datasets,
managing the feature extraction workflow, and providing interfaces for different feature extractors.
"""

from typing import Dict, Any, List, Optional, Union, Type
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import importlib
import inspect

from .base import FeatureExtractor
# AWS integration (optional)
try:
    from ..aws.s3.service import S3ServiceImpl
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    S3ServiceImpl = None


class FeatureExtractionFramework:
    """
    Main class for the feature extraction framework.
    
    This class creates the overall feature extraction pipeline architecture,
    manages the feature extraction workflow, provides interfaces for different
    feature extractors, and handles configuration and parameter management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature extraction framework.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.extractors = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS S3 service if AWS integration is enabled
        self.s3_service = None
        if self.config.get('aws_integration', False):
            self.s3_service = S3ServiceImpl(self.config.get('aws_config', {}))
        
        # Initialize components
        self._setup_logging()
        self._validate_config()
        self._load_extractors()
    
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
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_params = ['output_dir', 'extractors']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate extractors configuration
        if not isinstance(self.config['extractors'], list) or not self.config['extractors']:
            raise ValueError("'extractors' must be a non-empty list")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def _load_extractors(self) -> None:
        """
        Load feature extractors based on configuration.
        
        This method dynamically loads feature extractor classes specified in the configuration.
        """
        for extractor_config in self.config['extractors']:
            if 'type' not in extractor_config or 'config' not in extractor_config:
                self.logger.warning(f"Skipping invalid extractor config: {extractor_config}")
                continue
            
            extractor_type = extractor_config['type']
            extractor_class = self._get_extractor_class(extractor_type)
            
            if extractor_class is None:
                self.logger.warning(f"Could not find extractor class for type: {extractor_type}")
                continue
            
            try:
                extractor = extractor_class(extractor_config['config'])
                self.extractors[extractor_type] = extractor
                self.logger.info(f"Loaded extractor: {extractor_type}")
            except Exception as e:
                self.logger.error(f"Error initializing extractor {extractor_type}: {str(e)}")
    
    def _get_extractor_class(self, extractor_type: str) -> Optional[Type[FeatureExtractor]]:
        """
        Get the feature extractor class based on type.
        
        Args:
            extractor_type: Type of feature extractor
            
        Returns:
            Feature extractor class or None if not found
        """
        # Map extractor types to module paths
        extractor_modules = {
            'thermal': 'extractors.thermal',
            'gas': 'extractors.gas',
            'environmental': 'extractors.environmental'
        }
        
        if extractor_type not in extractor_modules:
            self.logger.warning(f"Unknown extractor type: {extractor_type}")
            return None
        
        try:
            # Import the module dynamically
            module_path = f"{__package__}.{extractor_modules[extractor_type]}"
            module = importlib.import_module(module_path)
            
            # Find the extractor class in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, FeatureExtractor) and 
                    obj.__name__.endswith('FeatureExtractor') and
                    obj.__name__ != 'FeatureExtractor'):
                    return obj
            
            self.logger.warning(f"No suitable extractor class found in module: {module_path}")
            return None
        except ImportError as e:
            self.logger.error(f"Error importing extractor module: {str(e)}")
            return None
    
    def extract_features(self, 
                        dataset_path: str, 
                        output_path: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract features from a dataset.
        
        Args:
            dataset_path: Path to the dataset
            output_path: Optional path to save extracted features
            metadata: Optional metadata about the dataset
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        self.logger.info(f"Extracting features from dataset: {dataset_path}")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.config['output_dir'], 
                                      f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize results dictionary
        results = {
            'metadata': {
                'dataset_path': dataset_path,
                'extraction_time': datetime.now().isoformat(),
                'feature_version': str(uuid.uuid4()),
                'extractors_used': list(self.extractors.keys())
            },
            'features': {}
        }
        
        # Add provided metadata if available
        if metadata:
            results['metadata'].update(metadata)
        
        # Extract features using each extractor
        for extractor_type, extractor in self.extractors.items():
            try:
                self.logger.info(f"Running {extractor_type} feature extraction")
                
                # Load data specific to this extractor type
                data = self._load_data_for_extractor(dataset_path, extractor_type)
                
                # Extract features
                features = extractor.extract_features(data)
                
                # Save features
                extractor_output_path = os.path.join(output_path, f"{extractor_type}_features.json")
                extractor.save(features, extractor_output_path)
                
                # Add to results
                results['features'][extractor_type] = features
                
                self.logger.info(f"Completed {extractor_type} feature extraction")
            except Exception as e:
                self.logger.error(f"Error in {extractor_type} feature extraction: {str(e)}")
        
        # Save overall results
        results_path = os.path.join(output_path, "extraction_results.json")
        with open(results_path, 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"features/{os.path.basename(output_path)}/extraction_results.json"
            self.s3_service.upload_file(results_path, s3_key)
            self.logger.info(f"Uploaded extraction results to S3: {s3_key}")
        
        self.logger.info(f"Feature extraction completed. Results saved to: {output_path}")
        return results
    
    def _load_data_for_extractor(self, dataset_path: str, extractor_type: str) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Load data specific to an extractor type from the dataset.
        
        Args:
            dataset_path: Path to the dataset
            extractor_type: Type of extractor
            
        Returns:
            Data for the extractor
        """
        # This is a simplified implementation. In a real system, this would load
        # the appropriate data files based on the extractor type.
        if extractor_type == 'thermal':
            # Load thermal data (e.g., thermal image frames)
            thermal_path = os.path.join(dataset_path, 'thermal_data.json')
            if os.path.exists(thermal_path):
                with open(thermal_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Thermal data not found at: {thermal_path}")
                return {}
        
        elif extractor_type == 'gas':
            # Load gas concentration data
            gas_path = os.path.join(dataset_path, 'gas_data.csv')
            if os.path.exists(gas_path):
                return pd.read_csv(gas_path)
            else:
                self.logger.warning(f"Gas data not found at: {gas_path}")
                return pd.DataFrame()
        
        elif extractor_type == 'environmental':
            # Load environmental data
            env_path = os.path.join(dataset_path, 'environmental_data.csv')
            if os.path.exists(env_path):
                return pd.read_csv(env_path)
            else:
                self.logger.warning(f"Environmental data not found at: {env_path}")
                return pd.DataFrame()
        
        else:
            self.logger.warning(f"Unknown extractor type: {extractor_type}")
            return {}
    
    def get_available_extractors(self) -> List[str]:
        """
        Get a list of available feature extractors.
        
        Returns:
            List of extractor types
        """
        return list(self.extractors.keys())
    
    def get_extractor_info(self, extractor_type: str) -> Dict[str, Any]:
        """
        Get information about a specific feature extractor.
        
        Args:
            extractor_type: Type of extractor
            
        Returns:
            Dictionary with extractor information
        """
        if extractor_type not in self.extractors:
            return {'error': f"Extractor not found: {extractor_type}"}
        
        extractor = self.extractors[extractor_type]
        return {
            'type': extractor_type,
            'features': extractor.get_feature_names(),
            'config': self.config['extractors'][list(self.extractors.keys()).index(extractor_type)]['config']
        }