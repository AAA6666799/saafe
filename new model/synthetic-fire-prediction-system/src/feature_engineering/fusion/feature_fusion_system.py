"""
Feature Fusion System for the synthetic fire prediction system.

This module provides the core system for fusing features from different sources
(thermal, gas, environmental, temporal) to create more powerful predictive features.
"""

from typing import Dict, Any, List, Optional, Union, Type
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import importlib
import inspect

from ..base import FeatureFusion
from ...aws.s3.service import S3ServiceImpl


class FeatureFusionSystem:
    """
    Main class for the feature fusion system.
    
    This class creates the overall feature fusion architecture,
    manages the feature fusion workflow, provides interfaces for different
    fusion techniques, and handles configuration and parameter management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature fusion system.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.fusion_components = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS S3 service if AWS integration is enabled
        self.s3_service = None
        if self.config.get('aws_integration', False):
            self.s3_service = S3ServiceImpl(self.config.get('aws_config', {}))
        
        # Initialize components
        self._setup_logging()
        self._validate_config()
        self._load_fusion_components()
    
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
        required_params = ['output_dir', 'fusion_components']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate fusion components configuration
        if not isinstance(self.config['fusion_components'], list) or not self.config['fusion_components']:
            raise ValueError("'fusion_components' must be a non-empty list")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def _load_fusion_components(self) -> None:
        """
        Load fusion components based on configuration.
        
        This method dynamically loads fusion component classes specified in the configuration.
        """
        for component_config in self.config['fusion_components']:
            if 'type' not in component_config or 'config' not in component_config:
                self.logger.warning(f"Skipping invalid fusion component config: {component_config}")
                continue
            
            component_type = component_config['type']
            component_class = self._get_fusion_component_class(component_type)
            
            if component_class is None:
                self.logger.warning(f"Could not find fusion component class for type: {component_type}")
                continue
            
            try:
                component = component_class(component_config['config'])
                self.fusion_components[component_type] = component
                self.logger.info(f"Loaded fusion component: {component_type}")
            except Exception as e:
                self.logger.error(f"Error initializing fusion component {component_type}: {str(e)}")
    
    def _get_fusion_component_class(self, component_type: str) -> Optional[Type[FeatureFusion]]:
        """
        Get the fusion component class based on type.
        
        Args:
            component_type: Type of fusion component
            
        Returns:
            Fusion component class or None if not found
        """
        # Map component types to module paths
        component_modules = {
            'early': 'early',
            'late': 'late',
            'hybrid': 'hybrid',
            'selection': 'selection'
        }
        
        # Extract the category (early, late, hybrid, selection)
        parts = component_type.split('.')
        if len(parts) < 2 or parts[0] not in component_modules:
            self.logger.warning(f"Unknown fusion component type: {component_type}")
            return None
        
        category = parts[0]
        component_name = parts[1]
        
        try:
            # Import the module dynamically
            module_path = f"{__package__}.{component_modules[category]}.{component_name.lower()}"
            module = importlib.import_module(module_path)
            
            # Find the fusion component class in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, FeatureFusion) and 
                    obj.__name__ == component_name):
                    return obj
            
            self.logger.warning(f"No suitable fusion component class found in module: {module_path}")
            return None
        except ImportError as e:
            self.logger.error(f"Error importing fusion component module: {str(e)}")
            return None
    
    def fuse_features(self, 
                     thermal_features: Dict[str, Any],
                     gas_features: Dict[str, Any],
                     environmental_features: Dict[str, Any],
                     temporal_features: Optional[Dict[str, Any]] = None,
                     output_path: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fuse features from different sources.
        
        Args:
            thermal_features: Features extracted from thermal data
            gas_features: Features extracted from gas data
            environmental_features: Features extracted from environmental data
            temporal_features: Optional features extracted from temporal data
            output_path: Optional path to save fused features
            metadata: Optional metadata about the features
            
        Returns:
            Dictionary containing the fused features and metadata
        """
        self.logger.info("Fusing features from different sources")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.config['output_dir'], 
                                     f"fused_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize results dictionary
        results = {
            'metadata': {
                'fusion_time': datetime.now().isoformat(),
                'fusion_version': self.config.get('version', '1.0.0'),
                'components_used': list(self.fusion_components.keys())
            },
            'fused_features': {}
        }
        
        # Add provided metadata if available
        if metadata:
            results['metadata'].update(metadata)
        
        # Fuse features using each component
        for component_type, component in self.fusion_components.items():
            try:
                self.logger.info(f"Running {component_type} feature fusion")
                
                # Fuse features
                fused_features = component.fuse_features(
                    thermal_features=thermal_features,
                    gas_features=gas_features,
                    environmental_features=environmental_features
                )
                
                # Calculate risk score
                risk_score = component.calculate_risk_score(fused_features)
                fused_features['risk_score'] = risk_score
                
                # Save fused features
                component_output_path = os.path.join(output_path, f"{component_type}_fused_features.json")
                with open(component_output_path, 'w') as f:
                    json.dump(fused_features, f, indent=2)
                
                # Add to results
                results['fused_features'][component_type] = fused_features
                
                self.logger.info(f"Completed {component_type} feature fusion with risk score: {risk_score}")
            except Exception as e:
                self.logger.error(f"Error in {component_type} feature fusion: {str(e)}")
        
        # Save overall results
        results_path = os.path.join(output_path, "fusion_results.json")
        with open(results_path, 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        
        # Upload to S3 if AWS integration is enabled
        if self.s3_service is not None:
            s3_key = f"fused_features/{os.path.basename(output_path)}/fusion_results.json"
            self.s3_service.upload_file(results_path, s3_key)
            self.logger.info(f"Uploaded fusion results to S3: {s3_key}")
        
        self.logger.info(f"Feature fusion completed. Results saved to: {output_path}")
        return results
    
    def get_available_components(self) -> List[str]:
        """
        Get a list of available fusion components.
        
        Returns:
            List of fusion component types
        """
        return list(self.fusion_components.keys())
    
    def get_component_info(self, component_type: str) -> Dict[str, Any]:
        """
        Get information about a specific fusion component.
        
        Args:
            component_type: Type of fusion component
            
        Returns:
            Dictionary with component information
        """
        if component_type not in self.fusion_components:
            return {'error': f"Fusion component not found: {component_type}"}
        
        component = self.fusion_components[component_type]
        return {
            'type': component_type,
            'config': self.config['fusion_components'][list(self.fusion_components.keys()).index(component_type)]['config']
        }


class FusionPipeline:
    """
    Pipeline for combining feature extraction and fusion.
    
    This class combines feature extraction and fusion into a single pipeline,
    manages the flow of data through the pipeline, provides configuration options,
    and handles error cases and reporting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fusion pipeline.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Import required modules
        from ..framework import FeatureExtractionFramework
        
        # Initialize components
        self.extraction_framework = FeatureExtractionFramework(config.get('extraction_config', {}))
        self.fusion_system = FeatureFusionSystem(config.get('fusion_config', {}))
        
        # Set up progress tracking
        self.progress = {
            'total_steps': 0,
            'completed_steps': 0,
            'current_step': '',
            'errors': []
        }
    
    def process_dataset(self, 
                       dataset_path: str, 
                       output_path: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a dataset through the complete pipeline.
        
        Args:
            dataset_path: Path to the dataset
            output_path: Optional path to save results
            metadata: Optional metadata about the dataset
            
        Returns:
            Dictionary containing the pipeline results
        """
        self.logger.info(f"Processing dataset: {dataset_path}")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.config.get('output_dir', 'output'), 
                                     f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize progress tracking
        self._reset_progress(total_steps=2)  # Extract and fuse
        
        try:
            # Step 1: Extract features
            self._update_progress(step="Extracting features")
            extraction_results = self.extraction_framework.extract_features(
                dataset_path=dataset_path,
                output_path=os.path.join(output_path, "extracted_features"),
                metadata=metadata
            )
            self._increment_progress()
            
            # Step 2: Fuse features
            self._update_progress(step="Fusing features")
            fusion_results = self.fusion_system.fuse_features(
                thermal_features=extraction_results['features'].get('thermal', {}),
                gas_features=extraction_results['features'].get('gas', {}),
                environmental_features=extraction_results['features'].get('environmental', {}),
                output_path=os.path.join(output_path, "fused_features"),
                metadata=extraction_results['metadata']
            )
            self._increment_progress()
            
            # Combine results
            pipeline_results = {
                'metadata': {
                    'dataset_path': dataset_path,
                    'processing_time': datetime.now().isoformat(),
                    'extraction_metadata': extraction_results['metadata'],
                    'fusion_metadata': fusion_results['metadata']
                },
                'extracted_features': extraction_results['features'],
                'fused_features': fusion_results['fused_features']
            }
            
            # Save pipeline results
            results_path = os.path.join(output_path, "pipeline_results.json")
            with open(results_path, 'w') as f:
                json.dump(pipeline_results['metadata'], f, indent=2)
            
            self.logger.info(f"Pipeline processing completed. Results saved to: {output_path}")
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Error in pipeline processing: {str(e)}"
            self.logger.error(error_msg)
            self._add_error(error_msg)
            
            # Return partial results if available
            return {
                'metadata': {
                    'dataset_path': dataset_path,
                    'processing_time': datetime.now().isoformat(),
                    'error': str(e),
                    'progress': self.progress
                }
            }
    
    def process_stream(self, 
                      thermal_data: Dict[str, Any],
                      gas_data: Dict[str, Any],
                      environmental_data: Dict[str, Any],
                      timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process streaming data through the pipeline.
        
        Args:
            thermal_data: Thermal data for processing
            gas_data: Gas data for processing
            environmental_data: Environmental data for processing
            timestamp: Optional timestamp for the data
            
        Returns:
            Dictionary containing the processing results
        """
        self.logger.info("Processing streaming data")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Extract features from each data source
            thermal_features = self._extract_stream_features('thermal', thermal_data)
            gas_features = self._extract_stream_features('gas', gas_data)
            environmental_features = self._extract_stream_features('environmental', environmental_data)
            
            # Fuse features
            fusion_results = self.fusion_system.fuse_features(
                thermal_features=thermal_features,
                gas_features=gas_features,
                environmental_features=environmental_features,
                metadata={'timestamp': timestamp.isoformat()}
            )
            
            # Combine results
            stream_results = {
                'metadata': {
                    'timestamp': timestamp.isoformat(),
                    'processing_time': datetime.now().isoformat()
                },
                'extracted_features': {
                    'thermal': thermal_features,
                    'gas': gas_features,
                    'environmental': environmental_features
                },
                'fused_features': fusion_results['fused_features']
            }
            
            self.logger.info("Stream processing completed")
            return stream_results
            
        except Exception as e:
            error_msg = f"Error in stream processing: {str(e)}"
            self.logger.error(error_msg)
            
            # Return error information
            return {
                'metadata': {
                    'timestamp': timestamp.isoformat(),
                    'processing_time': datetime.now().isoformat(),
                    'error': str(e)
                }
            }
    
    def _extract_stream_features(self, extractor_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from streaming data for a specific extractor type.
        
        Args:
            extractor_type: Type of extractor
            data: Data for processing
            
        Returns:
            Dictionary containing the extracted features
        """
        if extractor_type not in self.extraction_framework.extractors:
            self.logger.warning(f"Extractor not found: {extractor_type}")
            return {}
        
        extractor = self.extraction_framework.extractors[extractor_type]
        return extractor.extract_features(data)
    
    def _reset_progress(self, total_steps: int) -> None:
        """
        Reset progress tracking.
        
        Args:
            total_steps: Total number of steps in the pipeline
        """
        self.progress = {
            'total_steps': total_steps,
            'completed_steps': 0,
            'current_step': '',
            'errors': []
        }
    
    def _update_progress(self, step: str) -> None:
        """
        Update the current step in progress tracking.
        
        Args:
            step: Current step description
        """
        self.progress['current_step'] = step
        self.logger.info(f"Progress: {step} ({self.progress['completed_steps']}/{self.progress['total_steps']})")
    
    def _increment_progress(self) -> None:
        """
        Increment the completed steps count in progress tracking.
        """
        self.progress['completed_steps'] += 1
        completion_percentage = (self.progress['completed_steps'] / self.progress['total_steps']) * 100
        self.logger.info(f"Progress: {completion_percentage:.1f}% complete")
    
    def _add_error(self, error_msg: str) -> None:
        """
        Add an error to the progress tracking.
        
        Args:
            error_msg: Error message
        """
        self.progress['errors'].append({
            'time': datetime.now().isoformat(),
            'message': error_msg
        })