"""
Model Registry for the fire prediction system.

This module implements the ModelRegistry class that registers all available models,
provides a mechanism to discover and instantiate models, manages model dependencies,
and handles model versioning and tracking.
"""

import os
import json
import importlib
import inspect
import logging
from typing import Dict, Any, Optional, Type, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
import pickle

from ..base import FireModel, FireClassificationModel, FireIdentificationModel, FireProgressionModel, ConfidenceEstimationModel
from ..models.classification import BinaryFireClassifier, MultiClassFireClassifier, DeepLearningFireClassifier, EnsembleClassifier
from ..models.identification import ElectricalFireIdentifier, ChemicalFireIdentifier, SmolderingFireIdentifier, RapidCombustionIdentifier
from ..models.progression import FireGrowthPredictor, SpreadRateEstimator, TimeToThresholdPredictor, FirePathPredictor
from ..models.confidence import UncertaintyEstimator, ConfidenceScorer, CalibrationModel, EnsembleVarianceAnalyzer


class ModelRegistry:
    """
    Model registry for the fire prediction system.
    
    This class provides a central registry for all available models,
    with functionality to discover, instantiate, and manage models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model registry.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set up registry paths
        self.registry_dir = config.get('registry_dir', 'models')
        self.metadata_dir = config.get('metadata_dir', 'metadata')
        self.version_file = config.get('version_file', 'versions.json')
        
        # AWS configuration
        self.use_aws = config.get('use_aws', False)
        self.s3_bucket = config.get('s3_bucket', 'fire-prediction-models')
        self.s3_prefix = config.get('s3_prefix', 'registry')
        
        # Initialize AWS clients if needed
        self.s3_client = None
        self.sagemaker_client = None
        if self.use_aws:
            self.s3_client = boto3.client('s3')
            self.sagemaker_client = boto3.client('sagemaker')
        
        # Initialize model registry
        self.models = {}
        self.model_instances = {}
        self.model_versions = {}
        self.model_dependencies = {}
        
        # Register all available models
        self._register_models()
        
        # Load model versions
        self._load_versions()
    
    def _register_models(self) -> None:
        """
        Register all available models in the registry.
        """
        self.logger.info("Registering models in the registry")
        
        # Register classification models
        self.models['binary_classifier'] = {
            'class': BinaryFireClassifier,
            'type': 'classification',
            'description': 'Binary fire classifier (fire vs. no fire)',
            'default_config': {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'class_weight': 'balanced'
            }
        }
        
        self.models['multi_class_classifier'] = {
            'class': MultiClassFireClassifier,
            'type': 'classification',
            'description': 'Multi-class fire classifier (fire types)',
            'default_config': {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'class_weight': 'balanced'
            }
        }
        
        self.models['deep_learning_classifier'] = {
            'class': DeepLearningFireClassifier,
            'type': 'classification',
            'description': 'Deep learning fire classifier',
            'default_config': {
                'hidden_layers': [64, 32],
                'activation': 'relu',
                'dropout_rate': 0.2
            }
        }
        
        self.models['ensemble_classifier'] = {
            'class': EnsembleClassifier,
            'type': 'classification',
            'description': 'Ensemble fire classifier',
            'default_config': {
                'base_models': ['binary_classifier', 'multi_class_classifier'],
                'ensemble_method': 'voting'
            },
            'dependencies': ['binary_classifier', 'multi_class_classifier']
        }
        
        # Register identification models
        self.models['electrical_fire_identifier'] = {
            'class': ElectricalFireIdentifier,
            'type': 'identification',
            'description': 'Electrical fire identifier',
            'default_config': {
                'algorithm': 'gradient_boosting',
                'n_estimators': 100,
                'feature_scaling': True
            }
        }
        
        self.models['chemical_fire_identifier'] = {
            'class': ChemicalFireIdentifier,
            'type': 'identification',
            'description': 'Chemical fire identifier',
            'default_config': {
                'algorithm': 'svm',
                'feature_scaling': True
            }
        }
        
        self.models['smoldering_fire_identifier'] = {
            'class': SmolderingFireIdentifier,
            'type': 'identification',
            'description': 'Smoldering fire identifier',
            'default_config': {
                'algorithm': 'mlp',
                'feature_scaling': True
            }
        }
        
        self.models['rapid_combustion_identifier'] = {
            'class': RapidCombustionIdentifier,
            'type': 'identification',
            'description': 'Rapid combustion fire identifier',
            'default_config': {
                'algorithm': 'xgboost',
                'n_estimators': 100,
                'feature_scaling': True
            }
        }
        
        # Register progression models
        self.models['fire_growth_predictor'] = {
            'class': FireGrowthPredictor,
            'type': 'progression',
            'description': 'Fire growth predictor',
            'default_config': {
                'algorithm': 'gradient_boosting',
                'n_estimators': 100,
                'feature_scaling': True,
                'output_features': ['size', 'intensity', 'temperature']
            }
        }
        
        self.models['spread_rate_estimator'] = {
            'class': SpreadRateEstimator,
            'type': 'progression',
            'description': 'Fire spread rate estimator',
            'default_config': {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'feature_scaling': True,
                'directions': ['north', 'east', 'south', 'west']
            }
        }
        
        self.models['time_to_threshold_predictor'] = {
            'class': TimeToThresholdPredictor,
            'type': 'progression',
            'description': 'Time to threshold predictor',
            'default_config': {
                'algorithm': 'gradient_boosting',
                'n_estimators': 100,
                'feature_scaling': True,
                'thresholds': {
                    'temperature': [100, 200, 300, 400, 500],
                    'size': [10, 50, 100, 500, 1000],
                    'intensity': [5, 10, 20, 50, 100]
                }
            }
        }
        
        self.models['fire_path_predictor'] = {
            'class': FirePathPredictor,
            'type': 'progression',
            'description': 'Fire path predictor',
            'default_config': {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'feature_scaling': True,
                'grid_resolution': 10
            }
        }
        
        # Register confidence models
        self.models['uncertainty_estimator'] = {
            'class': UncertaintyEstimator,
            'type': 'confidence',
            'description': 'Uncertainty estimator',
            'default_config': {
                'method': 'gaussian_process',
                'feature_scaling': True
            }
        }
        
        self.models['confidence_scorer'] = {
            'class': ConfidenceScorer,
            'type': 'confidence',
            'description': 'Confidence scorer',
            'default_config': {
                'algorithm': 'gradient_boosting',
                'n_estimators': 100,
                'feature_scaling': True,
                'calibration_method': 'isotonic'
            }
        }
        
        self.models['calibration_model'] = {
            'class': CalibrationModel,
            'type': 'confidence',
            'description': 'Probability calibration model',
            'default_config': {
                'method': 'isotonic',
                'feature_scaling': True,
                'n_bins': 10
            }
        }
        
        self.models['ensemble_variance_analyzer'] = {
            'class': EnsembleVarianceAnalyzer,
            'type': 'confidence',
            'description': 'Ensemble variance analyzer',
            'default_config': {
                'task_type': 'regression',
                'n_estimators': 100,
                'feature_scaling': True,
                'bootstrap_fraction': 0.8
            }
        }
        
        # Build dependency graph
        self._build_dependencies()
        
        self.logger.info(f"Registered {len(self.models)} models in the registry")
    
    def _build_dependencies(self) -> None:
        """
        Build the dependency graph for models.
        """
        for model_name, model_info in self.models.items():
            if 'dependencies' in model_info:
                self.model_dependencies[model_name] = model_info['dependencies']
            else:
                self.model_dependencies[model_name] = []
    
    def _load_versions(self) -> None:
        """
        Load model versions from the version file.
        """
        version_path = os.path.join(self.registry_dir, self.version_file)
        
        if os.path.exists(version_path):
            with open(version_path, 'r') as f:
                self.model_versions = json.load(f)
            self.logger.info(f"Loaded model versions from {version_path}")
        else:
            self.model_versions = {}
            self.logger.info("No version file found, initializing empty versions")
    
    def _save_versions(self) -> None:
        """
        Save model versions to the version file.
        """
        # Create directory if it doesn't exist
        os.makedirs(self.registry_dir, exist_ok=True)
        
        version_path = os.path.join(self.registry_dir, self.version_file)
        
        with open(version_path, 'w') as f:
            json.dump(self.model_versions, f, indent=2)
        
        self.logger.info(f"Saved model versions to {version_path}")
        
        # Upload to S3 if using AWS
        if self.use_aws and self.s3_client:
            s3_key = f"{self.s3_prefix}/{self.version_file}"
            self.s3_client.upload_file(version_path, self.s3_bucket, s3_key)
            self.logger.info(f"Uploaded model versions to S3: s3://{self.s3_bucket}/{s3_key}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a registered model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return self.models[model_name]
    
    def get_model_class(self, model_name: str) -> Type[FireModel]:
        """
        Get the class for a registered model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model class
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return self.models[model_name]['class']
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get the default configuration for a registered model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Default configuration dictionary
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return self.models[model_name]['default_config']
    
    def get_model_dependencies(self, model_name: str) -> List[str]:
        """
        Get the dependencies for a registered model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of dependency model names
        """
        if model_name not in self.model_dependencies:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return self.model_dependencies[model_name]
    
    def get_model_versions(self, model_name: str) -> List[str]:
        """
        Get available versions for a registered model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        if model_name not in self.model_versions:
            return []
        
        return list(self.model_versions[model_name].keys())
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get the latest version for a registered model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string, or None if no versions exist
        """
        versions = self.get_model_versions(model_name)
        
        if not versions:
            return None
        
        # Sort versions by timestamp
        sorted_versions = sorted(
            versions,
            key=lambda v: self.model_versions[model_name][v]['timestamp'],
            reverse=True
        )
        
        return sorted_versions[0]
    
    def create_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> FireModel:
        """
        Create a new instance of a registered model.
        
        Args:
            model_name: Name of the model
            config: Optional configuration dictionary (will use default if not provided)
            
        Returns:
            Model instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_class = self.models[model_name]['class']
        
        # Use provided config or default
        if config is None:
            config = self.models[model_name]['default_config'].copy()
        
        # Create model instance
        model = model_class(config)
        
        return model
    
    def register_model_instance(self, model_name: str, model: FireModel, instance_name: str) -> None:
        """
        Register a model instance for later use.
        
        Args:
            model_name: Name of the model type
            model: Model instance
            instance_name: Name for this specific instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        self.model_instances[instance_name] = {
            'model_name': model_name,
            'model': model,
            'registered_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Registered model instance {instance_name} of type {model_name}")
    
    def get_model_instance(self, instance_name: str) -> FireModel:
        """
        Get a registered model instance.
        
        Args:
            instance_name: Name of the model instance
            
        Returns:
            Model instance
        """
        if instance_name not in self.model_instances:
            raise ValueError(f"Model instance {instance_name} not found in registry")
        
        return self.model_instances[instance_name]['model']
    
    def save_model(self, 
                  model: FireModel, 
                  model_name: str, 
                  version: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  to_s3: bool = False) -> str:
        """
        Save a model to the registry.
        
        Args:
            model: Model instance to save
            model_name: Name of the model type
            version: Optional version string (will generate if not provided)
            metadata: Optional metadata dictionary
            to_s3: Whether to also save to S3
            
        Returns:
            Path to the saved model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories if they don't exist
        model_dir = os.path.join(self.registry_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        metadata_dir = os.path.join(self.metadata_dir, model_name)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{version}.pkl")
        model.save(model_path)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        # Add standard metadata
        metadata.update({
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_type': self.models[model_name]['type'],
            'model_class': self.models[model_name]['class'].__name__
        })
        
        metadata_path = os.path.join(metadata_dir, f"{version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update version registry
        if model_name not in self.model_versions:
            self.model_versions[model_name] = {}
        
        self.model_versions[model_name][version] = {
            'path': model_path,
            'metadata_path': metadata_path,
            'timestamp': metadata['timestamp']
        }
        
        # Save version registry
        self._save_versions()
        
        # Upload to S3 if requested
        if to_s3 and self.use_aws and self.s3_client:
            s3_model_key = f"{self.s3_prefix}/{model_name}/{version}.pkl"
            s3_metadata_key = f"{self.s3_prefix}/{model_name}/{version}.json"
            
            self.s3_client.upload_file(model_path, self.s3_bucket, s3_model_key)
            self.s3_client.upload_file(metadata_path, self.s3_bucket, s3_metadata_key)
            
            # Update version registry with S3 paths
            self.model_versions[model_name][version]['s3_path'] = f"s3://{self.s3_bucket}/{s3_model_key}"
            self.model_versions[model_name][version]['s3_metadata_path'] = f"s3://{self.s3_bucket}/{s3_metadata_key}"
            
            # Save updated version registry
            self._save_versions()
            
            self.logger.info(f"Uploaded model to S3: s3://{self.s3_bucket}/{s3_model_key}")
        
        self.logger.info(f"Saved model {model_name} version {version} to {model_path}")
        
        return model_path
    
    def load_model(self, 
                 model_name: str, 
                 version: Optional[str] = None,
                 from_s3: bool = False) -> FireModel:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model type
            version: Version string (will use latest if not provided)
            from_s3: Whether to load from S3
            
        Returns:
            Loaded model instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Get version to load
        if version is None:
            version = self.get_latest_version(model_name)
            
            if version is None:
                raise ValueError(f"No versions found for model {model_name}")
        
        if model_name not in self.model_versions or version not in self.model_versions[model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        version_info = self.model_versions[model_name][version]
        
        # Load from S3 if requested
        if from_s3 and self.use_aws and self.s3_client:
            if 's3_path' not in version_info:
                raise ValueError(f"S3 path not found for model {model_name} version {version}")
            
            # Parse S3 path
            s3_path = version_info['s3_path']
            s3_bucket = s3_path.split('/')[2]
            s3_key = '/'.join(s3_path.split('/')[3:])
            
            # Download to temporary file
            temp_path = os.path.join(self.registry_dir, f"temp_{model_name}_{version}.pkl")
            self.s3_client.download_file(s3_bucket, s3_key, temp_path)
            
            # Load model
            with open(temp_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            self.logger.info(f"Loaded model {model_name} version {version} from S3")
        else:
            # Load from local path
            model_path = version_info['path']
            
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            
            # Create model instance
            model_class = self.models[model_name]['class']
            model = model_class(self.models[model_name]['default_config'])
            
            # Load model data
            model.load(model_path)
            
            self.logger.info(f"Loaded model {model_name} version {version} from {model_path}")
        
        return model
    
    def get_model_metadata(self, 
                         model_name: str, 
                         version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a model version.
        
        Args:
            model_name: Name of the model type
            version: Version string (will use latest if not provided)
            
        Returns:
            Metadata dictionary
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Get version to load
        if version is None:
            version = self.get_latest_version(model_name)
            
            if version is None:
                raise ValueError(f"No versions found for model {model_name}")
        
        if model_name not in self.model_versions or version not in self.model_versions[model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        version_info = self.model_versions[model_name][version]
        metadata_path = version_info['metadata_path']
        
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def delete_model_version(self, 
                           model_name: str, 
                           version: str,
                           delete_files: bool = True) -> None:
        """
        Delete a model version from the registry.
        
        Args:
            model_name: Name of the model type
            version: Version string
            delete_files: Whether to delete the model and metadata files
        """
        if model_name not in self.model_versions or version not in self.model_versions[model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        version_info = self.model_versions[model_name][version]
        
        # Delete files if requested
        if delete_files:
            # Delete model file
            if os.path.exists(version_info['path']):
                os.remove(version_info['path'])
            
            # Delete metadata file
            if os.path.exists(version_info['metadata_path']):
                os.remove(version_info['metadata_path'])
            
            # Delete from S3 if applicable
            if self.use_aws and self.s3_client and 's3_path' in version_info:
                # Parse S3 path
                s3_path = version_info['s3_path']
                s3_bucket = s3_path.split('/')[2]
                s3_key = '/'.join(s3_path.split('/')[3:])
                
                # Delete from S3
                self.s3_client.delete_object(Bucket=s3_bucket, Key=s3_key)
                
                # Delete metadata from S3
                if 's3_metadata_path' in version_info:
                    s3_metadata_path = version_info['s3_metadata_path']
                    s3_metadata_key = '/'.join(s3_metadata_path.split('/')[3:])
                    self.s3_client.delete_object(Bucket=s3_bucket, Key=s3_metadata_key)
        
        # Remove from version registry
        del self.model_versions[model_name][version]
        
        # Remove model entry if no versions left
        if not self.model_versions[model_name]:
            del self.model_versions[model_name]
        
        # Save version registry
        self._save_versions()
        
        self.logger.info(f"Deleted model {model_name} version {version}")
    
    def deploy_to_sagemaker(self,
                          model: FireModel,
                          model_name: str,
                          version: Optional[str] = None,
                          instance_type: str = 'ml.m5.large',
                          initial_instance_count: int = 1) -> str:
        """
        Deploy a model to SageMaker.
        
        Args:
            model: Model instance to deploy
            model_name: Name of the model type
            version: Version string (will generate if not provided)
            instance_type: SageMaker instance type
            initial_instance_count: Number of initial instances
            
        Returns:
            SageMaker endpoint name
        """
        if not self.use_aws or self.sagemaker_client is None:
            raise ValueError("AWS not configured for SageMaker deployment")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model to S3
        model_path = self.save_model(model, model_name, version, to_s3=True)
        s3_path = self.model_versions[model_name][version]['s3_path']
        
        # Create SageMaker model
        sagemaker_model_name = f"{model_name}-{version}"
        
        # TODO: Implement SageMaker deployment logic
        # This would involve:
        # 1. Creating a SageMaker model
        # 2. Creating an endpoint configuration
        # 3. Creating an endpoint
        
        # For now, just log the intent
        self.logger.info(f"SageMaker deployment not fully implemented")
        self.logger.info(f"Would deploy model {model_name} version {version} to SageMaker")
        self.logger.info(f"Model path: {s3_path}")
        self.logger.info(f"Instance type: {instance_type}")
        self.logger.info(f"Initial instance count: {initial_instance_count}")
        
        # Return a placeholder endpoint name
        endpoint_name = f"{model_name}-endpoint-{version}"
        
        return endpoint_name
    
    def list_models(self, model_type: Optional[str] = None) -> List[str]:
        """
        List all registered models, optionally filtered by type.
        
        Args:
            model_type: Optional model type filter
            
        Returns:
            List of model names
        """
        if model_type is None:
            return list(self.models.keys())
        
        return [name for name, info in self.models.items() if info['type'] == model_type]
    
    def list_model_instances(self) -> List[str]:
        """
        List all registered model instances.
        
        Returns:
            List of instance names
        """
        return list(self.model_instances.keys())
    
    def get_model_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the complete model dependency graph.
        
        Returns:
            Dictionary mapping model names to lists of dependencies
        """
        return self.model_dependencies.copy()
    
    def get_model_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model registry.
        
        Returns:
            Dictionary with registry summary
        """
        summary = {
            'total_models': len(self.models),
            'total_instances': len(self.model_instances),
            'models_by_type': {},
            'version_counts': {}
        }
        
        # Count models by type
        for model_name, model_info in self.models.items():
            model_type = model_info['type']
            if model_type not in summary['models_by_type']:
                summary['models_by_type'][model_type] = 0
            summary['models_by_type'][model_type] += 1
        
        # Count versions
        for model_name, versions in self.model_versions.items():
            summary['version_counts'][model_name] = len(versions)
        
        return summary