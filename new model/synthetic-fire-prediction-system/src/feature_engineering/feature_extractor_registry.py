"""
Feature Extractor Registry for managing feature extractors.

This module provides a registry for feature extractors, allowing them to be registered,
retrieved, and managed centrally.
"""

from typing import Dict, Any, List, Type, Optional, Union
import logging
import importlib
import inspect
import os
import json
from pathlib import Path

from .base import FeatureExtractor
from .base_temporal import TemporalFeatureExtractor


class FeatureExtractorRegistry:
    """
    Registry for feature extractors.
    
    This class provides a central registry for feature extractors, allowing them to be
    registered, retrieved, and managed. It supports automatic discovery of feature
    extractors in the project structure.
    """
    
    def __init__(self):
        """
        Initialize the feature extractor registry.
        """
        self.extractors: Dict[str, Type[FeatureExtractor]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, name: str, extractor_class: Type[FeatureExtractor]) -> None:
        """
        Register a feature extractor.
        
        Args:
            name: Name of the feature extractor
            extractor_class: Feature extractor class
        """
        if name in self.extractors:
            self.logger.warning(f"Feature extractor '{name}' already registered, overwriting")
        
        self.extractors[name] = extractor_class
        self.logger.info(f"Registered feature extractor '{name}'")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a feature extractor.
        
        Args:
            name: Name of the feature extractor
        """
        if name in self.extractors:
            del self.extractors[name]
            self.logger.info(f"Unregistered feature extractor '{name}'")
        else:
            self.logger.warning(f"Feature extractor '{name}' not found in registry")
    
    def get(self, name: str) -> Optional[Type[FeatureExtractor]]:
        """
        Get a feature extractor by name.
        
        Args:
            name: Name of the feature extractor
            
        Returns:
            Feature extractor class, or None if not found
        """
        return self.extractors.get(name)
    
    def create(self, name: str, config: Dict[str, Any]) -> Optional[FeatureExtractor]:
        """
        Create a feature extractor instance.
        
        Args:
            name: Name of the feature extractor
            config: Configuration for the feature extractor
            
        Returns:
            Feature extractor instance, or None if not found
        """
        extractor_class = self.get(name)
        
        if extractor_class is None:
            self.logger.warning(f"Feature extractor '{name}' not found in registry")
            return None
        
        try:
            extractor = extractor_class(config)
            extractor.validate_config()
            return extractor
        except Exception as e:
            self.logger.error(f"Error creating feature extractor '{name}': {str(e)}")
            return None
    
    def list_extractors(self) -> List[str]:
        """
        List all registered feature extractors.
        
        Returns:
            List of feature extractor names
        """
        return list(self.extractors.keys())
    
    def get_extractor_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a feature extractor.
        
        Args:
            name: Name of the feature extractor
            
        Returns:
            Dictionary containing information about the feature extractor
        """
        extractor_class = self.get(name)
        
        if extractor_class is None:
            return {}
        
        # Get class docstring
        docstring = extractor_class.__doc__ or ""
        
        # Get feature names
        feature_names = []
        
        try:
            # Create a temporary instance to get feature names
            temp_instance = extractor_class({})
            feature_names = temp_instance.get_feature_names()
        except Exception:
            pass
        
        # Determine extractor type
        extractor_type = "unknown"
        
        if issubclass(extractor_class, TemporalFeatureExtractor):
            extractor_type = "temporal"
        elif "thermal" in name.lower():
            extractor_type = "thermal"
        elif "gas" in name.lower():
            extractor_type = "gas"
        elif "environmental" in name.lower():
            extractor_type = "environmental"
        
        return {
            "name": name,
            "type": extractor_type,
            "description": docstring.strip(),
            "feature_names": feature_names,
            "module": extractor_class.__module__
        }
    
    def discover_extractors(self, package_path: str = "feature_engineering.extractors") -> None:
        """
        Discover and register feature extractors in the specified package.
        
        Args:
            package_path: Path to the package containing feature extractors
        """
        self.logger.info(f"Discovering feature extractors in package '{package_path}'")
        
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Get the package directory
            package_dir = os.path.dirname(package.__file__)
            
            # Discover extractors in subdirectories
            for subdir in os.listdir(package_dir):
                subdir_path = os.path.join(package_dir, subdir)
                
                # Skip non-directories and special directories
                if not os.path.isdir(subdir_path) or subdir.startswith("__"):
                    continue
                
                # Discover extractors in the subdirectory
                self._discover_extractors_in_directory(f"{package_path}.{subdir}")
        
        except Exception as e:
            self.logger.error(f"Error discovering feature extractors: {str(e)}")
    
    def _discover_extractors_in_directory(self, package_path: str) -> None:
        """
        Discover and register feature extractors in the specified directory.
        
        Args:
            package_path: Path to the package containing feature extractors
        """
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Get the package directory
            package_dir = os.path.dirname(package.__file__)
            
            # Discover extractors in Python files
            for filename in os.listdir(package_dir):
                if filename.endswith('.py') and filename != '__init__.py':
                    module_name = filename[:-3]  # Remove .py extension
                    full_module_path = f"{package_path}.{module_name}"
                    
                    try:
                        # Import the module
                        module = importlib.import_module(full_module_path)
                        
                        # Find FeatureExtractor subclasses
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, FeatureExtractor) and 
                                obj != FeatureExtractor and
                                obj != TemporalFeatureExtractor):
                                # Register the extractor
                                extractor_name = f"{package_path.split('.')[-1]}.{name}"
                                self.register(extractor_name, obj)
                                
                    except Exception as e:
                        self.logger.warning(f"Error importing module {full_module_path}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error discovering feature extractors in '{package_path}': {str(e)}")
    
    def _get_extractor_name(self, extractor_class: Type[FeatureExtractor], module_name: str) -> str:
        """
        Get a name for a feature extractor.
        
        Args:
            extractor_class: Feature extractor class
            module_name: Module name
            
        Returns:
            Name for the feature extractor
        """
        # Use class name as extractor name
        return extractor_class.__name__
    
    def save_registry(self, filepath: str) -> None:
        """
        Save the registry to a file.
        
        Args:
            filepath: Path to save the registry
        """
        # Create registry data
        registry_data = {
            "extractors": {}
        }
        
        # Add extractor information
        for name in self.list_extractors():
            registry_data["extractors"][name] = self.get_extractor_info(name)
        
        # Save to file
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save as JSON
            with open(filepath, "w") as f:
                json.dump(registry_data, f, indent=2)
            
            self.logger.info(f"Saved registry to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error saving registry to {filepath}: {str(e)}")
    
    def load_registry(self, filepath: str) -> None:
        """
        Load the registry from a file.
        
        Args:
            filepath: Path to load the registry from
        """
        try:
            # Load from file
            with open(filepath, "r") as f:
                registry_data = json.load(f)
            
            # Clear existing registry
            self.extractors.clear()
            
            # Import and register extractors
            for name, info in registry_data.get("extractors", {}).items():
                module_name = info.get("module")
                
                if module_name:
                    try:
                        # Import the module
                        module = importlib.import_module(module_name)
                        
                        # Find the extractor class
                        for class_name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, FeatureExtractor) and 
                                obj != FeatureExtractor and
                                obj != TemporalFeatureExtractor and
                                class_name == name):
                                
                                # Register the feature extractor
                                self.register(name, obj)
                                break
                    
                    except Exception as e:
                        self.logger.warning(f"Error importing module '{module_name}': {str(e)}")
            
            self.logger.info(f"Loaded registry from {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error loading registry from {filepath}: {str(e)}")
    
    def get_extractors_by_type(self, extractor_type: str) -> List[str]:
        """
        Get feature extractors by type.
        
        Args:
            extractor_type: Type of feature extractors to get
            
        Returns:
            List of feature extractor names
        """
        extractors = []
        
        for name in self.list_extractors():
            info = self.get_extractor_info(name)
            
            if info.get("type") == extractor_type:
                extractors.append(name)
        
        return extractors
    
    def create_extractor_pipeline(self, config: Dict[str, Any]) -> List[FeatureExtractor]:
        """
        Create a pipeline of feature extractors.
        
        Args:
            config: Configuration for the pipeline
            
        Returns:
            List of feature extractor instances
        """
        pipeline = []
        
        # Get extractors from configuration
        extractors_config = config.get("extractors", [])
        
        for extractor_config in extractors_config:
            # Get extractor name and configuration
            extractor_name = extractor_config.get("name")
            extractor_params = extractor_config.get("config", {})
            
            if not extractor_name:
                self.logger.warning("Missing extractor name in pipeline configuration")
                continue
            
            # Create extractor
            extractor = self.create(extractor_name, extractor_params)
            
            if extractor:
                pipeline.append(extractor)
            else:
                self.logger.warning(f"Failed to create extractor '{extractor_name}' for pipeline")
        
        return pipeline
    
    def extract_features(self, data: Any, pipeline: List[FeatureExtractor]) -> Dict[str, Any]:
        """
        Extract features using a pipeline of feature extractors.
        
        Args:
            data: Input data
            pipeline: List of feature extractors
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Extract features using each extractor in the pipeline
        for extractor in pipeline:
            try:
                # Extract features
                extractor_features = extractor.extract_features(data)
                
                # Add to features dictionary
                extractor_name = extractor.__class__.__name__
                features[extractor_name] = extractor_features
            
            except Exception as e:
                self.logger.error(f"Error extracting features with '{extractor.__class__.__name__}': {str(e)}")
        
        return features


# Create a global registry instance
registry = FeatureExtractorRegistry()