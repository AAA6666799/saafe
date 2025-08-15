"""
Model management and persistence system for Saafe MVP.

This module provides comprehensive model management including loading, validation,
fallback mechanisms, and GPU/CPU device management with automatic fallback.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import logging
import json
import hashlib
import time
from dataclasses import dataclass, asdict
import warnings

from .transformer import SpatioTemporalTransformer, ModelConfig
from .model_loader import ModelLoader
from .anti_hallucination import AntiHallucinationEngine
from ..utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, safe_execute

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a loaded model."""
    model_id: str
    model_path: str
    config: Dict[str, Any]
    checksum: str
    load_time: float
    validation_status: str
    device: str
    model_size_mb: float
    last_used: float


class ModelRegistry:
    """Registry for managing multiple models and their metadata."""
    
    def __init__(self):
        self.models: Dict[str, SpatioTemporalTransformer] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.primary_model_id: Optional[str] = None
        
    def register_model(self, model_id: str, model: SpatioTemporalTransformer, 
                      metadata: ModelMetadata) -> None:
        """Register a model in the registry."""
        self.models[model_id] = model
        self.metadata[model_id] = metadata
        
        if self.primary_model_id is None:
            self.primary_model_id = model_id
            
        logger.info(f"Model registered: {model_id}")
    
    def get_model(self, model_id: str) -> Optional[SpatioTemporalTransformer]:
        """Get a model by ID."""
        if model_id in self.models:
            self.metadata[model_id].last_used = time.time()
            return self.models[model_id]
        return None
    
    def get_primary_model(self) -> Optional[SpatioTemporalTransformer]:
        """Get the primary model."""
        if self.primary_model_id:
            return self.get_model(self.primary_model_id)
        return None
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.models.keys())
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        if model_id in self.models:
            del self.models[model_id]
            del self.metadata[model_id]
            
            if self.primary_model_id == model_id:
                self.primary_model_id = next(iter(self.models.keys()), None)
            
            logger.info(f"Model removed: {model_id}")
            return True
        return False
    
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        return self.metadata.get(model_id)
    
    def get_all_metadata(self) -> Dict[str, ModelMetadata]:
        """Get metadata for all models."""
        return self.metadata.copy()


class ModelManager:
    """
    Comprehensive model management system with loading, validation, and fallback mechanisms.
    """
    
    def __init__(self, device: Optional[torch.device] = None, model_dir: Optional[Path] = None, 
                 error_handler: Optional[ErrorHandler] = None):
        """
        Initialize the model manager.
        
        Args:
            device (torch.device): Device for model operations (auto-detected if None)
            model_dir (Path): Directory containing model files
            error_handler (ErrorHandler): Error handling system
        """
        # Initialize error handler
        from ..utils.error_handler import get_error_handler
        self.error_handler = error_handler or get_error_handler()
        
        self.device = device or self._detect_device()
        self.model_dir = Path(model_dir) if model_dir else Path("models/saved")
        
        # Create model directory with error handling
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, "ModelManager",
                context={'operation': 'create_model_dir', 'path': str(self.model_dir)}
            )
            # Use fallback directory
            self.model_dir = Path("./models_fallback")
            self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = ModelLoader(self.device)
        self.registry = ModelRegistry()
        
        # Fallback configurations
        self.fallback_configs = [
            ModelConfig(num_layers=2, d_model=128),  # Lightweight fallback
            ModelConfig(num_layers=1, d_model=64),   # Minimal fallback
        ]
        
        logger.info(f"ModelManager initialized with device: {self.device}")
        logger.info(f"Model directory: {self.model_dir}")
    
    def _detect_device(self) -> torch.device:
        """Detect the best available device with fallback."""
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality
                test_tensor = torch.randn(10, 10, device='cuda')
                _ = test_tensor @ test_tensor.T
                device = torch.device('cuda')
                logger.info(f"CUDA detected and functional: {torch.cuda.get_device_name(0)}")
                return device
            except Exception as e:
                if hasattr(self, 'error_handler'):
                    self.error_handler.handle_error(
                        e, ErrorCategory.SYSTEM_ERROR, "ModelManager",
                        context={'operation': 'cuda_test'},
                        severity=ErrorSeverity.MEDIUM
                    )
                logger.warning(f"CUDA available but not functional: {e}")
        
        device = torch.device('cpu')
        logger.info("Using CPU device")
        return device
    
    def load_model(self, model_path: Union[str, Path], 
                   model_id: Optional[str] = None,
                   set_as_primary: bool = True,
                   validate: bool = True) -> Tuple[bool, str]:
        """
        Load a model with comprehensive validation and fallback.
        
        Args:
            model_path (Union[str, Path]): Path to model file
            model_id (str): Unique identifier for the model
            set_as_primary (bool): Whether to set as primary model
            validate (bool): Whether to validate the model
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        model_path = Path(model_path)
        
        if model_id is None:
            model_id = model_path.stem
        
        start_time = time.time()
        
        try:
            # Check if file exists
            if not model_path.exists():
                return False, f"Model file not found: {model_path}"
            
            # Calculate checksum
            checksum = self._calculate_checksum(model_path)
            
            # Load model
            model = self.loader.load_model(model_path, strict=False)
            load_time = time.time() - start_time
            
            # Validate model if requested
            validation_status = "not_validated"
            if validate:
                validation_result = self.loader.validate_model(model)
                validation_status = "valid" if validation_result['valid'] else "invalid"
                
                if not validation_result['valid']:
                    logger.warning(f"Model validation failed: {validation_result['errors']}")
            
            # Calculate model size
            model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_path=str(model_path),
                config=asdict(model.config),
                checksum=checksum,
                load_time=load_time,
                validation_status=validation_status,
                device=str(self.device),
                model_size_mb=model_size_mb,
                last_used=time.time()
            )
            
            # Register model
            self.registry.register_model(model_id, model, metadata)
            
            if set_as_primary:
                self.registry.primary_model_id = model_id
            
            logger.info(f"Model loaded successfully: {model_id} ({load_time:.2f}s)")
            return True, f"Model loaded: {model_id}"
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            return False, f"Load failed: {str(e)}"
    
    def load_multiple_models(self, model_paths: List[Union[str, Path]], 
                           validate: bool = True) -> Dict[str, Tuple[bool, str]]:
        """
        Load multiple models for ensemble use.
        
        Args:
            model_paths (List[Union[str, Path]]): List of model file paths
            validate (bool): Whether to validate models
            
        Returns:
            Dict[str, Tuple[bool, str]]: Results for each model
        """
        results = {}
        
        for i, path in enumerate(model_paths):
            model_id = f"model_{i+1}"
            success, message = self.load_model(
                path, 
                model_id=model_id, 
                set_as_primary=(i == 0),
                validate=validate
            )
            results[model_id] = (success, message)
        
        return results
    
    def create_fallback_model(self, config_index: int = 0) -> Tuple[bool, str]:
        """
        Create a fallback model when loading fails.
        
        Args:
            config_index (int): Index of fallback configuration to use
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if config_index >= len(self.fallback_configs):
                config_index = len(self.fallback_configs) - 1
            
            config = self.fallback_configs[config_index]
            model = SpatioTemporalTransformer(config)
            model = model.to(self.device)
            
            model_id = f"fallback_{config_index}"
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_path="fallback",
                config=asdict(config),
                checksum="fallback",
                load_time=0.0,
                validation_status="fallback",
                device=str(self.device),
                model_size_mb=sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024,
                last_used=time.time()
            )
            
            self.registry.register_model(model_id, model, metadata)
            
            logger.info(f"Fallback model created: {model_id}")
            return True, f"Fallback model created: {model_id}"
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {str(e)}")
            return False, f"Fallback creation failed: {str(e)}"
    
    def get_model(self, model_id: Optional[str] = None) -> Optional[SpatioTemporalTransformer]:
        """
        Get a model with automatic fallback.
        
        Args:
            model_id (str): Model ID (uses primary if None)
            
        Returns:
            SpatioTemporalTransformer: Model instance or None
        """
        if model_id is None:
            model = self.registry.get_primary_model()
        else:
            model = self.registry.get_model(model_id)
        
        # If no model available, try to create fallback
        if model is None and len(self.registry.models) == 0:
            success, _ = self.create_fallback_model()
            if success:
                model = self.registry.get_primary_model()
        
        return model
    
    def get_ensemble_models(self, max_models: int = 3) -> List[SpatioTemporalTransformer]:
        """
        Get models for ensemble use.
        
        Args:
            max_models (int): Maximum number of models to return
            
        Returns:
            List[SpatioTemporalTransformer]: List of models
        """
        models = []
        model_ids = self.registry.list_models()[:max_models]
        
        for model_id in model_ids:
            model = self.registry.get_model(model_id)
            if model is not None:
                models.append(model)
        
        # If we don't have enough models, create fallbacks
        while len(models) < min(max_models, 2):  # At least 2 for ensemble
            success, _ = self.create_fallback_model(len(models))
            if success:
                fallback_id = f"fallback_{len(models)}"
                fallback_model = self.registry.get_model(fallback_id)
                if fallback_model:
                    models.append(fallback_model)
            else:
                break
        
        return models
    
    def create_anti_hallucination_engine(self) -> Optional[AntiHallucinationEngine]:
        """
        Create an anti-hallucination engine with available models.
        
        Returns:
            AntiHallucinationEngine: Engine instance or None
        """
        models = self.get_ensemble_models()
        
        if len(models) == 0:
            logger.error("No models available for anti-hallucination engine")
            return None
        
        try:
            engine = AntiHallucinationEngine(models, device=self.device)
            logger.info(f"Anti-hallucination engine created with {len(models)} models")
            return engine
        except Exception as e:
            logger.error(f"Failed to create anti-hallucination engine: {str(e)}")
            return None
    
    def validate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all registered models.
        
        Returns:
            Dict[str, Dict[str, Any]]: Validation results for each model
        """
        results = {}
        
        for model_id in self.registry.list_models():
            model = self.registry.get_model(model_id)
            if model:
                validation_result = self.loader.validate_model(model)
                results[model_id] = validation_result
                
                # Update metadata
                metadata = self.registry.get_metadata(model_id)
                if metadata:
                    metadata.validation_status = "valid" if validation_result['valid'] else "invalid"
        
        return results
    
    def save_model(self, model_id: str, save_path: Optional[Union[str, Path]] = None,
                   **kwargs) -> Tuple[bool, str]:
        """
        Save a registered model.
        
        Args:
            model_id (str): ID of model to save
            save_path (Union[str, Path]): Save path (auto-generated if None)
            **kwargs: Additional arguments for saving
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        model = self.registry.get_model(model_id)
        if model is None:
            return False, f"Model not found: {model_id}"
        
        if save_path is None:
            save_path = self.model_dir / f"{model_id}.pth"
        
        try:
            self.loader.save_model(model, save_path, **kwargs)
            logger.info(f"Model saved: {model_id} -> {save_path}")
            return True, f"Model saved: {save_path}"
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {str(e)}")
            return False, f"Save failed: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dict[str, Any]: System status information
        """
        status = {
            'device': str(self.device),
            'device_available': True,
            'model_count': len(self.registry.models),
            'primary_model': self.registry.primary_model_id,
            'models': {},
            'memory_info': {},
            'fallback_available': len(self.fallback_configs) > 0
        }
        
        # Device information
        if self.device.type == 'cuda':
            try:
                status['memory_info'] = {
                    'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                    'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
                    'device_name': torch.cuda.get_device_name(0)
                }
            except Exception as e:
                status['device_available'] = False
                status['device_error'] = str(e)
        
        # Model information
        for model_id, metadata in self.registry.get_all_metadata().items():
            status['models'][model_id] = {
                'validation_status': metadata.validation_status,
                'size_mb': metadata.model_size_mb,
                'load_time': metadata.load_time,
                'last_used': metadata.last_used,
                'device': metadata.device
            }
        
        return status
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def cleanup_unused_models(self, max_age_hours: float = 24.0) -> int:
        """
        Remove models that haven't been used recently.
        
        Args:
            max_age_hours (float): Maximum age in hours before cleanup
            
        Returns:
            int: Number of models removed
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        models_to_remove = []
        for model_id, metadata in self.registry.get_all_metadata().items():
            if model_id == self.registry.primary_model_id:
                continue  # Never remove primary model
            
            age = current_time - metadata.last_used
            if age > max_age_seconds:
                models_to_remove.append(model_id)
        
        removed_count = 0
        for model_id in models_to_remove:
            if self.registry.remove_model(model_id):
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} unused models")
        
        return removed_count
    
    def switch_device(self, new_device: Union[str, torch.device]) -> Tuple[bool, str]:
        """
        Switch all models to a new device.
        
        Args:
            new_device (Union[str, torch.device]): New device
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if isinstance(new_device, str):
            new_device = torch.device(new_device)
        
        try:
            # Test device availability
            test_tensor = torch.randn(10, 10, device=new_device)
            _ = test_tensor @ test_tensor.T
            
            # Move all models
            moved_count = 0
            for model_id in self.registry.list_models():
                model = self.registry.get_model(model_id)
                if model:
                    model.to(new_device)
                    metadata = self.registry.get_metadata(model_id)
                    if metadata:
                        metadata.device = str(new_device)
                    moved_count += 1
            
            self.device = new_device
            self.loader.device = new_device
            
            logger.info(f"Switched {moved_count} models to device: {new_device}")
            return True, f"Switched to device: {new_device}"
            
        except Exception as e:
            logger.error(f"Failed to switch to device {new_device}: {str(e)}")
            return False, f"Device switch failed: {str(e)}"


# Convenience functions
def create_model_manager(device: Optional[torch.device] = None, 
                        model_dir: Optional[Path] = None) -> ModelManager:
    """
    Create a model manager instance.
    
    Args:
        device (torch.device): Device for models
        model_dir (Path): Model directory
        
    Returns:
        ModelManager: Manager instance
    """
    return ModelManager(device, model_dir)


def load_models_from_directory(model_dir: Union[str, Path], 
                              manager: Optional[ModelManager] = None) -> ModelManager:
    """
    Load all models from a directory.
    
    Args:
        model_dir (Union[str, Path]): Directory containing model files
        manager (ModelManager): Existing manager (creates new if None)
        
    Returns:
        ModelManager: Manager with loaded models
    """
    if manager is None:
        manager = ModelManager(model_dir=model_dir)
    
    model_dir = Path(model_dir)
    model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
    
    for model_file in model_files:
        success, message = manager.load_model(model_file, set_as_primary=False)
        if success:
            logger.info(f"Loaded model: {model_file.name}")
        else:
            logger.warning(f"Failed to load {model_file.name}: {message}")
    
    return manager