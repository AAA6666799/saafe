"""
Model loading and initialization utilities for the Spatio-Temporal Transformer.

This module provides functions to load, save, and initialize models with proper
device management and error handling.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
import json
import os

from .transformer import SpatioTemporalTransformer, ModelConfig


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Utility class for loading and managing Spatio-Temporal Transformer models.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the model loader.
        
        Args:
            device (torch.device): Device to load models on (auto-detected if None)
        """
        self.device = device or self._detect_device()
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def _detect_device(self) -> torch.device:
        """
        Automatically detect the best available device.
        
        Returns:
            torch.device: Best available device (CUDA > CPU)
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        
        return device
    
    def load_model(self, 
                   model_path: Union[str, Path], 
                   config: Optional[ModelConfig] = None,
                   strict: bool = True) -> SpatioTemporalTransformer:
        """
        Load a pre-trained Spatio-Temporal Transformer model.
        
        Args:
            model_path (Union[str, Path]): Path to the model file (.pth or .pt)
            config (ModelConfig): Model configuration (loaded from file if None)
            strict (bool): Whether to strictly enforce state dict loading
            
        Returns:
            SpatioTemporalTransformer: Loaded model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract configuration if available
            if config is None:
                if 'config' in checkpoint:
                    config_dict = checkpoint['config']
                    config = ModelConfig(**config_dict)
                    logger.info("Loaded model configuration from checkpoint")
                else:
                    config = ModelConfig()
                    logger.warning("No configuration found in checkpoint, using default")
            
            # Create model
            model = SpatioTemporalTransformer(config)
            model = model.to(self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=strict)
            
            # Set to evaluation mode
            model.eval()
            
            logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Log additional checkpoint info if available
            if 'epoch' in checkpoint:
                logger.info(f"Model trained for {checkpoint['epoch']} epochs")
            if 'loss' in checkpoint:
                logger.info(f"Final training loss: {checkpoint['loss']:.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def save_model(self, 
                   model: SpatioTemporalTransformer, 
                   save_path: Union[str, Path],
                   epoch: Optional[int] = None,
                   loss: Optional[float] = None,
                   optimizer_state: Optional[Dict] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a Spatio-Temporal Transformer model.
        
        Args:
            model (SpatioTemporalTransformer): Model to save
            save_path (Union[str, Path]): Path to save the model
            epoch (int): Training epoch (optional)
            loss (float): Training loss (optional)
            optimizer_state (Dict): Optimizer state dict (optional)
            metadata (Dict): Additional metadata (optional)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to: {save_path}")
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'num_sensors': model.config.num_sensors,
                'feature_dim': model.config.feature_dim,
                'd_model': model.config.d_model,
                'num_heads': model.config.num_heads,
                'num_layers': model.config.num_layers,
                'max_seq_length': model.config.max_seq_length,
                'dropout': model.config.dropout,
                'num_classes': model.config.num_classes
            }
        }
        
        # Add optional information
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        # Save checkpoint
        try:
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise RuntimeError(f"Model saving failed: {str(e)}")
    
    def create_model(self, config: Optional[ModelConfig] = None) -> SpatioTemporalTransformer:
        """
        Create a new Spatio-Temporal Transformer model.
        
        Args:
            config (ModelConfig): Model configuration
            
        Returns:
            SpatioTemporalTransformer: New model instance
        """
        if config is None:
            config = ModelConfig()
        
        model = SpatioTemporalTransformer(config)
        model = model.to(self.device)
        
        logger.info(f"Created new model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def validate_model(self, model: SpatioTemporalTransformer) -> Dict[str, Any]:
        """
        Validate model architecture and perform basic functionality tests.
        
        Args:
            model (SpatioTemporalTransformer): Model to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info("Validating model architecture and functionality")
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Test model forward pass with dummy data
            batch_size = 2
            seq_len = 60
            num_sensors = model.config.num_sensors
            feature_dim = model.config.feature_dim
            
            dummy_input = torch.randn(batch_size, seq_len, num_sensors, feature_dim, device=self.device)
            
            with torch.no_grad():
                outputs = model(dummy_input)
            
            # Validate output shapes
            expected_logits_shape = (batch_size, model.config.num_classes)
            expected_risk_shape = (batch_size, 1)
            expected_features_shape = (batch_size, model.config.d_model)
            
            if outputs['logits'].shape != expected_logits_shape:
                results['errors'].append(f"Logits shape mismatch: {outputs['logits'].shape} != {expected_logits_shape}")
                results['valid'] = False
            
            if outputs['risk_score'].shape != expected_risk_shape:
                results['errors'].append(f"Risk score shape mismatch: {outputs['risk_score'].shape} != {expected_risk_shape}")
                results['valid'] = False
            
            if outputs['features'].shape != expected_features_shape:
                results['errors'].append(f"Features shape mismatch: {outputs['features'].shape} != {expected_features_shape}")
                results['valid'] = False
            
            # Validate risk score range
            risk_scores = outputs['risk_score']
            if not (torch.all(risk_scores >= 0) and torch.all(risk_scores <= 100)):
                results['errors'].append(f"Risk scores out of range [0, 100]: {risk_scores.min().item():.2f} - {risk_scores.max().item():.2f}")
                results['valid'] = False
            
            # Collect model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results['info'] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
                'device': str(model.device) if hasattr(model, 'device') else str(self.device),
                'config': {
                    'num_sensors': model.config.num_sensors,
                    'feature_dim': model.config.feature_dim,
                    'd_model': model.config.d_model,
                    'num_heads': model.config.num_heads,
                    'num_layers': model.config.num_layers,
                    'num_classes': model.config.num_classes
                }
            }
            
            if results['valid']:
                logger.info("Model validation passed successfully")
            else:
                logger.error(f"Model validation failed with {len(results['errors'])} errors")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Validation failed with exception: {str(e)}")
            logger.error(f"Model validation failed: {str(e)}")
        
        return results


def load_model(model_path: Union[str, Path], 
               device: Optional[torch.device] = None,
               config: Optional[ModelConfig] = None) -> SpatioTemporalTransformer:
    """
    Convenience function to load a model.
    
    Args:
        model_path (Union[str, Path]): Path to model file
        device (torch.device): Device to load on
        config (ModelConfig): Model configuration
        
    Returns:
        SpatioTemporalTransformer: Loaded model
    """
    loader = ModelLoader(device)
    return loader.load_model(model_path, config)


def create_model(config: Optional[ModelConfig] = None, 
                device: Optional[torch.device] = None) -> SpatioTemporalTransformer:
    """
    Convenience function to create a new model.
    
    Args:
        config (ModelConfig): Model configuration
        device (torch.device): Device to create on
        
    Returns:
        SpatioTemporalTransformer: New model instance
    """
    loader = ModelLoader(device)
    return loader.create_model(config)


def save_model(model: SpatioTemporalTransformer, 
               save_path: Union[str, Path],
               **kwargs) -> None:
    """
    Convenience function to save a model.
    
    Args:
        model (SpatioTemporalTransformer): Model to save
        save_path (Union[str, Path]): Save path
        **kwargs: Additional arguments for ModelLoader.save_model
    """
    loader = ModelLoader()
    loader.save_model(model, save_path, **kwargs)


def validate_model(model: SpatioTemporalTransformer, 
                  device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Convenience function to validate a model.
    
    Args:
        model (SpatioTemporalTransformer): Model to validate
        device (torch.device): Device for validation
        
    Returns:
        Dict[str, Any]: Validation results
    """
    loader = ModelLoader(device)
    return loader.validate_model(model)