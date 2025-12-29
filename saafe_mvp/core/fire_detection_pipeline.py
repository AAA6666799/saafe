"""
Main fire detection pipeline that orchestrates model inference.

This module implements the FireDetectionPipeline class that coordinates
data preprocessing, model inference, anti-hallucination validation,
and result formatting with performance monitoring.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .data_models import SensorReading, PredictionResult, ValidationResult
from ..models.model_manager import ModelManager
from ..models.anti_hallucination import AntiHallucinationEngine
from ..utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, safe_execute
from ..utils.fallback_manager import FallbackManager

logger = logging.getLogger(__name__)



class DataPreprocessor:
    """
    Handles data preprocessing and normalization for model input.
    Converts sensor readings to model-compatible tensor format.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize the data preprocessor.
        
        Args:
            device (torch.device): Device for tensor operations
        """
        self.device = device or torch.device('cpu')
        
        # Normalization parameters (mean and std for each feature)
        # These should ideally be computed from training data
        self.normalization_params = {
            'temperature': {'mean': 22.0, 'std': 15.0},
            'pm25': {'mean': 25.0, 'std': 30.0},
            'co2': {'mean': 500.0, 'std': 200.0},
            'audio_level': {'mean': 40.0, 'std': 15.0}
        }
        
        # Sequence parameters
        self.max_sequence_length = 60  # Maximum sequence length for model
        self.num_sensors = 4  # Number of sensor locations
        self.feature_dim = 4  # Number of features per sensor
        
        logger.info("DataPreprocessor initialized")
    
    def preprocess_sensor_readings(self, 
                                 readings: List[SensorReading],
                                 sequence_length: int = None) -> torch.Tensor:
        """
        Convert sensor readings to model input tensor.
        
        Args:
            readings (List[SensorReading]): List of sensor readings
            sequence_length (int): Desired sequence length (uses max if None)
            
        Returns:
            torch.Tensor: Preprocessed tensor (1, seq_len, num_sensors, features)
        """
        if not readings:
            raise ValueError("No sensor readings provided")
        
        if sequence_length is None:
            sequence_length = min(len(readings), self.max_sequence_length)
        
        # Take the most recent readings
        recent_readings = readings[-sequence_length:] if len(readings) >= sequence_length else readings
        
        # Create tensor
        batch_size = 1
        actual_seq_len = len(recent_readings)
        
        # Initialize tensor with zeros
        tensor = torch.zeros(
            batch_size, sequence_length, self.num_sensors, self.feature_dim,
            device=self.device, dtype=torch.float32
        )
        
        # Fill tensor with sensor data
        for t, reading in enumerate(recent_readings):
            # For now, replicate single reading across all sensor locations
            # In a real system, you'd have readings from multiple locations
            sensor_values = self._normalize_reading(reading)
            
            for sensor_idx in range(self.num_sensors):
                tensor[0, t, sensor_idx, :] = sensor_values
        
        # If we have fewer readings than sequence_length, pad with the last reading
        if actual_seq_len < sequence_length:
            last_reading = recent_readings[-1]
            last_values = self._normalize_reading(last_reading)
            
            for t in range(actual_seq_len, sequence_length):
                for sensor_idx in range(self.num_sensors):
                    tensor[0, t, sensor_idx, :] = last_values
        
        return tensor
    
    def _normalize_reading(self, reading: SensorReading) -> torch.Tensor:
        """
        Normalize a single sensor reading.
        
        Args:
            reading (SensorReading): Raw sensor reading
            
        Returns:
            torch.Tensor: Normalized feature vector
        """
        # Extract raw values
        raw_values = [
            reading.temperature,
            reading.pm25,
            reading.co2,
            reading.audio_level
        ]
        
        # Normalize each feature
        normalized_values = []
        feature_names = ['temperature', 'pm25', 'co2', 'audio_level']
        
        for value, feature_name in zip(raw_values, feature_names):
            params = self.normalization_params[feature_name]
            normalized = (value - params['mean']) / params['std']
            normalized_values.append(normalized)
        
        return torch.tensor(normalized_values, device=self.device, dtype=torch.float32)
    
    def create_feature_importance_map(self, 
                                    gradients: torch.Tensor,
                                    readings: List[SensorReading]) -> Dict[str, float]:
        """
        Create feature importance map from model gradients.
        
        Args:
            gradients (torch.Tensor): Model gradients w.r.t. input
            readings (List[SensorReading]): Original sensor readings
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if gradients.dim() != 4:  # (batch, seq, sensors, features)
            logger.warning(f"Unexpected gradient shape: {gradients.shape}")
            return {'temperature': 0.25, 'pm25': 0.25, 'co2': 0.25, 'audio_level': 0.25}
        
        # Average gradients across batch, sequence, and sensors
        avg_gradients = gradients.abs().mean(dim=(0, 1, 2))  # (features,)
        
        # Normalize to sum to 1
        total_importance = avg_gradients.sum()
        if total_importance > 0:
            normalized_gradients = avg_gradients / total_importance
        else:
            normalized_gradients = torch.ones_like(avg_gradients) / len(avg_gradients)
        
        feature_names = ['temperature', 'pm25', 'co2', 'audio_level']
        importance_map = {}
        
        for i, feature_name in enumerate(feature_names):
            importance_map[feature_name] = float(normalized_gradients[i])
        
        return importance_map


class FireDetectionPipeline:
    """
    Main fire detection pipeline that orchestrates model inference.
    
    Coordinates data preprocessing, model inference, anti-hallucination validation,
    and result formatting with comprehensive performance monitoring.
    """
    
    def __init__(self, 
                 model_manager: ModelManager,
                 device: torch.device = None,
                 enable_anti_hallucination: bool = True,
                 error_handler: ErrorHandler = None,
                 fallback_manager: FallbackManager = None):
        """
        Initialize the fire detection pipeline.
        
        Args:
            model_manager (ModelManager): Model management system
            device (torch.device): Device for computations
            enable_anti_hallucination (bool): Whether to enable anti-hallucination
            error_handler (ErrorHandler): Error handling system
            fallback_manager (FallbackManager): Fallback management system
        """
        self.device = device or torch.device('cpu')
        self.model_manager = model_manager
        self.enable_anti_hallucination = enable_anti_hallucination
        
        # Initialize error handling and fallback systems
        from ..utils.error_handler import get_error_handler
        from ..utils.fallback_manager import get_fallback_manager
        self.error_handler = error_handler or get_error_handler()
        self.fallback_manager = fallback_manager or get_fallback_manager()
        
        # Initialize components with error handling
        try:
            self.preprocessor = DataPreprocessor(device=self.device)
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, "FireDetectionPipeline",
                context={'component': 'preprocessor_init'}
            )
            # Use fallback preprocessor
            self.preprocessor = self._create_fallback_preprocessor()
        
        # Initialize anti-hallucination engine if enabled
        self.anti_hallucination_engine = None
        if self.enable_anti_hallucination:
            try:
                self.anti_hallucination_engine = model_manager.create_anti_hallucination_engine()
                if self.anti_hallucination_engine is None:
                    logger.warning("Failed to create anti-hallucination engine")
                    self.enable_anti_hallucination = False
            except Exception as e:
                self.error_handler.handle_error(
                    e, ErrorCategory.MODEL_ERROR, "FireDetectionPipeline",
                    context={'component': 'anti_hallucination_init'}
                )
                self.enable_anti_hallucination = False
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'avg_processing_time': 0.0,
            'min_processing_time': float('inf'),
            'max_processing_time': 0.0,
            'error_count': 0,
            'fallback_count': 0
        }
        
        # Class mapping
        self.class_names = ['normal', 'cooking', 'fire']
        
        logger.info("FireDetectionPipeline initialized with error handling")
        logger.info(f"Anti-hallucination enabled: {self.enable_anti_hallucination}")
    
    def predict(self, 
                sensor_readings: List[SensorReading],
                return_gradients: bool = True) -> PredictionResult:
        """
        Generate fire detection prediction from sensor readings.
        
        Args:
            sensor_readings (List[SensorReading]): List of sensor readings
            return_gradients (bool): Whether to compute feature importance
            
        Returns:
            PredictionResult: Comprehensive prediction result
        """
        start_time = time.time()
        
        try:
            # Preprocess sensor data with error handling
            input_tensor = safe_execute(
                lambda: self.preprocessor.preprocess_sensor_readings(sensor_readings),
                ErrorCategory.DATA_ERROR, "FireDetectionPipeline.preprocess",
                default_return=None,
                context={'sensor_count': len(sensor_readings)}
            )
            
            if input_tensor is None:
                raise RuntimeError("Failed to preprocess sensor data")
            
            # Get primary model with fallback
            primary_model = self.model_manager.get_model()
            if primary_model is None:
                # Try fallback model
                primary_model = self.fallback_manager.handle_model_failure(self.device)
                if primary_model is None:
                    raise RuntimeError("No model available for prediction")
                self.performance_stats['fallback_count'] += 1
            
            # Enable gradient computation for feature importance
            if return_gradients:
                input_tensor.requires_grad_(True)
            
            # Model inference with error handling
            model_output = safe_execute(
                lambda: self._run_model_inference(primary_model, input_tensor, return_gradients),
                ErrorCategory.MODEL_ERROR, "FireDetectionPipeline.inference",
                default_return=None,
                context={'model_type': type(primary_model).__name__}
            )
            
            if model_output is None:
                raise RuntimeError("Model inference failed")
            
            # Extract predictions
            if isinstance(model_output, dict):
                risk_score = float(model_output.get('risk_score', torch.tensor(0.0)).squeeze())
                logits = model_output.get('logits', torch.zeros(3))
                features = model_output.get('features', torch.zeros(256))
            else:
                # Handle simple tensor output
                risk_score = float(model_output.squeeze()) if model_output.numel() == 1 else 0.0
                logits = torch.zeros(3)
                features = torch.zeros(256)
            
            # Ensure risk score is in valid range
            risk_score = max(0.0, min(100.0, risk_score))
            
            # Get predicted class
            predicted_class_idx = torch.argmax(logits).item()
            predicted_class = self.class_names[predicted_class_idx]
            
            # Calculate confidence from logits
            confidence = float(torch.softmax(logits, dim=-1).max())
            
            # Compute feature importance with error handling
            feature_importance = safe_execute(
                lambda: self._compute_feature_importance(input_tensor, sensor_readings, return_gradients),
                ErrorCategory.DATA_ERROR, "FireDetectionPipeline.feature_importance",
                default_return={'temperature': 0.25, 'pm25': 0.25, 'co2': 0.25, 'audio_level': 0.25}
            )
            
            # Anti-hallucination validation with error handling
            anti_hallucination_result = None
            ensemble_votes = {}
            
            if self.enable_anti_hallucination and self.anti_hallucination_engine:
                anti_hallucination_result = safe_execute(
                    lambda: self._run_anti_hallucination(risk_score, input_tensor),
                    ErrorCategory.MODEL_ERROR, "FireDetectionPipeline.anti_hallucination",
                    default_return=ValidationResult(
                        is_valid=True,
                        confidence=1.0,
                        reason="Anti-hallucination validation failed - using primary prediction"
                    )
                )
                
                if anti_hallucination_result:
                    # Apply confidence adjustment
                    if anti_hallucination_result.confidence_adjustment != 1.0:
                        risk_score *= anti_hallucination_result.confidence_adjustment
                        risk_score = max(0.0, min(100.0, risk_score))
                    
                    ensemble_votes = {}  # ValidationResult doesn't have ensemble_votes
                    
                    # Update predicted class based on anti-hallucination
                    if hasattr(anti_hallucination_result, 'cooking_detected') and anti_hallucination_result.cooking_detected:
                        predicted_class = 'cooking'
                    elif not anti_hallucination_result.is_valid and risk_score > 50:
                        # Reduce confidence for invalidated high-risk predictions
                        confidence *= 0.5
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update performance statistics
            self._update_performance_stats(processing_time)
            
            # Get model metadata
            model_metadata = self._get_model_metadata()
            
            # Create result
            result = PredictionResult(
                risk_score=risk_score,
                confidence=confidence,
                predicted_class=predicted_class,
                feature_importance=feature_importance,
                processing_time=processing_time,
                ensemble_votes=ensemble_votes,
                anti_hallucination=anti_hallucination_result,
                timestamp=datetime.now(),
                model_metadata=model_metadata
            )
            
            logger.debug(f"Prediction completed: risk={risk_score:.1f}, class={predicted_class}, time={processing_time:.1f}ms")
            
            return result
            
        except Exception as e:
            self.performance_stats['error_count'] += 1
            
            # Handle error with comprehensive error handling
            error_info = self.error_handler.handle_error(
                e, ErrorCategory.MODEL_ERROR, "FireDetectionPipeline.predict",
                context={
                    'sensor_count': len(sensor_readings),
                    'return_gradients': return_gradients,
                    'anti_hallucination_enabled': self.enable_anti_hallucination
                },
                severity=ErrorSeverity.HIGH
            )
            
            # Return safe fallback result
            processing_time = (time.time() - start_time) * 1000
            return self._create_fallback_result(processing_time, str(e))
    
    def predict_batch(self, 
                     batch_readings: List[List[SensorReading]]) -> List[PredictionResult]:
        """
        Generate predictions for a batch of sensor reading sequences.
        
        Args:
            batch_readings (List[List[SensorReading]]): Batch of reading sequences
            
        Returns:
            List[PredictionResult]: List of prediction results
        """
        results = []
        
        for readings in batch_readings:
            result = self.predict(readings, return_gradients=False)  # Skip gradients for batch
            results.append(result)
        
        return results
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics."""
        self.performance_stats['total_predictions'] += 1
        
        # Update processing time statistics
        current_avg = self.performance_stats['avg_processing_time']
        count = self.performance_stats['total_predictions']
        
        self.performance_stats['avg_processing_time'] = (
            (current_avg * (count - 1) + processing_time) / count
        )
        
        self.performance_stats['min_processing_time'] = min(
            self.performance_stats['min_processing_time'], processing_time
        )
        
        self.performance_stats['max_processing_time'] = max(
            self.performance_stats['max_processing_time'], processing_time
        )
    
    def _get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current model."""
        try:
            system_status = self.model_manager.get_system_status()
            return {
                'device': system_status['device'],
                'model_count': system_status['model_count'],
                'primary_model': system_status['primary_model'],
                'anti_hallucination_enabled': self.enable_anti_hallucination
            }
        except Exception as e:
            logger.warning(f"Failed to get model metadata: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics and system status
        """
        metrics = self.performance_stats.copy()
        
        # Add system metrics
        try:
            system_status = self.model_manager.get_system_status()
            metrics.update({
                'system_status': system_status,
                'anti_hallucination_enabled': self.enable_anti_hallucination,
                'device': str(self.device)
            })
        except Exception as e:
            metrics['system_error'] = str(e)
        
        return metrics
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_predictions': 0,
            'avg_processing_time': 0.0,
            'min_processing_time': float('inf'),
            'max_processing_time': 0.0,
            'error_count': 0
        }
        logger.info("Performance statistics reset")
    
    def set_anti_hallucination_enabled(self, enabled: bool):
        """
        Enable or disable anti-hallucination validation.
        
        Args:
            enabled (bool): Whether to enable anti-hallucination
        """
        if enabled and self.anti_hallucination_engine is None:
            # Try to create engine
            self.anti_hallucination_engine = self.model_manager.create_anti_hallucination_engine()
            if self.anti_hallucination_engine is None:
                logger.error("Cannot enable anti-hallucination: engine creation failed")
                return False
        
        self.enable_anti_hallucination = enabled
        logger.info(f"Anti-hallucination {'enabled' if enabled else 'disabled'}")
        return True
    
    def update_normalization_params(self, params: Dict[str, Dict[str, float]]):
        """
        Update data normalization parameters.
        
        Args:
            params (Dict): New normalization parameters
        """
        self.preprocessor.normalization_params.update(params)
        logger.info("Normalization parameters updated")
    
    def _create_fallback_preprocessor(self) -> 'DataPreprocessor':
        """Create a fallback preprocessor with minimal functionality."""
        class FallbackPreprocessor:
            def __init__(self, device):
                self.device = device
                self.normalization_params = {
                    'temperature': {'mean': 22.0, 'std': 15.0},
                    'pm25': {'mean': 25.0, 'std': 30.0},
                    'co2': {'mean': 500.0, 'std': 200.0},
                    'audio_level': {'mean': 40.0, 'std': 15.0}
                }
            
            def preprocess_sensor_readings(self, readings, sequence_length=None):
                if not readings:
                    return torch.zeros(1, 10, 4, 4, device=self.device)
                
                # Simple preprocessing - just take the last reading
                last_reading = readings[-1]
                values = torch.tensor([
                    last_reading.temperature,
                    last_reading.pm25,
                    last_reading.co2,
                    last_reading.audio_level
                ], device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                
                return values.expand(1, 10, 4, 4)  # Expand to expected shape
            
            def create_feature_importance_map(self, gradients, readings):
                return {'temperature': 0.25, 'pm25': 0.25, 'co2': 0.25, 'audio_level': 0.25}
        
        return FallbackPreprocessor(self.device)
    
    def _run_model_inference(self, model, input_tensor, return_gradients):
        """Run model inference with proper gradient handling."""
        with torch.set_grad_enabled(return_gradients):
            return model(input_tensor)
    
    def _compute_feature_importance(self, input_tensor, sensor_readings, return_gradients):
        """Compute feature importance from gradients."""
        if return_gradients and input_tensor.grad is not None:
            return self.preprocessor.create_feature_importance_map(
                input_tensor.grad, sensor_readings
            )
        else:
            return {'temperature': 0.25, 'pm25': 0.25, 'co2': 0.25, 'audio_level': 0.25}
    
    def _run_anti_hallucination(self, risk_score, input_tensor):
        """Run anti-hallucination validation."""
        return self.anti_hallucination_engine.validate_fire_prediction(
            risk_score, input_tensor.squeeze(0)
        )
    
    def _create_fallback_result(self, processing_time, error_message):
        """Create a safe fallback prediction result."""
        return PredictionResult(
            risk_score=0.0,
            confidence=0.0,
            predicted_class='normal',
            feature_importance={'temperature': 0.25, 'pm25': 0.25, 'co2': 0.25, 'audio_level': 0.25},
            processing_time=processing_time,
            ensemble_votes={},
            anti_hallucination=ValidationResult(
                is_valid=False,
                confidence=0.0,
                reason=f"Prediction error: {error_message}"
            ),
            timestamp=datetime.now(),
            model_metadata={'error': error_message, 'fallback': True}
        )


# Convenience functions
def create_fire_detection_pipeline(model_manager: ModelManager = None,
                                 device: torch.device = None,
                                 enable_anti_hallucination: bool = True) -> FireDetectionPipeline:
    """
    Create a fire detection pipeline with default configuration.
    
    Args:
        model_manager (ModelManager): Model manager (creates default if None)
        device (torch.device): Device for computations
        enable_anti_hallucination (bool): Whether to enable anti-hallucination
        
    Returns:
        FireDetectionPipeline: Configured pipeline
    """
    if model_manager is None:
        from ..models.model_manager import ModelManager
        model_manager = ModelManager(device=device)
    
    return FireDetectionPipeline(
        model_manager=model_manager,
        device=device,
        enable_anti_hallucination=enable_anti_hallucination
    )