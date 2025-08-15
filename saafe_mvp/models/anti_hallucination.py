"""
Anti-hallucination logic and ensemble voting for fire detection.

This module implements the safety layer that prevents false alarms while maintaining
high fire detection sensitivity through ensemble voting, cooking pattern detection,
and comprehensive fire signature validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of anti-hallucination validation."""
    is_valid: bool
    confidence_adjustment: float
    reasoning: str
    ensemble_votes: Dict[str, float]
    cooking_detected: bool = False
    fire_signatures_confirmed: bool = False


class EnsembleFireDetector:
    """
    Ensemble fire detection system that combines multiple models with voting strategy.
    Requires at least 2 models in agreement for critical alerts to prevent false positives.
    """
    
    def __init__(self, models: List[nn.Module], voting_strategy: str = 'weighted', device: Optional[torch.device] = None):
        """
        Initialize the ensemble fire detector with multiple models.
        
        Args:
            models (List[nn.Module]): List of trained fire detection models
            voting_strategy (str): Voting strategy ('weighted', 'majority', 'conservative')
            device (torch.device): Device for model inference
        """
        self.models = models
        self.voting_strategy = voting_strategy
        self.device = device or torch.device('cpu')
        self.num_models = len(models)
        
        # Model confidence weights (can be learned or set based on validation performance)
        self.model_weights = torch.ones(self.num_models, device=self.device) / self.num_models
        
        # Critical alert thresholds
        self.critical_threshold = 85.0
        self.agreement_threshold = 2  # Minimum models that must agree
        
        # Move all models to device
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        logger.info(f"EnsembleFireDetector initialized with {self.num_models} models")
        logger.info(f"Voting strategy: {voting_strategy}")
        logger.info(f"Agreement threshold: {self.agreement_threshold} models")
    
    def predict(self, data: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Generate ensemble prediction with individual model scores and confidence.
        
        Args:
            data (torch.Tensor): Input sensor data (batch_size, seq_len, num_sensors, features)
            
        Returns:
            Tuple[float, Dict[str, float]]: (ensemble_score, individual_scores_dict)
        """
        individual_scores = {}
        predictions = torch.zeros(self.num_models, device=self.device)
        confidences = torch.zeros(self.num_models, device=self.device)
        
        # Get predictions from all models
        with torch.no_grad():
            for i, model in enumerate(self.models):
                try:
                    # Get model prediction
                    output = model(data)
                    
                    # Extract risk score and confidence
                    if isinstance(output, dict):
                        risk_score = output.get('risk_score', output.get('logits', torch.tensor(0.0)))
                        confidence = torch.tensor(1.0)  # Default confidence
                    elif isinstance(output, tuple):
                        risk_score = output[0]
                        confidence = output[1] if len(output) > 1 else torch.tensor(1.0)
                    else:
                        risk_score = output
                        confidence = torch.tensor(1.0)
                    
                    # Ensure risk score is in 0-100 range
                    if risk_score.dim() > 0:
                        risk_score = risk_score.mean()  # Average if batch
                    
                    risk_score = torch.clamp(risk_score, 0, 100)
                    
                    predictions[i] = risk_score
                    confidences[i] = confidence
                    individual_scores[f'model_{i+1}'] = float(risk_score)
                    
                except Exception as e:
                    logger.warning(f"Model {i+1} prediction failed: {e}")
                    predictions[i] = 0.0
                    confidences[i] = 0.0
                    individual_scores[f'model_{i+1}'] = 0.0
        
        # Calculate ensemble score based on voting strategy
        ensemble_score = self._calculate_ensemble_score(predictions, confidences)
        
        # Add ensemble metadata
        individual_scores['ensemble_score'] = float(ensemble_score)
        individual_scores['agreement_count'] = int(self._count_agreements(predictions))
        individual_scores['confidence_weighted_score'] = float(self._weighted_score(predictions, confidences))
        
        return float(ensemble_score), individual_scores
    
    def _calculate_ensemble_score(self, predictions: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        """
        Calculate ensemble score based on the selected voting strategy.
        
        Args:
            predictions (torch.Tensor): Individual model predictions
            confidences (torch.Tensor): Model confidence scores
            
        Returns:
            torch.Tensor: Ensemble prediction score
        """
        if self.voting_strategy == 'weighted':
            # Weighted average based on model weights and confidences
            total_weights = self.model_weights * confidences
            if total_weights.sum() > 0:
                ensemble_score = (predictions * total_weights).sum() / total_weights.sum()
            else:
                ensemble_score = predictions.mean()
                
        elif self.voting_strategy == 'majority':
            # Simple majority voting (average of predictions)
            ensemble_score = predictions.mean()
            
        elif self.voting_strategy == 'conservative':
            # Conservative approach: require agreement for high scores
            agreement_count = self._count_agreements(predictions)
            
            if agreement_count >= self.agreement_threshold:
                # Use weighted average if sufficient agreement
                ensemble_score = self._weighted_score(predictions, confidences)
            else:
                # Apply penalty for lack of agreement
                base_score = predictions.mean()
                agreement_penalty = (self.agreement_threshold - agreement_count) * 10.0
                ensemble_score = torch.clamp(base_score - agreement_penalty, 0, 100)
        
        else:
            # Default to simple average
            ensemble_score = predictions.mean()
        
        return ensemble_score
    
    def _count_agreements(self, predictions: torch.Tensor, threshold: float = 85.0) -> int:
        """
        Count how many models agree on critical alert (score > threshold).
        
        Args:
            predictions (torch.Tensor): Model predictions
            threshold (float): Critical alert threshold
            
        Returns:
            int: Number of models agreeing on critical alert
        """
        critical_predictions = predictions > threshold
        return int(critical_predictions.sum())
    
    def _weighted_score(self, predictions: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        """
        Calculate confidence-weighted score.
        
        Args:
            predictions (torch.Tensor): Model predictions
            confidences (torch.Tensor): Model confidences
            
        Returns:
            torch.Tensor: Weighted prediction score
        """
        weights = confidences * self.model_weights
        if weights.sum() > 0:
            return (predictions * weights).sum() / weights.sum()
        else:
            return predictions.mean()
    
    def get_model_agreements(self, predictions: torch.Tensor = None) -> Dict[str, bool]:
        """
        Get agreement status for each model on critical alerts.
        
        Args:
            predictions (torch.Tensor): Model predictions (optional, uses last if None)
            
        Returns:
            Dict[str, bool]: Agreement status for each model
        """
        if predictions is None:
            # Would need to store last predictions - simplified for now
            return {f'model_{i+1}': False for i in range(self.num_models)}
        
        agreements = {}
        for i, pred in enumerate(predictions):
            agreements[f'model_{i+1}'] = bool(pred > self.critical_threshold)
        
        return agreements
    
    def update_model_weights(self, validation_scores: List[float]):
        """
        Update model weights based on validation performance.
        
        Args:
            validation_scores (List[float]): Validation accuracy scores for each model
        """
        if len(validation_scores) != self.num_models:
            logger.warning(f"Expected {self.num_models} scores, got {len(validation_scores)}")
            return
        
        # Convert to tensor and normalize
        scores = torch.tensor(validation_scores, device=self.device)
        self.model_weights = F.softmax(scores, dim=0)
        
        logger.info("Model weights updated based on validation performance:")
        for i, weight in enumerate(self.model_weights):
            logger.info(f"   Model {i+1}: {float(weight):.3f}")
    
    def set_voting_strategy(self, strategy: str):
        """
        Update the voting strategy.
        
        Args:
            strategy (str): New voting strategy
        """
        valid_strategies = ['weighted', 'majority', 'conservative']
        if strategy in valid_strategies:
            self.voting_strategy = strategy
            logger.info(f"Voting strategy updated to: {strategy}")
        else:
            logger.warning(f"Invalid strategy. Valid options: {valid_strategies}")


class CookingPatternDetector:
    """
    Detects cooking-specific patterns to prevent false fire alarms during cooking activities.
    Identifies characteristic cooking signatures: elevated PM2.5/CO₂ without sustained high temperature.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the cooking pattern detector.
        
        Args:
            device (torch.device): Device for tensor operations
        """
        self.device = device or torch.device('cpu')
        
        # Cooking pattern thresholds
        self.cooking_thresholds = {
            'pm25_elevated': 30.0,      # PM2.5 threshold for cooking detection
            'co2_elevated': 600.0,      # CO₂ threshold for cooking detection
            'temp_max_cooking': 35.0,   # Maximum temperature for cooking (not fire)
            'temp_gradient_max': 2.0,   # Max temperature gradient for cooking
            'duration_threshold': 10,   # Minimum duration for pattern detection
            'pm25_co2_ratio_min': 0.02, # Minimum PM2.5/CO₂ ratio for cooking
            'pm25_co2_ratio_max': 0.15  # Maximum PM2.5/CO₂ ratio for cooking
        }
        
        logger.info("CookingPatternDetector initialized")
    
    def detect_cooking_patterns(self, sensor_data: torch.Tensor, window_size: int = 20) -> Dict[str, Any]:
        """
        Detect cooking-specific patterns in sensor data.
        
        Args:
            sensor_data (torch.Tensor): Sensor data (time_steps, num_sensors, features)
            window_size (int): Analysis window size for pattern detection
            
        Returns:
            Dict[str, Any]: Cooking pattern detection results
        """
        if sensor_data.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {sensor_data.dim()}D")
        
        time_steps, num_sensors, features = sensor_data.shape
        
        # Extract features (assuming order: temperature, PM2.5, CO₂, audio)
        temperature = sensor_data[:, :, 0]  # Temperature
        pm25 = sensor_data[:, :, 1]         # PM2.5
        co2 = sensor_data[:, :, 2]          # CO₂
        audio = sensor_data[:, :, 3]        # Audio
        
        # Calculate sensor averages across locations
        temp_avg = temperature.mean(dim=1)
        pm25_avg = pm25.mean(dim=1)
        co2_avg = co2.mean(dim=1)
        audio_avg = audio.mean(dim=1)
        
        # Analyze recent window
        window_start = max(0, time_steps - window_size)
        recent_temp = temp_avg[window_start:]
        recent_pm25 = pm25_avg[window_start:]
        recent_co2 = co2_avg[window_start:]
        recent_audio = audio_avg[window_start:]
        
        # Cooking pattern indicators
        results = {
            'is_cooking': False,
            'confidence': 0.0,
            'indicators': {},
            'pattern_strength': 0.0
        }
        
        # Indicator 1: Elevated PM2.5 without extreme temperature
        pm25_elevated = recent_pm25.max() > self.cooking_thresholds['pm25_elevated']
        temp_moderate = recent_temp.max() < self.cooking_thresholds['temp_max_cooking']
        
        # Indicator 2: Elevated CO₂ with controlled temperature rise
        co2_elevated = recent_co2.max() > self.cooking_thresholds['co2_elevated']
        temp_gradient = self._calculate_temperature_gradient(recent_temp)
        temp_gradient_controlled = temp_gradient < self.cooking_thresholds['temp_gradient_max']
        
        # Indicator 3: PM2.5/CO₂ ratio characteristic of cooking
        pm25_co2_ratio = self._calculate_pm25_co2_ratio(recent_pm25, recent_co2)
        ratio_in_cooking_range = (
            self.cooking_thresholds['pm25_co2_ratio_min'] <= pm25_co2_ratio <= 
            self.cooking_thresholds['pm25_co2_ratio_max']
        )
        
        # Indicator 4: Gradual onset (not sudden spike)
        gradual_onset = self._detect_gradual_onset(recent_pm25, recent_co2)
        
        # Indicator 5: Audio levels consistent with cooking (not fire alarms)
        audio_cooking_range = self._check_audio_cooking_range(recent_audio)
        
        # Store individual indicators
        results['indicators'] = {
            'pm25_elevated': bool(pm25_elevated),
            'temp_moderate': bool(temp_moderate),
            'co2_elevated': bool(co2_elevated),
            'temp_gradient_controlled': bool(temp_gradient_controlled),
            'ratio_in_cooking_range': bool(ratio_in_cooking_range),
            'gradual_onset': bool(gradual_onset),
            'audio_cooking_range': bool(audio_cooking_range),
            'pm25_co2_ratio': float(pm25_co2_ratio),
            'temp_gradient': float(temp_gradient),
            'max_temp': float(recent_temp.max()),
            'max_pm25': float(recent_pm25.max()),
            'max_co2': float(recent_co2.max())
        }
        
        # Calculate cooking confidence based on indicators
        cooking_indicators = [
            pm25_elevated and temp_moderate,
            co2_elevated and temp_gradient_controlled,
            ratio_in_cooking_range,
            gradual_onset,
            audio_cooking_range
        ]
        
        positive_indicators = sum(cooking_indicators)
        results['confidence'] = positive_indicators / len(cooking_indicators)
        results['pattern_strength'] = positive_indicators
        
        # Determine if cooking pattern is detected (require at least 3 indicators)
        results['is_cooking'] = positive_indicators >= 3
        
        return results
    
    def _calculate_temperature_gradient(self, temperature: torch.Tensor) -> float:
        """
        Calculate the maximum temperature gradient (rate of change).
        
        Args:
            temperature (torch.Tensor): Temperature time series
            
        Returns:
            float: Maximum temperature gradient
        """
        if len(temperature) < 2:
            return 0.0
        
        gradients = torch.diff(temperature)
        return float(gradients.abs().max())
    
    def _calculate_pm25_co2_ratio(self, pm25: torch.Tensor, co2: torch.Tensor) -> float:
        """
        Calculate the average PM2.5/CO₂ ratio.
        
        Args:
            pm25 (torch.Tensor): PM2.5 time series
            co2 (torch.Tensor): CO₂ time series
            
        Returns:
            float: Average PM2.5/CO₂ ratio
        """
        # Avoid division by zero
        co2_safe = torch.clamp(co2, min=1.0)
        ratios = pm25 / co2_safe
        return float(ratios.mean())
    
    def _detect_gradual_onset(self, pm25: torch.Tensor, co2: torch.Tensor, threshold: float = 0.7) -> bool:
        """
        Detect gradual onset characteristic of cooking (vs sudden fire spike).
        
        Args:
            pm25 (torch.Tensor): PM2.5 time series
            co2 (torch.Tensor): CO₂ time series
            threshold (float): Correlation threshold for gradual onset
            
        Returns:
            bool: True if gradual onset detected
        """
        if len(pm25) < 5:
            return False
        
        # Check if increases are correlated and gradual
        pm25_increases = torch.diff(pm25) > 0
        co2_increases = torch.diff(co2) > 0
        
        # Calculate correlation between PM2.5 and CO₂ increases
        if len(pm25_increases) > 0:
            correlation = (pm25_increases.float() * co2_increases.float()).mean()
            return float(correlation) > threshold
        
        return False
    
    def _check_audio_cooking_range(self, audio: torch.Tensor) -> bool:
        """
        Check if audio levels are consistent with cooking activities.
        
        Args:
            audio (torch.Tensor): Audio level time series
            
        Returns:
            bool: True if audio levels suggest cooking
        """
        # Cooking audio: moderate levels, some variation but not extreme
        audio_mean = float(audio.mean())
        audio_max = float(audio.max())
        audio_std = float(audio.std())
        
        # Cooking characteristics: 30-60 dB average, max < 80 dB, moderate variation
        cooking_range = 30.0 <= audio_mean <= 60.0
        not_too_loud = audio_max < 80.0
        moderate_variation = 2.0 <= audio_std <= 15.0
        
        return cooking_range and not_too_loud and moderate_variation


class FireSignatureValidator:
    """
    Validates presence of multiple fire indicators simultaneously to confirm fire events.
    Checks for comprehensive fire signatures across all sensor modalities.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the fire signature validator.
        
        Args:
            device (torch.device): Device for tensor operations
        """
        self.device = device or torch.device('cpu')
        
        # Fire signature thresholds
        self.fire_thresholds = {
            'temp_critical': 60.0,      # Critical temperature threshold
            'temp_gradient_fire': 5.0,  # Rapid temperature rise for fire
            'pm25_fire': 100.0,         # PM2.5 threshold for fire
            'co2_fire': 1000.0,         # CO₂ threshold for fire
            'audio_fire': 70.0,         # Audio threshold for fire/alarms
            'temp_duration': 5,         # Sustained high temperature duration
            'multi_sensor_agreement': 0.75,  # Fraction of sensors that must agree
            'signature_completeness': 0.8    # Required completeness of fire signature
        }
        
        logger.info("FireSignatureValidator initialized")
    
    def validate_fire_signatures(self, sensor_data: torch.Tensor, window_size: int = 15) -> Dict[str, Any]:
        """
        Validate comprehensive fire signatures across multiple indicators.
        
        Args:
            sensor_data (torch.Tensor): Sensor data (time_steps, num_sensors, features)
            window_size (int): Analysis window size
            
        Returns:
            Dict[str, Any]: Fire signature validation results
        """
        if sensor_data.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {sensor_data.dim()}D")
        
        time_steps, num_sensors, features = sensor_data.shape
        
        # Extract features
        temperature = sensor_data[:, :, 0]
        pm25 = sensor_data[:, :, 1]
        co2 = sensor_data[:, :, 2]
        audio = sensor_data[:, :, 3]
        
        # Analyze recent window
        window_start = max(0, time_steps - window_size)
        recent_temp = temperature[window_start:]
        recent_pm25 = pm25[window_start:]
        recent_co2 = co2[window_start:]
        recent_audio = audio[window_start:]
        
        results = {
            'fire_confirmed': False,
            'confidence': 0.0,
            'signatures': {},
            'completeness_score': 0.0,
            'sensor_agreement': 0.0
        }
        
        # Fire Signature 1: Rapid temperature rise
        temp_signature = self._validate_temperature_signature(recent_temp)
        
        # Fire Signature 2: Extreme PM2.5 levels
        pm25_signature = self._validate_pm25_signature(recent_pm25)
        
        # Fire Signature 3: High CO₂ concentration
        co2_signature = self._validate_co2_signature(recent_co2)
        
        # Fire Signature 4: Audio anomalies (alarms, crackling)
        audio_signature = self._validate_audio_signature(recent_audio)
        
        # Fire Signature 5: Multi-sensor spatial agreement
        spatial_agreement = self._validate_spatial_agreement(
            recent_temp, recent_pm25, recent_co2, recent_audio
        )
        
        # Store signature results
        results['signatures'] = {
            'temperature': temp_signature,
            'pm25': pm25_signature,
            'co2': co2_signature,
            'audio': audio_signature,
            'spatial_agreement': spatial_agreement
        }
        
        # Calculate completeness score
        signature_scores = [
            temp_signature['score'],
            pm25_signature['score'],
            co2_signature['score'],
            audio_signature['score'],
            spatial_agreement['score']
        ]
        
        results['completeness_score'] = sum(signature_scores) / len(signature_scores)
        results['sensor_agreement'] = spatial_agreement['agreement_fraction']
        
        # Fire confirmation logic: require high completeness and multiple signatures
        critical_signatures = sum([
            temp_signature['critical'],
            pm25_signature['critical'],
            co2_signature['critical']
        ])
        
        results['confidence'] = results['completeness_score']
        results['fire_confirmed'] = (
            results['completeness_score'] >= self.fire_thresholds['signature_completeness'] and
            critical_signatures >= 2 and  # At least 2 critical signatures
            results['sensor_agreement'] >= self.fire_thresholds['multi_sensor_agreement']
        )
        
        return results
    
    def _validate_temperature_signature(self, temperature: torch.Tensor) -> Dict[str, Any]:
        """Validate temperature-based fire signature."""
        temp_max = temperature.max()
        temp_mean = temperature.mean()
        
        # Calculate temperature gradient
        if temperature.shape[0] > 1:
            temp_gradients = torch.diff(temperature.mean(dim=1))
            max_gradient = temp_gradients.max() if len(temp_gradients) > 0 else torch.tensor(0.0)
        else:
            max_gradient = torch.tensor(0.0)
        
        # Sustained high temperature
        high_temp_duration = (temperature.mean(dim=1) > self.fire_thresholds['temp_critical']).sum()
        
        signature = {
            'max_temp': float(temp_max),
            'mean_temp': float(temp_mean),
            'max_gradient': float(max_gradient),
            'high_temp_duration': int(high_temp_duration),
            'critical': bool(temp_max > self.fire_thresholds['temp_critical']),
            'rapid_rise': bool(max_gradient > self.fire_thresholds['temp_gradient_fire']),
            'sustained': bool(high_temp_duration >= self.fire_thresholds['temp_duration'])
        }
        
        # Calculate signature score
        score_components = [
            signature['critical'],
            signature['rapid_rise'],
            signature['sustained']
        ]
        signature['score'] = sum(score_components) / len(score_components)
        
        return signature
    
    def _validate_pm25_signature(self, pm25: torch.Tensor) -> Dict[str, Any]:
        """Validate PM2.5-based fire signature."""
        pm25_max = pm25.max()
        pm25_mean = pm25.mean()
        pm25_std = pm25.std()
        
        signature = {
            'max_pm25': float(pm25_max),
            'mean_pm25': float(pm25_mean),
            'std_pm25': float(pm25_std),
            'critical': bool(pm25_max > self.fire_thresholds['pm25_fire']),
            'elevated_mean': bool(pm25_mean > self.fire_thresholds['pm25_fire'] * 0.5),
            'high_variation': bool(pm25_std > 20.0)
        }
        
        score_components = [
            signature['critical'],
            signature['elevated_mean'],
            signature['high_variation']
        ]
        signature['score'] = sum(score_components) / len(score_components)
        
        return signature
    
    def _validate_co2_signature(self, co2: torch.Tensor) -> Dict[str, Any]:
        """Validate CO₂-based fire signature."""
        co2_max = co2.max()
        co2_mean = co2.mean()
        
        signature = {
            'max_co2': float(co2_max),
            'mean_co2': float(co2_mean),
            'critical': bool(co2_max > self.fire_thresholds['co2_fire']),
            'elevated_mean': bool(co2_mean > self.fire_thresholds['co2_fire'] * 0.6)
        }
        
        score_components = [
            signature['critical'],
            signature['elevated_mean']
        ]
        signature['score'] = sum(score_components) / len(score_components)
        
        return signature
    
    def _validate_audio_signature(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Validate audio-based fire signature."""
        audio_max = audio.max()
        audio_mean = audio.mean()
        audio_std = audio.std()
        
        signature = {
            'max_audio': float(audio_max),
            'mean_audio': float(audio_mean),
            'std_audio': float(audio_std),
            'critical': bool(audio_max > self.fire_thresholds['audio_fire']),
            'high_variation': bool(audio_std > 10.0)
        }
        
        score_components = [
            signature['critical'],
            signature['high_variation']
        ]
        signature['score'] = sum(score_components) / len(score_components)
        
        return signature
    
    def _validate_spatial_agreement(self, temperature: torch.Tensor, pm25: torch.Tensor, 
                                  co2: torch.Tensor, audio: torch.Tensor) -> Dict[str, Any]:
        """Validate spatial agreement across multiple sensors."""
        num_sensors = temperature.shape[1]
        
        # Check agreement for each sensor location
        sensor_agreements = []
        
        for sensor_idx in range(num_sensors):
            temp_high = temperature[:, sensor_idx].max() > self.fire_thresholds['temp_critical']
            pm25_high = pm25[:, sensor_idx].max() > self.fire_thresholds['pm25_fire']
            co2_high = co2[:, sensor_idx].max() > self.fire_thresholds['co2_fire']
            
            # Count indicators for this sensor
            indicators = [temp_high, pm25_high, co2_high]
            agreement_score = sum(indicators) / len(indicators)
            sensor_agreements.append(agreement_score)
        
        agreement_fraction = sum(s > 0.5 for s in sensor_agreements) / num_sensors
        
        signature = {
            'sensor_scores': sensor_agreements,
            'agreement_fraction': float(agreement_fraction),
            'strong_agreement': bool(agreement_fraction >= self.fire_thresholds['multi_sensor_agreement'])
        }
        
        signature['score'] = agreement_fraction
        
        return signature


class AntiHallucinationEngine:
    """
    Main anti-hallucination engine that combines ensemble voting, cooking detection,
    and fire signature validation to prevent false alarms.
    """
    
    def __init__(self, models: List[nn.Module], device: Optional[torch.device] = None):
        """
        Initialize the anti-hallucination engine.
        
        Args:
            models (List[nn.Module]): List of fire detection models
            device (torch.device): Device for computations
        """
        self.device = device or torch.device('cpu')
        
        # Initialize components
        self.ensemble_detector = EnsembleFireDetector(models, voting_strategy='conservative', device=self.device)
        self.cooking_detector = CookingPatternDetector(device=self.device)
        self.fire_validator = FireSignatureValidator(device=self.device)
        
        logger.info("AntiHallucinationEngine initialized")
    
    def validate_fire_prediction(self, 
                                primary_pred: float,
                                sensor_data: torch.Tensor) -> ValidationResult:
        """
        Validate fire prediction using multiple methods.
        
        Args:
            primary_pred (float): Primary model prediction (0-100)
            sensor_data (torch.Tensor): Raw sensor data for analysis
            
        Returns:
            ValidationResult: Comprehensive validation result
        """
        # Get ensemble prediction
        ensemble_score, ensemble_votes = self.ensemble_detector.predict(sensor_data.unsqueeze(0))
        
        # Detect cooking patterns
        cooking_result = self.cooking_detector.detect_cooking_patterns(sensor_data)
        
        # Validate fire signatures
        fire_result = self.fire_validator.validate_fire_signatures(sensor_data)
        
        # Decision logic
        is_valid = True
        confidence_adjustment = 1.0
        reasoning_parts = []
        
        # Check for cooking patterns (suppress fire alerts)
        if cooking_result['is_cooking'] and primary_pred > 50:
            confidence_adjustment *= 0.3  # Reduce confidence significantly
            reasoning_parts.append(f"Cooking pattern detected (confidence: {cooking_result['confidence']:.2f})")
            
        # Check ensemble agreement
        agreement_count = ensemble_votes.get('agreement_count', 0)
        if primary_pred > 85 and agreement_count < 2:
            confidence_adjustment *= 0.5
            reasoning_parts.append(f"Insufficient model agreement ({agreement_count}/3 models)")
            
        # Check fire signature completeness
        if primary_pred > 85 and not fire_result['fire_confirmed']:
            confidence_adjustment *= 0.6
            reasoning_parts.append(f"Incomplete fire signatures (completeness: {fire_result['completeness_score']:.2f})")
        
        # Apply conservative thresholds
        adjusted_score = primary_pred * confidence_adjustment
        
        if adjusted_score < 30:
            is_valid = True
            reasoning_parts.append("Low risk - validation passed")
        elif cooking_result['is_cooking'] and adjusted_score < 70:
            is_valid = True
            reasoning_parts.append("Cooking detected - fire alert suppressed")
        elif fire_result['fire_confirmed'] and ensemble_score > 80:
            is_valid = True
            reasoning_parts.append("Fire signatures confirmed by ensemble")
        elif adjusted_score > 85 and not fire_result['fire_confirmed']:
            is_valid = False
            reasoning_parts.append("High risk but insufficient validation")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Standard validation"
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            reasoning=reasoning,
            ensemble_votes=ensemble_votes,
            cooking_detected=cooking_result['is_cooking'],
            fire_signatures_confirmed=fire_result['fire_confirmed']
        )