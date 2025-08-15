"""
Fallback mechanisms for graceful degradation when components fail.

This module provides CPU fallback, simplified model loading, cached data patterns,
and offline mode when services are unavailable.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock torch for testing
    class MockTorch:
        class device:
            def __init__(self, device_type='cpu'):
                self.type = device_type
            def __str__(self):
                return self.type
        
        class nn:
            class Module:
                def __init__(self):
                    pass
                def to(self, device):
                    return self
                def cpu(self):
                    return self
        
        class cuda:
            @staticmethod
            def is_available():
                return False
            @staticmethod
            def synchronize():
                pass
            @staticmethod
            def get_device_name(idx):
                return "Mock GPU"
            @staticmethod
            def memory_allocated():
                return 0
            @staticmethod
            def memory_reserved():
                return 0
        
        @staticmethod
        def randn(*args, **kwargs):
            return MockTensor()
        
        @staticmethod
        def zeros(*args, **kwargs):
            return MockTensor()
        
        @staticmethod
        def tensor(data, **kwargs):
            return MockTensor()
    
    class MockTensor:
        def __init__(self):
            pass
        def __matmul__(self, other):
            return MockTensor()
        @property
        def T(self):
            return MockTensor()
    
    torch = MockTorch()

try:
    import numpy as np
except ImportError:
    # Create minimal numpy mock for basic operations
    class MockNumpy:
        class random:
            @staticmethod
            def normal(mean, std, size=None):
                # Simple fallback without numpy
                import random
                return random.gauss(mean, std)
    np = MockNumpy()

from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, safe_execute
from ..core.data_models import SensorReading


@dataclass
class FallbackResult:
    """Result of a fallback operation."""
    success: bool
    fallback_used: str = None
    message: str = None
    performance_impact: float = 0.0
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class FallbackStrategy:
    """Defines a fallback strategy."""
    name: str
    handler: Callable
    priority: int = 0
    description: str = None
    
    def execute(self, context: Dict[str, Any]) -> FallbackResult:
        """Execute the fallback strategy."""
        try:
            return self.handler(context)
        except Exception as e:
            return FallbackResult(
                success=False,
                message=f"Fallback strategy '{self.name}' failed: {str(e)}",
                context=context
            )


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms"""
    enable_cpu_fallback: bool = True
    enable_cached_data: bool = True
    enable_offline_mode: bool = True
    enable_simplified_models: bool = True
    cache_duration_hours: float = 24.0
    max_cache_size_mb: float = 100.0


class CPUFallbackManager:
    """Manages CPU fallback when GPU is unavailable"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        self.gpu_available = torch.cuda.is_available()
        self.current_device = torch.device('cuda' if self.gpu_available else 'cpu')
        self.fallback_active = False
        
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available and functional"""
        if not TORCH_AVAILABLE:
            return False
            
        if not torch.cuda.is_available():
            return False
        
        try:
            # Test GPU functionality
            test_tensor = torch.randn(100, 100, device='cuda')
            _ = test_tensor @ test_tensor.T
            torch.cuda.synchronize()
            return True
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, "CPUFallbackManager",
                context={'operation': 'gpu_test'}
            )
            return False
    
    def switch_to_cpu(self, model) -> Any:
        """Switch model to CPU processing"""
        try:
            if self.current_device.type == 'cuda':
                self.logger.warning("Switching to CPU processing due to GPU issues")
                model = model.cpu()
                self.current_device = torch.device('cpu')
                self.fallback_active = True
                self.logger.info("Successfully switched to CPU processing")
            return model
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, "CPUFallbackManager",
                context={'operation': 'cpu_switch'}
            )
            return model
    
    def get_optimal_device(self):
        """Get the optimal device for processing"""
        if self.fallback_active:
            return torch.device('cpu')
        
        if self.check_gpu_availability():
            return torch.device('cuda')
        else:
            self.fallback_active = True
            return torch.device('cpu')
    
    def get_status(self) -> Dict[str, Any]:
        """Get CPU fallback status"""
        return {
            'gpu_available': self.gpu_available,
            'gpu_functional': self.check_gpu_availability(),
            'current_device': str(self.current_device),
            'fallback_active': self.fallback_active
        }


class SimplifiedModelManager:
    """Manages simplified models when full models fail"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        self.simplified_models = {}
        self.model_cache_dir = Path("models/simplified")
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def create_minimal_model(self, device) -> Any:
        """Create a minimal fire detection model"""
        try:
            if TORCH_AVAILABLE:
                class MinimalFireDetector(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = torch.nn.Linear(4, 16)  # 4 sensor inputs
                        self.fc2 = torch.nn.Linear(16, 8)
                        self.fc3 = torch.nn.Linear(8, 1)   # Risk score output
                        self.relu = torch.nn.ReLU()
                        self.sigmoid = torch.nn.Sigmoid()
                    
                    def forward(self, x):
                        # Flatten input if needed
                        if x.dim() > 2:
                            x = x.view(x.size(0), -1)
                            # Take mean across sequence and sensors
                            x = x.view(x.size(0), -1, 4).mean(dim=1)
                        
                        x = self.relu(self.fc1(x))
                        x = self.relu(self.fc2(x))
                        risk_score = self.sigmoid(self.fc3(x)) * 100  # Scale to 0-100
                        
                        return {
                            'risk_score': risk_score,
                            'logits': torch.zeros(x.size(0), 3),  # Dummy logits
                            'features': torch.zeros(x.size(0), 16)  # Dummy features
                        }
                
                model = MinimalFireDetector().to(device)
            else:
                # Mock model for testing
                class MockMinimalFireDetector:
                    def __init__(self):
                        pass
                    
                    def to(self, device):
                        return self
                    
                    def __call__(self, x):
                        return {
                            'risk_score': torch.tensor([[25.0]]),  # Mock risk score
                            'logits': torch.zeros(1, 3),
                            'features': torch.zeros(1, 16)
                        }
                
                model = MockMinimalFireDetector()
            
            self.logger.info("Created minimal fire detection model")
            return model
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.MODEL_ERROR, "SimplifiedModelManager",
                context={'operation': 'create_minimal_model'}
            )
            return None
    
    def create_rule_based_model(self, device) -> Any:
        """Create a rule-based fire detection model"""
        try:
            if TORCH_AVAILABLE:
                class RuleBasedFireDetector(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        # Thresholds for fire detection
                        self.temp_threshold = 50.0  # Celsius
                        self.pm25_threshold = 100.0  # μg/m³
                        self.co2_threshold = 1000.0  # ppm
                        self.audio_threshold = 60.0  # dB
                    
                    def forward(self, x):
                        batch_size = x.size(0)
                        
                        # Extract sensor values (take mean across sequence and sensors)
                        if x.dim() == 4:  # (batch, seq, sensors, features)
                            sensor_values = x.mean(dim=(1, 2))  # (batch, features)
                        else:
                            sensor_values = x.view(batch_size, -1)[:, :4]
                        
                        temp = sensor_values[:, 0]
                        pm25 = sensor_values[:, 1] 
                        co2 = sensor_values[:, 2]
                        audio = sensor_values[:, 3]
                        
                        # Rule-based risk calculation
                        risk_scores = torch.zeros(batch_size, device=x.device)
                        
                        # Temperature contribution (0-40 points)
                        temp_risk = torch.clamp((temp - 20) / 30 * 40, 0, 40)
                        risk_scores += temp_risk
                        
                        # PM2.5 contribution (0-30 points)
                        pm25_risk = torch.clamp(pm25 / 200 * 30, 0, 30)
                        risk_scores += pm25_risk
                        
                        # CO2 contribution (0-20 points)
                        co2_risk = torch.clamp((co2 - 400) / 600 * 20, 0, 20)
                        risk_scores += co2_risk
                        
                        # Audio contribution (0-10 points)
                        audio_risk = torch.clamp((audio - 40) / 20 * 10, 0, 10)
                        risk_scores += audio_risk
                        
                        # Clamp to 0-100 range
                        risk_scores = torch.clamp(risk_scores, 0, 100)
                        
                        # Create class predictions based on risk score
                        logits = torch.zeros(batch_size, 3, device=x.device)
                        for i in range(batch_size):
                            if risk_scores[i] < 30:
                                logits[i, 0] = 1.0  # Normal
                            elif risk_scores[i] < 70:
                                logits[i, 1] = 1.0  # Cooking
                            else:
                                logits[i, 2] = 1.0  # Fire
                        
                        return {
                            'risk_score': risk_scores.unsqueeze(1),
                            'logits': logits,
                            'features': sensor_values
                        }
                
                model = RuleBasedFireDetector().to(device)
            else:
                # Mock rule-based model for testing
                class MockRuleBasedFireDetector:
                    def __init__(self):
                        pass
                    
                    def to(self, device):
                        return self
                    
                    def __call__(self, x):
                        # Simple rule-based logic for testing
                        # Assume input represents sensor values
                        risk_score = 45.0  # Mock moderate risk
                        return {
                            'risk_score': torch.tensor([[risk_score]]),
                            'logits': torch.tensor([[0.0, 1.0, 0.0]]),  # Cooking class
                            'features': torch.zeros(1, 4)
                        }
                
                model = MockRuleBasedFireDetector()
            
            self.logger.info("Created rule-based fire detection model")
            return model
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.MODEL_ERROR, "SimplifiedModelManager",
                context={'operation': 'create_rule_based_model'}
            )
            return None
    
    def get_fallback_model(self, device, model_type: str = "minimal") -> Any:
        """Get a fallback model"""
        if model_type == "rule_based":
            return self.create_rule_based_model(device)
        else:
            return self.create_minimal_model(device)


class CachedDataManager:
    """Manages cached data patterns when data generation fails"""
    
    def __init__(self, error_handler: ErrorHandler, config: FallbackConfig):
        self.error_handler = error_handler
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cached patterns
        self.cached_patterns = {}
        self.cache_timestamps = {}
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load cached data patterns from disk"""
        cache_file = self.cache_dir / "sensor_patterns.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.cached_patterns = cache_data.get('patterns', {})
                    self.cache_timestamps = cache_data.get('timestamps', {})
                self.logger.info(f"Loaded {len(self.cached_patterns)} cached patterns")
            except Exception as e:
                self.error_handler.handle_error(
                    e, ErrorCategory.DATA_ERROR, "CachedDataManager",
                    context={'operation': 'load_cache'}
                )
    
    def _save_cache(self):
        """Save cached data patterns to disk"""
        try:
            cache_data = {
                'patterns': self.cached_patterns,
                'timestamps': self.cache_timestamps
            }
            cache_file = self.cache_dir / "sensor_patterns.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.DATA_ERROR, "CachedDataManager",
                context={'operation': 'save_cache'}
            )
    
    def cache_pattern(self, pattern_name: str, data: List[SensorReading]):
        """Cache a data pattern"""
        if not self.config.enable_cached_data:
            return
        
        try:
            # Convert to serializable format
            pattern_data = [
                {
                    'timestamp': reading.timestamp.isoformat(),
                    'temperature': reading.temperature,
                    'pm25': reading.pm25,
                    'co2': reading.co2,
                    'audio_level': reading.audio_level,
                    'location': reading.location
                }
                for reading in data
            ]
            
            self.cached_patterns[pattern_name] = pattern_data
            self.cache_timestamps[pattern_name] = datetime.now()
            
            # Save to disk
            self._save_cache()
            
            self.logger.info(f"Cached pattern: {pattern_name} ({len(data)} readings)")
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.DATA_ERROR, "CachedDataManager",
                context={'operation': 'cache_pattern', 'pattern': pattern_name}
            )
    
    def get_cached_pattern(self, pattern_name: str) -> Optional[List[SensorReading]]:
        """Get a cached data pattern"""
        if not self.config.enable_cached_data:
            return None
        
        try:
            if pattern_name not in self.cached_patterns:
                return None
            
            # Check if cache is still valid
            cache_time = self.cache_timestamps.get(pattern_name)
            if cache_time:
                age = datetime.now() - cache_time
                if age.total_seconds() > self.config.cache_duration_hours * 3600:
                    # Cache expired
                    del self.cached_patterns[pattern_name]
                    del self.cache_timestamps[pattern_name]
                    self._save_cache()
                    return None
            
            # Convert back to SensorReading objects
            pattern_data = self.cached_patterns[pattern_name]
            readings = []
            
            for data in pattern_data:
                reading = SensorReading(
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    temperature=data['temperature'],
                    pm25=data['pm25'],
                    co2=data['co2'],
                    audio_level=data['audio_level'],
                    location=data['location']
                )
                readings.append(reading)
            
            self.logger.info(f"Retrieved cached pattern: {pattern_name} ({len(readings)} readings)")
            return readings
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.DATA_ERROR, "CachedDataManager",
                context={'operation': 'get_cached_pattern', 'pattern': pattern_name}
            )
            return None
    
    def generate_fallback_data(self, scenario: str, duration: int = 60) -> List[SensorReading]:
        """Generate fallback data when generators fail"""
        try:
            # Try to get cached pattern first
            cached = self.get_cached_pattern(f"fallback_{scenario}")
            if cached:
                return cached[:duration]  # Return requested duration
            
            # Generate simple fallback patterns
            readings = []
            base_time = datetime.now()
            
            if scenario == "normal":
                base_temp, base_pm25, base_co2, base_audio = 22.0, 15.0, 450.0, 35.0
                temp_var, pm25_var, co2_var, audio_var = 2.0, 5.0, 50.0, 5.0
            elif scenario == "cooking":
                base_temp, base_pm25, base_co2, base_audio = 28.0, 45.0, 600.0, 42.0
                temp_var, pm25_var, co2_var, audio_var = 3.0, 15.0, 100.0, 8.0
            elif scenario == "fire":
                base_temp, base_pm25, base_co2, base_audio = 65.0, 150.0, 1200.0, 65.0
                temp_var, pm25_var, co2_var, audio_var = 10.0, 50.0, 200.0, 15.0
            else:
                # Default to normal
                base_temp, base_pm25, base_co2, base_audio = 22.0, 15.0, 450.0, 35.0
                temp_var, pm25_var, co2_var, audio_var = 2.0, 5.0, 50.0, 5.0
            
            for i in range(duration):
                # Add some random variation
                temp = base_temp + np.random.normal(0, temp_var)
                pm25 = max(0, base_pm25 + np.random.normal(0, pm25_var))
                co2 = max(300, base_co2 + np.random.normal(0, co2_var))
                audio = max(20, base_audio + np.random.normal(0, audio_var))
                
                reading = SensorReading(
                    timestamp=base_time + timedelta(seconds=i),
                    temperature=temp,
                    pm25=pm25,
                    co2=co2,
                    audio_level=audio,
                    location="fallback_sensor"
                )
                readings.append(reading)
            
            # Cache the generated pattern
            self.cache_pattern(f"fallback_{scenario}", readings)
            
            self.logger.info(f"Generated fallback data for {scenario}: {len(readings)} readings")
            return readings
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.DATA_ERROR, "CachedDataManager",
                context={'operation': 'generate_fallback_data', 'scenario': scenario}
            )
            return []
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        try:
            current_time = datetime.now()
            expired_patterns = []
            
            for pattern_name, cache_time in self.cache_timestamps.items():
                age = current_time - cache_time
                if age.total_seconds() > self.config.cache_duration_hours * 3600:
                    expired_patterns.append(pattern_name)
            
            for pattern_name in expired_patterns:
                del self.cached_patterns[pattern_name]
                del self.cache_timestamps[pattern_name]
            
            if expired_patterns:
                self._save_cache()
                self.logger.info(f"Cleaned up {len(expired_patterns)} expired cache entries")
                
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.DATA_ERROR, "CachedDataManager",
                context={'operation': 'cleanup_cache'}
            )


class OfflineModeManager:
    """Manages offline mode when network services are unavailable"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        self.offline_mode = False
        self.offline_notifications = []
        self.offline_data = {}
    
    def enable_offline_mode(self, reason: str = "Network unavailable"):
        """Enable offline mode"""
        if not self.offline_mode:
            self.offline_mode = True
            self.logger.warning(f"Offline mode enabled: {reason}")
    
    def disable_offline_mode(self):
        """Disable offline mode and process queued operations"""
        if self.offline_mode:
            self.offline_mode = False
            self.logger.info("Offline mode disabled")
            
            # Process queued notifications
            if self.offline_notifications:
                self.logger.info(f"Processing {len(self.offline_notifications)} queued notifications")
                # Note: Actual processing would be handled by notification manager
                self.offline_notifications.clear()
    
    def queue_notification(self, notification_data: Dict[str, Any]):
        """Queue notification for when online"""
        if self.offline_mode:
            self.offline_notifications.append({
                'data': notification_data,
                'timestamp': datetime.now()
            })
            self.logger.info("Notification queued for offline processing")
    
    def store_offline_data(self, key: str, data: Any):
        """Store data for offline processing"""
        self.offline_data[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def get_offline_status(self) -> Dict[str, Any]:
        """Get offline mode status"""
        return {
            'offline_mode': self.offline_mode,
            'queued_notifications': len(self.offline_notifications),
            'stored_data_keys': list(self.offline_data.keys())
        }


class FallbackManager:
    """Main fallback manager coordinating all fallback mechanisms"""
    
    def __init__(self, config: FallbackConfig = None, error_handler: ErrorHandler = None):
        self.config = config or FallbackConfig()
        self.error_handler = error_handler or ErrorHandler()
        self.logger = logging.getLogger(__name__)
        
        # Initialize fallback components
        self.cpu_fallback = CPUFallbackManager(self.error_handler)
        self.simplified_models = SimplifiedModelManager(self.error_handler)
        self.cached_data = CachedDataManager(self.error_handler, self.config)
        self.offline_mode = OfflineModeManager(self.error_handler)
        
        self.logger.info("FallbackManager initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive fallback system status"""
        return {
            'cpu_fallback': self.cpu_fallback.get_status(),
            'cached_data_enabled': self.config.enable_cached_data,
            'offline_mode': self.offline_mode.get_offline_status(),
            'simplified_models_enabled': self.config.enable_simplified_models,
            'config': {
                'cache_duration_hours': self.config.cache_duration_hours,
                'max_cache_size_mb': self.config.max_cache_size_mb
            }
        }
    
    def handle_model_failure(self, device) -> Any:
        """Handle model loading failure with fallbacks"""
        self.logger.warning("Model failure detected, attempting fallback")
        
        # Try CPU fallback first
        if self.config.enable_cpu_fallback and str(device) == 'cuda':
            device = torch.device('cpu')
            self.cpu_fallback.switch_to_cpu(None)  # Signal CPU switch
        
        # Try simplified model
        if self.config.enable_simplified_models:
            model = self.simplified_models.get_fallback_model(device, "rule_based")
            if model is not None:
                self.logger.info("Using rule-based fallback model")
                return model
            
            model = self.simplified_models.get_fallback_model(device, "minimal")
            if model is not None:
                self.logger.info("Using minimal fallback model")
                return model
        
        self.logger.error("All model fallback mechanisms failed")
        return None
    
    def handle_data_failure(self, scenario: str) -> List[SensorReading]:
        """Handle data generation failure with cached patterns"""
        self.logger.warning(f"Data generation failure for {scenario}, using fallback")
        
        if self.config.enable_cached_data:
            return self.cached_data.generate_fallback_data(scenario)
        
        return []
    
    def handle_network_failure(self):
        """Handle network failure by enabling offline mode"""
        self.offline_mode.enable_offline_mode("Network failure detected")
    
    def cleanup(self):
        """Cleanup fallback resources"""
        if self.config.enable_cached_data:
            self.cached_data.cleanup_cache()
        
        self.logger.info("Fallback cleanup completed")


# Global fallback manager instance
_global_fallback_manager: Optional[FallbackManager] = None


def get_fallback_manager() -> FallbackManager:
    """Get global fallback manager instance"""
    global _global_fallback_manager
    if _global_fallback_manager is None:
        _global_fallback_manager = FallbackManager()
    return _global_fallback_manager