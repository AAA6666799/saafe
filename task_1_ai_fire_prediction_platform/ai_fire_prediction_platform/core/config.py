"""
Configuration Management System

Provides centralized configuration management for all system components
with validation, environment-specific settings, and runtime updates.
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class ConfigFormat(Enum):
    """Supported configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    YML = "yml"


@dataclass
class SystemConfig:
    """Main system configuration"""
    # System settings
    system_name: str = "synthetic_fire_prediction_system"
    version: str = "0.1.0"
    environment: str = "development"  # development, testing, production
    
    # Performance settings
    max_processing_latency_ms: int = 100
    max_prediction_latency_ms: int = 500
    batch_size: int = 32
    num_workers: int = 4
    
    # Data settings
    thermal_image_resolution: tuple = (384, 288)
    feature_vector_size: int = 20
    sequence_length: int = 10
    
    # Model settings
    model_accuracy_threshold: float = 0.90
    false_positive_threshold: float = 0.05
    false_negative_threshold: float = 0.01
    confidence_threshold: float = 0.8
    
    # Agent settings
    agent_communication_timeout: int = 5
    agent_retry_attempts: int = 3
    
    # Hardware settings
    sensor_health_check_interval: int = 30
    calibration_check_interval: int = 3600


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    # Thermal data settings
    thermal_noise_level: float = 0.1
    hotspot_min_size: int = 5
    hotspot_max_size: int = 50
    temperature_range: tuple = (15.0, 100.0)
    
    # Gas data settings
    gas_types: list = None
    concentration_range: tuple = (0.0, 1000.0)
    diffusion_rate: float = 0.1
    sensor_drift_rate: float = 0.01
    
    # Environmental data settings
    temperature_range_env: tuple = (10.0, 40.0)
    humidity_range: tuple = (20.0, 80.0)
    pressure_range: tuple = (980.0, 1020.0)
    voc_range: tuple = (0.0, 500.0)
    
    # Scenario settings
    normal_scenario_hours: int = 1000
    fire_scenarios_per_type: int = 100
    false_positive_scenarios: int = 200
    
    def __post_init__(self):
        if self.gas_types is None:
            self.gas_types = ["methane", "propane", "hydrogen", "co", "co2"]


@dataclass
class ModelConfig:
    """Configuration for machine learning models"""
    # Model types to use
    use_temporal_models: bool = True
    use_baseline_models: bool = True
    use_ensemble: bool = True
    
    # Temporal model settings
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Baseline model settings
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2


@dataclass
class AgentConfig:
    """Configuration for agent system"""
    # Agent types to enable
    enable_monitoring_agent: bool = True
    enable_analysis_agent: bool = True
    enable_response_agent: bool = True
    enable_learning_agent: bool = True
    
    # Monitoring agent settings
    anomaly_detection_threshold: float = 2.0
    sensor_health_threshold: float = 0.8
    attention_priority_weights: dict = None
    
    # Analysis agent settings
    historical_correlation_window: int = 3600
    confidence_calculation_method: str = "ensemble_agreement"
    risk_assessment_weights: dict = None
    
    # Response agent settings
    alert_distribution_channels: list = None
    escalation_thresholds: dict = None
    response_timeout: int = 300
    
    # Learning agent settings
    performance_tracking_window: int = 86400
    retraining_threshold: float = 0.05
    error_analysis_window: int = 3600
    
    def __post_init__(self):
        if self.attention_priority_weights is None:
            self.attention_priority_weights = {
                "thermal_anomaly": 0.4,
                "gas_anomaly": 0.3,
                "environmental_anomaly": 0.2,
                "sensor_health": 0.1
            }
        
        if self.risk_assessment_weights is None:
            self.risk_assessment_weights = {
                "fire_probability": 0.5,
                "confidence_score": 0.3,
                "sensor_agreement": 0.2
            }
        
        if self.alert_distribution_channels is None:
            self.alert_distribution_channels = ["console", "log", "email"]
        
        if self.escalation_thresholds is None:
            self.escalation_thresholds = {
                "LOW": 0.3,
                "MEDIUM": 0.6,
                "HIGH": 0.8,
                "CRITICAL": 0.95
            }


class ConfigurationManager:
    """Central configuration management system"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize default configurations
        self.system_config = SystemConfig()
        self.synthetic_data_config = SyntheticDataConfig()
        self.model_config = ModelConfig()
        self.agent_config = AgentConfig()
        
        # Component-specific configurations
        self._component_configs: Dict[str, Dict[str, Any]] = {}
        
        # Load configurations if they exist
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Load configurations from files"""
        config_files = {
            "system": self.system_config,
            "synthetic_data": self.synthetic_data_config,
            "model": self.model_config,
            "agent": self.agent_config
        }
        
        for config_name, config_obj in config_files.items():
            config_path = self._find_config_file(config_name)
            if config_path:
                try:
                    data = self._load_config_file(config_path)
                    # Update config object with loaded data
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
                except Exception as e:
                    print(f"Warning: Failed to load {config_name} config: {e}")
    
    def _find_config_file(self, config_name: str) -> Optional[Path]:
        """Find configuration file with supported extensions"""
        for ext in ["json", "yaml", "yml"]:
            config_path = self.config_dir / f"{config_name}.{ext}"
            if config_path.exists():
                return config_path
        return None
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                return json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def save_configurations(self, format: ConfigFormat = ConfigFormat.JSON) -> None:
        """Save all configurations to files"""
        configs = {
            "system": self.system_config,
            "synthetic_data": self.synthetic_data_config,
            "model": self.model_config,
            "agent": self.agent_config
        }
        
        for config_name, config_obj in configs.items():
            self._save_config_file(config_name, asdict(config_obj), format)
    
    def _save_config_file(self, config_name: str, data: Dict[str, Any], 
                         format: ConfigFormat) -> None:
        """Save configuration to file"""
        config_path = self.config_dir / f"{config_name}.{format.value}"
        
        with open(config_path, 'w') as f:
            if format == ConfigFormat.JSON:
                json.dump(data, f, indent=2, default=str)
            elif format in [ConfigFormat.YAML, ConfigFormat.YML]:
                yaml.dump(data, f, default_flow_style=False)
    
    def get_component_config(self, component_id: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        return self._component_configs.get(component_id, {})
    
    def set_component_config(self, component_id: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific component"""
        self._component_configs[component_id] = config
    
    def update_config(self, config_section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section at runtime"""
        config_map = {
            "system": self.system_config,
            "synthetic_data": self.synthetic_data_config,
            "model": self.model_config,
            "agent": self.agent_config
        }
        
        if config_section in config_map:
            config_obj = config_map[config_section]
            for key, value in updates.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                else:
                    print(f"Warning: Unknown config key '{key}' in section '{config_section}'")
        else:
            print(f"Warning: Unknown config section '{config_section}'")
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate all configurations and return any errors"""
        errors = {}
        
        # Validate system config
        system_errors = []
        if self.system_config.max_processing_latency_ms <= 0:
            system_errors.append("max_processing_latency_ms must be positive")
        if self.system_config.model_accuracy_threshold < 0 or self.system_config.model_accuracy_threshold > 1:
            system_errors.append("model_accuracy_threshold must be between 0 and 1")
        if system_errors:
            errors["system"] = system_errors
        
        # Validate synthetic data config
        data_errors = []
        if self.synthetic_data_config.thermal_noise_level < 0:
            data_errors.append("thermal_noise_level must be non-negative")
        if len(self.synthetic_data_config.gas_types) == 0:
            data_errors.append("gas_types cannot be empty")
        if data_errors:
            errors["synthetic_data"] = data_errors
        
        # Validate model config
        model_errors = []
        if self.model_config.learning_rate <= 0:
            model_errors.append("learning_rate must be positive")
        if self.model_config.validation_split < 0 or self.model_config.validation_split >= 1:
            model_errors.append("validation_split must be between 0 and 1")
        if model_errors:
            errors["model"] = model_errors
        
        return errors
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides"""
        env = self.system_config.environment
        env_config_path = self.config_dir / f"environment_{env}.json"
        
        if env_config_path.exists():
            return self._load_config_file(env_config_path)
        return {}


# Global configuration manager instance
config_manager = ConfigurationManager()