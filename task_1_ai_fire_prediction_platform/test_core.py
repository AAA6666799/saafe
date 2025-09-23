"""
Unit tests for core components
"""

import pytest
import numpy as np
from synthetic_fire_system.core.interfaces import SensorData, FeatureVector, PredictionResult, RiskAssessment
from synthetic_fire_system.core.config import ConfigurationManager, SystemConfig
from synthetic_fire_system.core.utils import (
    validate_array_shape, validate_probability, validate_positive,
    safe_divide, normalize_array, calculate_z_score
)
from synthetic_fire_system.core.exceptions import ValidationError


class TestDataStructures:
    """Test core data structures"""
    
    def test_sensor_data_creation(self):
        """Test SensorData creation"""
        thermal_frame = np.random.rand(384, 288)
        gas_readings = {"methane": 10.5, "co": 2.1}
        env_data = {"temperature": 25.0, "humidity": 60.0}
        
        sensor_data = SensorData(
            timestamp=1234567890.0,
            thermal_frame=thermal_frame,
            gas_readings=gas_readings,
            environmental_data=env_data
        )
        
        assert sensor_data.timestamp == 1234567890.0
        assert sensor_data.thermal_frame.shape == (384, 288)
        assert sensor_data.gas_readings["methane"] == 10.5
        assert sensor_data.environmental_data["temperature"] == 25.0
    
    def test_feature_vector_creation(self):
        """Test FeatureVector creation"""
        feature_vector = FeatureVector(
            timestamp=1234567890.0,
            thermal_features=np.array([1.0, 2.0, 3.0]),
            gas_features=np.array([4.0, 5.0]),
            environmental_features=np.array([6.0, 7.0]),
            fusion_features=np.array([8.0]),
            feature_quality=0.95
        )
        
        assert feature_vector.timestamp == 1234567890.0
        assert len(feature_vector.thermal_features) == 3
        assert feature_vector.feature_quality == 0.95
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation"""
        prediction = PredictionResult(
            timestamp=1234567890.0,
            fire_probability=0.85,
            confidence_score=0.92,
            lead_time_estimate=45.0,
            contributing_factors={"thermal": 0.6, "gas": 0.4},
            model_ensemble_votes={"lstm": 0.9, "rf": 0.8}
        )
        
        assert prediction.fire_probability == 0.85
        assert prediction.confidence_score == 0.92
        assert prediction.contributing_factors["thermal"] == 0.6


class TestConfigurationManager:
    """Test configuration management"""
    
    def test_system_config_defaults(self):
        """Test default system configuration"""
        config = SystemConfig()
        
        assert config.system_name == "synthetic_fire_prediction_system"
        assert config.version == "0.1.0"
        assert config.max_processing_latency_ms == 100
        assert config.model_accuracy_threshold == 0.90
        assert config.thermal_image_resolution == (384, 288)
    
    def test_configuration_manager_initialization(self):
        """Test configuration manager initialization"""
        config_manager = ConfigurationManager()
        
        assert config_manager.system_config is not None
        assert config_manager.synthetic_data_config is not None
        assert config_manager.model_config is not None
        assert config_manager.agent_config is not None
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config_manager = ConfigurationManager()
        
        # Test valid configuration
        errors = config_manager.validate_configuration()
        assert isinstance(errors, dict)
        
        # Test invalid configuration
        config_manager.system_config.max_processing_latency_ms = -1
        errors = config_manager.validate_configuration()
        assert "system" in errors
        assert len(errors["system"]) > 0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_validate_array_shape(self):
        """Test array shape validation"""
        array = np.random.rand(10, 5)
        
        # Should not raise exception
        validate_array_shape(array, (10, 5))
        
        # Should raise exception
        with pytest.raises(ValueError):
            validate_array_shape(array, (5, 10))
    
    def test_validate_probability(self):
        """Test probability validation"""
        # Valid probabilities
        validate_probability(0.0)
        validate_probability(0.5)
        validate_probability(1.0)
        
        # Invalid probabilities
        with pytest.raises(ValueError):
            validate_probability(-0.1)
        
        with pytest.raises(ValueError):
            validate_probability(1.1)
    
    def test_validate_positive(self):
        """Test positive value validation"""
        # Valid positive values
        validate_positive(0.1)
        validate_positive(1.0)
        validate_positive(100.0)
        
        # Invalid values
        with pytest.raises(ValueError):
            validate_positive(0.0)
        
        with pytest.raises(ValueError):
            validate_positive(-1.0)
    
    def test_safe_divide(self):
        """Test safe division"""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=1.0) == 1.0
    
    def test_normalize_array(self):
        """Test array normalization"""
        array = np.array([1, 2, 3, 4, 5])
        normalized = normalize_array(array, 0.0, 1.0)
        
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        
        # Test with custom range
        normalized_custom = normalize_array(array, -1.0, 1.0)
        assert np.min(normalized_custom) == -1.0
        assert np.max(normalized_custom) == 1.0
    
    def test_calculate_z_score(self):
        """Test z-score calculation"""
        assert calculate_z_score(10, 5, 2) == 2.5
        assert calculate_z_score(5, 5, 2) == 0.0
        assert calculate_z_score(5, 5, 0) == 0.0  # Handle zero std