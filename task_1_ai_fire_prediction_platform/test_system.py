"""
Integration test for the Synthetic Fire Prediction System
"""

import sys
import os
import time
import numpy as np

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.core.config import ConfigurationManager
from synthetic_fire_system.system.manager import SystemManager
from synthetic_fire_system.hardware.mock_interface import MockHardwareInterface
from synthetic_fire_system.feature_engineering.fusion import FeatureFusionEngine
from synthetic_fire_system.models.ensemble import EnsembleModel


def test_system_components():
    """Test individual system components"""
    print("Testing Synthetic Fire Prediction System components...")
    print("=" * 60)
    
    # Test 1: Configuration Manager
    print("1. Testing Configuration Manager...")
    try:
        config_manager = ConfigurationManager()
        errors = config_manager.validate_configuration()
        assert len(errors) == 0, f"Configuration validation failed: {errors}"
        print("   ✓ Configuration Manager working correctly")
    except Exception as e:
        print(f"   ✗ Configuration Manager failed: {e}")
        return False
    
    # Test 2: Mock Hardware Interface
    print("2. Testing Mock Hardware Interface...")
    try:
        hardware = MockHardwareInterface()
        sensor_data = hardware.get_sensor_data()
        assert sensor_data is not None, "Failed to get sensor data"
        assert sensor_data.thermal_frame is not None, "Thermal frame missing"
        assert sensor_data.gas_readings is not None, "Gas readings missing"
        assert sensor_data.environmental_data is not None, "Environmental data missing"
        print("   ✓ Mock Hardware Interface working correctly")
    except Exception as e:
        print(f"   ✗ Mock Hardware Interface failed: {e}")
        return False
    
    # Test 3: Feature Fusion Engine
    print("3. Testing Feature Fusion Engine...")
    try:
        fusion_engine = FeatureFusionEngine(config_manager.synthetic_data_config.__dict__)
        feature_vector = fusion_engine.extract_features(sensor_data)
        assert feature_vector is not None, "Failed to extract features"
        assert feature_vector.thermal_features is not None, "Thermal features missing"
        assert feature_vector.gas_features is not None, "Gas features missing"
        assert feature_vector.environmental_features is not None, "Environmental features missing"
        assert feature_vector.fusion_features is not None, "Fusion features missing"
        print("   ✓ Feature Fusion Engine working correctly")
    except Exception as e:
        print(f"   ✗ Feature Fusion Engine failed: {e}")
        return False
    
    # Test 4: Ensemble Model
    print("4. Testing Ensemble Model...")
    try:
        model = EnsembleModel(config_manager.model_config.__dict__)
        
        # Create mock training data
        num_samples = 100
        feature_size = len(feature_vector.thermal_features) + \
                      len(feature_vector.gas_features) + \
                      len(feature_vector.environmental_features) + \
                      len(feature_vector.fusion_features)
        
        training_features = np.random.rand(num_samples, feature_size)
        training_labels = np.random.randint(0, 2, num_samples)
        
        # Train model
        model.train(training_features, training_labels)
        
        # Test prediction
        test_features = np.random.rand(feature_size)
        probability, confidence = model.predict(test_features)
        assert 0.0 <= probability <= 1.0, "Invalid probability value"
        assert 0.0 <= confidence <= 1.0, "Invalid confidence value"
        print("   ✓ Ensemble Model working correctly")
    except Exception as e:
        print(f"   ✗ Ensemble Model failed: {e}")
        return False
    
    print("\n✓ All system components working correctly!")
    return True


def test_system_manager():
    """Test the system manager"""
    print("\nTesting System Manager...")
    print("=" * 60)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigurationManager()
        
        # Initialize system manager
        system_manager = SystemManager(config_manager)
        
        # Check status
        status = system_manager.get_status()
        print(f"   Initial status: {status}")
        
        # Start system
        print("   Starting system...")
        if system_manager.start():
            print("   ✓ System started successfully")
            
            # Let it run for a few seconds
            time.sleep(3)
            
            # Check status again
            status = system_manager.get_status()
            print(f"   Status after running: {status}")
            
            # Get latest prediction
            prediction = system_manager.get_latest_prediction()
            if prediction:
                print(f"   Latest prediction: Fire probability = {prediction.fire_probability:.3f}, "
                      f"Confidence = {prediction.confidence_score:.3f}")
            
            # Get current risk assessment
            risk_assessment = system_manager.get_current_risk_assessment()
            if risk_assessment:
                print(f"   Current risk assessment: Level = {risk_assessment.risk_level}, "
                      f"Probability = {risk_assessment.fire_probability:.3f}")
            
            # Stop system
            print("   Stopping system...")
            system_manager.stop()
            print("   ✓ System stopped successfully")
            
            return True
        else:
            print("   ✗ Failed to start system")
            return False
            
    except Exception as e:
        print(f"   ✗ System Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Synthetic Fire Prediction System - Integration Tests")
    print("=" * 60)
    
    # Run component tests
    if not test_system_components():
        print("\n❌ Component tests failed!")
        return 1
    
    # Run system manager test
    if not test_system_manager():
        print("\n❌ System manager test failed!")
        return 1
    
    print("\n🎉 All tests passed! The system is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())