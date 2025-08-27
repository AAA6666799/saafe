#!/usr/bin/env python3
"""
Test script to validate the synthetic fire prediction system components.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_thermal_generation():
    """Test thermal image generation"""
    print("🔥 Testing thermal image generation...")
    
    try:
        from data_generation.thermal.thermal_image_generator import ThermalImageGenerator
        
        config = {
            'resolution': (288, 384),
            'min_temperature': 20.0,
            'max_temperature': 500.0,
            'output_formats': ['numpy'],
            'hotspot_config': {},
            'temporal_config': {},
            'noise_config': {
                'noise_types': ['gaussian', 'salt_and_pepper'],
                'noise_level': 0.1,
                'seed': 42,
                'noise_params': {
                    'gaussian': {'mean': 0, 'std': 0.1},
                    'salt_and_pepper': {'amount': 0.01, 'salt_vs_pepper': 0.5}
                }
            }
        }
        
        generator = ThermalImageGenerator(config)
        print("✓ Thermal generator initialized")
        
        # Generate a single frame
        frame = generator.generate_frame(datetime.now())
        print(f"✓ Generated thermal frame: {frame.shape}")
        
        # Generate a sequence
        sequence = generator.generate(
            timestamp=datetime.now(),
            duration_seconds=10,
            sample_rate_hz=1.0,
            seed=42
        )
        print(f"✓ Generated sequence with {len(sequence['frames'])} frames")
        
        return True
        
    except Exception as e:
        print(f"✗ Thermal generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gas_generation():
    """Test gas concentration generation"""
    print("\n💨 Testing gas concentration generation...")
    
    try:
        from data_generation.gas.gas_concentration_generator import GasConcentrationGenerator
        
        config = {
            'gas_types': ['methane', 'propane', 'hydrogen'],
            'diffusion_config': {},
            'temporal_config': {},
            'sensor_configs': {}
        }
        
        generator = GasConcentrationGenerator(config)
        print("✓ Gas generator initialized")
        
        # Generate gas data
        gas_data = generator.generate(
            timestamp=datetime.now(),
            duration_seconds=60,
            sample_rate_hz=0.1,
            seed=42  # Valid seed range
        )
        print(f"✓ Generated gas data for {len(gas_data['gas_data'])} gas types")
        
        return True
        
    except Exception as e:
        print(f"✗ Gas generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test feature extraction"""
    print("\n📊 Testing feature extraction...")
    
    try:
        from feature_engineering.feature_extractor_registry import registry
        from feature_engineering.extractors.gas.gas_concentration_extractor import GasConcentrationExtractor
        
        print("✓ Feature registry imported")
        
        # Test registry
        config = {
            'gas_column': 'concentration', 
            'threshold': 70.0,
            'gas_types': ['methane', 'propane'],
            'window_sizes': [5, 10],
            'baseline_window': 20
        }
        extractor = GasConcentrationExtractor(config)
        print("✓ Gas concentration extractor created")
        
        # Test with dummy data
        data = pd.DataFrame({
            'concentration': [10, 20, 30, 40, 50],
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='1H')
        })
        
        features = extractor.extract_features(data)
        print(f"✓ Extracted {len(features)} feature groups")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_models():
    """Test ML model interfaces"""
    print("\n🤖 Testing ML model interfaces...")
    
    try:
        from ml.base import FireModel, FireClassificationModel
        print("✓ ML base classes imported")
        
        from ml.models.classification import BinaryFireClassifier
        print("✓ Binary fire classifier imported")
        
        return True
        
    except Exception as e:
        print(f"✗ ML model error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_integration():
    """Test system integration"""
    print("\n🏗️ Testing system integration...")
    
    try:
        from system import SystemManager
        
        # Create system manager with minimal config
        system = SystemManager()
        print("✓ System manager created")
        
        print(f"✓ System initialized: {system.is_initialized}")
        
        return True
        
    except Exception as e:
        print(f"✗ System integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Saafe Fire Detection System Tests")
    print("=" * 60)
    
    tests = [
        test_thermal_generation,
        test_gas_generation, 
        test_feature_extraction,
        test_ml_models,
        test_system_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for development.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)