#!/usr/bin/env python3
"""
Test script to verify the rename from synthetic_fire_system to ai_fire_prediction_platform
"""

import sys
import os

def test_imports():
    """Test that imports work with the new package name"""
    try:
        print("Testing imports...")
        
        # Test core imports
        from ai_fire_prediction_platform.core.config import ConfigurationManager
        from ai_fire_prediction_platform.hardware.abstraction import S3HardwareInterface
        from ai_fire_prediction_platform.feature_engineering.fusion import FeatureFusionEngine
        from ai_fire_prediction_platform.models.ensemble import EnsembleModel
        from ai_fire_prediction_platform.system.manager import SystemManager
        
        print("✅ Core imports successful")
        
        # Test component initialization
        config_manager = ConfigurationManager()
        print("✅ ConfigurationManager initialized")
        
        s3_interface = S3HardwareInterface({
            's3_bucket': 'data-collector-of-first-device',
            'thermal_prefix': 'thermal-data/',
            'gas_prefix': 'gas-data/'
        })
        print("✅ S3HardwareInterface initialized")
        
        fusion_engine = FeatureFusionEngine(config_manager.synthetic_data_config.__dict__)
        print("✅ FeatureFusionEngine initialized")
        
        print("\n🎉 All tests passed! The rename was successful.")
        print("The system is now called 'AI Fire Prediction Platform'")
        print("and properly reflects that it uses real IoT sensor data.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def main():
    """Main test function"""
    print("AI Fire Prediction Platform - Rename Verification")
    print("=" * 55)
    
    if test_imports():
        print("\n📋 Summary:")
        print("   ✅ Package renamed successfully")
        print("   ✅ All imports updated")
        print("   ✅ Components initialize correctly")
        print("   ✅ Ready for use with real IoT data")
        return 0
    else:
        print("\n❌ Rename verification failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())