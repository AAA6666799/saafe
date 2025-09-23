"""
Test S3 integration for the Synthetic Fire Prediction System
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.core.config import ConfigurationManager
from synthetic_fire_system.hardware.abstraction import S3HardwareInterface
from synthetic_fire_system.feature_engineering.fusion import FeatureFusionEngine


def test_s3_integration():
    """Test S3 integration"""
    print("Testing S3 Integration...")
    print("=" * 50)
    
    try:
        # Initialize S3 hardware interface
        print("1. Initializing S3 Hardware Interface...")
        s3_interface = S3HardwareInterface({
            's3_bucket': 'data-collector-of-first-device',
            'thermal_prefix': 'thermal-data/',
            'gas_prefix': 'gas-data/'
        })
        
        if not s3_interface.is_connected():
            print("   ✗ Failed to connect to S3")
            return False
        print("   ✓ Connected to S3 successfully")
        
        # Test getting sensor data
        print("2. Getting sensor data from S3...")
        sensor_data = s3_interface.get_sensor_data()
        
        if sensor_data is None:
            print("   ✗ Failed to get sensor data from S3")
            return False
        print("   ✓ Got sensor data from S3")
        
        # Display sensor data info
        print(f"   Timestamp: {sensor_data.timestamp}")
        if sensor_data.thermal_frame is not None:
            print(f"   Thermal frame shape: {sensor_data.thermal_frame.shape}")
            print(f"   Thermal frame min/max: {sensor_data.thermal_frame.min():.2f}/{sensor_data.thermal_frame.max():.2f}°C")
        if sensor_data.gas_readings:
            print(f"   Gas readings: {sensor_data.gas_readings}")
        if sensor_data.environmental_data:
            print(f"   Environmental data: {sensor_data.environmental_data}")
        
        # Test feature extraction
        print("3. Testing feature extraction...")
        config_manager = ConfigurationManager()
        fusion_engine = FeatureFusionEngine(config_manager.synthetic_data_config.__dict__)
        
        feature_vector = fusion_engine.extract_features(sensor_data)
        
        if feature_vector is None:
            print("   ✗ Failed to extract features")
            return False
        print("   ✓ Features extracted successfully")
        
        # Display feature info
        if feature_vector.thermal_features is not None:
            print(f"   Thermal features: {feature_vector.thermal_features}")
        if feature_vector.gas_features is not None:
            print(f"   Gas features: {feature_vector.gas_features}")
        if feature_vector.environmental_features is not None:
            print(f"   Environmental features: {feature_vector.environmental_features}")
        if feature_vector.fusion_features is not None:
            print(f"   Fusion features: {feature_vector.fusion_features}")
        
        print("\n✓ S3 integration test passed!")
        return True
        
    except Exception as e:
        print(f"   ✗ S3 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Synthetic Fire Prediction System - S3 Integration Test")
    print("=" * 60)
    
    if test_s3_integration():
        print("\n🎉 S3 integration test completed successfully!")
        return 0
    else:
        print("\n❌ S3 integration test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())