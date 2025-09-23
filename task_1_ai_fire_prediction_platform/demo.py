"""
Demo script for the Synthetic Fire Prediction System
"""

import sys
import os
import time

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.core.config import ConfigurationManager
from synthetic_fire_system.system.manager import SystemManager


def demo_system():
    """Demonstrate the synthetic fire prediction system with real S3 data"""
    print("Synthetic Fire Prediction System - Real S3 Data Demo")
    print("=" * 55)
    
    # Initialize configuration manager
    config_manager = ConfigurationManager()
    
    # Initialize system manager
    print("Initializing system with S3 data integration...")
    system_manager = SystemManager(config_manager)
    
    # Start system
    print("Starting system...")
    if system_manager.start():
        print("✅ System started successfully")
        print("\nSystem is now running and monitoring live S3 data...")
        print("Press Ctrl+C to stop the demo\n")
        
        # Run for 15 seconds to show some predictions
        start_time = time.time()
        try:
            while time.time() - start_time < 15:
                # Get current status
                status = system_manager.get_status()
                
                # Get latest prediction
                prediction = system_manager.get_latest_prediction()
                risk_assessment = system_manager.get_current_risk_assessment()
                
                if prediction and risk_assessment:
                    print(f"⏱️  Time: {prediction.timestamp:.0f}s | "
                          f"🔥 Fire Probability: {prediction.fire_probability:.3f} | "
                          f"📊 Confidence: {prediction.confidence_score:.3f} | "
                          f"⚠️  Risk Level: {risk_assessment.risk_level}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Demo interrupted by user")
        
        # Stop system
        print("\nStopping system...")
        system_manager.stop()
        print("✅ System stopped successfully")
        
        return True
    else:
        print("❌ Failed to start system")
        return False


def demo_s3_data():
    """Demonstrate S3 data processing"""
    print("\n\nSynthetic Fire Prediction System - S3 Data Processing Demo")
    print("=" * 60)
    
    try:
        # Import S3 hardware interface
        from synthetic_fire_system.hardware.abstraction import S3HardwareInterface
        
        print("Initializing S3 hardware interface...")
        s3_interface = S3HardwareInterface({
            's3_bucket': 'data-collector-of-first-device',
            'thermal_prefix': 'thermal-data/',
            'gas_prefix': 'gas-data/'
        })
        
        if not s3_interface.is_connected():
            print("❌ Failed to connect to S3")
            return False
        
        print("✅ Connected to S3 successfully")
        print("\nFetching live data from S3...")
        
        # Get several samples of data
        for i in range(5):
            sensor_data = s3_interface.get_sensor_data()
            if sensor_data:
                print(f"\nSample {i+1}:")
                print(f"  Timestamp: {sensor_data.timestamp:.0f}")
                if sensor_data.thermal_frame is not None:
                    print(f"  Thermal: {sensor_data.thermal_frame.shape} "
                          f"(min: {sensor_data.thermal_frame.min():.1f}°C, "
                          f"max: {sensor_data.thermal_frame.max():.1f}°C, "
                          f"mean: {sensor_data.thermal_frame.mean():.1f}°C)")
                if sensor_data.gas_readings:
                    print(f"  Gas readings: {sensor_data.gas_readings}")
            else:
                print(f"\nSample {i+1}: No data available")
            
            time.sleep(2)  # Wait 2 seconds between samples
        
        print("\n✅ S3 data processing demo completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ S3 data processing demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demo function"""
    print("🔥 Synthetic Fire Prediction System - Real S3 Data Demo 🔥")
    print("=" * 65)
    
    # Run S3 data demo
    if not demo_s3_data():
        return 1
    
    # Run system demo
    if not demo_system():
        return 1
    
    print("\n🎉 All demos completed successfully!")
    print("\nTo run the system continuously:")
    print("  python -m synthetic_fire_system.main")
    print("\nTo run tests:")
    print("  python test_system.py")
    print("  python test_s3_integration.py")
    print("\nTo train the system:")
    print("  python train_system.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())