"""
Test script for the Saafe Fire Detection Dashboard
"""

import sys
import os
import time
import requests

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


def test_dashboard_access():
    """Test that the dashboard is accessible"""
    print("Testing dashboard access...")
    
    try:
        # Try to access the dashboard
        response = requests.get("http://localhost:8505", timeout=5)
        
        if response.status_code == 200:
            print("✅ Dashboard is accessible")
            return True
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Dashboard is not accessible - connection refused")
        return False
    except requests.exceptions.Timeout:
        print("❌ Dashboard request timed out")
        return False
    except Exception as e:
        print(f"❌ Error accessing dashboard: {e}")
        return False


def test_dashboard_components():
    """Test dashboard components"""
    print("Testing dashboard components...")
    
    try:
        # Import the dashboard components
        from dashboard import initialize_system, get_latest_sensor_data
        from dashboard import create_thermal_image_plot, create_gas_readings_plot
        from dashboard import create_environmental_plot, create_risk_gauge
        
        # Initialize system
        config_manager, s3_interface, fusion_engine = initialize_system()
        
        if not s3_interface or not s3_interface.is_connected():
            print("❌ Failed to initialize S3 interface")
            return False
        
        print("✅ Dashboard components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing dashboard components: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Saafe Fire Detection Dashboard - Test Suite")
    print("=" * 50)
    
    # Test dashboard components
    if not test_dashboard_components():
        return 1
    
    print("\n🎉 All tests completed!")
    print("\nTo view the dashboard:")
    print("1. Make sure the dashboard is running: python run_dashboard.py")
    print("2. Open your browser to: http://localhost:8505")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())