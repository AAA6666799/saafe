#!/usr/bin/env python3
"""
Test script to verify that the data retrieval function works correctly
"""

import sys
import os

# Add the current directory to the path so we can import the dashboard functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_retrieval():
    """Test the data retrieval function"""
    print("🔍 Testing data retrieval function...")
    print("=" * 50)
    
    try:
        # Import the function from the dashboard
        from saafe_aws_dashboard import get_recent_sensor_data
        
        # Test the function
        print("Calling get_recent_sensor_data()...")
        sensor_data = get_recent_sensor_data()
        
        if sensor_data:
            print("✅ Function executed successfully")
            print(f"Thermal data entries: {len(sensor_data.get('thermal', []))}")
            print(f"Gas data entries: {len(sensor_data.get('gas', []))}")
            
            # Show details of the latest data
            latest_thermal = sensor_data.get('latest_thermal')
            latest_gas = sensor_data.get('latest_gas')
            
            if latest_thermal:
                print("\nLatest thermal data:")
                for key, value in list(latest_thermal.items())[:5]:  # Show first 5 items
                    print(f"  {key}: {value}")
                    
            if latest_gas:
                print("\nLatest gas data:")
                for key, value in list(latest_gas.items())[:5]:  # Show first 5 items
                    print(f"  {key}: {value}")
        else:
            print("❌ Function returned None")
            
    except Exception as e:
        print(f"❌ Error testing data retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_retrieval()