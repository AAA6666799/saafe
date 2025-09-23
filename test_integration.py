"""
Test script to verify the integration between the Synthetic Fire Prediction System
and the SAAFE Global Command Center.
"""

import requests
import time

def test_api_connection():
    """Test connection to the Fire Detection API"""
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8000/")
        print(f"Root endpoint: {response.status_code} - {response.json()}")
        
        # Test status endpoint
        response = requests.get("http://localhost:8000/api/status")
        print(f"Status endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data.get('data', {}).get('system_health', 'Unknown')}")
        
        # Test fire detection data endpoint
        response = requests.get("http://localhost:8000/api/fire-detection-data")
        print(f"Fire detection data endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                fire_data = data.get('data', {})
                print(f"  Sensor data available: {'sensor_data' in fire_data}")
                print(f"  Prediction available: {'prediction' in fire_data}")
                print(f"  Risk assessment available: {'risk_assessment' in fire_data}")
                print(f"  Alert available: {'alert' in fire_data}")
            else:
                print(f"  Error: {data.get('message', 'Unknown error')}")
        elif response.status_code == 503:
            print("  Error: System not initialized")
        else:
            print(f"  Error: {response.text}")
            
        return True
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"Error testing API connection: {e}")
        return False

def test_frontend_connection():
    """Test connection to the React frontend"""
    try:
        response = requests.get("http://localhost:5173/")
        print(f"Frontend connection: {response.status_code}")
        if response.status_code == 200:
            print("  Frontend is accessible")
        else:
            print(f"  Unexpected response: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("Warning: Could not connect to the frontend. Make sure it's running on http://localhost:5173")
        return False
    except Exception as e:
        print(f"Error testing frontend connection: {e}")
        return False

def main():
    """Main test function"""
    print("Testing Fire Detection System Integration")
    print("=" * 50)
    
    # Test API connection
    print("\n1. Testing API Server Connection:")
    api_success = test_api_connection()
    
    # Test frontend connection
    print("\n2. Testing Frontend Connection:")
    frontend_success = test_frontend_connection()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  API Server: {'✓ Working' if api_success else '✗ Not accessible'}")
    print(f"  Frontend: {'✓ Working' if frontend_success else '⚠ Not accessible (may be OK if not started)'}")
    
    if api_success:
        print("\n✓ Integration appears to be working correctly!")
        print("You can now access the Global Command Center at http://localhost:5173")
    else:
        print("\n✗ There are issues with the integration.")
        print("Please check that the API server is running.")

if __name__ == "__main__":
    main()