#!/usr/bin/env python3
"""
Test script to verify dashboard access on the correct port
"""

import requests
import time

def test_dashboard_access():
    """Test if the dashboard is accessible on port 8505"""
    print("🔍 Testing dashboard access on port 8505...")
    
    try:
        # Test port 8505 (correct port)
        response = requests.get("http://localhost:8505", timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard is accessible on port 8505")
            print("🔗 URL: http://localhost:8505")
            return True
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to dashboard on port 8505")
        return False
    except Exception as e:
        print(f"❌ Error testing dashboard access: {e}")
        return False

def test_incorrect_port():
    """Test that port 8501 is not being used for the main dashboard"""
    print("🔍 Verifying dashboard is NOT on port 8501...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        # If we get a response, it means something is running on 8501
        print("ℹ️  Port 8501 has a service running (this might be a different dashboard or service)")
    except:
        print("✅ Port 8501 is free (as expected)")

def main():
    """Main test function"""
    print("🧪 Dashboard Access Test")
    print("=" * 30)
    
    # Test dashboard access
    dashboard_ok = test_dashboard_access()
    
    # Test incorrect port
    test_incorrect_port()
    
    print("\n" + "=" * 30)
    if dashboard_ok:
        print("🎉 Dashboard is running correctly!")
        print("\n📋 Access Information:")
        print("   🔗 URL: http://localhost:8505")
        print("   🕐 Refresh: Dashboard auto-refreshes every 2 seconds")
        print("   📊 Data: Shows real-time sensor readings when available")
        print("\n💡 Next Steps:")
        print("   1. Open your browser and go to http://localhost:8505")
        print("   2. Select a scenario to begin monitoring")
        print("   3. The dashboard will show 'No recent live data' until devices send data to S3")
    else:
        print("❌ Dashboard access test failed!")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Check if the dashboard process is running:")
        print("      ps aux | grep streamlit")
        print("   2. If not running, start it with:")
        print("      ./deployment/start_kitchen_dashboard.sh")
        print("   3. Check for port conflicts:")
        print("      lsof -i :8505")

if __name__ == "__main__":
    main()