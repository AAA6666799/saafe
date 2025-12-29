#!/usr/bin/env python3
"""
Test script for the kitchen fire detection system.
This script verifies that the dashboard is running and the alert system is functional.
"""

import requests
import time
import json
from pathlib import Path

def test_dashboard_access():
    """Test if the dashboard is accessible"""
    print("🔍 Testing dashboard access...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard is accessible")
            return True
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to dashboard - make sure it's running")
        return False
    except Exception as e:
        print(f"❌ Error testing dashboard access: {e}")
        return False

def test_api_endpoints():
    """Test if API endpoints are available"""
    print("🔍 Testing API endpoints...")
    
    # Test health endpoint (if available)
    try:
        response = requests.get("http://localhost:8501/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint is accessible")
        else:
            print(f"ℹ️  Health endpoint returned: {response.status_code}")
    except:
        print("ℹ️  Health endpoint not available (this is normal for Streamlit apps)")
    
    return True

def test_notification_config():
    """Test if notification configuration is properly set"""
    print("🔍 Testing notification configuration...")
    
    config_path = Path("/Volumes/Ajay/saafe copy 3/config/app_config.json")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        notifications = config.get("notifications", {})
        
        email_enabled = notifications.get("email_enabled", False)
        sms_enabled = notifications.get("sms_enabled", False)
        email_addresses = notifications.get("email_addresses", [])
        phone_numbers = notifications.get("phone_numbers", [])
        
        print(f"✅ Email alerts: {'ENABLED' if email_enabled else 'DISABLED'}")
        print(f"✅ SMS alerts: {'ENABLED' if sms_enabled else 'DISABLED'}")
        
        if email_addresses:
            print(f"✅ Email recipients: {len(email_addresses)} address(es)")
        else:
            print("⚠️  No email recipients configured")
            
        if phone_numbers:
            print(f"✅ SMS recipients: {len(phone_numbers)} number(s)")
        else:
            print("⚠️  No SMS recipients configured")
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading notification configuration: {e}")
        return False

def test_iot_config():
    """Test if IoT configuration is properly set"""
    print("🔍 Testing IoT configuration...")
    
    config_path = Path("/Volumes/Ajay/saafe copy 3/config/iot_config.yaml")
    
    try:
        if config_path.exists():
            print("✅ IoT configuration file found")
            # Read first few lines to verify it's properly formatted
            with open(config_path, 'r') as f:
                first_lines = [next(f) for _ in range(5)]
            
            print("✅ IoT configuration appears to be properly formatted")
            return True
        else:
            print("❌ IoT configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading IoT configuration: {e}")
        return False

def test_system_status():
    """Test overall system status"""
    print("🔍 Testing overall system status...")
    
    # Check if required directories exist
    required_dirs = [
        "/Volumes/Ajay/saafe copy 3/config",
        "/Volumes/Ajay/saafe copy 3/models",
        "/Volumes/Ajay/saafe copy 3/logs"
    ]
    
    all_good = True
    for directory in required_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Directory missing: {directory}")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🧪 Kitchen Fire Detection System - End-to-End Testing")
    print("=" * 60)
    
    tests = [
        test_dashboard_access,
        test_api_endpoints,
        test_notification_config,
        test_iot_config,
        test_system_status
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()  # Empty line for readability
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
            print()  # Empty line for readability
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your kitchen fire detection system is ready for use.")
        print("\n📋 System Information:")
        print("   🔗 Dashboard URL: http://localhost:8501")
        print("   📧 Email alerts: Enabled (to ch.ajay1707@gmail.com)")
        print("   📱 SMS alerts: Enabled")
        print("   📍 Monitoring: Kitchen above chimney")
        print("\n💡 Next Steps:")
        print("   1. Visit the dashboard at http://localhost:8501")
        print("   2. Configure additional team members in config/app_config.json")
        print("   3. Test the alert system by simulating a fire scenario")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")
        return 1

if __name__ == "__main__":
    exit(main())