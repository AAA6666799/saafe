"""
Script to check if the Saafe Fire Detection Dashboard is running
"""

import sys
import os
import time
import requests

def check_dashboard_status():
    """Check if the dashboard is running"""
    print("Checking Saafe Fire Detection Dashboard status...")
    print("=" * 50)
    
    ports_to_check = [8501, 8502, 8503, 8504, 8505]
    
    for port in ports_to_check:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            if response.status_code == 200:
                print(f"✅ Dashboard is running on port {port}")
                print(f"   Access it at: http://localhost:{port}")
                return port
        except:
            pass
    
    print("❌ Dashboard is not running on any standard ports")
    print("   Please start the dashboard with: python run_dashboard.py")
    return None


def main():
    """Main function"""
    port = check_dashboard_status()
    
    if port:
        print(f"\n🎉 Dashboard is accessible!")
        print(f"   URL: http://localhost:{port}")
        return 0
    else:
        print(f"\n❌ Dashboard is not running")
        print("   Start it with: python run_dashboard.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())