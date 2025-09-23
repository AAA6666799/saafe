#!/usr/bin/env python3
"""
Script to run the Saafe AWS Dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the Saafe AWS Dashboard"""
    print("🔥 Starting Saafe Fire Detection AWS Dashboard...")
    print("=" * 50)
    
    # Check if Streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if required AWS libraries are installed
    try:
        import boto3
        import pytz
    except ImportError:
        print("❌ AWS libraries not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "boto3", "pytz"])
    
    # Check if Plotly is installed
    try:
        import plotly
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "plotly"])
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "saafe_aws_dashboard.py")
    
    # Run the Streamlit app
    print("🚀 Launching dashboard...")
    print("🌐 Access the dashboard at: http://localhost:8502")
    print("💡 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port", "8502",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    main()