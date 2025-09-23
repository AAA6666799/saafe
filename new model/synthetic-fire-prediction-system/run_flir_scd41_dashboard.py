#!/usr/bin/env python3
"""
Script to run the FLIR+SCD41 Fire Detection Dashboard.

This script launches the Streamlit dashboard for the FLIR Lepton 3.5 + SCD41 CO₂ 
sensor fire detection system.
"""

import subprocess
import sys
import os

def run_dashboard():
    """Run the FLIR+SCD41 dashboard using Streamlit."""
    # Get the path to the dashboard file
    dashboard_path = os.path.join(
        os.path.dirname(__file__), 
        'src', 
        'visualization', 
        'flir_scd41_dashboard.py'
    )
    
    # Check if the dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return 1
    
    # Run the Streamlit app
    try:
        print("🚀 Launching FLIR+SCD41 Fire Detection Dashboard...")
        print(f"📄 Dashboard file: {dashboard_path}")
        print("🌐 Access the dashboard at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the dashboard")
        print()
        
        # Run Streamlit
        result = subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', dashboard_path
        ])
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return 1

if __name__ == "__main__":
    # Check if Streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit is not installed. Please install it with:")
        print("   pip install streamlit")
        sys.exit(1)
    
    # Run the dashboard
    sys.exit(run_dashboard())