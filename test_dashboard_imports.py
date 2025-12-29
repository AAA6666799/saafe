#!/usr/bin/env python3
"""
Test script to verify that the AWS dashboard can import all required modules
"""

def test_dashboard_imports():
    """Test that all required modules for the dashboard can be imported"""
    print("🔍 Testing dashboard imports...")
    print("=" * 40)
    
    # Test Streamlit import
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    # Test boto3 import
    try:
        import boto3
        print("✅ Boto3 import successful")
    except ImportError as e:
        print(f"❌ Boto3 import failed: {e}")
        return False
    
    # Test pytz import
    try:
        import pytz
        print("✅ Pytz import successful")
    except ImportError as e:
        print(f"❌ Pytz import failed: {e}")
        return False
    
    # Test plotly import
    try:
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly import successful")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    # Test pandas import
    try:
        import pandas as pd
        print("✅ Pandas import successful")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("✅ All dashboard imports successful!")
    return True

if __name__ == "__main__":
    success = test_dashboard_imports()
    if success:
        print("\n🎉 Dashboard is ready to run!")
    else:
        print("\n❌ Dashboard has missing dependencies!")