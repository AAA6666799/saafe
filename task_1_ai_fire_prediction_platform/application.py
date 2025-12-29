"""
Main application file for Elastic Beanstalk
"""

import streamlit as st
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the dashboard
import dashboard
from app import application

if __name__ == "__main__":
    application.run()
    # Get the port from the environment variable or default to 8501
    port = int(os.environ.get("PORT", 8501))
    
    # Run the dashboard with the correct port
    import sys
    sys.argv = ["streamlit", "run", "dashboard.py", f"--server.port={port}", "--server.address=0.0.0.0"]
    import streamlit.web.bootstrap
    streamlit.web.bootstrap.run("dashboard.py", sys.argv[0], sys.argv[1:], flag_options={})