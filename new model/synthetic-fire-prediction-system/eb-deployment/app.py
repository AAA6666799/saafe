import subprocess
import sys
import os

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-dashboard.txt"])

# Import after installing dependencies
import streamlit as st
from flask import Flask, render_template_string
import threading
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Serve a simple health check page"""
    return "<h1>Fire Detection Dashboard</h1><p>Application is running. Streamlit dashboard should be available at <a href='/dashboard'>/dashboard</a>.</p>"

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Application is running"}

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=8080)