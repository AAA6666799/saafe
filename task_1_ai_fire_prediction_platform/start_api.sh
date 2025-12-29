#!/bin/bash

# Script to start the Synthetic Fire Prediction System API Server

echo "Starting Synthetic Fire Prediction System API Server..."

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r synthetic_fire_requirements.txt
pip install -r api_requirements.txt

# Start the API server
echo "Starting API server on http://0.0.0.0:8000"
python api_server.py