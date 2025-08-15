#!/bin/bash
echo "Starting Saafe Fire Detection MVP..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    echo "Please ensure Python 3 is installed"
    exit 1
fi

# Start the application
echo "Opening Saafe MVP..."
echo "Your browser will open automatically at http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo

python3 main.py
