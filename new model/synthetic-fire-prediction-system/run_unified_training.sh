#!/bin/bash

# FLIR+SCD41 Unified Training Execution Script
# This script sets up the environment and runs the unified training notebook

echo "🔥 FLIR+SCD41 Fire Detection System - Unified Training"
echo "====================================================="

# Check if we're in the right directory
if [[ ! -f "requirements_unified_notebook.txt" ]]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Create data directory
echo "📁 Setting up data directory..."
mkdir -p data/flir_scd41

# Check if virtual environment exists
if [[ -d ".venv" ]]; then
    echo "🐍 Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python -c "import numpy, pandas, sklearn, torch, xgboost, jupyter" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "⚠️  Some dependencies are missing"
    echo "📦 Installing required dependencies..."
    pip install -r requirements_unified_notebook.txt
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✅ All dependencies are available"
fi

# Run the unified training
echo "🚀 Executing unified training notebook..."
python run_unified_training.py

if [[ $? -eq 0 ]]; then
    echo "🎉 Unified training completed successfully!"
    echo "📂 Check the data/flir_scd41/ directory for output files"
else
    echo "❌ Unified training failed"
    exit 1
fi