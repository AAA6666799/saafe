#!/bin/bash

# Script to start the SAAFE Global Command Center frontend

echo "Starting SAAFE Global Command Center..."

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the development server
echo "Starting development server on http://localhost:5173"
npm run dev