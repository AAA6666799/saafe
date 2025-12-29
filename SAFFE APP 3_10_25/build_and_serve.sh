#!/bin/bash

# Script to build and serve the SAAFE Fire Detection Dashboard for production

echo "Building SAAFE Fire Detection Dashboard for production..."

# Build the React frontend
echo "Building frontend..."
npm run build

if [ $? -ne 0 ]; then
    echo "Frontend build failed!"
    exit 1
fi

echo "Frontend build completed successfully!"

# Copy built files to backend dist directory
echo "Copying built files to backend..."
cp -r dist/* backend/dist/

# Start the backend server which will serve the built frontend
echo "Starting production server on http://localhost:8000..."
cd backend && npm start