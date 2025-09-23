#!/bin/bash

# Script to start both the frontend and backend for the SAAFE Fire Detection Dashboard

echo "Starting SAAFE Fire Detection Dashboard..."

# Function to clean up background processes on exit
cleanup() {
    echo "Stopping dashboard services..."
    if [[ -n $BACKEND_PID ]]; then
        kill $BACKEND_PID
    fi
    if [[ -n $FRONTEND_PID ]]; then
        kill $FRONTEND_PID
    fi
    exit 0
}

# Trap exit signals to clean up
trap cleanup EXIT INT TERM

# Start backend server
echo "Starting backend server on http://localhost:8000..."
cd backend && npm start &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend development server
echo "Starting frontend development server on http://localhost:5173..."
npm run dev &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID
wait $FRONTEND_PID