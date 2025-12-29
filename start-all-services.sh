#!/bin/bash

# SAAFE Fire Detection System - Start All Services
# This script starts the backend, dashboard, and data sender

echo "🚀 Starting SAAFE Fire Detection System..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kill any existing processes on our ports
echo "${YELLOW}Cleaning up existing processes...${NC}"
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:5174 | xargs kill -9 2>/dev/null || true
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
sleep 2

# Start Backend (without email)
echo ""
echo "${BLUE}1. Starting Backend API (Port 8080)...${NC}"
cd "SAFFE APP 3_10_25/backend"
node server.js > /tmp/saafe-backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
cd ../..
sleep 3

# Check if backend started
if lsof -ti:8080 > /dev/null; then
    echo "${GREEN}   ✓ Backend started successfully${NC}"
else
    echo "   ✗ Backend failed to start"
    exit 1
fi

# Start Dashboard
echo ""
echo "${BLUE}2. Starting Fire Dashboard (Port 5174)...${NC}"
cd "fire-dashboard 21-10-25"
npm run dev > /tmp/saafe-dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "   Dashboard PID: $DASHBOARD_PID"
cd ..
sleep 5

# Check if dashboard started
if lsof -ti:5174 > /dev/null; then
    echo "${GREEN}   ✓ Dashboard started successfully${NC}"
else
    echo "   ✗ Dashboard failed to start"
    exit 1
fi

# Start Data Sender
echo ""
echo "${BLUE}3. Starting Data Sender (Port 8001)...${NC}"
cd "fire-data-sender-standalone"
python3 -m http.server 8001 > /tmp/saafe-datasender.log 2>&1 &
SENDER_PID=$!
echo "   Data Sender PID: $SENDER_PID"
cd ..
sleep 2

# Check if data sender started
if lsof -ti:8001 > /dev/null; then
    echo "${GREEN}   ✓ Data Sender started successfully${NC}"
else
    echo "   ✗ Data Sender failed to start"
    exit 1
fi

# Display status
echo ""
echo "${GREEN}========================================${NC}"
echo "${GREEN}   All Services Started Successfully!${NC}"
echo "${GREEN}========================================${NC}"
echo ""
echo "📊 Service URLs:"
echo "   Backend API:    http://localhost:8080"
echo "   Dashboard:      http://localhost:5174"
echo "   Data Sender:    http://localhost:8001"
echo ""
echo "📝 Log Files:"
echo "   Backend:        /tmp/saafe-backend.log"
echo "   Dashboard:      /tmp/saafe-dashboard.log"
echo "   Data Sender:    /tmp/saafe-datasender.log"
echo ""
echo "🎮 How to Test:"
echo "   1. Open Dashboard:    http://localhost:5174"
echo "   2. Open Data Sender:  http://localhost:8001"
echo "   3. Click any button in Data Sender"
echo "   4. Watch Dashboard update within 5 seconds!"
echo ""
echo "🛑 To stop all services:"
echo "   kill $BACKEND_PID $DASHBOARD_PID $SENDER_PID"
echo ""
echo "Press Ctrl+C to stop monitoring..."
echo ""

# Monitor logs
tail -f /tmp/saafe-backend.log /tmp/saafe-dashboard.log /tmp/saafe-datasender.log