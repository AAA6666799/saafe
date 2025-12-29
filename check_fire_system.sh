#!/bin/bash

# Script to check the status of the SAAFE fire detection system

echo "Checking SAAFE Fire Detection System Status..."
echo "=========================================="

# Check if required processes are running
echo "Checking running processes..."

# Check for Vite frontend server (port 5173)
if lsof -i :5173 >/dev/null 2>&1; then
    echo "✅ Frontend server: Running (port 5173)"
    FRONTEND_STATUS="RUNNING"
else
    echo "❌ Frontend server: Not running"
    FRONTEND_STATUS="STOPPED"
fi

# Check for backend server (port 8000)
if lsof -i :8000 >/dev/null 2>&1; then
    echo "✅ Backend server: Running (port 8000)"
    BACKEND_STATUS="RUNNING"
else
    echo "❌ Backend server: Not running"
    BACKEND_STATUS="STOPPED"
fi

# Check IoT device simulation (if running)
if pgrep -f "saafe_mvp" >/dev/null 2>&1; then
    echo "✅ IoT system: Running"
    IOT_STATUS="RUNNING"
else
    echo "⚠️  IoT system: Not detected"
    IOT_STATUS="UNKNOWN"
fi

echo ""
echo "System Components Status:"
echo "========================"
echo "Frontend Dashboard: $FRONTEND_STATUS"
echo "Backend API: $BACKEND_STATUS"
echo "IoT Device System: $IOT_STATUS"

echo ""
echo "Access Information:"
echo "=================="
if [ "$FRONTEND_STATUS" = "RUNNING" ]; then
    echo "Dashboard (Development): http://localhost:5173"
fi

if [ "$BACKEND_STATUS" = "RUNNING" ]; then
    echo "Dashboard (Production): http://localhost:8000"
    echo "API Endpoint: http://localhost:8000/api/fire-detection-data"
fi

echo ""
echo "Configuration:"
echo "============="
echo "Alert Email: ch.ajay1707@gmail.com"
echo "Device Location: Kitchen (above chimney)"

echo ""
if [ "$FRONTEND_STATUS" = "RUNNING" ] && [ "$BACKEND_STATUS" = "RUNNING" ]; then
    echo "✅ System is fully operational!"
elif [ "$FRONTEND_STATUS" = "RUNNING" ] || [ "$BACKEND_STATUS" = "RUNNING" ]; then
    echo "⚠️  System is partially operational"
else
    echo "❌ System is not running"
    echo ""
    echo "To start the system:"
    echo "  cd '/Volumes/Ajay/saafe copy 3/saafe-lovable'"
    echo "  ./start_dashboard.sh"
fi