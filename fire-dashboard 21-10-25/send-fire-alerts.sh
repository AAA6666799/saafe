#!/bin/bash

# Fire Alert Data Sender Script
# This script sends dummy fire alert data to the backend API
# Usage: ./send-fire-alerts.sh [backend-url] [alert-type]
# Example: ./send-fire-alerts.sh http://localhost:8080 fire

# Configuration
BACKEND_URL="${1:-http://localhost:8080}"
ALERT_TYPE="${2:-fire}"

echo "🔥 Fire Alert Data Sender"
echo "========================="
echo "Backend URL: $BACKEND_URL"
echo "Alert Type: $ALERT_TYPE"
echo ""

# Function to send non-fire alert
send_non_fire() {
    echo "📤 Sending NON-FIRE alert..."
    curl -X POST "$BACKEND_URL/api/alert-state" \
        -H "Content-Type: application/json" \
        -d '{
            "isActive": false,
            "level": 1,
            "message": "System operating normally",
            "riskScore": 10,
            "confidence": 0.9,
            "location": "Kitchen",
            "deviceId": "SAAFE-KITCHEN-001"
        }'
    echo -e "\n✅ Non-fire alert sent\n"
}

# Function to send fire predicted alert
send_fire_predicted() {
    echo "📤 Sending FIRE PREDICTED alert..."
    curl -X POST "$BACKEND_URL/api/alert-state" \
        -H "Content-Type: application/json" \
        -d '{
            "isActive": true,
            "level": 5,
            "message": "Fire predicted - elevated risk detected",
            "riskScore": 55,
            "confidence": 0.75,
            "location": "Kitchen",
            "deviceId": "SAAFE-KITCHEN-001"
        }'
    echo -e "\n⚠️  Fire predicted alert sent (emails will be sent if risk >= 40)\n"
}

# Function to send fire detected alert
send_fire_detected() {
    echo "📤 Sending FIRE DETECTED alert..."
    curl -X POST "$BACKEND_URL/api/alert-state" \
        -H "Content-Type: application/json" \
        -d '{
            "isActive": true,
            "level": 9,
            "message": "FIRE DETECTED - immediate action required",
            "riskScore": 95,
            "confidence": 0.95,
            "location": "Kitchen",
            "deviceId": "SAAFE-KITCHEN-001"
        }'
    echo -e "\n🔥 Fire detected alert sent (urgent emails will be sent)\n"
}

# Function to send caution alert
send_caution() {
    echo "📤 Sending CAUTION alert..."
    curl -X POST "$BACKEND_URL/api/alert-state" \
        -H "Content-Type: application/json" \
        -d '{
            "isActive": true,
            "level": 2,
            "message": "Elevated fire risk - monitoring recommended",
            "riskScore": 25,
            "confidence": 0.7,
            "location": "Kitchen",
            "deviceId": "SAAFE-KITCHEN-001"
        }'
    echo -e "\n⚡ Caution alert sent\n"
}

# Send alert based on type
case "$ALERT_TYPE" in
    "non-fire"|"normal")
        send_non_fire
        ;;
    "predicted"|"warning")
        send_fire_predicted
        ;;
    "fire"|"urgent")
        send_fire_detected
        ;;
    "caution")
        send_caution
        ;;
    "all")
        echo "🔄 Sending all alert types..."
        send_non_fire
        sleep 2
        send_caution
        sleep 2
        send_fire_predicted
        sleep 2
        send_fire_detected
        ;;
    *)
        echo "❌ Invalid alert type: $ALERT_TYPE"
        echo ""
        echo "Valid alert types:"
        echo "  - non-fire (or normal)    : Normal conditions, no fire"
        echo "  - caution                 : Elevated risk (score 20-39)"
        echo "  - predicted (or warning)  : Fire predicted (score 40-79)"
        echo "  - fire (or urgent)        : Fire detected (score 80+)"
        echo "  - all                     : Send all alert types in sequence"
        echo ""
        echo "Usage: $0 [backend-url] [alert-type]"
        echo "Example: $0 http://localhost:8080 fire"
        exit 1
        ;;
esac

echo "✅ Done!"