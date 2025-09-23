#!/bin/bash
# Kitchen Fire Detection Dashboard Startup Script

cd "/Volumes/Ajay/saafe copy 3"

# Start the dashboard on port 8505 as specified in project configuration
streamlit run saafe_mvp/main.py \
    --server.port=8505 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

echo "Dashboard started on http://0.0.0.0:8505"