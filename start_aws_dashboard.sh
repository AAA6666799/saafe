#!/bin/bash
# Script to start the Saafe AWS Dashboard

echo "🔥 Starting Saafe Fire Detection AWS Dashboard..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "saafe_aws_dashboard.py" ]; then
    echo "❌ Error: saafe_aws_dashboard.py not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if AWS credentials are configured
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "⚠️  Warning: AWS credentials not found in environment variables"
    echo "Make sure to configure AWS credentials using 'aws configure' or set environment variables"
    echo ""
fi

# Start the dashboard
echo "🚀 Launching dashboard..."
echo "🌐 Access the dashboard at: http://localhost:8502"
echo "💡 Press Ctrl+C to stop the dashboard"
echo ""

# Run the Streamlit app
python -m streamlit run saafe_aws_dashboard.py --server.port 8502 --server.address 0.0.0.0