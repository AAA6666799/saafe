#!/bin/bash

# Monitor the deployment status

echo "Monitoring deployment status..."

while true; do
    echo "Checking environment status at $(date)..."
    STATUS=$(aws elasticbeanstalk describe-environments --application-name saafe-fire-dashboard --environment-names saafe-fire-dashboard-env --region us-east-1 --query 'Environments[0].Status' --output text 2>/dev/null)
    HEALTH=$(aws elasticbeanstalk describe-environments --application-name saafe-fire-dashboard --environment-names saafe-fire-dashboard-env --region us-east-1 --query 'Environments[0].Health' --output text 2>/dev/null)
    CNAME=$(aws elasticbeanstalk describe-environments --application-name saafe-fire-dashboard --environment-names saafe-fire-dashboard-env --region us-east-1 --query 'Environments[0].CNAME' --output text 2>/dev/null)
    
    echo "Status: $STATUS"
    echo "Health: $HEALTH"
    echo "URL: http://$CNAME"
    
    if [ "$STATUS" == "Ready" ]; then
        echo "✅ Deployment completed successfully!"
        echo "🌐 Access your dashboard at: http://$CNAME"
        break
    fi
    
    if [ "$STATUS" == "Terminated" ]; then
        echo "❌ Deployment failed!"
        echo "Check the events for more information:"
        aws elasticbeanstalk describe-events --application-name saafe-fire-dashboard --environment-name saafe-fire-dashboard-env --region us-east-1
        break
    fi
    
    echo "Waiting 30 seconds before next check..."
    sleep 30
done