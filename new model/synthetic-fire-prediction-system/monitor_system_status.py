#!/usr/bin/env python3
"""
Real-time monitoring dashboard for the fire detection system.
This script provides continuous monitoring of the deployed system.
"""

import boto3
import time
import json
from datetime import datetime
from collections import defaultdict

class FireDetectionMonitor:
    """Monitor the fire detection system in real-time."""
    
    def __init__(self):
        """Initialize the monitoring system."""
        self.s3_client = boto3.client('s3')
        self.lambda_client = boto3.client('lambda', region_name='us-east-1')
        self.sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
        self.sns_client = boto3.client('sns', region_name='us-east-1')
        self.logs_client = boto3.client('logs', region_name='us-east-1')
        
        self.bucket_name = 'data-collector-of-first-device'
        self.lambda_function = 'saafe-s3-data-processor'
        self.sagemaker_endpoint = 'fire-mvp-xgb-endpoint'
        self.sns_topic = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'errors': 0,
            'alerts_sent': 0,
            'start_time': datetime.now()
        }
    
    def check_s3_status(self):
        """Check S3 bucket status and recent file count."""
        try:
            # Get recent files (last 10 minutes)
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=100
            )
            
            total_files = response.get('KeyCount', 0)
            
            # Count file types
            thermal_files = 0
            gas_files = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if 'thermal_data' in key:
                        thermal_files += 1
                    elif 'gas_data' in key:
                        gas_files += 1
            
            return {
                'status': 'OK',
                'total_files': total_files,
                'thermal_files': thermal_files,
                'gas_files': gas_files
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def check_lambda_status(self):
        """Check Lambda function status and recent invocations."""
        try:
            # Get function configuration
            response = self.lambda_client.get_function(FunctionName=self.lambda_function)
            config = response['Configuration']
            
            # Get CloudWatch metrics for the last hour
            metrics = self.get_lambda_metrics()
            
            return {
                'status': 'OK',
                'runtime': config.get('Runtime', 'Unknown'),
                'timeout': config.get('Timeout', 0),
                'memory': config.get('MemorySize', 0),
                'invocations': metrics.get('invocations', 0),
                'errors': metrics.get('errors', 0),
                'duration': metrics.get('duration', 0)
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def get_lambda_metrics(self):
        """Get Lambda function metrics from CloudWatch."""
        try:
            # Get invocations
            invocations = self.get_cloudwatch_metric(
                'AWS/Lambda', 'Invocations',
                [{'Name': 'FunctionName', 'Value': self.lambda_function}]
            )
            
            # Get errors
            errors = self.get_cloudwatch_metric(
                'AWS/Lambda', 'Errors',
                [{'Name': 'FunctionName', 'Value': self.lambda_function}]
            )
            
            # Get duration
            duration = self.get_cloudwatch_metric(
                'AWS/Lambda', 'Duration',
                [{'Name': 'FunctionName', 'Value': self.lambda_function}]
            )
            
            return {
                'invocations': invocations,
                'errors': errors,
                'duration': duration
            }
        except Exception as e:
            return {
                'invocations': 0,
                'errors': 0,
                'duration': 0
            }
    
    def get_cloudwatch_metric(self, namespace, metric_name, dimensions):
        """Get a CloudWatch metric."""
        try:
            response = self.logs_client.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=datetime.now().timestamp() - 3600,  # Last hour
                EndTime=datetime.now().timestamp(),
                Period=3600,  # 1 hour period
                Statistics=['Sum' if metric_name in ['Invocations', 'Errors'] else 'Average']
            )
            
            if response['Datapoints']:
                datapoint = response['Datapoints'][0]
                return datapoint.get('Sum' if metric_name in ['Invocations', 'Errors'] else 'Average', 0)
            return 0
        except Exception:
            return 0
    
    def check_sagemaker_status(self):
        """Check SageMaker endpoint status."""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=self.sagemaker_endpoint)
            status = response['EndpointStatus']
            
            return {
                'status': 'OK' if status == 'InService' else 'WARNING',
                'endpoint_status': status
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def check_sns_status(self):
        """Check SNS topic status."""
        try:
            response = self.sns_client.get_topic_attributes(TopicArn=self.sns_topic)
            subs_response = self.sns_client.list_subscriptions_by_topic(TopicArn=self.sns_topic)
            subscription_count = len(subs_response.get('Subscriptions', []))
            
            return {
                'status': 'OK',
                'subscriptions': subscription_count
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def get_recent_logs(self):
        """Get recent CloudWatch logs."""
        try:
            # Get recent log streams
            response = self.logs_client.describe_log_streams(
                logGroupName=f'/aws/lambda/{self.lambda_function}',
                orderBy='LastEventTime',
                descending=True,
                limit=1
            )
            
            if response.get('logStreams'):
                log_stream_name = response['logStreams'][0]['logStreamName']
                
                # Get recent log events
                events_response = self.logs_client.get_log_events(
                    logGroupName=f'/aws/lambda/{self.lambda_function}',
                    logStreamName=log_stream_name,
                    limit=10
                )
                
                events = events_response.get('events', [])
                return events
            return []
        except Exception as e:
            return []
    
    def display_dashboard(self):
        """Display the monitoring dashboard."""
        print("\033[2J\033[H")  # Clear screen
        print("🔥 Fire Detection System - Real-Time Monitoring Dashboard")
        print("=" * 60)
        print(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # S3 Status
        s3_status = self.check_s3_status()
        print("📂 S3 Data Ingestion Status:")
        if s3_status['status'] == 'OK':
            print(f"   ✅ Status: OK")
            print(f"   📁 Total Files: {s3_status['total_files']}")
            print(f"   🔥 Thermal Files: {s3_status['thermal_files']}")
            print(f"   🧪 Gas Files: {s3_status['gas_files']}")
        else:
            print(f"   ❌ Status: ERROR - {s3_status['error']}")
        print()
        
        # Lambda Status
        lambda_status = self.check_lambda_status()
        print("⚙️  Lambda Processing Status:")
        if lambda_status['status'] == 'OK':
            print(f"   ✅ Status: OK")
            print(f"   📋 Invocations (last hour): {lambda_status['invocations']}")
            print(f"   ❌ Errors (last hour): {lambda_status['errors']}")
            print(f"   🕐 Avg Duration: {lambda_status['duration']:.2f}ms")
        else:
            print(f"   ❌ Status: ERROR - {lambda_status['error']}")
        print()
        
        # SageMaker Status
        sagemaker_status = self.check_sagemaker_status()
        print("🧠 SageMaker Endpoint Status:")
        if sagemaker_status['status'] == 'OK':
            print(f"   ✅ Status: OK")
            print(f"   📋 Endpoint: {sagemaker_status['endpoint_status']}")
        elif sagemaker_status['status'] == 'WARNING':
            print(f"   ⚠️  Status: {sagemaker_status['endpoint_status']}")
        else:
            print(f"   ❌ Status: ERROR - {sagemaker_status['error']}")
        print()
        
        # SNS Status
        sns_status = self.check_sns_status()
        print("🚨 SNS Alerting Status:")
        if sns_status['status'] == 'OK':
            print(f"   ✅ Status: OK")
            print(f"   📋 Subscriptions: {sns_status['subscriptions']}")
        else:
            print(f"   ❌ Status: ERROR - {sns_status['error']}")
        print()
        
        # Recent Activity
        print("📋 Recent Activity:")
        recent_logs = self.get_recent_logs()
        if recent_logs:
            for event in recent_logs[-5:]:  # Show last 5 events
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000).strftime('%H:%M:%S')
                message = event['message']
                # Truncate long messages
                if len(message) > 80:
                    message = message[:77] + "..."
                print(f"   [{timestamp}] {message}")
        else:
            print("   No recent activity")
        print()
        
        # System Stats
        uptime = datetime.now() - self.stats['start_time']
        print("📊 System Statistics:")
        print(f"   ⏱️  Uptime: {str(uptime).split('.')[0]}")
        print(f"   📁 Files Processed: {self.stats['files_processed']}")
        print(f"   ❌ Errors: {self.stats['errors']}")
        print(f"   🚨 Alerts Sent: {self.stats['alerts_sent']}")
    
    def run_monitoring(self, duration_minutes=60):
        """Run continuous monitoring."""
        print("🚀 Starting real-time monitoring...")
        print(f"📊 Monitoring will run for {duration_minutes} minutes")
        print("Press Ctrl+C to stop monitoring")
        
        end_time = time.time() + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                self.display_dashboard()
                time.sleep(10)  # Update every 10 seconds
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
        
        print("✅ Monitoring session completed")

def main():
    """Main monitoring function."""
    print("🔥 Fire Detection System Monitoring")
    print("=" * 40)
    
    # Initialize monitor
    monitor = FireDetectionMonitor()
    
    # Run monitoring for 60 minutes
    monitor.run_monitoring(duration_minutes=60)

if __name__ == "__main__":
    main()