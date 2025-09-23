#!/usr/bin/env python3
"""
Quick monitoring check for the fire detection system.
This script provides a one-time status check of the deployed system.
"""

import boto3
import json
from datetime import datetime

class FireDetectionQuickMonitor:
    """Quick monitor for the fire detection system."""
    
    def __init__(self):
        """Initialize the monitoring system."""
        self.s3_client = boto3.client('s3')
        self.lambda_client = boto3.client('lambda', region_name='us-east-1')
        self.sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
        self.sns_client = boto3.client('sns', region_name='us-east-1')
        
        self.bucket_name = 'data-collector-of-first-device'
        self.lambda_function = 'saafe-s3-data-processor'
        self.sagemaker_endpoint = 'fire-mvp-xgb-endpoint'
        self.sns_topic = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
    
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
            
            return {
                'status': 'OK',
                'runtime': config.get('Runtime', 'Unknown'),
                'timeout': config.get('Timeout', 0),
                'memory': config.get('MemorySize', 0)
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
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
    
    def run_quick_check(self):
        """Run a quick status check."""
        print("🔥 Fire Detection System - Quick Status Check")
        print("=" * 50)
        print(f"🕒 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
            print(f"   📋 Runtime: {lambda_status['runtime']}")
            print(f"   📋 Timeout: {lambda_status['timeout']} seconds")
            print(f"   📋 Memory: {lambda_status['memory']} MB")
        else:
            print(f"   ❌ Status: ERROR - {lambda_status['error']}")
        print()
        
        # SageMaker Status
        sagemaker_status = self.check_sagemaker_status()
        print("🧠 SageMaker Endpoint Status:")
        if sagemaker_status['status'] == 'OK':
            print(f"   ✅ Status: OK")
            print(f"   📋 Endpoint Status: {sagemaker_status['endpoint_status']}")
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
        
        print("✅ Quick status check completed")

def main():
    """Main function."""
    monitor = FireDetectionQuickMonitor()
    monitor.run_quick_check()

if __name__ == "__main__":
    main()