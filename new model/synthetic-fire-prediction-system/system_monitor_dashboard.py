#!/usr/bin/env python3
"""
System Monitoring Dashboard for Fire Detection System
"""

import boto3
import json
from datetime import datetime

class FireDetectionSystemMonitor:
    """Monitor the fire detection system components."""
    
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
    
    def check_system_status(self):
        """Check the status of all system components."""
        print("🔥 Fire Detection System Status Dashboard")
        print("=" * 50)
        print(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check S3 status
        self._check_s3_status()
        
        # Check Lambda status
        self._check_lambda_status()
        
        # Check SageMaker status
        self._check_sagemaker_status()
        
        # Check SNS status
        self._check_sns_status()
        
        # Show next steps
        self._show_next_steps()
    
    def _check_s3_status(self):
        """Check S3 bucket status."""
        print("📂 S3 Data Ingestion Status:")
        try:
            # List objects in bucket
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
            
            print(f"   ✅ Status: OK")
            print(f"   📁 Total Files: {total_files}")
            print(f"   🔥 Thermal Files: {thermal_files}")
            print(f"   🧪 Gas Files: {gas_files}")
            
        except Exception as e:
            print(f"   ❌ Status: ERROR - {e}")
        print()
    
    def _check_lambda_status(self):
        """Check Lambda function status."""
        print("⚙️  Lambda Processing Status:")
        try:
            # Get function configuration
            response = self.lambda_client.get_function(FunctionName=self.lambda_function)
            config = response['Configuration']
            
            print(f"   ✅ Status: OK")
            print(f"   📋 Function Name: {config.get('FunctionName')}")
            print(f"   📋 Runtime: {config.get('Runtime')}")
            print(f"   📋 Timeout: {config.get('Timeout')} seconds")
            print(f"   📋 Memory: {config.get('MemorySize')} MB")
            
        except Exception as e:
            print(f"   ❌ Status: ERROR - {e}")
        print()
    
    def _check_sagemaker_status(self):
        """Check SageMaker endpoint status."""
        print("🧠 SageMaker Endpoint Status:")
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=self.sagemaker_endpoint)
            status = response['EndpointStatus']
            
            print(f"   {'✅' if status == 'InService' else '⚠️'} Status: {status}")
            print(f"   📋 Endpoint Name: {response.get('EndpointName')}")
            
        except Exception as e:
            print(f"   ❌ Status: ERROR - {e}")
        print()
    
    def _check_sns_status(self):
        """Check SNS topic status."""
        print("🚨 SNS Alerting Status:")
        try:
            response = self.sns_client.get_topic_attributes(TopicArn=self.sns_topic)
            subs_response = self.sns_client.list_subscriptions_by_topic(TopicArn=self.sns_topic)
            subscription_count = len(subs_response.get('Subscriptions', []))
            
            print(f"   ✅ Status: OK")
            print(f"   📋 Topic ARN: {response['Attributes'].get('TopicArn')}")
            print(f"   📋 Subscriptions: {subscription_count}")
            
        except Exception as e:
            print(f"   ❌ Status: ERROR - {e}")
        print()
    
    def _show_next_steps(self):
        """Show recommended next steps."""
        print("📋 Recommended Next Steps:")
        print("   1. Configure SNS subscriptions for alerting")
        print("   2. Monitor CloudWatch logs for processing activity")
        print("   3. Test end-to-end data flow with sample data")
        print("   4. Set up automated monitoring and alerting")
        print("   5. Document system operations procedures")

def main():
    """Main function."""
    monitor = FireDetectionSystemMonitor()
    monitor.check_system_status()

if __name__ == "__main__":
    main()