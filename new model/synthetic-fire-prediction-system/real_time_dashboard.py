#!/usr/bin/env python3
"""
Real-Time Dashboard for Fire Detection System
This script provides a text-based real-time dashboard showing the status of all system components.
"""

import boto3
import time
import json
from datetime import datetime
from collections import defaultdict

class FireDetectionDashboard:
    """Real-time dashboard for the fire detection system."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.s3_client = boto3.client('s3')
        self.lambda_client = boto3.client('lambda', region_name='us-east-1')
        self.sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
        self.sns_client = boto3.client('sns', region_name='us-east-1')
        
        self.bucket_name = 'data-collector-of-first-device'
        self.lambda_function = 'saafe-s3-data-processor'
        self.sagemaker_endpoint = 'fire-mvp-xgb-endpoint'
        self.sns_topic = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        
        # Status tracking
        self.system_status = {
            's3': {'status': 'UNKNOWN', 'details': {}},
            'lambda': {'status': 'UNKNOWN', 'details': {}},
            'sagemaker': {'status': 'UNKNOWN', 'details': {}},
            'sns': {'status': 'UNKNOWN', 'details': {}}
        }
    
    def check_s3_status(self):
        """Check S3 bucket status."""
        try:
            # List recent objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=50
            )
            
            total_files = response.get('KeyCount', 0)
            
            # Count file types
            thermal_files = 0
            gas_files = 0
            recent_files = []
            
            if 'Contents' in response:
                # Sort by last modified and get recent files
                sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                for obj in sorted_files[:5]:  # Get 5 most recent files
                    recent_files.append({
                        'key': obj['Key'],
                        'modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                # Count file types
                for obj in response['Contents']:
                    key = obj['Key']
                    if 'thermal_data' in key:
                        thermal_files += 1
                    elif 'gas_data' in key:
                        gas_files += 1
            
            self.system_status['s3'] = {
                'status': 'OPERATIONAL',
                'details': {
                    'total_files': total_files,
                    'thermal_files': thermal_files,
                    'gas_files': gas_files,
                    'recent_files': recent_files
                }
            }
            return True
        except Exception as e:
            self.system_status['s3'] = {
                'status': 'ERROR',
                'details': {'error': str(e)}
            }
            return False
    
    def check_lambda_status(self):
        """Check Lambda function status."""
        try:
            # Get function configuration
            response = self.lambda_client.get_function(FunctionName=self.lambda_function)
            config = response['Configuration']
            
            self.system_status['lambda'] = {
                'status': 'OPERATIONAL',
                'details': {
                    'function_name': config.get('FunctionName'),
                    'runtime': config.get('Runtime'),
                    'timeout': config.get('Timeout'),
                    'memory': config.get('MemorySize'),
                    'last_modified': config.get('LastModified')
                }
            }
            return True
        except Exception as e:
            self.system_status['lambda'] = {
                'status': 'ERROR',
                'details': {'error': str(e)}
            }
            return False
    
    def check_sagemaker_status(self):
        """Check SageMaker endpoint status."""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=self.sagemaker_endpoint)
            status = response['EndpointStatus']
            
            self.system_status['sagemaker'] = {
                'status': status,
                'details': {
                    'endpoint_name': response.get('EndpointName'),
                    'creation_time': response.get('CreationTime').strftime('%Y-%m-%d %H:%M:%S') if response.get('CreationTime') else 'N/A'
                }
            }
            return True
        except Exception as e:
            self.system_status['sagemaker'] = {
                'status': 'ERROR',
                'details': {'error': str(e)}
            }
            return False
    
    def check_sns_status(self):
        """Check SNS topic status."""
        try:
            response = self.sns_client.get_topic_attributes(TopicArn=self.sns_topic)
            subs_response = self.sns_client.list_subscriptions_by_topic(TopicArn=self.sns_topic)
            subscription_count = len(subs_response.get('Subscriptions', []))
            
            self.system_status['sns'] = {
                'status': 'OPERATIONAL',
                'details': {
                    'topic_arn': response['Attributes'].get('TopicArn'),
                    'subscriptions': subscription_count
                }
            }
            return True
        except Exception as e:
            self.system_status['sns'] = {
                'status': 'ERROR',
                'details': {'error': str(e)}
            }
            return False
    
    def refresh_status(self):
        """Refresh status of all system components."""
        print("🔄 Refreshing system status...")
        
        # Check all components
        self.check_s3_status()
        self.check_lambda_status()
        self.check_sagemaker_status()
        self.check_sns_status()
        
        print("✅ Status refresh complete")
    
    def display_dashboard(self):
        """Display the real-time dashboard."""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H")
        
        # Header
        print("=" * 80)
        print("🔥 FIRE DETECTION SYSTEM - REAL-TIME DASHBOARD")
        print("=" * 80)
        print(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()
        
        # Overall system status
        all_operational = all([
            self.system_status['s3']['status'] in ['OPERATIONAL', 'InService'],
            self.system_status['lambda']['status'] == 'OPERATIONAL',
            self.system_status['sagemaker']['status'] in ['OPERATIONAL', 'InService'],
            self.system_status['sns']['status'] == 'OPERATIONAL'
        ])
        
        status_emoji = "✅" if all_operational else "⚠️"
        print(f"{status_emoji} OVERALL SYSTEM STATUS: {'OPERATIONAL' if all_operational else 'ISSUES DETECTED'}")
        print()
        
        # S3 Status
        s3_status = self.system_status['s3']
        print("📂 S3 DATA INGESTION")
        print("─" * 30)
        if s3_status['status'] == 'OPERATIONAL':
            details = s3_status['details']
            print(f"   Status: ✅ OPERATIONAL")
            print(f"   Bucket: {self.bucket_name}")
            print(f"   Total Files: {details['total_files']}")
            print(f"   Thermal Files: {details['thermal_files']}")
            print(f"   Gas Files: {details['gas_files']}")
            print("   Recent Files:")
            for file in details['recent_files'][:3]:  # Show only top 3
                print(f"     • {file['key']} ({file['modified']} UTC)")
        else:
            print(f"   Status: ❌ ERROR - {s3_status['details'].get('error', 'Unknown error')}")
        print()
        
        # Lambda Status
        lambda_status = self.system_status['lambda']
        print("⚙️ LAMBDA PROCESSING")
        print("─" * 30)
        if lambda_status['status'] == 'OPERATIONAL':
            details = lambda_status['details']
            print(f"   Status: ✅ OPERATIONAL")
            print(f"   Function: {details['function_name']}")
            print(f"   Runtime: {details['runtime']}")
            print(f"   Memory: {details['memory']} MB")
            print(f"   Timeout: {details['timeout']} seconds")
        else:
            print(f"   Status: ❌ ERROR - {lambda_status['details'].get('error', 'Unknown error')}")
        print()
        
        # SageMaker Status
        sagemaker_status = self.system_status['sagemaker']
        print("🧠 SAGEMAKER INFERENCE")
        print("─" * 30)
        if sagemaker_status['status'] in ['OPERATIONAL', 'InService']:
            details = sagemaker_status['details']
            status_emoji = "✅" if sagemaker_status['status'] == 'InService' else "⚠️"
            print(f"   Status: {status_emoji} {sagemaker_status['status']}")
            print(f"   Endpoint: {details['endpoint_name']}")
            print(f"   Created: {details['creation_time']}")
        else:
            print(f"   Status: ❌ ERROR - {sagemaker_status['details'].get('error', 'Unknown error')}")
        print()
        
        # SNS Status
        sns_status = self.system_status['sns']
        print("🚨 SNS ALERTING")
        print("─" * 30)
        if sns_status['status'] == 'OPERATIONAL':
            details = sns_status['details']
            print(f"   Status: ✅ OPERATIONAL")
            print(f"   Topic: {details['topic_arn']}")
            subscription_text = f"{details['subscriptions']} subscription(s)" if details['subscriptions'] > 0 else "No subscriptions (alerts won't be sent)"
            print(f"   Subscriptions: {subscription_text}")
        else:
            print(f"   Status: ❌ ERROR - {sns_status['details'].get('error', 'Unknown error')}")
        print()
        
        # Data Flow Visualization
        print("📊 END-TO-END DATA FLOW")
        print("─" * 30)
        flow_status = []
        flow_status.append("✅" if s3_status['status'] == 'OPERATIONAL' else "❌")
        flow_status.append("✅" if lambda_status['status'] == 'OPERATIONAL' else "❌")
        flow_status.append("✅" if sagemaker_status['status'] in ['OPERATIONAL', 'InService'] else "❌")
        flow_status.append("✅" if sns_status['status'] == 'OPERATIONAL' else "❌")
        
        print(f"   Devices → S3 {flow_status[0]} → Lambda {flow_status[1]} → SageMaker {flow_status[2]} → SNS {flow_status[3]} → Alerts")
        print()
        
        # Instructions
        print("💡 INSTRUCTIONS:")
        print("   • Press Ctrl+C to exit dashboard")
        print("   • Dashboard refreshes every 30 seconds")
        print("   • Check SNS subscriptions to receive alerts")
        print()
        
        # Next steps if there are issues
        if not all_operational:
            print("🔧 RECOMMENDED ACTIONS:")
            if s3_status['status'] != 'OPERATIONAL':
                print("   • Check S3 bucket permissions and connectivity")
            if lambda_status['status'] != 'OPERATIONAL':
                print("   • Verify Lambda function configuration")
            if sagemaker_status['status'] not in ['OPERATIONAL', 'InService']:
                print("   • Check SageMaker endpoint status")
            if sns_status['status'] != 'OPERATIONAL':
                print("   • Configure SNS subscriptions to receive alerts")
            print()
    
    def run_dashboard(self, refresh_interval=30):
        """
        Run the real-time dashboard.
        
        Args:
            refresh_interval (int): Seconds between refreshes
        """
        print("🚀 Starting Fire Detection System Dashboard...")
        print(f"📊 Refresh interval: {refresh_interval} seconds")
        print("🛑 Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                # Refresh status
                self.refresh_status()
                
                # Display dashboard
                self.display_dashboard()
                
                # Wait for next refresh
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Dashboard stopped by user")
            print("✅ Thank you for using the Fire Detection System Dashboard!")
        
        except Exception as e:
            print(f"\n\n❌ Unexpected error: {e}")
            print("Please check your AWS credentials and permissions.")

def main():
    """Main function to run the dashboard."""
    print("🔥 Fire Detection System - Real-Time Dashboard")
    print("=" * 50)
    
    # Initialize dashboard
    dashboard = FireDetectionDashboard()
    
    # Run dashboard with 30-second refresh interval
    dashboard.run_dashboard(refresh_interval=30)

if __name__ == "__main__":
    main()