#!/usr/bin/env python3
"""
Monitor SageMaker Training Job Progress
"""

import boto3
import time
from datetime import datetime

def monitor_training(job_name):
    """Monitor training job progress"""
    
    sagemaker = boto3.client('sagemaker')
    
    print(f"🔍 Monitoring training job: {job_name}")
    print("=" * 50)
    
    while True:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            
            status = response['TrainingJobStatus']
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"[{timestamp}] Status: {status}")
            
            if status == 'Completed':
                print("🎉 Training completed successfully!")
                
                # Get output location
                output_path = response['OutputDataConfig']['S3OutputPath']
                print(f"📁 Model artifacts: {output_path}")
                print(f"💾 Download with: aws s3 sync {output_path} ./trained_models/")
                
                # Get training metrics
                if 'FinalMetricDataList' in response:
                    print("📊 Final Metrics:")
                    for metric in response['FinalMetricDataList']:
                        print(f"   {metric['MetricName']}: {metric['Value']}")
                
                break
                
            elif status == 'Failed':
                print("❌ Training failed!")
                if 'FailureReason' in response:
                    print(f"Reason: {response['FailureReason']}")
                break
                
            elif status == 'Stopped':
                print("⏹️ Training stopped!")
                break
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped (training continues)")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_training("saafe-iot-g5-1755382352")