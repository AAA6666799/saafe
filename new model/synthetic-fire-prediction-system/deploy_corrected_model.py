#!/usr/bin/env python3
"""
Deploy the corrected FLIR+SCD41 fire detection model to a SageMaker endpoint
"""

import boto3
import datetime
import time

def deploy_model():
    """Deploy the trained model to a SageMaker endpoint"""
    print("🚀 Deploying corrected FLIR+SCD41 fire detection model to SageMaker endpoint...")
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    # Generate unique timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f'flir-scd41-fire-detection-corrected-{timestamp}'
    endpoint_config_name = f'flir-scd41-fire-detection-corrected-config-{timestamp}'
    endpoint_name = f'flir-scd41-fire-detection-corrected-{timestamp}'
    
    try:
        # Create model
        print(f"📦 Creating model: {model_name}")
        model_response = sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
                'ModelDataUrl': 's3://fire-detection-training-691595239825/flir_scd41_training/models/flir-scd41-full-ensemble-fixed2-20250901-103338/output/model.tar.gz',
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'serve',
                    'SAGEMAKER_SUBMIT_DIRECTORY': 's3://fire-detection-training-691595239825/flir_scd41_training/code/corrected_code.tar.gz'
                }
            },
            ExecutionRoleArn='arn:aws:iam::691595239825:role/SageMakerExecutionRole'
        )
        print(f"✅ Model created successfully: {model_response['ModelArn']}")
        
        # Create endpoint configuration
        print(f"⚙️  Creating endpoint configuration: {endpoint_config_name}")
        config_response = sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large'
                }
            ]
        )
        print(f"✅ Endpoint configuration created successfully: {config_response['EndpointConfigArn']}")
        
        # Create endpoint
        print(f"🌐 Creating endpoint: {endpoint_name}")
        endpoint_response = sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"✅ Endpoint creation initiated successfully: {endpoint_response['EndpointArn']}")
        
        print("\n📋 Deployment Summary:")
        print(f"  Model Name: {model_name}")
        print(f"  Endpoint Config: {endpoint_config_name}")
        print(f"  Endpoint Name: {endpoint_name}")
        print(f"  Model Data URL: s3://fire-detection-training-691595239825/flir_scd41_training/models/flir-scd41-full-ensemble-fixed2-20250901-103338/output/model.tar.gz")
        print(f"  Code Package URL: s3://fire-detection-training-691595239825/flir_scd41_training/code/corrected_code.tar.gz")
        
        print(f"\n⏳ Endpoint deployment in progress...")
        print(f"   You can monitor the deployment status with:")
        print(f"   aws sagemaker describe-endpoint --endpoint-name {endpoint_name} --region us-east-1")
        
        # Wait for endpoint to be in service
        print(f"\n⏳ Waiting for endpoint to be in service...")
        while True:
            try:
                response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                print(f"   Current status: {status}")
                
                if status == 'InService':
                    print("✅ Endpoint is now in service!")
                    break
                elif status in ['Failed', 'OutOfService']:
                    print(f"❌ Endpoint failed with status: {status}")
                    break
                else:
                    print("   Waiting for endpoint to be ready...")
                    time.sleep(30)  # Wait 30 seconds before checking again
            except Exception as e:
                print(f"❌ Error checking endpoint status: {e}")
                break
        
        print(f"\n🚀 Once deployed, you can make predictions using:")
        print(f"   aws sagemaker-runtime invoke-endpoint \\")
        print(f"     --endpoint-name {endpoint_name} \\")
        print(f"     --body '{{\"t_mean\": 30.5, \"t_std\": 5.2, \"t_max\": 45.0, ...}}' \\")
        print(f"     --content-type 'application/json' \\")
        print(f"     output.json --region us-east-1")
        
        return endpoint_name
        
    except Exception as e:
        print(f"❌ Error during deployment: {e}")
        raise

def main():
    """Main function"""
    print("FLIR+SCD41 Fire Detection - Corrected Model Deployment")
    print("=" * 55)
    
    try:
        endpoint_name = deploy_model()
        print(f"\n🎉 Deployment completed successfully!")
        print(f"   Endpoint name: {endpoint_name}")
        
    except Exception as e:
        print(f"💥 Deployment failed: {e}")
        raise

if __name__ == "__main__":
    main()