#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Deploy 100K Trained Model
This script deploys trained models from SageMaker training jobs to endpoints.
"""

import boto3
import argparse
import sys
import time
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'

def deploy_model(model_s3_uri, model_name, instance_type='ml.t2.medium'):
    """
    Deploy a trained model to a SageMaker endpoint.
    
    Args:
        model_s3_uri (str): S3 URI of the trained model artifacts
        model_name (str): Name for the SageMaker model
        instance_type (str): EC2 instance type for the endpoint
    
    Returns:
        str: Name of the created endpoint
    """
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    print(f"🚀 Deploying model from {model_s3_uri}")
    print(f"🏷️  Model name: {model_name}")
    print(f"🖥️  Instance type: {instance_type}")
    
    try:
        # Create model
        print("\\n📦 Creating SageMaker model...")
        model_response = sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3',
                'ModelDataUrl': model_s3_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'serve'
                }
            },
            ExecutionRoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole"
        )
        
        print(f"✅ Model created: {model_response['ModelArn']}")
        
        # Create endpoint configuration
        print("\\n⚙️  Creating endpoint configuration...")
        config_name = f"{model_name}-config"
        config_response = sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': 1
                }
            ]
        )
        
        print(f"✅ Endpoint configuration created: {config_name}")
        
        # Create endpoint
        endpoint_name = f"{model_name}-endpoint"
        print(f"\\n🌐 Creating endpoint: {endpoint_name}")
        endpoint_response = sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        print(f"✅ Endpoint creation initiated: {endpoint_name}")
        print(f"🔗 Endpoint ARN: {endpoint_response['EndpointArn']}")
        
        return endpoint_name
        
    except Exception as e:
        print(f"❌ Error deploying model: {e}")
        raise

def deploy_ensemble_model(model_name_prefix="flir-scd41-ensemble"):
    """
    Deploy an ensemble model using multiple trained models.
    
    Args:
        model_name_prefix (str): Prefix for the ensemble model name
    
    Returns:
        str: Name of the created endpoint
    """
    # For ensemble deployment, we would typically:
    # 1. Combine multiple model artifacts
    # 2. Create a custom inference container that loads all models
    # 3. Implement ensemble logic in the inference script
    
    # This is a simplified version - in practice, you'd need to:
    # - Download all model artifacts
    # - Package them together
    # - Create a custom inference script that loads all models
    # - Deploy the combined model
    
    print("🚧 Ensemble deployment is a complex process that requires:")
    print("   1. Combining multiple model artifacts")
    print("   2. Creating a custom inference container")
    print("   3. Implementing ensemble logic")
    print("   4. Packaging and deploying")
    print("\\n📝 For now, use the single model deployment option.")
    
    return None

def wait_for_endpoint(endpoint_name, wait_time=600):
    """
    Wait for an endpoint to be in service.
    
    Args:
        endpoint_name (str): Name of the endpoint to wait for
        wait_time (int): Maximum time to wait in seconds
    
    Returns:
        bool: True if endpoint is in service, False otherwise
    """
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    
    print(f"\\n⏳ Waiting for endpoint {endpoint_name} to be in service...")
    print(f"⏰ Maximum wait time: {wait_time} seconds")
    
    start_time = time.time()
    
    while time.time() - start_time < wait_time:
        try:
            response = sagemaker.describe_endpoint(
                EndpointName=endpoint_name
            )
            
            status = response['EndpointStatus']
            print(f"   {datetime.now().strftime('%H:%M:%S')} - Status: {status}")
            
            if status == 'InService':
                print("✅ Endpoint is now in service!")
                return True
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"❌ Endpoint failed: {failure_reason}")
                return False
            elif status in ['Creating', 'Updating']:
                # Continue waiting
                time.sleep(30)
                continue
            else:
                print(f"⚠️  Unexpected endpoint status: {status}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking endpoint status: {e}")
            time.sleep(30)
    
    print(f"⏰ Wait time exceeded ({wait_time} seconds)")
    return False

def test_endpoint(endpoint_name, test_data=None):
    """
    Test the deployed endpoint with sample data.
    
    Args:
        endpoint_name (str): Name of the endpoint to test
        test_data (list): Sample data for testing (18 features)
    """
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    # Default test data representing normal conditions
    if test_data is None:
        test_data = [
            25.0,   # t_mean
            5.0,    # t_std
            45.0,   # t_max
            40.0,   # t_p95
            10.0,   # t_hot_area_pct
            5.0,    # t_hot_largest_blob_pct
            2.0,    # t_grad_mean
            1.0,    # t_grad_std
            3.0,    # t_diff_mean
            1.5,    # t_diff_std
            5.0,    # flow_mag_mean
            2.0,    # flow_mag_std
            30.0,   # tproxy_val
            5.0,    # tproxy_delta
            1.0,    # tproxy_vel
            450.0,  # gas_val
            50.0,   # gas_delta
            5.0     # gas_vel
        ]
    
    print(f"\\n🧪 Testing endpoint {endpoint_name}...")
    
    try:
        # Format test data as CSV
        test_csv = ','.join([str(x) for x in test_data])
        
        # Make prediction
        print("   Sending test request...")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=test_csv
        )
        
        result = response['Body'].read().decode()
        print(f"✅ Prediction successful!")
        print(f"📊 Result: {result}")
        
        # Try to parse result as probability
        try:
            probability = float(result.strip().strip('[]'))
            print(f"🔥 Fire probability: {probability:.4f} ({probability*100:.2f}%)")
            
            if probability > 0.7:
                print("🚨 HIGH RISK: Strong indication of fire detected!")
            elif probability > 0.5:
                print("⚠️  MEDIUM RISK: Possible fire detected")
            elif probability > 0.3:
                print("🟡 LOW RISK: Unusual conditions")
            else:
                print("✅ NORMAL: No fire detected")
                
        except ValueError:
            print("ℹ️  Result format: Raw output from model")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        raise

def main():
    """Main function to deploy trained models."""
    parser = argparse.ArgumentParser(description='Deploy FLIR+SCD41 trained models')
    parser.add_argument('--model-uri', required=True, help='S3 URI of the trained model artifacts')
    parser.add_argument('--model-name', required=True, help='Name for the SageMaker model')
    parser.add_argument('--instance-type', default='ml.t2.medium', help='EC2 instance type (default: ml.t2.medium)')
    parser.add_argument('--wait', action='store_true', help='Wait for endpoint to be in service')
    parser.add_argument('--test', action='store_true', help='Test the deployed endpoint')
    parser.add_argument('--ensemble', action='store_true', help='Deploy as ensemble model (not implemented)')
    
    args = parser.parse_args()
    
    print("🔥 FLIR+SCD41 Fire Detection System - Model Deployment")
    print("=" * 55)
    
    try:
        if args.ensemble:
            # Ensemble deployment (not fully implemented)
            endpoint_name = deploy_ensemble_model(args.model_name)
        else:
            # Single model deployment
            endpoint_name = deploy_model(
                model_s3_uri=args.model_uri,
                model_name=args.model_name,
                instance_type=args.instance_type
            )
        
        if not endpoint_name:
            print("❌ Deployment failed or not completed")
            return 1
        
        print(f"\\n🌐 Deployment initiated for endpoint: {endpoint_name}")
        
        # Wait for endpoint to be in service
        if args.wait:
            if wait_for_endpoint(endpoint_name):
                print(f"✅ Endpoint {endpoint_name} is ready for use!")
                
                # Test endpoint if requested
                if args.test:
                    test_endpoint(endpoint_name)
            else:
                print(f"❌ Endpoint {endpoint_name} failed to become ready")
                return 1
        else:
            print(f"ℹ️  Endpoint {endpoint_name} is being created in the background")
            print(f"🔧 Use '--wait' flag to wait for completion")
            print(f"🔍 Monitor status with:")
            print(f"   aws sagemaker describe-endpoint --endpoint-name {endpoint_name} --region {AWS_REGION}")
        
        print(f"\\n🎉 Deployment process completed!")
        print(f"🔗 Endpoint name: {endpoint_name}")
        print(f"📡 Ready for predictions!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())