#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - AWS 100K Training Pipeline
This script implements a complete end-to-end training pipeline using AWS resources for 100K+ samples.
"""

import os
import sys
import json
import boto3
import logging
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tarfile
import tempfile

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET = 'fire-detection-training-691595239825'
S3_PREFIX = 'flir_scd41_training'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AWS100KTrainingPipeline:
    def __init__(self, bucket_name=S3_BUCKET, prefix=S3_PREFIX):
        """Initialize the AWS 100K training pipeline."""
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
        self.sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
        
        # Create S3 prefix directories if they don't exist
        self._ensure_s3_directories()
        
    def _ensure_s3_directories(self):
        """Ensure required S3 directories exist."""
        directories = [
            f"{self.prefix}/data/",
            f"{self.prefix}/models/",
            f"{self.prefix}/results/",
            f"{self.prefix}/logs/",
            f"{self.prefix}/code/"
        ]
        
        for directory in directories:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=directory
                )
                logger.info(f"Ensured S3 directory exists: {directory}")
            except Exception as e:
                logger.error(f"Error creating S3 directory {directory}: {e}")
    
    def generate_synthetic_data_aws(self, num_samples=100000):
        """Generate 100K synthetic training data and upload to S3."""
        logger.info(f"Generating 100K synthetic data with {num_samples} samples")
        
        try:
            # Generate FLIR features (15 features) with more realistic distributions
            flir_features = np.zeros((num_samples, 15))
            
            # t_mean: Mean temperature (-40 to 330°C)
            flir_features[:, 0] = np.random.normal(25, 15, num_samples)
            flir_features[:, 0] = np.clip(flir_features[:, 0], -40, 330)
            
            # t_std: Temperature standard deviation (0 to 50°C)
            flir_features[:, 1] = np.random.gamma(2, 5, num_samples)
            flir_features[:, 1] = np.clip(flir_features[:, 1], 0, 50)
            
            # t_max: Maximum temperature (-40 to 330°C)
            flir_features[:, 2] = np.random.normal(45, 20, num_samples)
            flir_features[:, 2] = np.clip(flir_features[:, 2], -40, 330)
            
            # t_p95: 95th percentile temperature (-40 to 330°C)
            flir_features[:, 3] = np.random.normal(40, 18, num_samples)
            flir_features[:, 3] = np.clip(flir_features[:, 3], -40, 330)
            
            # t_hot_area_pct: Percentage of hot area (0-100%)
            flir_features[:, 4] = np.random.beta(2, 5, num_samples) * 100
            
            # t_hot_largest_blob_pct: Percentage of largest hot blob (0-50%)
            flir_features[:, 5] = np.random.beta(1, 4, num_samples) * 50
            
            # t_grad_mean: Mean temperature gradient (0-20)
            flir_features[:, 6] = np.random.gamma(2, 3, num_samples)
            flir_features[:, 6] = np.clip(flir_features[:, 6], 0, 20)
            
            # t_grad_std: Std of temperature gradient (0-10)
            flir_features[:, 7] = np.random.gamma(1, 2, num_samples)
            flir_features[:, 7] = np.clip(flir_features[:, 7], 0, 10)
            
            # t_diff_mean: Mean temperature difference (0-30)
            flir_features[:, 8] = np.random.gamma(2, 4, num_samples)
            flir_features[:, 8] = np.clip(flir_features[:, 8], 0, 30)
            
            # t_diff_std: Std of temperature difference (0-15)
            flir_features[:, 9] = np.random.gamma(1, 3, num_samples)
            flir_features[:, 9] = np.clip(flir_features[:, 9], 0, 15)
            
            # flow_mag_mean: Mean flow magnitude (0-15)
            flir_features[:, 10] = np.random.gamma(2, 2, num_samples)
            flir_features[:, 10] = np.clip(flir_features[:, 10], 0, 15)
            
            # flow_mag_std: Std of flow magnitude (0-8)
            flir_features[:, 11] = np.random.gamma(1, 1.5, num_samples)
            flir_features[:, 11] = np.clip(flir_features[:, 11], 0, 8)
            
            # tproxy_val: Temperature proxy value (0-100)
            flir_features[:, 12] = np.random.normal(30, 15, num_samples)
            flir_features[:, 12] = np.clip(flir_features[:, 12], 0, 100)
            
            # tproxy_delta: Temperature proxy delta (-50 to 50)
            flir_features[:, 13] = np.random.normal(0, 10, num_samples)
            flir_features[:, 13] = np.clip(flir_features[:, 13], -50, 50)
            
            # tproxy_vel: Temperature proxy velocity (-20 to 20)
            flir_features[:, 14] = np.random.normal(0, 5, num_samples)
            flir_features[:, 14] = np.clip(flir_features[:, 14], -20, 20)
            
            # Generate SCD41 features (3 features) with realistic distributions
            scd41_features = np.zeros((num_samples, 3))
            
            # gas_val: CO2 concentration (400-5000 ppm)
            scd41_features[:, 0] = np.random.normal(450, 200, num_samples)
            scd41_features[:, 0] = np.clip(scd41_features[:, 0], 400, 5000)
            
            # gas_delta: CO2 change rate (-500 to 500 ppm/min)
            scd41_features[:, 1] = np.random.normal(0, 100, num_samples)
            scd41_features[:, 1] = np.clip(scd41_features[:, 1], -500, 500)
            
            # gas_vel: CO2 velocity (-50 to 50 ppm/s)
            scd41_features[:, 2] = np.random.normal(0, 15, num_samples)
            scd41_features[:, 2] = np.clip(scd41_features[:, 2], -50, 50)
            
            # Combine all features (15 FLIR + 3 SCD41 = 18 features)
            all_features = np.concatenate([flir_features, scd41_features], axis=1)
            
            # Create more sophisticated fire detection logic
            # Fire probability based on multiple interacting factors
            fire_indicators = np.zeros(len(all_features))
            
            # High temperature indicators (weighted more heavily)
            fire_indicators += (flir_features[:, 2] > 60) * 0.3  # High max temperature
            fire_indicators += (flir_features[:, 3] > 55) * 0.25  # High 95th percentile
            fire_indicators += (flir_features[:, 0] > 40) * 0.2   # High mean temperature
            
            # Large hot area indicators
            fire_indicators += (flir_features[:, 4] > 15) * 0.2   # Large hot area
            fire_indicators += (flir_features[:, 5] > 5) * 0.15   # Large hot blob
            
            # Rapid temperature changes
            fire_indicators += (flir_features[:, 6] > 8) * 0.15   # High temperature gradient
            fire_indicators += (flir_features[:, 8] > 12) * 0.15  # High temperature difference
            
            # Elevated CO2 levels
            fire_indicators += (scd41_features[:, 0] > 800) * 0.25  # High CO2
            fire_indicators += (scd41_features[:, 1] > 150) * 0.2   # Rapid CO2 increase
            
            # Interaction effects (more realistic fire signatures)
            # High temperature + high CO2
            temp_co2_interaction = ((flir_features[:, 2] > 70) & (scd41_features[:, 0] > 1000)).astype(int) * 0.3
            fire_indicators += temp_co2_interaction
            
            # Large hot area + rapid temperature change
            area_gradient_interaction = ((flir_features[:, 4] > 20) & (flir_features[:, 6] > 10)).astype(int) * 0.2
            fire_indicators += area_gradient_interaction
            
            # Clip probabilities to [0, 1]
            fire_indicators = np.clip(fire_indicators, 0, 1)
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.05, len(fire_indicators))
            fire_probability = np.clip(fire_indicators + noise, 0, 1)
            
            # Generate labels based on fire probability
            labels = np.random.binomial(1, fire_probability)
            
            # Create DataFrame
            feature_names = [
                't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
                't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
                't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
                'tproxy_val', 'tproxy_delta', 'tproxy_vel',
                'gas_val', 'gas_delta', 'gas_vel'
            ]
            
            df = pd.DataFrame(all_features, columns=feature_names)
            df['fire_detected'] = labels
            
            logger.info(f"✅ Dataset created with shape: {df.shape}")
            logger.info(f"Fire samples: {sum(labels):,} ({sum(labels)/len(labels)*100:.2f}%)")
            
            # Convert to JSON format to match working example
            samples = []
            for _, row in df.iterrows():
                sample = {
                    "features": {col: float(row[col]) for col in feature_names},
                    "label": int(row['fire_detected'])
                }
                samples.append(sample)
            
            json_data = {"samples": samples}
            
            # Save to local file first
            local_file_path = f"/tmp/flir_scd41_data_{num_samples}.json"
            with open(local_file_path, 'w') as f:
                json.dump(json_data, f)
            logger.info(f"Saved synthetic data to {local_file_path}")
            
            # Upload to S3
            s3_key = f"{self.prefix}/data/flir_scd41_data_{num_samples}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded synthetic data to s3://{self.bucket_name}/{s3_key}")
            
            return s3_key
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def package_training_code(self):
        """Package training code into a tar.gz file for SageMaker."""
        logger.info("Packaging training code for SageMaker...")
        
        # Create a temporary directory for our code
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy our new ensemble training and inference scripts
            os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_sagemaker_training_100k_ensemble.py {temp_dir}/train")
            os.system(f"cp /Volumes/Ajay/saafe\\ copy\\ 3/new\\ model/synthetic-fire-prediction-system/flir_scd41_inference_100k_ensemble.py {temp_dir}/serve")
            
            # Make scripts executable
            os.chmod(f"{temp_dir}/train", 0o755)
            os.chmod(f"{temp_dir}/serve", 0o755)
            
            # Verify files exist before packaging
            if not os.path.exists(f"{temp_dir}/train"):
                raise FileNotFoundError(f"Training script not created at {temp_dir}/train")
            if not os.path.exists(f"{temp_dir}/serve"):
                raise FileNotFoundError(f"Inference script not created at {temp_dir}/serve")
            
            # Create tar.gz file
            code_tar_path = f"{temp_dir}/code_100k_{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
            with tarfile.open(code_tar_path, "w:gz") as tar:
                # Add files with correct names for SageMaker
                tar.add(f"{temp_dir}/train", arcname="train")
                tar.add(f"{temp_dir}/serve", arcname="serve")
            
            # Verify tar file was created
            if not os.path.exists(code_tar_path):
                raise FileNotFoundError(f"Tar file not created at {code_tar_path}")
            
            # Debug: List contents of tar file
            with tarfile.open(code_tar_path, "r:gz") as tar:
                members = tar.getnames()
                logger.info(f"Tar file contents: {members}")
            
            # Upload to S3
            s3_key = f"{self.prefix}/code/code_100k_{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
            self.s3_client.upload_file(code_tar_path, self.bucket_name, s3_key)
            
            logger.info(f"Code packaged and uploaded to s3://{self.bucket_name}/{s3_key}")
            return f"s3://{self.bucket_name}/{s3_key}"
    
    def create_training_job(self, job_name, instance_type='ml.m5.4xlarge'):
        """Create a SageMaker training job for 100K samples."""
        logger.info(f"Creating SageMaker training job: {job_name}")
        
        # Package code
        code_s3_uri = self.package_training_code()
        
        try:
            # Create training job
            response = self.sagemaker_client.create_training_job(
                TrainingJobName=job_name,
                RoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole",
                AlgorithmSpecification={
                    'TrainingInputMode': 'File',
                    'TrainingImage': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3'  # Changed back to 0.23-1 as in the working example
                },
                InputDataConfig=[
                    {
                        'ChannelName': 'training',
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': f's3://{self.bucket_name}/{self.prefix}/data/',
                                'S3DataDistributionType': 'FullyReplicated'
                            }
                        },
                        'ContentType': 'application/json',
                        'CompressionType': 'None'
                    }
                ],
                OutputDataConfig={
                    'S3OutputPath': f's3://{self.bucket_name}/{self.prefix}/models/'
                },
                ResourceConfig={
                    'InstanceType': instance_type,
                    'InstanceCount': 1,
                    'VolumeSizeInGB': 100  # Larger volume for 100K samples
                },
                StoppingCondition={
                    'MaxRuntimeInSeconds': 14400  # 4 hours for 100K samples
                },
                HyperParameters={
                    'sagemaker_program': 'train',
                    'sagemaker_submit_directory': code_s3_uri  # Added this to match successful jobs
                }
            )
            
            logger.info(f"Training job created successfully: {job_name}")
            logger.info(f"Training job ARN: {response['TrainingJobArn']}")
            return job_name
            
        except Exception as e:
            logger.error(f"Error creating training job: {e}")
            raise
    
    def create_ensemble_training_jobs(self):
        """Create multiple training jobs for ensemble methods."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        jobs = []
        
        # Random Forest training job
        rf_job_name = f"flir-scd41-rf-100k-{timestamp}"
        rf_job = self.create_training_job(rf_job_name, 'ml.m5.4xlarge')
        jobs.append(('Random Forest', rf_job))
        
        # Gradient Boosting training job
        gb_job_name = f"flir-scd41-gb-100k-{timestamp}"
        gb_job = self.create_training_job(gb_job_name, 'ml.m5.4xlarge')
        jobs.append(('Gradient Boosting', gb_job))
        
        # Logistic Regression training job
        lr_job_name = f"flir-scd41-lr-100k-{timestamp}"
        lr_job = self.create_training_job(lr_job_name, 'ml.m5.4xlarge')
        jobs.append(('Logistic Regression', lr_job))
        
        return jobs
    
    def monitor_training_jobs(self, job_names):
        """Monitor the status of training jobs."""
        logger.info("Monitoring training jobs...")
        
        completed_jobs = []
        failed_jobs = []
        
        for job_name in job_names:
            try:
                response = self.sagemaker_client.describe_training_job(
                    TrainingJobName=job_name
                )
                
                status = response['TrainingJobStatus']
                logger.info(f"{job_name}: {status}")
                
                if status == 'Completed':
                    completed_jobs.append(job_name)
                elif status == 'Failed':
                    failed_jobs.append(job_name)
                    failure_reason = response.get('FailureReason', 'Unknown')
                    logger.error(f"{job_name} failed: {failure_reason}")
                    
            except Exception as e:
                logger.error(f"Error monitoring {job_name}: {e}")
        
        return completed_jobs, failed_jobs
    
    def deploy_model(self, model_name, endpoint_name):
        """Deploy a trained model to a SageMaker endpoint."""
        logger.info(f"Deploying model {model_name} to endpoint {endpoint_name}")
        
        try:
            # Create model
            model_response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3',
                    'ModelDataUrl': f's3://{self.bucket_name}/{self.prefix}/models/{model_name}/output/model.tar.gz',
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'serve'
                    }
                },
                ExecutionRoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole"
            )
            
            # Create endpoint config
            config_name = f"{endpoint_name}-config"
            config_response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': model_name,
                        'InstanceType': 'ml.t2.medium',
                        'InitialInstanceCount': 1
                    }
                ]
            )
            
            # Create endpoint
            endpoint_response = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            
            logger.info(f"Endpoint creation initiated: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    def test_endpoint(self, endpoint_name, test_data):
        """Test the deployed endpoint with sample data."""
        logger.info(f"Testing endpoint {endpoint_name}")
        
        try:
            # Format test data as CSV
            if isinstance(test_data, list):
                test_csv = ','.join([str(x) for x in test_data])
            else:
                test_csv = test_data
            
            # Make prediction
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='text/csv',
                Body=test_csv
            )
            
            result = response['Body'].read().decode()
            logger.info(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error testing endpoint: {e}")
            raise

def main():
    """Main function to start the 100K training pipeline."""
    print("🔥 FLIR+SCD41 Fire Detection System - AWS 100K Training Pipeline")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = AWS100KTrainingPipeline()
    
    try:
        # Step 1: Generate 100K synthetic data
        print("\\n🔄 Step 1: Generating 100K synthetic data...")
        data_s3_key = pipeline.generate_synthetic_data_aws(num_samples=100000)
        print(f"✅ Data generated and uploaded to {data_s3_key}")
        
        # Step 2: Create training jobs
        print("\\n🚀 Step 2: Creating SageMaker training jobs...")
        jobs = pipeline.create_ensemble_training_jobs()
        
        print("\\n" + "=" * 70)
        print("TRAINING JOBS CREATED SUCCESSFULLY")
        print("=" * 70)
        for model_type, job_name in jobs:
            print(f"{model_type} job: {job_name}")
        
        print("\\n📋 To monitor training progress, use:")
        for model_type, job_name in jobs:
            print(f"aws sagemaker describe-training-job --training-job-name {job_name} --region {AWS_REGION}")
        
        print("\\nOr use the monitoring script:")
        print("python monitor_training_jobs.py")
        
        # Step 3: Provide instructions for next steps
        print("\\n🎯 Next steps:")
        print("1. Monitor training jobs until completion")
        print("2. Deploy the best performing model")
        print("3. Test the deployed endpoint")
        print("4. Validate model performance with test data")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)