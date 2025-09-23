#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - AWS Training Pipeline
This script implements the complete end-to-end training pipeline using AWS resources.
"""

import os
import sys
import json
import boto3
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import local modules
try:
    from src.data_generation.synthetic_data_generator import SyntheticFireDataGenerator
    from src.feature_engineering.feature_extractor import FeatureExtractor
    from src.ml.model_trainer import ModelTrainer
    from src.ml.ensemble_manager import EnsembleManager
    HAS_LOCAL_MODULES = True
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")
    HAS_LOCAL_MODULES = False

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

class AWSTrainingPipeline:
    def __init__(self, bucket_name=S3_BUCKET, prefix=S3_PREFIX):
        """Initialize the AWS training pipeline."""
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
            f"{self.prefix}/logs/"
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
    
    def generate_synthetic_data(self, num_samples=10000, output_path=None):
        """Generate synthetic training data."""
        logger.info(f"Generating synthetic data with {num_samples} samples")
        
        if not HAS_LOCAL_MODULES:
            # Create demo data if local modules aren't available
            logger.warning("Using demo data generation instead of actual modules")
            return self._generate_demo_data(num_samples, output_path)
        
        try:
            # Initialize data generator
            generator = SyntheticFireDataGenerator()
            
            # Generate data
            data = generator.generate_dataset(num_samples)
            
            # Save to local file first
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f)
                logger.info(f"Saved synthetic data to {output_path}")
            
            # Upload to S3
            s3_key = f"{self.prefix}/data/synthetic_data_{num_samples}.json"
            self.s3_client.upload_file(output_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded synthetic data to s3://{self.bucket_name}/{s3_key}")
            
            return data
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            # Fallback to demo data
            return self._generate_demo_data(num_samples, output_path)
    
    def _generate_demo_data(self, num_samples, output_path):
        """Generate demo data when local modules aren't available."""
        logger.info("Generating demo data")
        
        # Create simple demo data structure
        import numpy as np
        np.random.seed(42)
        
        # FLIR+SCD41 features (18 total)
        feature_names = [
            "t_mean", "t_std", "t_max", "t_p95", "t_hot_area_pct", 
            "t_hot_largest_blob_pct", "t_grad_mean", "t_grad_std", 
            "t_diff_mean", "t_diff_std", "flow_mag_mean", "flow_mag_std",
            "tproxy_val", "tproxy_delta", "tproxy_vel",
            "gas_val", "gas_delta", "gas_vel"
        ]
        
        # Generate synthetic data
        data = {
            "features": feature_names,
            "samples": []
        }
        
        for i in range(num_samples):
            # Generate realistic feature values
            sample = {
                "id": f"sample_{i}",
                "features": {
                    "t_mean": np.random.normal(25, 5),  # Celsius
                    "t_std": np.random.uniform(0, 10),
                    "t_max": np.random.normal(50, 15),
                    "t_p95": np.random.normal(45, 12),
                    "t_hot_area_pct": np.random.uniform(0, 100),
                    "t_hot_largest_blob_pct": np.random.uniform(0, 50),
                    "t_grad_mean": np.random.normal(0, 2),
                    "t_grad_std": np.random.uniform(0, 5),
                    "t_diff_mean": np.random.normal(0, 3),
                    "t_diff_std": np.random.uniform(0, 4),
                    "flow_mag_mean": np.random.uniform(0, 10),
                    "flow_mag_std": np.random.uniform(0, 5),
                    "tproxy_val": np.random.normal(30, 8),
                    "tproxy_delta": np.random.normal(0, 5),
                    "tproxy_vel": np.random.normal(0, 2),
                    "gas_val": np.random.normal(400, 100),  # CO2 ppm
                    "gas_delta": np.random.normal(0, 50),
                    "gas_vel": np.random.normal(0, 10)
                },
                "label": np.random.choice([0, 1], p=[0.7, 0.3])  # 30% fire cases
            }
            data["samples"].append(sample)
        
        # Save to file
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved demo data to {output_path}")
        
        # Upload to S3
        s3_key = f"{self.prefix}/data/demo_data_{num_samples}.json"
        self.s3_client.upload_file(output_path, self.bucket_name, s3_key)
        logger.info(f"Uploaded demo data to s3://{self.bucket_name}/{s3_key}")
        
        return data
    
    def extract_features(self, data, output_path=None):
        """Extract features from raw data."""
        logger.info("Extracting features from data")
        
        if not HAS_LOCAL_MODULES:
            logger.warning("Skipping feature extraction - using raw data")
            return data
        
        try:
            # Initialize feature extractor
            extractor = FeatureExtractor()
            
            # Extract features
            features = extractor.extract_features(data)
            
            # Save features
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(features, f)
                logger.info(f"Saved features to {output_path}")
            
            # Upload to S3
            s3_key = f"{self.prefix}/data/features_{len(data.get('samples', []))}.json"
            self.s3_client.upload_file(output_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded features to s3://{self.bucket_name}/{s3_key}")
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return data
    
    def train_models_sagemaker(self, data_path, instance_type='ml.m5.large', max_jobs=3):
        """Train models using SageMaker."""
        logger.info(f"Starting SageMaker training with instance type: {instance_type}")
        
        try:
            # Upload training script to S3
            training_script = "sagemaker_training_script.py"
            s3_script_key = f"{self.prefix}/scripts/{training_script}"
            
            # Create a simple training script
            script_content = '''
import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()
    
    # Load data
    data_files = os.listdir(args.data-path)
    data_file = [f for f in data_files if f.endswith(".json")][0]
    with open(os.path.join(args.data-path, data_file), "r") as f:
        data = json.load(f)
    
    # Convert to DataFrame
    samples = data["samples"]
    df = pd.DataFrame([{
        **sample["features"],
        "label": sample["label"]
    } for sample in samples])
    
    # Prepare features and target
    X = df.drop("label", axis=1)
    y = df["label"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    os.makedirs(args.model-dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model-dir, "model.joblib"))
    
    # Save evaluation results
    with open(os.path.join(args.model-dir, "evaluation.json"), "w") as f:
        json.dump({
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }, f)
'''
            
            # Save script locally
            script_local_path = f"/tmp/{training_script}"
            with open(script_local_path, 'w') as f:
                f.write(script_content)
            
            # Upload to S3
            self.s3_client.upload_file(script_local_path, self.bucket_name, s3_script_key)
            logger.info(f"Uploaded training script to s3://{self.bucket_name}/{s3_script_key}")
            
            # Create SageMaker training jobs
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            training_jobs = []
            
            for i in range(min(max_jobs, 3)):  # Limit to 3 jobs
                job_name = f"flir-scd41-training-{timestamp}-{i}"
                
                # Create training job
                response = self.sagemaker_client.create_training_job(
                    TrainingJobName=job_name,
                    RoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole",  # Replace with actual role
                    AlgorithmSpecification={
                        'TrainingInputMode': 'File',
                        'TrainingImage': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3'
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
                            }
                        }
                    ],
                    OutputDataConfig={
                        'S3OutputPath': f's3://{self.bucket_name}/{self.prefix}/models/'
                    },
                    ResourceConfig={
                        'InstanceType': instance_type,
                        'InstanceCount': 1,
                        'VolumeSizeInGB': 30
                    },
                    StoppingCondition={
                        'MaxRuntimeInSeconds': 3600  # 1 hour
                    },
                    HyperParameters={
                        'sagemaker_program': training_script,
                        'sagemaker_submit_directory': f's3://{self.bucket_name}/{s3_script_key}'
                    }
                )
                
                training_jobs.append(job_name)
                logger.info(f"Started training job: {job_name}")
            
            return training_jobs
            
        except Exception as e:
            logger.error(f"Error starting SageMaker training: {e}")
            return []
    
    def create_ensemble_sagemaker(self, model_paths):
        """Create ensemble model using SageMaker."""
        logger.info("Creating ensemble model")
        
        try:
            # Upload ensemble script to S3
            ensemble_script = "ensemble_script.py"
            s3_script_key = f"{self.prefix}/scripts/{ensemble_script}"
            
            # Create ensemble script
            script_content = '''
import argparse
import json
import os
import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--ensemble-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()
    
    # Load individual models
    models = []
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith(".joblib")]
    
    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        model = joblib.load(model_path)
        models.append((model_file.replace(".joblib", ""), model))
    
    # Create ensemble
    if len(models) > 1:
        ensemble = VotingClassifier(estimators=models, voting="soft")
        # Note: In practice, you would need to fit the ensemble on data
        # For this example, we'll save the individual models as the ensemble
        ensemble_model = {
            "models": models,
            "weights": [1.0/len(models)] * len(models)  # Equal weights
        }
    else:
        ensemble_model = {
            "models": models,
            "weights": [1.0]
        }
    
    # Save ensemble
    os.makedirs(args.ensemble_dir, exist_ok=True)
    joblib.dump(ensemble_model, os.path.join(args.ensemble_dir, "ensemble_model.joblib"))
    
    # Save ensemble info
    ensemble_info = {
        "model_count": len(models),
        "model_names": [name for name, _ in models],
        "weights": ensemble_model["weights"],
        "created_at": str(pd.Timestamp.now())
    }
    
    with open(os.path.join(args.ensemble_dir, "ensemble_info.json"), "w") as f:
        json.dump(ensemble_info, f, indent=2)
'''
            
            # Save script locally
            script_local_path = f"/tmp/{ensemble_script}"
            with open(script_local_path, 'w') as f:
                f.write(script_content)
            
            # Upload to S3
            self.s3_client.upload_file(script_local_path, self.bucket_name, s3_script_key)
            logger.info(f"Uploaded ensemble script to s3://{self.bucket_name}/{s3_script_key}")
            
            # Create ensemble processing job
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            processing_job_name = f"flir-scd41-ensemble-{timestamp}"
            
            response = self.sagemaker_client.create_processing_job(
                ProcessingJobName=processing_job_name,
                ProcessingInputs=[
                    {
                        'InputName': 'models',
                        'S3Input': {
                            'S3Uri': f's3://{self.bucket_name}/{self.prefix}/models/',
                            'LocalPath': '/opt/ml/processing/models',
                            'S3DataType': 'S3Prefix',
                            'S3InputMode': 'File',
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                ],
                ProcessingOutputConfig={
                    'Outputs': [
                        {
                            'OutputName': 'ensemble',
                            'S3Output': {
                                'S3Uri': f's3://{self.bucket_name}/{self.prefix}/ensemble/',
                                'LocalPath': '/opt/ml/processing/ensemble',
                                'S3UploadMode': 'EndOfJob'
                            }
                        }
                    ]
                },
                ProcessingResources={
                    'ClusterConfig': {
                        'InstanceCount': 1,
                        'InstanceType': 'ml.m5.large',
                        'VolumeSizeInGB': 30
                    }
                },
                StoppingCondition={
                    'MaxRuntimeInSeconds': 1800  # 30 minutes
                },
                AppSpecification={
                    'ImageUri': f'683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3',
                    'ContainerEntrypoint': ['python3', f'/opt/ml/processing/scripts/{ensemble_script}']
                },
                RoleArn="arn:aws:iam::691595239825:role/SageMakerExecutionRole"  # Replace with actual role
            )
            
            logger.info(f"Started ensemble processing job: {processing_job_name}")
            return processing_job_name
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            return None
    
    def run_complete_pipeline(self, num_samples=10000, instance_type='ml.m5.large'):
        """Run the complete training pipeline on AWS."""
        logger.info("Starting complete AWS training pipeline")
        
        try:
            # Step 1: Generate synthetic data
            data_file = "/tmp/synthetic_data.json"
            data = self.generate_synthetic_data(num_samples, data_file)
            
            # Step 2: Extract features
            features_file = "/tmp/features.json"
            features = self.extract_features(data, features_file)
            
            # Step 3: Train models using SageMaker
            logger.info("Starting model training on SageMaker")
            training_jobs = self.train_models_sagemaker(data_file, instance_type)
            
            # Wait for training jobs to complete
            logger.info("Waiting for training jobs to complete...")
            completed_jobs = []
            for job_name in training_jobs:
                try:
                    waiter = self.sagemaker_client.get_waiter('training_job_completed_or_stopped')
                    waiter.wait(TrainingJobName=job_name, WaiterConfig={'Delay': 30, 'MaxAttempts': 60})
                    completed_jobs.append(job_name)
                    logger.info(f"Training job completed: {job_name}")
                except Exception as e:
                    logger.error(f"Training job failed: {job_name} - {e}")
            
            # Step 4: Create ensemble
            if completed_jobs:
                logger.info("Creating ensemble model")
                ensemble_job = self.create_ensemble_sagemaker([])
                logger.info(f"Ensemble job started: {ensemble_job}")
            
            logger.info("AWS training pipeline completed successfully")
            return {
                "data_file": data_file,
                "features_file": features_file,
                "training_jobs": completed_jobs,
                "ensemble_job": ensemble_job if 'ensemble_job' in locals() else None
            }
            
        except Exception as e:
            logger.error(f"Error in complete pipeline: {e}")
            raise

def main():
    """Main function to run the AWS training pipeline."""
    parser = argparse.ArgumentParser(description="FLIR+SCD41 Fire Detection - AWS Training Pipeline")
    parser.add_argument("--samples", type=int, default=10000, help="Number of synthetic samples to generate")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large", help="SageMaker instance type")
    parser.add_argument("--bucket", type=str, default=S3_BUCKET, help="S3 bucket for storage")
    parser.add_argument("--prefix", type=str, default=S3_PREFIX, help="S3 prefix for storage")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AWSTrainingPipeline(bucket_name=args.bucket, prefix=args.prefix)
    
    # Run complete pipeline
    try:
        results = pipeline.run_complete_pipeline(
            num_samples=args.samples,
            instance_type=args.instance_type
        )
        
        print("\n" + "="*50)
        print("AWS TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Data file: {results['data_file']}")
        print(f"Features file: {results['features_file']}")
        print(f"Training jobs: {len(results['training_jobs'])}")
        if results['ensemble_job']:
            print(f"Ensemble job: {results['ensemble_job']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()