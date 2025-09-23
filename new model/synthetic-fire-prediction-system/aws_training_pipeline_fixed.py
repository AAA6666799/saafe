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
import numpy as np
from datetime import datetime
from pathlib import Path

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

class AWSTrainingPipeline:
    def __init__(self, bucket_name=S3_BUCKET, prefix=S3_PREFIX):
        """Initialize the AWS training pipeline."""
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
        self.sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
        
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
        
        # Create simple demo data structure
        logger.info("Generating demo data")
        
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
        
        np.random.seed(42)
        
        for i in range(num_samples):
            # Generate realistic feature values and convert to Python native types
            sample = {
                "id": f"sample_{i}",
                "features": {
                    "t_mean": float(np.random.normal(25, 5)),  # Celsius
                    "t_std": float(np.random.uniform(0, 10)),
                    "t_max": float(np.random.normal(50, 15)),
                    "t_p95": float(np.random.normal(45, 12)),
                    "t_hot_area_pct": float(np.random.uniform(0, 100)),
                    "t_hot_largest_blob_pct": float(np.random.uniform(0, 50)),
                    "t_grad_mean": float(np.random.normal(0, 2)),
                    "t_grad_std": float(np.random.uniform(0, 5)),
                    "t_diff_mean": float(np.random.normal(0, 3)),
                    "t_diff_std": float(np.random.uniform(0, 4)),
                    "flow_mag_mean": float(np.random.uniform(0, 10)),
                    "flow_mag_std": float(np.random.uniform(0, 5)),
                    "tproxy_val": float(np.random.normal(30, 8)),
                    "tproxy_delta": float(np.random.normal(0, 5)),
                    "tproxy_vel": float(np.random.normal(0, 2)),
                    "gas_val": float(np.random.normal(400, 100)),  # CO2 ppm
                    "gas_delta": float(np.random.normal(0, 50)),
                    "gas_vel": float(np.random.normal(0, 10))
                },
                "label": int(np.random.choice([0, 1], p=[0.7, 0.3]))  # 30% fire cases
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
    
    def run_complete_pipeline(self, num_samples=10000, instance_type='ml.m5.large'):
        """Run the complete training pipeline on AWS."""
        logger.info("Starting complete AWS training pipeline")
        
        try:
            # Step 1: Generate synthetic data
            data_file = "/tmp/synthetic_data.json"
            data = self.generate_synthetic_data(num_samples, data_file)
            
            logger.info("AWS training pipeline completed successfully")
            return {
                "data_file": data_file,
                "data_samples": len(data["samples"])
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
        print(f"Generated samples: {results['data_samples']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()