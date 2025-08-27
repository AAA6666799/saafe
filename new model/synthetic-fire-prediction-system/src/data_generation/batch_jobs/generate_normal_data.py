"""
AWS Batch job script for generating normal operation data.

This script is designed to be run as an AWS Batch job to generate
normal operation data for the synthetic fire prediction system.
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime

# Add parent directory to path to import from parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_generation.dataset_generator import DatasetGenerator
from aws.s3.service import S3ServiceImpl


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate normal operation data')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--num-hours', type=int, default=1000, help='Number of hours of data to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--s3-bucket', help='S3 bucket for storing results')
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting normal data generation: {args.num_hours} hours")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure dataset generator
    config = {
        'output_dir': args.output_dir,
        'num_normal_hours': args.num_hours,
        'scenario_duration': 3600,  # 1 hour in seconds
        'sample_rate': 1.0,  # 1 Hz
        'log_level': 'INFO'
    }
    
    # Add AWS integration if S3 bucket is provided
    if args.s3_bucket:
        config['aws_integration'] = True
        config['aws_config'] = {
            'default_bucket': args.s3_bucket
        }
    
    # Initialize dataset generator
    generator = DatasetGenerator(config)
    
    # Generate normal operation data
    result = generator.generate_normal_operation_data(
        num_hours=args.num_hours,
        output_dir=args.output_dir,
        parallel=True,
        seed=args.seed
    )
    
    # Save job result
    result_path = os.path.join(args.output_dir, 'job_result.json')
    with open(result_path, 'w') as f:
        json.dump({
            'job_type': 'normal_data_generation',
            'num_hours': args.num_hours,
            'output_dir': args.output_dir,
            'completion_time': datetime.now().isoformat(),
            'status': 'completed',
            'scenarios_generated': len(result.get('scenarios', []))
        }, f, indent=2)
    
    # Upload job result to S3 if configured
    if args.s3_bucket:
        s3_service = S3ServiceImpl({'default_bucket': args.s3_bucket})
        s3_key = f"batch_jobs/normal_data/{os.path.basename(args.output_dir)}_result.json"
        s3_service.upload_file(result_path, s3_key)
        logger.info(f"Uploaded job result to S3: {s3_key}")
    
    logger.info(f"Normal data generation complete: {len(result.get('scenarios', []))} scenarios")


if __name__ == '__main__':
    main()