#!/usr/bin/env python3
"""
AWS Training Options for Saafe Fire Detection Models.
Provides multiple training approaches using different AWS services.
"""

import boto3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSTrainingOptions:
    """Multiple AWS training options for different needs and budgets."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.session = boto3.Session(region_name=region)
        
        # AWS clients
        self.sagemaker = self.session.client('sagemaker')
        self.ec2 = self.session.client('ec2')
        self.s3 = self.session.client('s3')
        self.batch = self.session.client('batch')
        self.ecs = self.session.client('ecs')
        
        logger.info(f"AWS Training Options initialized for region: {region}")
    
    def get_training_options(self) -> Dict[str, Dict[str, Any]]:
        """Get all available training options with costs and specifications."""
        
        options = {
            'ultra_fast_gpu': {
                'name': '🚀 ULTRA-FAST GPU Training (MAXIMUM SPEED)',
                'description': 'Blazing fast training with 4x V100 GPUs - Cost no object!',
                'instance_type': 'ml.p3.8xlarge',
                'specs': {
                    'vcpus': 32,
                    'memory_gb': 244,
                    'gpu': '4x NVIDIA V100',
                    'gpu_memory_gb': 64
                },
                'estimated_cost_per_hour': 12.24,
                'estimated_training_time_hours': 0.5,  # 30 minutes!
                'total_estimated_cost': 6.12,
                'pros': [
                    'FASTEST possible training (30 minutes)',
                    '4x V100 GPUs for maximum parallel processing',
                    'Optimized model architecture',
                    'Mixed precision training',
                    'Production-ready in under 1 hour'
                ],
                'cons': [
                    'Higher cost per hour (but lower total cost due to speed)',
                    'Overkill for simple demos'
                ],
                'best_for': 'When speed is critical, production deployment, time-sensitive projects'
            },
            
            'sagemaker_gpu': {
                'name': 'SageMaker GPU Training (Recommended)',
                'description': 'High-performance GPU training with managed infrastructure',
                'instance_type': 'ml.p3.2xlarge',
                'specs': {
                    'vcpus': 8,
                    'memory_gb': 61,
                    'gpu': 'NVIDIA V100',
                    'gpu_memory_gb': 16
                },
                'estimated_cost_per_hour': 3.06,
                'estimated_training_time_hours': 1.5,
                'total_estimated_cost': 4.59,
                'pros': [
                    'Fast training time',
                    'Managed infrastructure',
                    'Auto-scaling',
                    'Built-in monitoring'
                ],
                'cons': [
                    'Higher cost per hour',
                    'GPU may be overkill for demo'
                ],
                'best_for': 'Production models, large datasets, balanced speed/cost'
            },
            
            'sagemaker_cpu': {
                'name': 'SageMaker CPU Training (Budget)',
                'description': 'Cost-effective CPU training with managed infrastructure',
                'instance_type': 'ml.m5.2xlarge',
                'specs': {
                    'vcpus': 8,
                    'memory_gb': 32,
                    'gpu': None,
                    'storage_gb': 'EBS optimized'
                },
                'estimated_cost_per_hour': 0.46,
                'estimated_training_time_hours': 3.0,
                'total_estimated_cost': 1.38,
                'pros': [
                    'Very cost-effective',
                    'Managed infrastructure',
                    'Good for smaller models'
                ],
                'cons': [
                    'Slower training',
                    'Limited by CPU performance'
                ],
                'best_for': 'Demo models, budget-conscious training, smaller datasets'
            },
            
            'ec2_spot_gpu': {
                'name': 'EC2 Spot GPU Training (Advanced)',
                'description': 'Up to 90% cost savings with spot instances',
                'instance_type': 'p3.2xlarge',
                'specs': {
                    'vcpus': 8,
                    'memory_gb': 61,
                    'gpu': 'NVIDIA V100',
                    'gpu_memory_gb': 16
                },
                'estimated_cost_per_hour': 0.31,  # Spot pricing
                'estimated_training_time_hours': 1.5,
                'total_estimated_cost': 0.47,
                'pros': [
                    'Massive cost savings (90%)',
                    'Same performance as on-demand',
                    'Full control over environment'
                ],
                'cons': [
                    'Can be interrupted',
                    'Requires setup and management',
                    'Need to handle interruptions'
                ],
                'best_for': 'Cost-sensitive production training, fault-tolerant workloads'
            },
            
            'batch_training': {
                'name': 'AWS Batch Training (Scalable)',
                'description': 'Containerized training with automatic scaling',
                'instance_type': 'c5.2xlarge',
                'specs': {
                    'vcpus': 8,
                    'memory_gb': 16,
                    'gpu': None,
                    'containers': 'Docker-based'
                },
                'estimated_cost_per_hour': 0.34,
                'estimated_training_time_hours': 2.5,
                'total_estimated_cost': 0.85,
                'pros': [
                    'Automatic scaling',
                    'Queue management',
                    'Container-based',
                    'Good for multiple experiments'
                ],
                'cons': [
                    'More complex setup',
                    'Longer startup time'
                ],
                'best_for': 'Multiple training runs, experimentation, batch processing'
            },
            
            'fargate_training': {
                'name': 'ECS Fargate Training (Serverless)',
                'description': 'Serverless container training',
                'instance_type': 'fargate',
                'specs': {
                    'vcpus': 4,
                    'memory_gb': 8,
                    'gpu': None,
                    'serverless': True
                },
                'estimated_cost_per_hour': 0.20,
                'estimated_training_time_hours': 4.0,
                'total_estimated_cost': 0.80,
                'pros': [
                    'No infrastructure management',
                    'Pay only for compute time',
                    'Automatic scaling'
                ],
                'cons': [
                    'Limited compute resources',
                    'Slower training',
                    'No GPU support'
                ],
                'best_for': 'Lightweight models, serverless architecture, minimal management'
            }
        }
        
        return options
    
    def display_training_options(self):
        """Display all training options in a user-friendly format."""
        
        options = self.get_training_options()
        
        print("\n" + "=" * 80)
        print("🚀 AWS TRAINING OPTIONS FOR SAAFE FIRE DETECTION MODELS")
        print("=" * 80)
        
        for i, (key, option) in enumerate(options.items(), 1):
            print(f"\n{i}. {option['name']}")
            print(f"   {option['description']}")
            print(f"   Instance: {option['instance_type']}")
            
            specs = option['specs']
            print(f"   Specs: {specs['vcpus']} vCPUs, {specs['memory_gb']}GB RAM", end="")
            if specs.get('gpu'):
                print(f", {specs['gpu']} GPU")
            else:
                print()
            
            print(f"   Cost: ${option['estimated_cost_per_hour']:.2f}/hour")
            print(f"   Training Time: ~{option['estimated_training_time_hours']} hours")
            print(f"   Total Cost: ~${option['total_estimated_cost']:.2f}")
            
            print(f"   Best for: {option['best_for']}")
            
            print(f"   Pros: {', '.join(option['pros'])}")
            print(f"   Cons: {', '.join(option['cons'])}")
        
        print("\n" + "=" * 80)
    
    def recommend_option(self, budget: float = None, urgency: str = 'medium', 
                        experience: str = 'beginner') -> str:
        """Recommend the best training option based on requirements."""
        
        options = self.get_training_options()
        
        if urgency == 'high' and (budget is None or budget > 4):
            return 'sagemaker_gpu'
        elif budget is not None and budget < 1:
            return 'ec2_spot_gpu' if experience == 'advanced' else 'fargate_training'
        elif budget is not None and budget < 2:
            return 'sagemaker_cpu'
        elif experience == 'beginner':
            return 'sagemaker_cpu'
        elif experience == 'advanced' and urgency == 'low':
            return 'ec2_spot_gpu'
        else:
            return 'sagemaker_gpu'
    
    def setup_sagemaker_training(self, instance_type: str = 'ml.p3.2xlarge') -> Dict[str, str]:
        """Set up SageMaker training job."""
        
        logger.info(f"Setting up SageMaker training with {instance_type}")
        
        # This would integrate with the main training pipeline
        setup_commands = {
            'install_dependencies': 'pip install boto3 sagemaker torch',
            'run_training': f'python aws_training_pipeline.py --instance-type {instance_type}',
            'monitor': 'aws sagemaker list-training-jobs --status-equals InProgress'
        }
        
        return setup_commands
    
    def setup_ec2_spot_training(self) -> Dict[str, str]:
        """Set up EC2 Spot instance training."""
        
        logger.info("Setting up EC2 Spot training")
        
        # Create launch template for spot instances
        user_data_script = '''#!/bin/bash
# Install Docker
yum update -y
yum install -y docker
service docker start
usermod -a -G docker ec2-user

# Install NVIDIA drivers for GPU instances
if lspci | grep -i nvidia; then
    yum install -y gcc kernel-devel-$(uname -r)
    wget https://developer.download.nvidia.com/compute/cuda/repos/amzn2/x86_64/cuda-repo-amzn2-10.2.89-1.x86_64.rpm
    rpm -i cuda-repo-amzn2-10.2.89-1.x86_64.rpm
    yum clean all
    yum install -y cuda
fi

# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | tee /etc/yum.repos.d/nvidia-docker.repo
yum install -y nvidia-docker2
systemctl restart docker

# Download and run training container
docker pull pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
docker run --gpus all -d --name saafe-training pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime sleep infinity

# Download training code from S3
aws s3 cp s3://your-bucket/training-code.tar.gz /tmp/
tar -xzf /tmp/training-code.tar.gz -C /tmp/

# Run training
docker exec saafe-training python /tmp/train.py
'''
        
        setup_commands = {
            'create_launch_template': 'aws ec2 create-launch-template --launch-template-name saafe-spot-training',
            'request_spot_instance': 'aws ec2 request-spot-instances --spot-price 0.50 --instance-count 1',
            'monitor_spot_requests': 'aws ec2 describe-spot-instance-requests'
        }
        
        return setup_commands
    
    def setup_batch_training(self) -> Dict[str, str]:
        """Set up AWS Batch training."""
        
        logger.info("Setting up AWS Batch training")
        
        # Batch job definition
        job_definition = {
            'jobDefinitionName': 'saafe-training-job',
            'type': 'container',
            'containerProperties': {
                'image': 'pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime',
                'vcpus': 8,
                'memory': 16384,
                'jobRoleArn': 'arn:aws:iam::account:role/BatchExecutionRole'
            }
        }
        
        setup_commands = {
            'create_compute_environment': 'aws batch create-compute-environment --compute-environment-name saafe-training',
            'create_job_queue': 'aws batch create-job-queue --job-queue-name saafe-queue',
            'submit_job': 'aws batch submit-job --job-name saafe-training --job-queue saafe-queue'
        }
        
        return setup_commands
    
    def estimate_costs(self, option_key: str, training_hours: float = None) -> Dict[str, float]:
        """Estimate detailed costs for a training option."""
        
        options = self.get_training_options()
        option = options.get(option_key)
        
        if not option:
            return {}
        
        hours = training_hours or option['estimated_training_time_hours']
        
        costs = {
            'compute_cost': option['estimated_cost_per_hour'] * hours,
            'storage_cost': 0.10,  # S3 storage for models/data
            'data_transfer_cost': 0.05,  # Minimal data transfer
            'total_cost': 0
        }
        
        costs['total_cost'] = sum(costs.values())
        
        return costs
    
    def create_training_budget_alert(self, budget_limit: float = 10.0) -> str:
        """Create CloudWatch billing alert for training costs."""
        
        cloudwatch = self.session.client('cloudwatch')
        sns = self.session.client('sns')
        
        # Create SNS topic for alerts
        topic_name = 'saafe-training-budget-alerts'
        
        try:
            response = sns.create_topic(Name=topic_name)
            topic_arn = response['TopicArn']
            
            # Create billing alarm
            alarm_name = 'SaafeTrainingBudgetAlert'
            
            cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName='EstimatedCharges',
                Namespace='AWS/Billing',
                Period=86400,  # 24 hours
                Statistic='Maximum',
                Threshold=budget_limit,
                ActionsEnabled=True,
                AlarmActions=[topic_arn],
                AlarmDescription=f'Alert when Saafe training costs exceed ${budget_limit}',
                Dimensions=[
                    {
                        'Name': 'Currency',
                        'Value': 'USD'
                    }
                ]
            )
            
            logger.info(f"Budget alert created: ${budget_limit} limit")
            return topic_arn
            
        except Exception as e:
            logger.error(f"Error creating budget alert: {e}")
            return ""


def interactive_training_setup():
    """Interactive setup for AWS training."""
    
    print("\n🚀 Welcome to Saafe AWS Training Setup!")
    print("=" * 50)
    
    # Initialize options
    aws_options = AWSTrainingOptions()
    
    # Display options
    aws_options.display_training_options()
    
    # Get user preferences
    print("\n📋 Let's find the best training option for you:")
    
    budget = input("What's your budget for training? (Enter amount in USD, or 'no limit'): ").strip()
    if budget.lower() in ['no limit', 'unlimited', '']:
        budget = None
    else:
        try:
            budget = float(budget)
        except ValueError:
            budget = None
    
    urgency = input("How urgent is this training? (low/medium/high): ").strip().lower()
    if urgency not in ['low', 'medium', 'high']:
        urgency = 'medium'
    
    experience = input("What's your AWS experience level? (beginner/intermediate/advanced): ").strip().lower()
    if experience not in ['beginner', 'intermediate', 'advanced']:
        experience = 'beginner'
    
    # Get recommendation
    recommended = aws_options.recommend_option(budget, urgency, experience)
    options = aws_options.get_training_options()
    
    print(f"\n🎯 RECOMMENDED OPTION: {options[recommended]['name']}")
    print(f"   Estimated Cost: ${options[recommended]['total_estimated_cost']:.2f}")
    print(f"   Training Time: ~{options[recommended]['estimated_training_time_hours']} hours")
    print(f"   Why: {options[recommended]['best_for']}")
    
    # Confirm choice
    proceed = input(f"\nProceed with {options[recommended]['name']}? (y/N): ").strip().lower()
    
    if proceed == 'y':
        print(f"\n✅ Setting up {options[recommended]['name']}...")
        
        if recommended.startswith('sagemaker'):
            commands = aws_options.setup_sagemaker_training(options[recommended]['instance_type'])
        elif recommended == 'ec2_spot_gpu':
            commands = aws_options.setup_ec2_spot_training()
        elif recommended == 'batch_training':
            commands = aws_options.setup_batch_training()
        else:
            commands = {'info': 'Setup instructions will be provided'}
        
        print("\n📝 Next Steps:")
        for step, command in commands.items():
            print(f"   {step}: {command}")
        
        # Budget alert
        if budget:
            alert_budget = budget * 1.2  # 20% buffer
            create_alert = input(f"\nCreate budget alert at ${alert_budget:.2f}? (y/N): ").strip().lower()
            if create_alert == 'y':
                topic_arn = aws_options.create_training_budget_alert(alert_budget)
                if topic_arn:
                    print(f"✅ Budget alert created!")
        
        print(f"\n🚀 Ready to start training!")
        print(f"Run: python aws_training_pipeline.py --option {recommended}")
        
    else:
        print("\n👍 No problem! Run this script again when you're ready.")
    
    print("\n" + "=" * 50)


def main():
    """Main function."""
    
    print("AWS Training Options for Saafe Fire Detection")
    print("=" * 50)
    
    choice = input("Choose an option:\n1. Interactive setup\n2. View all options\n3. Quick start (recommended)\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        interactive_training_setup()
    elif choice == '2':
        aws_options = AWSTrainingOptions()
        aws_options.display_training_options()
    elif choice == '3':
        print("\n🚀 Quick Start: SageMaker GPU Training")
        print("This is the recommended option for most users.")
        print("Estimated cost: ~$4.59")
        print("Training time: ~1.5 hours")
        print("\nTo start training:")
        print("python aws_training_pipeline.py")
    else:
        print("Invalid choice. Please run again.")


if __name__ == "__main__":
    main()