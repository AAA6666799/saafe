#!/usr/bin/env python3
"""
SageMaker Storage Optimization Configuration
Optimizes storage for 50M dataset fire detection training
"""

import boto3
import json

def get_optimal_storage_config():
    """Optimal storage configuration for 50M dataset training"""
    return {
        "instance_type": "ml.g5.2xlarge",  # Better than g5.xlarge for your needs
        "volume_size_gb": 500,  # Minimum for 50M dataset + processing
        "volume_type": "gp3",   # Best price/performance ratio
        "volume_config": {
            "throughput": 1000,  # Max for gp3 (1000 MiB/s)
            "iops": 16000       # Max for gp3 (16000 IOPS)
        },
        "alternative_high_performance": {
            "volume_type": "io1",
            "volume_size_gb": 500,
            "iops": 25000,      # Higher IOPS for extreme performance
            "cost_note": "~3x more expensive than gp3"
        }
    }

def create_sagemaker_notebook_config():
    """SageMaker notebook instance configuration"""
    return {
        "NotebookInstanceName": "fire-detection-50m-training",
        "InstanceType": "ml.g5.2xlarge",
        "RoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole",
        "VolumeSizeInGB": 500,
        "DefaultCodeRepository": "your-code-repo-url",
        "AdditionalModelDataUrl": "s3://your-bucket/models/",
        "Tags": [
            {"Key": "Project", "Value": "FireDetection50M"},
            {"Key": "CostCenter", "Value": "MLTraining"}
        ]
    }

def estimate_costs():
    """Cost estimation for different configurations"""
    return {
        "ml.g5.2xlarge": {
            "hourly_cost": "$1.624/hour",
            "daily_cost": "$38.98/day",
            "storage_gp3_500gb": "$50/month",
            "storage_io1_500gb": "$150/month"
        },
        "ml.p3.2xlarge": {
            "hourly_cost": "$3.825/hour", 
            "daily_cost": "$91.80/day",
            "note": "Your current instance - more expensive!"
        },
        "recommendations": {
            "training_time_estimate": "6-12 hours for 50M dataset",
            "total_cost_g5": "$10-20 for training + $1.67/day storage",
            "savings_vs_p3": "~60% cheaper than P3"
        }
    }

if __name__ == "__main__":
    config = get_optimal_storage_config()
    costs = estimate_costs()
    
    print("🚀 OPTIMAL SAGEMAKER CONFIGURATION FOR 50M FIRE DETECTION")
    print("=" * 60)
    print(f"💰 Recommended Instance: {config['instance_type']}")
    print(f"💾 Storage Size: {config['volume_size_gb']} GB")
    print(f"⚡ Storage Type: {config['volume_type']}")
    print(f"🔥 Throughput: {config['volume_config']['throughput']} MiB/s")
    print(f"📊 IOPS: {config['volume_config']['iops']:,}")
    
    print("\n💡 PERFORMANCE BENEFITS:")
    print("✅ 50% more GPU memory (24GB vs 16GB)")
    print("✅ 60% cost savings vs P3 instances") 
    print("✅ 1000 MiB/s storage throughput")
    print("✅ 16,000 IOPS for fast dataset loading")
    
    print(f"\n💰 COST ESTIMATES:")
    print(f"🏷️ Training Cost: $10-20 (6-12 hours)")
    print(f"💾 Storage Cost: $50/month (GP3 500GB)")
    print(f"📈 Total Monthly: ~$70 including storage")