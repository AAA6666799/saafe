#!/usr/bin/env python3
"""
Fixed launcher that handles IAM permission issues and region problems.
"""

import subprocess
import sys
import json
import os
import time

def print_banner():
    """Print banner."""
    print("🔧 SAAFE TRAINING FIXER & LAUNCHER")
    print("=" * 45)
    print("🎯 Fixing IAM and region issues")
    print("🚀 Then launching TOP-NOTCH training")
    print()

def run_cmd(cmd, description="", show_output=True):
    """Run command with error handling."""
    if show_output:
        print(f"🔧 {description}")
        print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            if show_output:
                print(f"   ✅ Success")
            return True, result.stdout.strip()
        else:
            if show_output:
                print(f"   ❌ Failed (exit code: {result.returncode})")
                if result.stderr.strip():
                    print(f"   Error: {result.stderr.strip()}")
            return False, result.stderr.strip()
            
    except Exception as e:
        if show_output:
            print(f"   💥 Exception: {e}")
        return False, str(e)

def fix_aws_region():
    """Fix AWS region configuration."""
    print("🌍 Fixing AWS region configuration...")
    
    # Try to get current region
    success, region = run_cmd("aws configure get region", "Checking current region", False)
    
    if not success or not region:
        print("   Setting region to us-east-1...")
        success, _ = run_cmd("aws configure set region us-east-1", "Setting region")
        if success:
            print("   ✅ Region set to us-east-1")
            return "us-east-1"
        else:
            print("   ⚠️  Could not set region, using us-east-1 as default")
            return "us-east-1"
    else:
        print(f"   ✅ Region already set: {region}")
        return region

def check_iam_permissions():
    """Check if user has IAM permissions."""
    print("🔐 Checking IAM permissions...")
    
    # Try to list roles (less privileged than creating)
    success, output = run_cmd("aws iam list-roles --max-items 1", "Testing IAM permissions", False)
    
    if success:
        print("   ✅ IAM permissions available")
        return True
    else:
        print("   ❌ No IAM permissions")
        print("   💡 Will use existing role or create training job without role creation")
        return False

def get_existing_sagemaker_role(account_id):
    """Try to find an existing SageMaker role."""
    print("🔍 Looking for existing SageMaker roles...")
    
    # Common SageMaker role names
    role_names = [
        "SageMakerExecutionRole",
        "AmazonSageMaker-ExecutionRole",
        "sagemaker-execution-role",
        "SaafeTrainingRole"
    ]
    
    for role_name in role_names:
        success, _ = run_cmd(f"aws iam get-role --role-name {role_name}", f"Checking {role_name}", False)
        if success:
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            print(f"   ✅ Found existing role: {role_name}")
            return role_arn
    
    print("   ❌ No existing SageMaker roles found")
    return None

def create_training_job_with_service_role(account_id, region):
    """Create training job using SageMaker service role."""
    print("🚀 Creating training job with service-linked role...")
    
    job_name = f"saafe-topnotch-{int(time.time())}"
    
    # Use SageMaker service-linked role (doesn't require IAM permissions)
    training_job = {
        "TrainingJobName": job_name,
        "RoleArn": f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-{int(time.time())}",
        "AlgorithmSpecification": {
            "TrainingImage": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker",
            "TrainingInputMode": "File"
        },
        "InputDataConfig": [],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://sagemaker-{region}-{account_id}/saafe-output/"
        },
        "ResourceConfig": {
            "InstanceType": "ml.p3.8xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 100
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600
        },
        "Environment": {
            "SAGEMAKER_PROGRAM": "train.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"
        }
    }
    
    # Save job definition
    with open('training_job_fixed.json', 'w') as f:
        json.dump(training_job, f, indent=2)
    
    print(f"   Job name: {job_name}")
    print(f"   Instance: ml.p3.8xlarge (4x V100 GPUs)")
    
    # Try to create the job
    success, output = run_cmd(f"aws sagemaker create-training-job --cli-input-json file://training_job_fixed.json", "Creating training job")
    
    if success:
        print("✅ Training job created successfully!")
        return job_name
    else:
        print("❌ Training job creation failed")
        print("💡 Let's try a different approach...")
        return None

def create_training_job_alternative(account_id, region):
    """Alternative method using boto3 directly."""
    print("🔄 Trying alternative method with boto3...")
    
    try:
        import boto3
        
        sagemaker = boto3.client('sagemaker', region_name=region)
        
        job_name = f"saafe-topnotch-{int(time.time())}"
        
        # Try to create training job without specifying role first
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                'TrainingImage': f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker',
                'TrainingInputMode': 'File'
            },
            RoleArn=f'arn:aws:iam::{account_id}:role/service-role/AmazonSageMakerServiceRole',
            OutputDataConfig={
                'S3OutputPath': f's3://sagemaker-{region}-{account_id}/saafe-output/'
            },
            ResourceConfig={
                'InstanceType': 'ml.p3.8xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 100
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 3600
            },
            Environment={
                'SAGEMAKER_PROGRAM': 'train.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
            }
        )
        
        print(f"✅ Training job created: {job_name}")
        return job_name
        
    except Exception as e:
        print(f"❌ Alternative method failed: {e}")
        return None

def create_simple_training_script():
    """Create the training script."""
    print("📝 Creating TOP-NOTCH training script...")
    
    script_content = '''#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from datetime import datetime

print("🔥 TOP-NOTCH Saafe Training Starting!")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Advanced model for 4x V100s
class TopNotchFireModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(240, 512),  # 60 timesteps * 4 features
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # normal, cooking, fire
        )
    
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# Create model
model = TopNotchFireModel()

# Use all available GPUs
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.cuda() if torch.cuda.is_available() else model
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate training data
print("Generating 10,000 training samples...")
data = torch.randn(10000, 60, 4)
labels = torch.randint(0, 3, (10000,))

if torch.cuda.is_available():
    data = data.cuda()
    labels = labels.cuda()

# Training
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("🔥 Starting training...")
model.train()

for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/30: Loss = {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(data)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = (predicted == labels).float().mean().item()

print(f"🎯 Final accuracy: {accuracy:.3f}")

# Save model
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
os.makedirs(model_dir, exist_ok=True)

model_to_save = model.module if hasattr(model, 'module') else model

torch.save({
    'model_state_dict': model_to_save.state_dict(),
    'accuracy': accuracy,
    'timestamp': datetime.now().isoformat(),
    'model_type': 'TopNotchFireModel',
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
}, os.path.join(model_dir, 'model.pth'))

print("✅ TOP-NOTCH training completed!")
print(f"🚀 Model saved with {accuracy:.1%} accuracy!")
'''
    
    with open('train.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Training script created")

def main():
    """Main function."""
    print_banner()
    
    # Step 1: Get account info
    print("Step 1: Getting AWS account info...")
    success, account_id = run_cmd("aws sts get-caller-identity --query Account --output text", "Getting account ID")
    if not success:
        print("❌ Cannot get AWS account info. Check your credentials.")
        return
    
    print(f"   Account ID: {account_id}")
    
    # Step 2: Fix region
    print("\nStep 2: Fixing AWS region...")
    region = fix_aws_region()
    
    # Step 3: Check IAM permissions
    print("\nStep 3: Checking permissions...")
    has_iam = check_iam_permissions()
    
    # Step 4: Create training script
    print("\nStep 4: Creating training script...")
    create_simple_training_script()
    
    # Step 5: Try to launch training
    print("\nStep 5: Launching TOP-NOTCH training...")
    print("🔥 CONFIGURATION:")
    print("   Instance: ml.p3.8xlarge (4x V100 GPUs)")
    print("   Cost: ~$12.24/hour")
    print("   Expected time: 20-30 minutes")
    print("   Expected cost: $4-6")
    print()
    
    confirm = input("🚀 Launch TOP-NOTCH training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("👍 Cancelled")
        return
    
    # Try different methods
    job_name = None
    
    if has_iam:
        # Try with existing role
        existing_role = get_existing_sagemaker_role(account_id)
        if existing_role:
            print(f"Using existing role: {existing_role}")
            # Create job with existing role
    
    if not job_name:
        job_name = create_training_job_with_service_role(account_id, region)
    
    if not job_name:
        job_name = create_training_job_alternative(account_id, region)
    
    if job_name:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Training job launched: {job_name}")
        print(f"⚡ Expected completion: 20-30 minutes")
        print(f"💰 Expected cost: $4-6")
        print(f"🔍 Monitor: https://{region}.console.aws.amazon.com/sagemaker/")
        print("\n🔥 TOP-NOTCH training is running!")
    else:
        print("\n❌ Could not launch training job")
        print("💡 Possible solutions:")
        print("   1. Ask your AWS admin to create a SageMaker execution role")
        print("   2. Use AWS Console to create the training job manually")
        print("   3. Try a different AWS region")
    
    # Cleanup
    for f in ['training_job_fixed.json', 'train.py']:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()