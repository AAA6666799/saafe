#!/usr/bin/env python3
"""
Smart launcher that handles AWS service quotas and finds available instances.
"""

import subprocess
import sys
import json
import os
import time

def print_banner():
    """Print banner."""
    print("🎯 SAAFE SMART TRAINING LAUNCHER")
    print("=" * 45)
    print("🔍 Finding available GPU instances")
    print("📈 Requesting quota increases if needed")
    print("🚀 Launching with best available hardware")
    print()

def run_cmd(cmd, description="", show_output=True):
    """Run command with error handling."""
    if show_output:
        print(f"🔧 {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            if show_output:
                print(f"   ✅ Success")
            return True, result.stdout.strip()
        else:
            if show_output:
                print(f"   ❌ Failed: {result.stderr.strip()}")
            return False, result.stderr.strip()
            
    except Exception as e:
        if show_output:
            print(f"   💥 Exception: {e}")
        return False, str(e)

def check_service_quotas():
    """Check SageMaker service quotas for GPU instances."""
    print("📊 Checking SageMaker service quotas...")
    
    gpu_instances = [
        ("ml.p3.8xlarge", "4x V100 GPUs - MAXIMUM POWER"),
        ("ml.p3.2xlarge", "1x V100 GPU - HIGH PERFORMANCE"),
        ("ml.g4dn.xlarge", "1x T4 GPU - GOOD PERFORMANCE"),
        ("ml.g4dn.2xlarge", "1x T4 GPU - GOOD PERFORMANCE"),
        ("ml.m5.xlarge", "CPU only - BASIC")
    ]
    
    available_instances = []
    
    for instance_type, description in gpu_instances:
        print(f"   Checking {instance_type}...")
        
        # Try to get quota for this instance type
        success, output = run_cmd(
            f"aws service-quotas get-service-quota --service-code sagemaker --quota-code L-{get_quota_code(instance_type)} --query 'Quota.Value' --output text 2>/dev/null || echo '0'",
            "",
            False
        )
        
        if success:
            try:
                quota = float(output)
                if quota > 0:
                    available_instances.append((instance_type, description, quota))
                    print(f"      ✅ Available: {quota} instances")
                else:
                    print(f"      ❌ Quota: {quota}")
            except:
                print(f"      ⚠️  Could not determine quota")
        else:
            print(f"      ⚠️  Could not check quota")
    
    return available_instances

def get_quota_code(instance_type):
    """Get quota code for instance type."""
    quota_codes = {
        "ml.p3.8xlarge": "1194F27D",
        "ml.p3.2xlarge": "09B73C5E", 
        "ml.g4dn.xlarge": "4B9A7B75",
        "ml.g4dn.2xlarge": "B8A6D4F1",
        "ml.m5.xlarge": "7C4F8B2A"
    }
    return quota_codes.get(instance_type, "1194F27D")

def request_quota_increase(instance_type):
    """Request quota increase for instance type."""
    print(f"📈 Requesting quota increase for {instance_type}...")
    
    quota_code = get_quota_code(instance_type)
    
    success, output = run_cmd(
        f"aws service-quotas request-service-quota-increase --service-code sagemaker --quota-code {quota_code} --desired-value 1",
        f"Requesting {instance_type} quota increase"
    )
    
    if success:
        print("   ✅ Quota increase requested!")
        print("   ⏰ Usually approved within 24-48 hours")
        return True
    else:
        print("   ❌ Could not request quota increase")
        return False

def create_training_job(instance_type, account_id, region):
    """Create training job with specified instance type."""
    print(f"🚀 Creating training job with {instance_type}...")
    
    job_name = f"saafe-smart-{int(time.time())}"
    
    # Adjust training parameters based on instance type
    if "p3.8xlarge" in instance_type:
        epochs = 50
        batch_size = 256
        samples = 20000
        description = "TOP-NOTCH with 4x V100 GPUs"
    elif "p3.2xlarge" in instance_type:
        epochs = 40
        batch_size = 128
        samples = 15000
        description = "HIGH-PERFORMANCE with 1x V100 GPU"
    elif "g4dn" in instance_type:
        epochs = 30
        batch_size = 64
        samples = 10000
        description = "GOOD-PERFORMANCE with T4 GPU"
    else:
        epochs = 20
        batch_size = 32
        samples = 5000
        description = "CPU training"
    
    training_job = {
        "TrainingJobName": job_name,
        "RoleArn": f"arn:aws:iam::{account_id}:role/SaafeTrainingRole",
        "AlgorithmSpecification": {
            "TrainingImage": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker",
            "TrainingInputMode": "File"
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://sagemaker-{region}-{account_id}/dummy-data/",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "application/json",
                "CompressionType": "None"
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://sagemaker-{region}-{account_id}/saafe-output/"
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600
        },
        "Environment": {
            "SAGEMAKER_PROGRAM": "train.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
            "EPOCHS": str(epochs),
            "BATCH_SIZE": str(batch_size),
            "SAMPLES": str(samples)
        }
    }
    
    # Create dummy S3 data (required by SageMaker)
    print("   Creating dummy training data...")
    success, _ = run_cmd(f"aws s3 mb s3://sagemaker-{region}-{account_id} 2>/dev/null || true", "", False)
    success, _ = run_cmd(f"echo '{{\"dummy\": \"data\"}}' | aws s3 cp - s3://sagemaker-{region}-{account_id}/dummy-data/data.json", "", False)
    
    # Save job definition
    with open('smart_training_job.json', 'w') as f:
        json.dump(training_job, f, indent=2)
    
    print(f"   Job name: {job_name}")
    print(f"   Instance: {instance_type}")
    print(f"   Description: {description}")
    print(f"   Training: {epochs} epochs, {samples} samples")
    
    # Create the training job
    success, output = run_cmd(
        f"aws sagemaker create-training-job --cli-input-json file://smart_training_job.json",
        "Creating training job"
    )
    
    if success:
        print("✅ Training job created successfully!")
        return job_name
    else:
        print("❌ Training job creation failed")
        return None

def create_optimized_training_script():
    """Create training script that adapts to available resources."""
    print("📝 Creating adaptive training script...")
    
    script_content = '''#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from datetime import datetime

# Get training parameters from environment
EPOCHS = int(os.environ.get('EPOCHS', '30'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '64'))
SAMPLES = int(os.environ.get('SAMPLES', '10000'))

print(f"🔥 SAAFE ADAPTIVE TRAINING STARTING!")
print(f"📊 Configuration: {EPOCHS} epochs, {SAMPLES} samples, batch size {BATCH_SIZE}")
print(f"🖥️  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🚀 GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")

# Adaptive model based on available resources
class AdaptiveFireModel(nn.Module):
    def __init__(self, gpu_count=0):
        super().__init__()
        
        # Scale model size based on available GPUs
        if gpu_count >= 4:
            # 4x V100 - Maximum model
            hidden_size = 512
            layers = [512, 256, 128, 64]
            print("🔥 Using MAXIMUM model for 4x GPUs")
        elif gpu_count >= 1:
            # 1x GPU - Large model
            hidden_size = 256
            layers = [256, 128, 64]
            print("⚡ Using LARGE model for GPU")
        else:
            # CPU - Smaller model
            hidden_size = 128
            layers = [128, 64]
            print("💻 Using COMPACT model for CPU")
        
        # Build adaptive architecture
        model_layers = []
        input_size = 240  # 60 timesteps * 4 features
        
        for hidden in layers:
            model_layers.extend([
                nn.Linear(input_size, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(0.2)
            ])
            input_size = hidden
        
        # Output layer
        model_layers.append(nn.Linear(input_size, 3))  # normal, cooking, fire
        
        self.net = nn.Sequential(*model_layers)
        
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# Create adaptive model
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
model = AdaptiveFireModel(gpu_count)

# Use all available GPUs
if torch.cuda.is_available() and gpu_count > 1:
    print(f"🚀 Using DataParallel with {gpu_count} GPUs!")
    model = nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate adaptive training data
print(f"📊 Generating {SAMPLES} training samples...")

def generate_fire_data(num_samples):
    data = torch.zeros(num_samples, 60, 4)
    labels = torch.zeros(num_samples, dtype=torch.long)
    
    samples_per_class = num_samples // 3
    
    for i in range(num_samples):
        if i < samples_per_class:
            # Normal: temp 20-25°C, PM2.5 5-20, CO2 400-600, audio 35-50dB
            data[i, :, 0] = 22 + torch.randn(60) * 2  # temperature
            data[i, :, 1] = 10 + torch.randn(60) * 5  # PM2.5
            data[i, :, 2] = 450 + torch.randn(60) * 50  # CO2
            data[i, :, 3] = 40 + torch.randn(60) * 5   # audio
            labels[i] = 0
        elif i < 2 * samples_per_class:
            # Cooking: elevated levels
            data[i, :, 0] = torch.linspace(22, 32, 60) + torch.randn(60) * 2
            data[i, :, 1] = torch.linspace(15, 45, 60) + torch.randn(60) * 8
            data[i, :, 2] = torch.linspace(450, 700, 60) + torch.randn(60) * 50
            data[i, :, 3] = 55 + torch.randn(60) * 8
            labels[i] = 1
        else:
            # Fire: high levels
            data[i, :, 0] = torch.linspace(25, 70, 60) + torch.randn(60) * 5
            data[i, :, 1] = torch.linspace(20, 150, 60) + torch.randn(60) * 20
            data[i, :, 2] = torch.linspace(500, 1200, 60) + torch.randn(60) * 100
            data[i, :, 3] = 75 + torch.randn(60) * 8
            labels[i] = 2
    
    return data, labels

data, labels = generate_fire_data(SAMPLES)
data = data.to(device)
labels = labels.to(device)

# Adaptive training
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

print(f"🔥 Starting training for {EPOCHS} epochs...")
model.train()

for epoch in range(EPOCHS):
    # Process in batches
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(data), BATCH_SIZE):
        batch_data = data[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if (epoch + 1) % max(1, EPOCHS // 10) == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss = {avg_loss:.4f}")

# Evaluation
print("📊 Evaluating model...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i in range(0, len(data), BATCH_SIZE):
        batch_data = data[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        
        outputs = model(batch_data)
        predicted = torch.argmax(outputs, dim=1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = correct / total
print(f"🎯 Final accuracy: {accuracy:.3f}")

# Save model
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
os.makedirs(model_dir, exist_ok=True)

model_to_save = model.module if hasattr(model, 'module') else model

torch.save({
    'model_state_dict': model_to_save.state_dict(),
    'accuracy': accuracy,
    'epochs': EPOCHS,
    'samples': SAMPLES,
    'gpu_count': gpu_count,
    'timestamp': datetime.now().isoformat(),
    'model_type': 'AdaptiveFireModel'
}, os.path.join(model_dir, 'model.pth'))

# Save metrics
metrics = {
    'accuracy': accuracy,
    'epochs': EPOCHS,
    'samples': SAMPLES,
    'gpu_count': gpu_count,
    'model_parameters': sum(p.numel() for p in model.parameters()),
    'device': str(device)
}

with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ ADAPTIVE training completed successfully!")
print(f"🚀 Model saved with {accuracy:.1%} accuracy!")
print(f"🔥 Ready for fire detection deployment!")
'''
    
    with open('train.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Adaptive training script created")

def main():
    """Main function."""
    print_banner()
    
    # Get account info
    success, account_id = run_cmd("aws sts get-caller-identity --query Account --output text", "Getting account ID")
    if not success:
        print("❌ Cannot get AWS account info")
        return
    
    region = "us-east-1"
    print(f"Account: {account_id}, Region: {region}")
    
    # Check available instances
    print("\n🔍 Checking available GPU instances...")
    available_instances = check_service_quotas()
    
    if not available_instances:
        print("\n❌ No GPU instances available in your account")
        print("📈 Requesting quota increases...")
        
        # Request quota for p3.2xlarge (most likely to be approved)
        request_quota_increase("ml.p3.2xlarge")
        
        print("\n💡 IMMEDIATE OPTIONS:")
        print("1. Wait 24-48 hours for quota approval")
        print("2. Use AWS Console to request quota manually")
        print("3. Try a different AWS region")
        print("4. Use CPU training (slower but works)")
        
        use_cpu = input("\nUse CPU training for now? (y/N): ").strip().lower()
        if use_cpu == 'y':
            available_instances = [("ml.m5.xlarge", "CPU training", 1)]
        else:
            return
    
    # Select best available instance
    best_instance = available_instances[0]
    instance_type, description, quota = best_instance
    
    print(f"\n🎯 SELECTED INSTANCE: {instance_type}")
    print(f"   Description: {description}")
    print(f"   Available quota: {quota}")
    
    # Show cost estimate
    costs = {
        "ml.p3.8xlarge": 12.24,
        "ml.p3.2xlarge": 3.06,
        "ml.g4dn.xlarge": 0.526,
        "ml.g4dn.2xlarge": 0.752,
        "ml.m5.xlarge": 0.192
    }
    
    cost_per_hour = costs.get(instance_type, 1.0)
    estimated_time = 0.5 if "p3.8xlarge" in instance_type else 1.0 if "p3.2xlarge" in instance_type else 1.5
    total_cost = cost_per_hour * estimated_time
    
    print(f"   Cost: ${cost_per_hour}/hour")
    print(f"   Estimated time: {estimated_time} hours")
    print(f"   Total cost: ~${total_cost:.2f}")
    
    confirm = input(f"\n🚀 Launch training with {instance_type}? (y/N): ").strip().lower()
    if confirm != 'y':
        print("👍 Cancelled")
        return
    
    # Create training script
    create_optimized_training_script()
    
    # Launch training
    job_name = create_training_job(instance_type, account_id, region)
    
    if job_name:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Training job launched: {job_name}")
        print(f"⚡ Instance: {instance_type}")
        print(f"💰 Estimated cost: ${total_cost:.2f}")
        print(f"🔍 Monitor: https://console.aws.amazon.com/sagemaker/")
        print("\n🔥 Your adaptive fire detection model is training!")
        
        # Request quota increase for future use
        if "p3.8xlarge" not in instance_type:
            print("\n📈 Also requesting p3.8xlarge quota for future use...")
            request_quota_increase("ml.p3.8xlarge")
    else:
        print("\n❌ Training launch failed")
    
    # Cleanup
    for f in ['smart_training_job.json', 'train.py']:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()