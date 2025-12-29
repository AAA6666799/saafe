#!/usr/bin/env python3
"""
Check your specific AWS quotas and launch training with what's available.
"""

import subprocess
import sys
import json
import os
import time

def print_banner():
    """Print banner."""
    print("🔍 CHECKING YOUR SPECIFIC AWS QUOTAS")
    print("=" * 45)
    print("📊 Finding exactly what GPU instances you can use")
    print("🚀 Launching with your available resources")
    print()

def run_cmd(cmd, description="", show_output=True):
    """Run command with error handling."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            if show_output and description:
                print(f"   ✅ {description}")
            return True, result.stdout.strip()
        else:
            if show_output and description:
                print(f"   ❌ {description}: {result.stderr.strip()}")
            return False, result.stderr.strip()
            
    except Exception as e:
        if show_output and description:
            print(f"   💥 {description}: {e}")
        return False, str(e)

def check_specific_quotas():
    """Check specific SageMaker training instance quotas."""
    print("📊 Checking your SageMaker training quotas...")
    
    # Common SageMaker training instances and their quota codes
    instances_to_check = [
        ("ml.m5.large", "ml.m5.large for training job usage"),
        ("ml.m5.xlarge", "ml.m5.xlarge for training job usage"), 
        ("ml.m5.2xlarge", "ml.m5.2xlarge for training job usage"),
        ("ml.m5.4xlarge", "ml.m5.4xlarge for training job usage"),
        ("ml.c5.xlarge", "ml.c5.xlarge for training job usage"),
        ("ml.c5.2xlarge", "ml.c5.2xlarge for training job usage"),
        ("ml.g4dn.xlarge", "ml.g4dn.xlarge for training job usage"),
        ("ml.g4dn.2xlarge", "ml.g4dn.2xlarge for training job usage"),
        ("ml.p3.2xlarge", "ml.p3.2xlarge for training job usage"),
        ("ml.p3.8xlarge", "ml.p3.8xlarge for training job usage")
    ]
    
    available_instances = []
    
    for instance_type, quota_name in instances_to_check:
        print(f"   Checking {instance_type}...")
        
        # Try to get the quota using service quotas
        success, output = run_cmd(
            f"aws service-quotas list-service-quotas --service-code sagemaker --query 'Quotas[?QuotaName==`{quota_name}`].Value' --output text",
            "",
            False
        )
        
        if success and output and output != "None" and output.strip():
            try:
                quota_value = float(output.strip())
                if quota_value > 0:
                    available_instances.append((instance_type, quota_value))
                    print(f"      ✅ Available: {quota_value} instances")
                else:
                    print(f"      ❌ Quota: {quota_value}")
            except:
                print(f"      ⚠️  Could not parse quota: {output}")
        else:
            # Try alternative method - check if we can describe the instance type
            success2, _ = run_cmd(
                f"aws ec2 describe-instance-types --instance-types {instance_type.replace('ml.', '')} --query 'InstanceTypes[0].InstanceType' --output text 2>/dev/null",
                "",
                False
            )
            if success2:
                print(f"      ⚠️  Instance type exists, quota unknown (likely available)")
                available_instances.append((instance_type, 1))  # Assume 1 if we can't check
            else:
                print(f"      ❌ Not available")
    
    return available_instances

def get_instance_specs(instance_type):
    """Get specifications for instance type."""
    specs = {
        "ml.m5.large": {"cpu": 2, "memory": 8, "gpu": None, "cost_hour": 0.096, "description": "CPU - Basic"},
        "ml.m5.xlarge": {"cpu": 4, "memory": 16, "gpu": None, "cost_hour": 0.192, "description": "CPU - Good"},
        "ml.m5.2xlarge": {"cpu": 8, "memory": 32, "gpu": None, "cost_hour": 0.384, "description": "CPU - Better"},
        "ml.m5.4xlarge": {"cpu": 16, "memory": 64, "gpu": None, "cost_hour": 0.768, "description": "CPU - Best"},
        "ml.c5.xlarge": {"cpu": 4, "memory": 8, "gpu": None, "cost_hour": 0.17, "description": "CPU - Compute Optimized"},
        "ml.c5.2xlarge": {"cpu": 8, "memory": 16, "gpu": None, "cost_hour": 0.34, "description": "CPU - Compute Optimized"},
        "ml.g4dn.xlarge": {"cpu": 4, "memory": 16, "gpu": "1x T4", "cost_hour": 0.526, "description": "GPU - T4 (Good)"},
        "ml.g4dn.2xlarge": {"cpu": 8, "memory": 32, "gpu": "1x T4", "cost_hour": 0.752, "description": "GPU - T4 (Better)"},
        "ml.p3.2xlarge": {"cpu": 8, "memory": 61, "gpu": "1x V100", "cost_hour": 3.06, "description": "GPU - V100 (Excellent)"},
        "ml.p3.8xlarge": {"cpu": 32, "memory": 244, "gpu": "4x V100", "cost_hour": 12.24, "description": "GPU - 4x V100 (Maximum)"}
    }
    
    return specs.get(instance_type, {"cpu": 2, "memory": 8, "gpu": None, "cost_hour": 0.1, "description": "Unknown"})

def create_training_job(instance_type, account_id, region):
    """Create training job with the selected instance."""
    print(f"🚀 Creating training job with {instance_type}...")
    
    specs = get_instance_specs(instance_type)
    job_name = f"saafe-available-{int(time.time())}"
    
    # Adjust training based on instance capabilities
    if specs["gpu"]:
        if "V100" in specs["gpu"]:
            epochs = 50 if "4x" in specs["gpu"] else 40
            batch_size = 128
            samples = 15000
        else:  # T4 GPU
            epochs = 30
            batch_size = 64
            samples = 10000
    else:  # CPU
        epochs = 20
        batch_size = 32
        samples = 5000
    
    # Create training job configuration
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
                        "S3Uri": f"s3://sagemaker-{region}-{account_id}/saafe-data/",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "application/json"
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
            "MaxRuntimeInSeconds": 7200  # 2 hours max
        },
        "Environment": {
            "SAGEMAKER_PROGRAM": "train.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
            "EPOCHS": str(epochs),
            "BATCH_SIZE": str(batch_size),
            "SAMPLES": str(samples)
        }
    }
    
    # Create dummy training data in S3
    print("   Setting up S3 data...")
    run_cmd(f"aws s3 mb s3://sagemaker-{region}-{account_id} 2>/dev/null || true", "", False)
    run_cmd(f"echo '{{\"data\": \"dummy\"}}' | aws s3 cp - s3://sagemaker-{region}-{account_id}/saafe-data/train.json", "", False)
    
    # Save and create job
    with open('available_training_job.json', 'w') as f:
        json.dump(training_job, f, indent=2)
    
    print(f"   Job: {job_name}")
    print(f"   Config: {epochs} epochs, {samples} samples, batch {batch_size}")
    
    success, output = run_cmd(
        f"aws sagemaker create-training-job --cli-input-json file://available_training_job.json",
        "Creating training job"
    )
    
    if success:
        return job_name
    else:
        print(f"   Error: {output}")
        return None

def create_adaptive_training_script():
    """Create training script that works with any instance."""
    print("📝 Creating adaptive training script...")
    
    script_content = '''#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from datetime import datetime

# Get configuration from environment
EPOCHS = int(os.environ.get('EPOCHS', '20'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
SAMPLES = int(os.environ.get('SAMPLES', '5000'))

print("🔥 SAAFE FIRE DETECTION TRAINING")
print("=" * 40)
print(f"📊 Training: {EPOCHS} epochs, {SAMPLES} samples")
print(f"🖥️  CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"🚀 GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("💻 Using CPU")

# Adaptive Fire Detection Model
class FireDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Model scales based on available compute
        if torch.cuda.is_available():
            # GPU model - larger
            self.features = nn.Sequential(
                nn.Linear(240, 256),  # 60 timesteps * 4 sensors
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU()
            )
        else:
            # CPU model - smaller but efficient
            self.features = nn.Sequential(
                nn.Linear(240, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # Output layers
        self.classifier = nn.Linear(64, 3)  # normal, cooking, fire
        self.risk_regressor = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1, will scale to 0-100
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        features = self.features(x)
        
        classification = self.classifier(features)
        risk_score = self.risk_regressor(features) * 100  # Scale to 0-100
        
        return {'logits': classification, 'risk_score': risk_score}

# Create model
model = FireDetectionModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use multiple GPUs if available
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = nn.DataParallel(model)

model = model.to(device)
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate realistic fire detection data
print(f"📊 Generating {SAMPLES} fire detection samples...")

def create_fire_data():
    data = torch.zeros(SAMPLES, 60, 4)  # 60 timesteps, 4 sensors
    labels = torch.zeros(SAMPLES, dtype=torch.long)
    
    samples_per_class = SAMPLES // 3
    
    for i in range(SAMPLES):
        if i < samples_per_class:
            # Normal environment
            data[i, :, 0] = 22 + torch.randn(60) * 2  # temp 20-24°C
            data[i, :, 1] = 10 + torch.randn(60) * 3  # PM2.5 7-13
            data[i, :, 2] = 450 + torch.randn(60) * 50  # CO2 400-500
            data[i, :, 3] = 40 + torch.randn(60) * 5   # audio 35-45dB
            labels[i] = 0
        elif i < 2 * samples_per_class:
            # Cooking scenario
            temp_rise = torch.linspace(0, 8, 60)  # gradual temp rise
            data[i, :, 0] = 22 + temp_rise + torch.randn(60) * 1
            data[i, :, 1] = 15 + torch.linspace(0, 25, 60) + torch.randn(60) * 5  # PM2.5 rise
            data[i, :, 2] = 450 + torch.linspace(0, 200, 60) + torch.randn(60) * 30  # CO2 rise
            data[i, :, 3] = 50 + torch.randn(60) * 8  # cooking sounds
            labels[i] = 1
        else:
            # Fire scenario
            temp_spike = torch.linspace(0, 40, 60)  # rapid temp rise
            data[i, :, 0] = 25 + temp_spike + torch.randn(60) * 3
            data[i, :, 1] = 20 + torch.linspace(0, 120, 60) + torch.randn(60) * 15  # high PM2.5
            data[i, :, 2] = 500 + torch.linspace(0, 800, 60) + torch.randn(60) * 50  # high CO2
            data[i, :, 3] = 60 + torch.randn(60) * 10  # fire/alarm sounds
            labels[i] = 2
    
    return data, labels

data, labels = create_fire_data()
data = data.to(device)
labels = labels.to(device)

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
class_criterion = nn.CrossEntropyLoss()
risk_criterion = nn.MSELoss()

print(f"🔥 Starting training for {EPOCHS} epochs...")

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    
    # Process in batches
    for i in range(0, len(data), BATCH_SIZE):
        batch_data = data[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        
        # Create risk scores based on labels
        batch_risks = torch.where(batch_labels == 0, torch.rand(len(batch_labels)) * 20,  # normal: 0-20
                                torch.where(batch_labels == 1, 20 + torch.rand(len(batch_labels)) * 40,  # cooking: 20-60
                                           80 + torch.rand(len(batch_labels)) * 20))  # fire: 80-100
        batch_risks = batch_risks.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        
        class_loss = class_criterion(outputs['logits'], batch_labels)
        risk_loss = risk_criterion(outputs['risk_score'], batch_risks)
        loss = class_loss + 0.5 * risk_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if (epoch + 1) % max(1, EPOCHS // 5) == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss = {avg_loss:.4f}")

# Evaluation
print("📊 Evaluating model...")
model.eval()
correct = 0
total = 0
risk_errors = []

with torch.no_grad():
    for i in range(0, len(data), BATCH_SIZE):
        batch_data = data[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        
        outputs = model(batch_data)
        predicted = torch.argmax(outputs['logits'], dim=1)
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
    'timestamp': datetime.now().isoformat(),
    'model_type': 'AdaptiveFireDetectionModel'
}, os.path.join(model_dir, 'model.pth'))

# Save metrics
metrics = {
    'accuracy': accuracy,
    'epochs': EPOCHS,
    'samples': SAMPLES,
    'model_parameters': sum(p.numel() for p in model.parameters()),
    'device': str(device),
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
}

with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ FIRE DETECTION TRAINING COMPLETED!")
print(f"🎯 Accuracy: {accuracy:.1%}")
print(f"🔥 Model ready for fire detection!")
'''
    
    with open('train.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Adaptive training script created")

def main():
    """Main function."""
    print_banner()
    
    # Get AWS info
    success, account_id = run_cmd("aws sts get-caller-identity --query Account --output text", "Getting account")
    if not success:
        print("❌ Cannot access AWS account")
        return
    
    region = "us-east-1"
    print(f"Account: {account_id}, Region: {region}")
    
    # Check your specific quotas
    available_instances = check_specific_quotas()
    
    if not available_instances:
        print("\n❌ No training instances found with available quota")
        print("💡 This might mean:")
        print("   1. All quotas are 0 (new account)")
        print("   2. Quota checking failed")
        print("   3. Need to request increases")
        
        # Try with a basic instance anyway
        print("\n🔄 Trying with ml.m5.large (usually available)...")
        available_instances = [("ml.m5.large", 1)]
    
    # Sort by preference (GPU first, then by power)
    def sort_key(item):
        instance_type, quota = item
        if "p3.8xlarge" in instance_type:
            return 0
        elif "p3.2xlarge" in instance_type:
            return 1
        elif "g4dn" in instance_type:
            return 2
        elif "m5.4xlarge" in instance_type:
            return 3
        elif "m5.2xlarge" in instance_type:
            return 4
        else:
            return 5
    
    available_instances.sort(key=sort_key)
    
    print(f"\n🎯 AVAILABLE INSTANCES:")
    for i, (instance_type, quota) in enumerate(available_instances[:5], 1):
        specs = get_instance_specs(instance_type)
        gpu_info = f" ({specs['gpu']})" if specs['gpu'] else " (CPU)"
        cost_info = f"${specs['cost_hour']:.2f}/hr"
        print(f"   {i}. {instance_type}{gpu_info} - {specs['description']} - {cost_info}")
    
    # Select best instance
    best_instance, quota = available_instances[0]
    specs = get_instance_specs(best_instance)
    
    estimated_time = 0.5 if "p3.8xlarge" in best_instance else 1.0 if "p3" in best_instance else 1.5
    total_cost = specs['cost_hour'] * estimated_time
    
    print(f"\n🚀 RECOMMENDED: {best_instance}")
    print(f"   {specs['description']}")
    print(f"   Cost: ${specs['cost_hour']}/hour")
    print(f"   Estimated time: {estimated_time} hours")
    print(f"   Total cost: ~${total_cost:.2f}")
    
    confirm = input(f"\n🔥 Launch training with {best_instance}? (y/N): ").strip().lower()
    if confirm != 'y':
        print("👍 Cancelled")
        return
    
    # Create training script and launch
    create_adaptive_training_script()
    job_name = create_training_job(best_instance, account_id, region)
    
    if job_name:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Training job: {job_name}")
        print(f"⚡ Instance: {best_instance}")
        print(f"💰 Cost: ~${total_cost:.2f}")
        print(f"🔍 Monitor: https://console.aws.amazon.com/sagemaker/")
        print("\n🔥 Your fire detection model is training!")
    else:
        print("\n❌ Training launch failed")
        print("💡 Try using AWS Console to create the job manually")
    
    # Cleanup
    for f in ['available_training_job.json', 'train.py']:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()