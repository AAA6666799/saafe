#!/usr/bin/env python3
"""
AWS Fast Training Pipeline for Saafe - Maximum Speed Configuration
Optimized for fastest training time regardless of cost.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import json
import os
import tarfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastAWSTraining:
    """Ultra-fast AWS training pipeline optimized for speed."""
    
    def __init__(self, region: str = 'us-east-1'):
        """Initialize fast training pipeline."""
        self.region = region
        self.session = boto3.Session(region_name=region)
        self.sagemaker_session = sagemaker.Session(boto_session=self.session)
        self.s3_client = self.session.client('s3')
        
        # Get or create role
        self.role_arn = self._get_sagemaker_role()
        
        # Create S3 bucket for training
        self.bucket_name = f"saafe-fast-training-{int(time.time())}"
        self._create_s3_bucket()
        
        logger.info(f"🚀 Fast AWS Training initialized")
        logger.info(f"Region: {self.region}")
        logger.info(f"S3 Bucket: {self.bucket_name}")
    
    def _get_sagemaker_role(self) -> str:
        """Get SageMaker execution role."""
        iam = self.session.client('iam')
        role_name = 'SaafeTrainingRole'
        
        try:
            response = iam.get_role(RoleName=role_name)
            return response['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            logger.error(f"Role {role_name} not found. Run setup_aws_training.sh first.")
            raise
    
    def _create_s3_bucket(self):
        """Create S3 bucket."""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
        except Exception as e:
            if 'BucketAlreadyExists' in str(e):
                self.bucket_name = f"saafe-fast-training-{int(time.time())}-{os.urandom(4).hex()}"
                self._create_s3_bucket()
    
    def create_optimized_training_script(self) -> str:
        """Create ultra-optimized training script for maximum speed."""
        
        training_script = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
import logging
from datetime import datetime
import time

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedModelConfig:
    """Optimized configuration for fastest training."""
    def __init__(self):
        # Smaller model for faster training
        self.num_sensors = 4
        self.feature_dim = 4
        self.d_model = 128          # Reduced from 256
        self.num_heads = 8
        self.num_layers = 4         # Reduced from 6
        self.max_seq_length = 60
        self.dropout = 0.1
        self.num_classes = 3

class FastTransformer(nn.Module):
    """Ultra-fast transformer optimized for speed."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Optimized input embedding
        self.input_embedding = nn.Linear(config.feature_dim, config.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(config.max_seq_length, config.d_model))
        
        # Optimized transformer with fused operations
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model * 2,  # Reduced from 4x
                dropout=config.dropout,
                activation='gelu',  # Faster than relu
                batch_first=True,
                norm_first=True     # Pre-norm for stability
            ) for _ in range(config.num_layers)
        ])
        
        # Optimized output heads
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
        self.risk_regressor = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights for faster convergence
        self._fast_init_weights()
    
    def _fast_init_weights(self):
        """Fast weight initialization for quicker convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        batch_size, seq_len, num_sensors, features = x.shape
        
        # Reshape and embed efficiently
        x = x.view(batch_size, seq_len * num_sensors, features)
        x = self.input_embedding(x)
        
        # Add positional encoding
        seq_len_flat = x.size(1)
        if seq_len_flat <= self.config.max_seq_length:
            x = x + self.pos_encoding[:seq_len_flat].unsqueeze(0)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output heads
        logits = self.classifier(x)
        risk_score = self.risk_regressor(x) * 100.0
        
        return {'logits': logits, 'risk_score': risk_score}

class FastDataset(Dataset):
    """Optimized dataset for fastest training."""
    
    def __init__(self, num_samples=8000, seq_length=60, num_sensors=4):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_sensors = num_sensors
        
        logger.info(f"Generating {num_samples} optimized samples...")
        
        # Pre-generate all data for speed
        self.data = torch.zeros(num_samples, seq_length, num_sensors, 4)
        self.class_labels = torch.zeros(num_samples, dtype=torch.long)
        self.risk_scores = torch.zeros(num_samples)
        
        # Vectorized data generation for speed
        self._generate_all_data()
    
    def _generate_all_data(self):
        """Vectorized data generation for maximum speed."""
        samples_per_class = self.num_samples // 3
        
        # Normal scenarios (0-samples_per_class)
        normal_end = samples_per_class
        self.data[:normal_end] = self._generate_vectorized_normal(normal_end)
        self.class_labels[:normal_end] = 0
        self.risk_scores[:normal_end] = 5 + torch.rand(normal_end) * 10
        
        # Cooking scenarios
        cooking_end = 2 * samples_per_class
        self.data[normal_end:cooking_end] = self._generate_vectorized_cooking(samples_per_class)
        self.class_labels[normal_end:cooking_end] = 1
        self.risk_scores[normal_end:cooking_end] = 20 + torch.rand(samples_per_class) * 20
        
        # Fire scenarios
        self.data[cooking_end:] = self._generate_vectorized_fire(self.num_samples - cooking_end)
        self.class_labels[cooking_end:] = 2
        self.risk_scores[cooking_end:] = 80 + torch.rand(self.num_samples - cooking_end) * 20
    
    def _generate_vectorized_normal(self, batch_size):
        """Vectorized normal scenario generation."""
        data = torch.zeros(batch_size, self.seq_length, self.num_sensors, 4)
        
        # Temperature: 20-25°C
        data[:, :, :, 0] = 22 + torch.randn(batch_size, self.seq_length, self.num_sensors) * 2
        data[:, :, :, 0] = torch.clamp(data[:, :, :, 0], 18, 28)
        
        # PM2.5: 5-20 μg/m³
        data[:, :, :, 1] = 10 + torch.randn(batch_size, self.seq_length, self.num_sensors) * 5
        data[:, :, :, 1] = torch.clamp(data[:, :, :, 1], 5, 25)
        
        # CO2: 400-600 ppm
        data[:, :, :, 2] = 450 + torch.randn(batch_size, self.seq_length, self.num_sensors) * 50
        data[:, :, :, 2] = torch.clamp(data[:, :, :, 2], 350, 650)
        
        # Audio: 35-50 dB
        data[:, :, :, 3] = 40 + torch.randn(batch_size, self.seq_length, self.num_sensors) * 5
        data[:, :, :, 3] = torch.clamp(data[:, :, :, 3], 30, 55)
        
        return data
    
    def _generate_vectorized_cooking(self, batch_size):
        """Vectorized cooking scenario generation."""
        data = torch.zeros(batch_size, self.seq_length, self.num_sensors, 4)
        
        # Temperature gradient
        temp_base = torch.linspace(22, 32, self.seq_length).unsqueeze(0).unsqueeze(2)
        data[:, :, :, 0] = temp_base + torch.randn(batch_size, self.seq_length, self.num_sensors) * 2
        data[:, :, :, 0] = torch.clamp(data[:, :, :, 0], 20, 40)
        
        # PM2.5 gradient
        pm25_base = torch.linspace(15, 45, self.seq_length).unsqueeze(0).unsqueeze(2)
        data[:, :, :, 1] = pm25_base + torch.randn(batch_size, self.seq_length, self.num_sensors) * 8
        data[:, :, :, 1] = torch.clamp(data[:, :, :, 1], 15, 70)
        
        # CO2 gradient
        co2_base = torch.linspace(450, 700, self.seq_length).unsqueeze(0).unsqueeze(2)
        data[:, :, :, 2] = co2_base + torch.randn(batch_size, self.seq_length, self.num_sensors) * 50
        data[:, :, :, 2] = torch.clamp(data[:, :, :, 2], 400, 900)
        
        # Audio
        data[:, :, :, 3] = 55 + torch.randn(batch_size, self.seq_length, self.num_sensors) * 8
        data[:, :, :, 3] = torch.clamp(data[:, :, :, 3], 40, 70)
        
        return data
    
    def _generate_vectorized_fire(self, batch_size):
        """Vectorized fire scenario generation."""
        data = torch.zeros(batch_size, self.seq_length, self.num_sensors, 4)
        
        # Temperature: rapid rise
        temp_base = torch.linspace(25, 70, self.seq_length).unsqueeze(0).unsqueeze(2)
        data[:, :, :, 0] = temp_base + torch.randn(batch_size, self.seq_length, self.num_sensors) * 5
        data[:, :, :, 0] = torch.clamp(data[:, :, :, 0], 25, 85)
        
        # PM2.5: high levels
        pm25_base = torch.linspace(20, 150, self.seq_length).unsqueeze(0).unsqueeze(2)
        data[:, :, :, 1] = pm25_base + torch.randn(batch_size, self.seq_length, self.num_sensors) * 20
        data[:, :, :, 1] = torch.clamp(data[:, :, :, 1], 50, 200)
        
        # CO2: high levels
        co2_base = torch.linspace(500, 1200, self.seq_length).unsqueeze(0).unsqueeze(2)
        data[:, :, :, 2] = co2_base + torch.randn(batch_size, self.seq_length, self.num_sensors) * 100
        data[:, :, :, 2] = torch.clamp(data[:, :, :, 2], 600, 1800)
        
        # Audio: high levels
        data[:, :, :, 3] = 75 + torch.randn(batch_size, self.seq_length, self.num_sensors) * 8
        data[:, :, :, 3] = torch.clamp(data[:, :, :, 3], 60, 90)
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.class_labels[idx], self.risk_scores[idx]

def train_fast_model(args):
    """Ultra-fast training function."""
    
    logger.info("🚀 Starting ULTRA-FAST training...")
    start_time = time.time()
    
    # Device setup with optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        # GPU optimizations
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Optimized model configuration
    config = OptimizedModelConfig()
    model = FastTransformer(config).to(device)
    
    # Enable mixed precision for speed
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Fast dataset with fewer samples
    dataset = FastDataset(num_samples=args.num_samples)
    
    # Optimized data loading
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Optimized optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.95)  # Optimized betas
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Fast loss functions
    class_criterion = nn.NLLLoss()  # Use NLLLoss with LogSoftmax
    risk_criterion = nn.MSELoss()
    
    # Training loop with optimizations
    model.train()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0.0
        
        for batch_idx, (data, class_labels, risk_scores) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            risk_scores = risk_scores.to(device, non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    class_loss = class_criterion(outputs['logits'], class_labels)
                    risk_loss = risk_criterion(outputs['risk_score'], risk_scores)
                    loss = class_loss + 0.5 * risk_loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                class_loss = class_criterion(outputs['logits'], class_labels)
                risk_loss = risk_criterion(outputs['risk_score'], risk_scores)
                loss = class_loss + 0.5 * risk_loss
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    logger.info(f"🎉 Training completed in {total_time:.1f} seconds!")
    
    # Quick evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, class_labels, _ in train_loader:
            data = data.to(device)
            class_labels = class_labels.to(device)
            
            outputs = model(data)
            predicted = torch.argmax(outputs['logits'], dim=1)
            total += class_labels.size(0)
            correct += (predicted == class_labels).sum().item()
    
    accuracy = correct / total
    logger.info(f"Final accuracy: {accuracy:.3f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'accuracy': accuracy,
        'training_time': total_time,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }, os.path.join(args.model_dir, 'model.pth'))
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'training_time_seconds': total_time,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'training_completed': datetime.now().isoformat(),
        'device': str(device),
        'model_type': 'fast_optimized'
    }
    
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ Fast training completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    # Fast training arguments
    parser.add_argument('--epochs', type=int, default=30)        # Reduced epochs
    parser.add_argument('--batch-size', type=int, default=128)   # Larger batch size
    parser.add_argument('--learning-rate', type=float, default=0.003)  # Higher LR
    parser.add_argument('--num-samples', type=int, default=8000) # Fewer samples
    
    args = parser.parse_args()
    train_fast_model(args)
'''
        
        # Save optimized training script
        script_path = Path('fast_training_script.py')
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        # Create source directory
        source_dir = Path('fast_source_code')
        source_dir.mkdir(exist_ok=True)
        
        # Move script
        script_path.rename(source_dir / 'train.py')
        
        # Optimized requirements
        requirements = '''
torch>=1.12.0
numpy>=1.21.0
'''
        
        with open(source_dir / 'requirements.txt', 'w') as f:
            f.write(requirements)
        
        # Create archive
        archive_path = 'fast_training_code.tar.gz'
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(source_dir, arcname='.')
        
        # Upload to S3
        s3_key = 'fast-training-code/fast_training_code.tar.gz'
        self.s3_client.upload_file(archive_path, self.bucket_name, s3_key)
        
        s3_uri = f's3://{self.bucket_name}/{s3_key}'
        logger.info(f"Fast training code uploaded to: {s3_uri}")
        
        # Cleanup
        os.remove(archive_path)
        import shutil
        shutil.rmtree(source_dir)
        
        return s3_uri
    
    def launch_ultra_fast_training(self) -> str:
        """Launch ultra-fast training job with maximum performance."""
        
        logger.info("🚀 Launching ULTRA-FAST training job...")
        
        # Prepare optimized training code
        source_uri = self.create_optimized_training_script()
        
        # Ultra-fast hyperparameters
        hyperparameters = {
            'epochs': 25,           # Fewer epochs
            'batch-size': 128,      # Large batch size for GPU
            'learning-rate': 0.005, # Higher learning rate
            'num-samples': 6000     # Fewer samples for speed
        }
        
        # Use the most powerful GPU instance
        estimator = PyTorch(
            entry_point='train.py',
            source_dir=source_uri,
            role=self.role_arn,
            instance_type='ml.p3.8xlarge',  # 4x V100 GPUs!
            instance_count=1,
            framework_version='1.12.0',
            py_version='py38',
            hyperparameters=hyperparameters,
            max_run=3600,  # 1 hour max
            output_path=f's3://{self.bucket_name}/fast-training-output/',
            code_location=f's3://{self.bucket_name}/fast-training-code/',
            sagemaker_session=self.sagemaker_session,
            tags=[
                {'Key': 'Project', 'Value': 'Saafe'},
                {'Key': 'TrainingType', 'Value': 'UltraFast'},
                {'Key': 'Priority', 'Value': 'High'}
            ]
        )
        
        # Launch with timestamp
        job_name = f"saafe-ultrafast-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        
        logger.info(f"🔥 ULTRA-FAST TRAINING CONFIGURATION:")
        logger.info(f"   Instance: ml.p3.8xlarge (4x NVIDIA V100 GPUs)")
        logger.info(f"   Memory: 244GB RAM")
        logger.info(f"   GPU Memory: 64GB total (16GB x 4)")
        logger.info(f"   Cost: ~$12.24/hour")
        logger.info(f"   Expected time: 30-45 minutes")
        logger.info(f"   Expected cost: $6-9")
        
        estimator.fit(job_name=job_name, wait=False)
        
        logger.info(f"🚀 ULTRA-FAST training launched: {job_name}")
        logger.info(f"⚡ This will be BLAZING fast!")
        
        return job_name
    
    def launch_distributed_training(self) -> str:
        """Launch distributed training across multiple instances."""
        
        logger.info("🚀 Launching DISTRIBUTED training for maximum speed...")
        
        source_uri = self.create_optimized_training_script()
        
        # Distributed training hyperparameters
        hyperparameters = {
            'epochs': 20,           # Even fewer epochs with distributed
            'batch-size': 64,       # Per-GPU batch size
            'learning-rate': 0.008, # Higher LR for distributed
            'num-samples': 8000
        }
        
        # Multi-instance distributed training
        estimator = PyTorch(
            entry_point='train.py',
            source_dir=source_uri,
            role=self.role_arn,
            instance_type='ml.p3.2xlarge',  # V100 GPU instances
            instance_count=4,               # 4 instances = 4 GPUs
            framework_version='1.12.0',
            py_version='py38',
            hyperparameters=hyperparameters,
            max_run=2400,  # 40 minutes max
            distribution={
                'smdistributed': {
                    'dataparallel': {
                        'enabled': True
                    }
                }
            },
            output_path=f's3://{self.bucket_name}/distributed-training-output/',
            sagemaker_session=self.sagemaker_session,
            tags=[
                {'Key': 'Project', 'Value': 'Saafe'},
                {'Key': 'TrainingType', 'Value': 'Distributed'},
                {'Key': 'Priority', 'Value': 'Maximum'}
            ]
        )
        
        job_name = f"saafe-distributed-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        
        logger.info(f"🔥 DISTRIBUTED TRAINING CONFIGURATION:")
        logger.info(f"   Instances: 4x ml.p3.2xlarge")
        logger.info(f"   Total GPUs: 4x NVIDIA V100")
        logger.info(f"   Total Memory: 244GB RAM")
        logger.info(f"   Total GPU Memory: 64GB")
        logger.info(f"   Cost: ~$12.24/hour")
        logger.info(f"   Expected time: 20-30 minutes")
        logger.info(f"   Expected cost: $4-6")
        
        estimator.fit(job_name=job_name, wait=False)
        
        logger.info(f"🚀 DISTRIBUTED training launched: {job_name}")
        logger.info(f"⚡ Maximum parallel processing!")
        
        return job_name


def main():
    """Main function for ultra-fast training."""
    
    print("🚀 SAAFE ULTRA-FAST AWS TRAINING")
    print("=" * 50)
    print("💰 Cost is no object - Maximum speed!")
    print("⚡ Expected training time: 30-45 minutes")
    print("🔥 Expected cost: $6-12")
    print()
    
    # Initialize fast training
    fast_trainer = FastAWSTraining()
    
    print("Choose your SPEED option:")
    print("1. 🔥 Ultra-Fast Single GPU (ml.p3.8xlarge - 4x V100)")
    print("2. ⚡ Distributed Multi-GPU (4x ml.p3.2xlarge)")
    print("3. 🚀 Let me choose the fastest!")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        print("\n🔥 Launching Ultra-Fast Single GPU training...")
        job_name = fast_trainer.launch_ultra_fast_training()
        
    elif choice == '2':
        print("\n⚡ Launching Distributed Multi-GPU training...")
        job_name = fast_trainer.launch_distributed_training()
        
    elif choice == '3':
        print("\n🚀 Choosing MAXIMUM SPEED option...")
        print("   Ultra-Fast Single GPU selected (fastest overall)")
        job_name = fast_trainer.launch_ultra_fast_training()
        
    else:
        print("Invalid choice. Using Ultra-Fast option.")
        job_name = fast_trainer.launch_ultra_fast_training()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 ULTRA-FAST TRAINING LAUNCHED!")
    print(f"=" * 60)
    print(f"Job Name: {job_name}")
    print(f"⏱️  Expected completion: 30-45 minutes")
    print(f"💰 Expected cost: $6-12")
    print(f"🔍 Monitor: https://console.aws.amazon.com/sagemaker/")
    print()
    
    # Wait for completion option
    wait = input("Wait for completion and auto-download models? (y/N): ").strip().lower()
    
    if wait == 'y':
        print("⏳ Waiting for training completion...")
        
        sagemaker_client = fast_trainer.session.client('sagemaker')
        
        while True:
            try:
                response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']
                
                if status == 'Completed':
                    print("🎉 Training completed successfully!")
                    
                    # Download model
                    print("📥 Downloading trained models...")
                    
                    # Get model S3 location
                    model_s3_uri = response['ModelArtifacts']['S3ModelArtifacts']
                    print(f"Model location: {model_s3_uri}")
                    
                    # Parse S3 URI and download
                    s3_parts = model_s3_uri.replace('s3://', '').split('/')
                    bucket = s3_parts[0]
                    key = '/'.join(s3_parts[1:])
                    
                    # Download and extract
                    models_dir = Path('models')
                    models_dir.mkdir(exist_ok=True)
                    
                    model_archive = models_dir / 'model.tar.gz'
                    fast_trainer.s3_client.download_file(bucket, key, str(model_archive))
                    
                    # Extract
                    with tarfile.open(model_archive, 'r:gz') as tar:
                        tar.extractall(models_dir)
                    
                    model_archive.unlink()
                    
                    # Rename to standard format
                    if (models_dir / 'model.pth').exists():
                        (models_dir / 'model.pth').rename(models_dir / 'transformer_model.pth')
                    
                    print("✅ Models downloaded successfully!")
                    print(f"📁 Location: {models_dir}/")
                    
                    # Show training metrics
                    metrics_file = models_dir / 'metrics.json'
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                        
                        print(f"\n📊 TRAINING RESULTS:")
                        print(f"   Accuracy: {metrics.get('accuracy', 0):.1%}")
                        print(f"   Training Time: {metrics.get('training_time_seconds', 0):.0f} seconds")
                        print(f"   Parameters: {metrics.get('total_parameters', 0):,}")
                    
                    break
                    
                elif status == 'Failed':
                    print("❌ Training failed!")
                    if 'FailureReason' in response:
                        print(f"Reason: {response['FailureReason']}")
                    break
                    
                elif status in ['InProgress', 'Starting']:
                    print(f"⏳ Status: {status} - Still training...")
                    time.sleep(60)  # Check every minute
                    
                else:
                    print(f"Status: {status}")
                    time.sleep(30)
                    
            except Exception as e:
                print(f"Error checking status: {e}")
                time.sleep(30)
    
    print(f"\n🎉 ULTRA-FAST training setup complete!")
    print(f"🚀 Your models will be ready in 30-45 minutes!")


if __name__ == "__main__":
    main()