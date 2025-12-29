#!/usr/bin/env python3
"""
AWS-based training pipeline for Saafe fire detection models.
Uses SageMaker, S3, and EC2 for scalable model training.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
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


class AWSTrainingPipeline:
    """AWS-based training pipeline for Saafe models."""
    
    def __init__(self, region: str = 'us-east-1', role_arn: Optional[str] = None):
        """
        Initialize AWS training pipeline.
        
        Args:
            region (str): AWS region
            role_arn (str): SageMaker execution role ARN
        """
        self.region = region
        self.session = boto3.Session(region_name=region)
        self.sagemaker_session = sagemaker.Session(boto_session=self.session)
        self.s3_client = self.session.client('s3')
        self.ec2_client = self.session.client('ec2')
        
        # Get or create SageMaker role
        self.role_arn = role_arn or self._get_sagemaker_role()
        
        # S3 bucket for training artifacts
        self.bucket_name = f"saafe-training-{self.session.region_name}-{int(time.time())}"
        self._create_s3_bucket()
        
        logger.info(f"AWS Training Pipeline initialized")
        logger.info(f"Region: {self.region}")
        logger.info(f"S3 Bucket: {self.bucket_name}")
        logger.info(f"SageMaker Role: {self.role_arn}")
    
    def _get_sagemaker_role(self) -> str:
        """Get or create SageMaker execution role."""
        iam = self.session.client('iam')
        role_name = 'SaafeTrainingRole'
        
        try:
            # Try to get existing role
            response = iam.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            logger.info(f"Using existing SageMaker role: {role_arn}")
            return role_arn
        except iam.exceptions.NoSuchEntityException:
            # Create new role
            logger.info("Creating new SageMaker execution role...")
            
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            # Create role
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='SageMaker execution role for Saafe training'
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach policies
            policies = [
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess',
                'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
            ]
            
            for policy_arn in policies:
                iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            
            # Wait for role to be available
            time.sleep(10)
            
            logger.info(f"Created SageMaker role: {role_arn}")
            return role_arn
    
    def _create_s3_bucket(self):
        """Create S3 bucket for training artifacts."""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"Created S3 bucket: {self.bucket_name}")
        except Exception as e:
            if 'BucketAlreadyExists' in str(e):
                # Use existing bucket with timestamp
                self.bucket_name = f"saafe-training-{self.region}-{int(time.time())}"
                self._create_s3_bucket()
            else:
                logger.error(f"Failed to create S3 bucket: {e}")
                raise
    
    def prepare_training_code(self) -> str:
        """Prepare and upload training code to S3."""
        logger.info("Preparing training code for SageMaker...")
        
        # Create training script
        training_script = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model classes (these would be included in the source code)
class ModelConfig:
    def __init__(self, **kwargs):
        self.num_sensors = kwargs.get('num_sensors', 4)
        self.feature_dim = kwargs.get('feature_dim', 4)
        self.d_model = kwargs.get('d_model', 256)
        self.num_heads = kwargs.get('num_heads', 8)
        self.num_layers = kwargs.get('num_layers', 6)
        self.max_seq_length = kwargs.get('max_seq_length', 60)
        self.dropout = kwargs.get('dropout', 0.1)
        self.num_classes = kwargs.get('num_classes', 3)

class SimplifiedTransformer(nn.Module):
    """Simplified transformer for SageMaker training."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(config.feature_dim, config.d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes)
        )
        
        self.risk_regressor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, num_sensors, features)
        batch_size, seq_len, num_sensors, features = x.shape
        
        # Reshape and embed
        x = x.view(batch_size, seq_len * num_sensors, features)
        x = self.input_embedding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output heads
        logits = self.classifier(x)
        risk_score = self.risk_regressor(x) * 100.0
        
        return {'logits': logits, 'risk_score': risk_score}

class SyntheticDataset(Dataset):
    """Synthetic dataset for training."""
    
    def __init__(self, num_samples=10000, seq_length=60, num_sensors=4):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_sensors = num_sensors
        
        # Generate all data at once for efficiency
        self.data, self.class_labels, self.risk_scores = self._generate_data()
    
    def _generate_data(self):
        logger.info(f"Generating {self.num_samples} synthetic samples...")
        
        data = torch.zeros(self.num_samples, self.seq_length, self.num_sensors, 4)
        class_labels = torch.zeros(self.num_samples, dtype=torch.long)
        risk_scores = torch.zeros(self.num_samples)
        
        # Generate samples by category
        samples_per_class = self.num_samples // 3
        
        for i in range(self.num_samples):
            if i < samples_per_class:
                # Normal scenario
                data[i] = self._generate_normal()
                class_labels[i] = 0
                risk_scores[i] = 5 + torch.rand(1) * 10
            elif i < 2 * samples_per_class:
                # Cooking scenario
                data[i] = self._generate_cooking()
                class_labels[i] = 1
                risk_scores[i] = 20 + torch.rand(1) * 20
            else:
                # Fire scenario
                data[i] = self._generate_fire()
                class_labels[i] = 2
                risk_scores[i] = 80 + torch.rand(1) * 20
        
        return data, class_labels, risk_scores
    
    def _generate_normal(self):
        sample = torch.zeros(self.seq_length, self.num_sensors, 4)
        
        # Temperature: 20-25°C
        sample[:, :, 0] = 22 + torch.randn(self.seq_length, self.num_sensors) * 2
        sample[:, :, 0] = torch.clamp(sample[:, :, 0], 18, 28)
        
        # PM2.5: 5-20 μg/m³
        sample[:, :, 1] = 10 + torch.randn(self.seq_length, self.num_sensors) * 5
        sample[:, :, 1] = torch.clamp(sample[:, :, 1], 5, 25)
        
        # CO2: 400-600 ppm
        sample[:, :, 2] = 450 + torch.randn(self.seq_length, self.num_sensors) * 50
        sample[:, :, 2] = torch.clamp(sample[:, :, 2], 350, 650)
        
        # Audio: 35-50 dB
        sample[:, :, 3] = 40 + torch.randn(self.seq_length, self.num_sensors) * 5
        sample[:, :, 3] = torch.clamp(sample[:, :, 3], 30, 55)
        
        return sample
    
    def _generate_cooking(self):
        sample = torch.zeros(self.seq_length, self.num_sensors, 4)
        
        # Temperature: gradual rise to 28-35°C
        temp_base = torch.linspace(22, 32, self.seq_length).unsqueeze(1)
        sample[:, :, 0] = temp_base + torch.randn(self.seq_length, self.num_sensors) * 2
        sample[:, :, 0] = torch.clamp(sample[:, :, 0], 20, 40)
        
        # PM2.5: elevated 25-60 μg/m³
        pm25_base = torch.linspace(15, 45, self.seq_length).unsqueeze(1)
        sample[:, :, 1] = pm25_base + torch.randn(self.seq_length, self.num_sensors) * 8
        sample[:, :, 1] = torch.clamp(sample[:, :, 1], 15, 70)
        
        # CO2: elevated 500-800 ppm
        co2_base = torch.linspace(450, 700, self.seq_length).unsqueeze(1)
        sample[:, :, 2] = co2_base + torch.randn(self.seq_length, self.num_sensors) * 50
        sample[:, :, 2] = torch.clamp(sample[:, :, 2], 400, 900)
        
        # Audio: 45-65 dB
        sample[:, :, 3] = 55 + torch.randn(self.seq_length, self.num_sensors) * 8
        sample[:, :, 3] = torch.clamp(sample[:, :, 3], 40, 70)
        
        return sample
    
    def _generate_fire(self):
        sample = torch.zeros(self.seq_length, self.num_sensors, 4)
        
        # Temperature: rapid rise to 50-80°C
        temp_base = torch.linspace(25, 70, self.seq_length).unsqueeze(1)
        sample[:, :, 0] = temp_base + torch.randn(self.seq_length, self.num_sensors) * 5
        sample[:, :, 0] = torch.clamp(sample[:, :, 0], 25, 85)
        
        # PM2.5: high 80-200 μg/m³
        pm25_base = torch.linspace(20, 150, self.seq_length).unsqueeze(1)
        sample[:, :, 1] = pm25_base + torch.randn(self.seq_length, self.num_sensors) * 20
        sample[:, :, 1] = torch.clamp(sample[:, :, 1], 50, 200)
        
        # CO2: high 800-1500 ppm
        co2_base = torch.linspace(500, 1200, self.seq_length).unsqueeze(1)
        sample[:, :, 2] = co2_base + torch.randn(self.seq_length, self.num_sensors) * 100
        sample[:, :, 2] = torch.clamp(sample[:, :, 2], 600, 1800)
        
        # Audio: 60-85 dB
        sample[:, :, 3] = 75 + torch.randn(self.seq_length, self.num_sensors) * 8
        sample[:, :, 3] = torch.clamp(sample[:, :, 3], 60, 90)
        
        return sample
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.class_labels[idx], self.risk_scores[idx]

def train_model(args):
    """Main training function for SageMaker."""
    
    logger.info("Starting SageMaker training job...")
    logger.info(f"Arguments: {args}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model configuration
    config = ModelConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Create model
    model = SimplifiedTransformer(config).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset
    dataset = SyntheticDataset(num_samples=args.num_samples)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Loss functions and optimizer
    class_criterion = nn.CrossEntropyLoss()
    risk_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_class_loss = 0.0
        train_risk_loss = 0.0
        
        for batch_data, batch_classes, batch_risks in train_loader:
            batch_data = batch_data.to(device)
            batch_classes = batch_classes.to(device)
            batch_risks = batch_risks.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            class_loss = class_criterion(outputs['logits'], batch_classes)
            risk_loss = risk_criterion(outputs['risk_score'], batch_risks)
            total_loss = class_loss + 0.5 * risk_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_class_loss += class_loss.item()
            train_risk_loss += risk_loss.item()
        
        # Validation phase
        model.eval()
        val_class_loss = 0.0
        val_risk_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data, batch_classes, batch_risks in val_loader:
                batch_data = batch_data.to(device)
                batch_classes = batch_classes.to(device)
                batch_risks = batch_risks.to(device).unsqueeze(1)
                
                outputs = model(batch_data)
                
                class_loss = class_criterion(outputs['logits'], batch_classes)
                risk_loss = risk_criterion(outputs['risk_score'], batch_risks)
                
                val_class_loss += class_loss.item()
                val_risk_loss += risk_loss.item()
                
                predicted_classes = torch.argmax(outputs['logits'], dim=1)
                correct_predictions += (predicted_classes == batch_classes).sum().item()
                total_samples += batch_classes.size(0)
        
        scheduler.step()
        
        # Calculate averages
        train_class_loss /= len(train_loader)
        train_risk_loss /= len(train_loader)
        val_class_loss /= len(val_loader)
        val_risk_loss /= len(val_loader)
        val_accuracy = correct_predictions / total_samples
        
        val_total_loss = val_class_loss + 0.5 * val_risk_loss
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'epoch': epoch,
                'val_loss': val_total_loss,
                'val_accuracy': val_accuracy
            }, os.path.join(args.model_dir, 'model.pth'))
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                       f"Train Loss: {train_class_loss:.4f}/{train_risk_loss:.4f}, "
                       f"Val Loss: {val_class_loss:.4f}/{val_risk_loss:.4f}, "
                       f"Val Acc: {val_accuracy:.3f}")
    
    logger.info("Training completed successfully!")
    
    # Save final metrics
    metrics = {
        'final_val_accuracy': val_accuracy,
        'final_val_loss': val_total_loss,
        'best_val_loss': best_val_loss,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'training_completed': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-samples', type=int, default=15000)
    
    # Model arguments
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    train_model(args)
'''
        
        # Save training script
        script_path = Path('sagemaker_training_script.py')
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        # Create source code archive
        source_dir = Path('source_code')
        source_dir.mkdir(exist_ok=True)
        
        # Copy necessary files
        script_path.rename(source_dir / 'train.py')
        
        # Create requirements.txt for SageMaker
        requirements = '''
torch>=1.12.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
'''
        
        with open(source_dir / 'requirements.txt', 'w') as f:
            f.write(requirements)
        
        # Create tar.gz archive
        archive_path = 'saafe_training_code.tar.gz'
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(source_dir, arcname='.')
        
        # Upload to S3
        s3_key = 'training-code/saafe_training_code.tar.gz'
        self.s3_client.upload_file(archive_path, self.bucket_name, s3_key)
        
        s3_uri = f's3://{self.bucket_name}/{s3_key}'
        logger.info(f"Training code uploaded to: {s3_uri}")
        
        # Cleanup
        os.remove(archive_path)
        import shutil
        shutil.rmtree(source_dir)
        
        return s3_uri
    
    def launch_training_job(self, 
                          instance_type: str = 'ml.p3.2xlarge',
                          instance_count: int = 1,
                          max_runtime_hours: int = 2,
                          hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """Launch SageMaker training job."""
        
        logger.info("Launching SageMaker training job...")
        
        # Prepare training code
        source_uri = self.prepare_training_code()
        
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                'epochs': 100,
                'batch-size': 32,
                'learning-rate': 0.001,
                'num-samples': 15000,
                'd-model': 256,
                'num-heads': 8,
                'num-layers': 6,
                'dropout': 0.1
            }
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point='train.py',
            source_dir=source_uri,
            role=self.role_arn,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='1.12.0',
            py_version='py38',
            hyperparameters=hyperparameters,
            max_run=max_runtime_hours * 3600,  # Convert to seconds
            output_path=f's3://{self.bucket_name}/training-output/',
            code_location=f's3://{self.bucket_name}/training-code/',
            sagemaker_session=self.sagemaker_session,
            tags=[
                {'Key': 'Project', 'Value': 'Saafe'},
                {'Key': 'Environment', 'Value': 'Training'},
                {'Key': 'Owner', 'Value': 'SaafeAI'}
            ]
        )
        
        # Launch training job
        job_name = f"saafe-training-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        
        logger.info(f"Starting training job: {job_name}")
        logger.info(f"Instance type: {instance_type}")
        logger.info(f"Instance count: {instance_count}")
        logger.info(f"Max runtime: {max_runtime_hours} hours")
        
        estimator.fit(job_name=job_name, wait=False)
        
        logger.info(f"Training job launched successfully!")
        logger.info(f"Job name: {job_name}")
        logger.info(f"Monitor progress in AWS Console or use: aws sagemaker describe-training-job --training-job-name {job_name}")
        
        return job_name
    
    def monitor_training_job(self, job_name: str) -> Dict[str, Any]:
        """Monitor training job progress."""
        
        sagemaker_client = self.session.client('sagemaker')
        
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            
            status = response['TrainingJobStatus']
            
            logger.info(f"Training Job: {job_name}")
            logger.info(f"Status: {status}")
            
            if 'TrainingStartTime' in response:
                logger.info(f"Started: {response['TrainingStartTime']}")
            
            if 'TrainingEndTime' in response:
                logger.info(f"Ended: {response['TrainingEndTime']}")
            
            if 'BillableTimeInSeconds' in response:
                billable_time = response['BillableTimeInSeconds']
                logger.info(f"Billable time: {billable_time // 60} minutes")
            
            if status == 'Failed' and 'FailureReason' in response:
                logger.error(f"Failure reason: {response['FailureReason']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error monitoring training job: {e}")
            return {}
    
    def download_trained_model(self, job_name: str, local_path: str = 'models/') -> bool:
        """Download trained model from S3."""
        
        logger.info(f"Downloading trained model from job: {job_name}")
        
        try:
            # Get training job details
            sagemaker_client = self.session.client('sagemaker')
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            
            if response['TrainingJobStatus'] != 'Completed':
                logger.error(f"Training job not completed. Status: {response['TrainingJobStatus']}")
                return False
            
            # Download model artifacts
            model_s3_uri = response['ModelArtifacts']['S3ModelArtifacts']
            
            # Parse S3 URI
            s3_parts = model_s3_uri.replace('s3://', '').split('/')
            bucket = s3_parts[0]
            key = '/'.join(s3_parts[1:])
            
            # Download model.tar.gz
            local_model_path = Path(local_path)
            local_model_path.mkdir(parents=True, exist_ok=True)
            
            model_archive = local_model_path / 'model.tar.gz'
            self.s3_client.download_file(bucket, key, str(model_archive))
            
            # Extract model
            with tarfile.open(model_archive, 'r:gz') as tar:
                tar.extractall(local_model_path)
            
            # Remove archive
            model_archive.unlink()
            
            logger.info(f"Model downloaded to: {local_model_path}")
            
            # Convert to our format if needed
            self._convert_sagemaker_model(local_model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
    
    def _convert_sagemaker_model(self, model_path: Path):
        """Convert SageMaker model to our format."""
        
        try:
            # Load SageMaker model
            sagemaker_model_path = model_path / 'model.pth'
            if not sagemaker_model_path.exists():
                logger.warning("SageMaker model file not found")
                return
            
            checkpoint = torch.load(sagemaker_model_path, map_location='cpu')
            
            # Create our model metadata
            metadata = {
                'version': '2.0.0',
                'created': datetime.now().isoformat(),
                'description': 'Saafe MVP AI models - AWS SageMaker trained',
                'transformer_model': {
                    'architecture': 'SpatioTemporalTransformer',
                    'parameters': checkpoint.get('total_parameters', 0),
                    'accuracy': checkpoint.get('val_accuracy', 0.0),
                    'training_loss': checkpoint.get('val_loss', 0.0)
                },
                'training_info': {
                    'data_source': 'synthetic_aws',
                    'platform': 'AWS SageMaker',
                    'device': 'GPU',
                    'epochs_trained': checkpoint.get('epoch', 0)
                }
            }
            
            # Save metadata
            with open(model_path / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Rename model file to our convention
            sagemaker_model_path.rename(model_path / 'transformer_model.pth')
            
            logger.info("Model converted to Saafe format")
            
        except Exception as e:
            logger.error(f"Error converting model: {e}")
    
    def create_model_endpoint(self, job_name: str, endpoint_name: Optional[str] = None) -> str:
        """Create SageMaker endpoint for inference."""
        
        if endpoint_name is None:
            endpoint_name = f"saafe-endpoint-{int(time.time())}"
        
        logger.info(f"Creating SageMaker endpoint: {endpoint_name}")
        
        try:
            # Get training job details
            sagemaker_client = self.session.client('sagemaker')
            training_job = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            
            if training_job['TrainingJobStatus'] != 'Completed':
                logger.error("Training job not completed")
                return ""
            
            # Create model
            model_name = f"saafe-model-{int(time.time())}"
            
            sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': training_job['AlgorithmSpecification']['TrainingImage'],
                    'ModelDataUrl': training_job['ModelArtifacts']['S3ModelArtifacts'],
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                    }
                },
                ExecutionRoleArn=self.role_arn
            )
            
            # Create endpoint configuration
            config_name = f"saafe-config-{int(time.time())}"
            
            sagemaker_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.m5.large',
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            # Create endpoint
            sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            
            logger.info(f"Endpoint creation initiated: {endpoint_name}")
            logger.info("Endpoint will be available in 5-10 minutes")
            
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Error creating endpoint: {e}")
            return ""
    
    def cleanup_resources(self):
        """Clean up AWS resources."""
        
        logger.info("Cleaning up AWS resources...")
        
        try:
            # Delete S3 bucket contents
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            
            if 'Contents' in response:
                objects = [{'Key': obj['Key']} for obj in response['Contents']]
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': objects}
                )
            
            # Delete S3 bucket
            self.s3_client.delete_bucket(Bucket=self.bucket_name)
            
            logger.info("AWS resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error cleaning up resources: {e}")


def main():
    """Main function to run AWS training pipeline."""
    
    logger.info("🚀 Starting Saafe AWS Training Pipeline")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = AWSTrainingPipeline(region='us-east-1')
    
    # Training configuration
    training_config = {
        'instance_type': 'ml.p3.2xlarge',  # GPU instance for faster training
        'instance_count': 1,
        'max_runtime_hours': 2,
        'hyperparameters': {
            'epochs': 150,
            'batch-size': 64,  # Larger batch size for GPU
            'learning-rate': 0.001,
            'num-samples': 20000,  # More samples for better training
            'd-model': 256,
            'num-heads': 8,
            'num-layers': 6,
            'dropout': 0.1
        }
    }
    
    # Launch training job
    job_name = pipeline.launch_training_job(**training_config)
    
    logger.info(f"\n📊 Training Job Details:")
    logger.info(f"  Job Name: {job_name}")
    logger.info(f"  Instance: {training_config['instance_type']}")
    logger.info(f"  Max Runtime: {training_config['max_runtime_hours']} hours")
    logger.info(f"  Estimated Cost: $3-6 USD")
    
    logger.info(f"\n🔍 Monitor Progress:")
    logger.info(f"  AWS Console: https://console.aws.amazon.com/sagemaker/")
    logger.info(f"  CLI: aws sagemaker describe-training-job --training-job-name {job_name}")
    
    # Wait for completion (optional)
    wait_for_completion = input("\nWait for training completion? (y/N): ").strip().lower()
    
    if wait_for_completion == 'y':
        logger.info("Waiting for training completion...")
        
        while True:
            status = pipeline.monitor_training_job(job_name)
            
            if status.get('TrainingJobStatus') == 'Completed':
                logger.info("✅ Training completed successfully!")
                
                # Download model
                if pipeline.download_trained_model(job_name):
                    logger.info("✅ Model downloaded successfully!")
                
                # Create endpoint (optional)
                create_endpoint = input("Create SageMaker endpoint? (y/N): ").strip().lower()
                if create_endpoint == 'y':
                    endpoint_name = pipeline.create_model_endpoint(job_name)
                    if endpoint_name:
                        logger.info(f"✅ Endpoint creation started: {endpoint_name}")
                
                break
                
            elif status.get('TrainingJobStatus') == 'Failed':
                logger.error("❌ Training failed!")
                break
                
            elif status.get('TrainingJobStatus') in ['InProgress', 'Starting']:
                logger.info("⏳ Training in progress...")
                time.sleep(60)  # Wait 1 minute
                
            else:
                logger.info(f"Status: {status.get('TrainingJobStatus', 'Unknown')}")
                time.sleep(30)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 AWS Training Pipeline Summary:")
    logger.info(f"  ✅ Training job launched: {job_name}")
    logger.info(f"  📊 Monitor in AWS Console")
    logger.info(f"  💰 Estimated cost: $3-6 USD")
    logger.info(f"  ⏱️  Expected completion: 1-2 hours")
    
    # Cleanup option
    cleanup = input("\nCleanup AWS resources now? (y/N): ").strip().lower()
    if cleanup == 'y':
        pipeline.cleanup_resources()


if __name__ == "__main__":
    main()