#!/usr/bin/env python3
"""
Production Deployment System for Fire Detection Models
Downloads all trained models and creates production-ready AWS deployment
"""

import boto3
import json
import os
import zipfile
import torch
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
import tarfile
import shutil

class FireDetectionDeploymentManager:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.iam_client = boto3.client('iam', region_name=region)
        
        # Deployment configuration
        self.deployment_config = {
            'model_bucket': 'processedd-synthetic-data',
            'deployment_bucket': 'fire-detection-production',
            'model_prefix': 'fire-models/',
            'deployment_prefix': 'production-models/',
            'endpoint_name': f'fire-detection-endpoint-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            'model_name': f'fire-detection-model-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            'instance_type': 'ml.m5.xlarge',  # Production instance
            'min_capacity': 1,
            'max_capacity': 10
        }
        
        print("🔥 Fire Detection Production Deployment Manager Initialized")
        print(f"Region: {region}")
        print(f"Model Bucket: {self.deployment_config['model_bucket']}")
    
    def download_all_models(self, local_dir='./production_models'):
        """Download all trained models from S3"""
        print("\n📥 Downloading all trained models from S3...")
        
        # Create local directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # List all model files
        response = self.s3_client.list_objects_v2(
            Bucket=self.deployment_config['model_bucket'],
            Prefix=self.deployment_config['model_prefix']
        )
        
        downloaded_models = {}
        
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                local_path = os.path.join(local_dir, filename)
                
                # Download file
                self.s3_client.download_file(
                    self.deployment_config['model_bucket'],
                    key,
                    local_path
                )
                
                # Categorize models
                if filename.endswith('.pth'):
                    model_type = 'pytorch'
                elif filename.endswith('.joblib'):
                    model_type = 'sklearn'
                elif filename.endswith('.json'):
                    model_type = 'metadata'
                else:
                    model_type = 'other'
                
                if model_type not in downloaded_models:
                    downloaded_models[model_type] = []
                
                downloaded_models[model_type].append({
                    'filename': filename,
                    'local_path': local_path,
                    's3_key': key,
                    'size_mb': obj['Size'] / (1024 * 1024)
                })
                
                print(f"  ✅ Downloaded: {filename} ({obj['Size'] / (1024 * 1024):.1f} MB)")
        
        print(f"\n📊 Download Summary:")
        for model_type, models in downloaded_models.items():
            print(f"  {model_type}: {len(models)} files")
        
        return downloaded_models, local_dir
    
    def create_inference_code(self, local_dir):
        """Create production inference code"""
        print("\n🔧 Creating production inference code...")
        
        inference_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import json
import os
from typing import Dict, List, Any

class ProductionFireDetector:
    """Production-ready fire detection inference system"""
    
    def __init__(self, model_dir="/opt/ml/model"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all models
        self._load_models()
        
    def _load_models(self):
        """Load all trained models"""
        print("Loading production models...")
        
        # Load PyTorch models
        for file in os.listdir(self.model_dir):
            if file.endswith('.pth'):
                model_name = file.replace('.pth', '').split('_model_')[0]
                model_path = os.path.join(self.model_dir, file)
                
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Recreate model based on class name
                    if 'transformer' in model_name.lower():
                        model = QuickFireTransformer()
                    elif 'lstm' in model_name.lower():
                        model = QuickLSTMCNN()
                    else:
                        continue  # Skip unknown models
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    model.eval()
                    
                    self.models[model_name] = model
                    print(f"  ✅ Loaded PyTorch model: {model_name}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to load {file}: {e}")
            
            elif file.endswith('.joblib'):
                model_name = file.replace('.joblib', '').split('_model_')[0]
                model_path = os.path.join(self.model_dir, file)
                
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    print(f"  ✅ Loaded sklearn model: {model_name}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to load {file}: {e}")
            
            elif file.endswith('.json'):
                with open(os.path.join(self.model_dir, file), 'r') as f:
                    self.metadata[file] = json.load(f)
    
    def preprocess_input(self, data):
        """Preprocess input data for inference"""
        if isinstance(data, dict):
            # Handle area-specific data
            processed_data = []
            for area_name in ['kitchen', 'electrical', 'laundry_hvac', 'living_bedroom', 'basement_storage']:
                if area_name in data:
                    area_data = np.array(data[area_name])
                    if area_data.ndim == 1:
                        area_data = area_data.reshape(1, -1)
                    processed_data.append(area_data)
            
            # Combine all areas
            if processed_data:
                combined_data = np.concatenate(processed_data, axis=1)
                return combined_data
        
        elif isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data
        
        return data
    
    def predict_ensemble(self, input_data):
        """Get ensemble predictions from all models"""
        predictions = {}
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # PyTorch models
        if len(processed_data.shape) == 2:
            # Reshape for sequence models (batch, seq_len, features)
            seq_len = min(60, processed_data.shape[1] // 3)  # Assume 3 features max
            features = processed_data.shape[1] // seq_len
            
            if processed_data.shape[1] >= seq_len * 3:
                tensor_data = torch.FloatTensor(
                    processed_data[:, :seq_len*3].reshape(-1, seq_len, 3)
                ).to(self.device)
            else:
                # Pad if necessary
                padded_data = np.zeros((processed_data.shape[0], seq_len * 3))
                padded_data[:, :processed_data.shape[1]] = processed_data
                tensor_data = torch.FloatTensor(
                    padded_data.reshape(-1, seq_len, 3)
                ).to(self.device)
        
        # Get predictions from PyTorch models
        for model_name, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                try:
                    with torch.no_grad():
                        output = model(tensor_data)
                        
                        predictions[model_name] = {
                            'fire_probability': output['fire_probability'].cpu().numpy(),
                            'lead_time_logits': output['lead_time_logits'].cpu().numpy(),
                            'lead_time_probs': torch.softmax(output['lead_time_logits'], dim=1).cpu().numpy()
                        }
                except Exception as e:
                    print(f"Error in {model_name}: {e}")
        
        # Get predictions from sklearn models (using engineered features)
        try:
            features = self._engineer_features(processed_data)
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(features)
                        predictions[model_name] = {
                            'lead_time_probs': proba,
                            'predicted_class': model.predict(features)
                        }
                    except Exception as e:
                        print(f"Error in {model_name}: {e}")
        
        except Exception as e:
            print(f"Feature engineering error: {e}")
        
        return predictions
    
    def _engineer_features(self, data):
        """Engineer features for sklearn models"""
        if data.ndim == 2 and data.shape[0] == 1:
            data = data.flatten()
        
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.median(data)
        ])
        
        # Percentiles
        features.extend([
            np.percentile(data, 25),
            np.percentile(data, 75)
        ])
        
        # Trend
        if len(data) > 1:
            slope = np.polyfit(range(len(data)), data, 1)[0]
            features.append(slope)
        else:
            features.append(0)
        
        # Changes
        if len(data) > 1:
            diff = np.diff(data)
            features.extend([
                np.mean(np.abs(diff)),
                np.std(diff)
            ])
        else:
            features.extend([0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def get_final_prediction(self, input_data):
        """Get final ensemble prediction with confidence"""
        predictions = self.predict_ensemble(input_data)
        
        if not predictions:
            return {
                'error': 'No models available for prediction',
                'fire_probability': 0.0,
                'lead_time': 'unknown',
                'confidence': 0.0
            }
        
        # Aggregate predictions
        fire_probs = []
        lead_time_probs = []
        
        for model_name, pred in predictions.items():
            if 'fire_probability' in pred:
                fire_probs.append(pred['fire_probability'].flatten()[0])
            
            if 'lead_time_probs' in pred:
                lead_time_probs.append(pred['lead_time_probs'][0])
        
        # Ensemble fire probability
        final_fire_prob = np.mean(fire_probs) if fire_probs else 0.0
        
        # Ensemble lead time
        if lead_time_probs:
            avg_lead_time_probs = np.mean(lead_time_probs, axis=0)
            predicted_lead_time_class = np.argmax(avg_lead_time_probs)
            lead_time_confidence = np.max(avg_lead_time_probs)
        else:
            predicted_lead_time_class = 3
            lead_time_confidence = 0.0
        
        # Map lead time class to description
        lead_time_map = {
            0: 'immediate',
            1: 'hours', 
            2: 'days',
            3: 'weeks'
        }
        
        # Overall confidence (agreement between models)
        confidence = 1.0 - np.std(fire_probs) if len(fire_probs) > 1 else 0.5
        
        return {
            'fire_probability': float(final_fire_prob),
            'lead_time': lead_time_map.get(predicted_lead_time_class, 'unknown'),
            'lead_time_class': int(predicted_lead_time_class),
            'confidence': float(confidence),
            'num_models': len(predictions),
            'individual_predictions': predictions,
            'alert_level': self._get_alert_level(final_fire_prob, predicted_lead_time_class)
        }
    
    def _get_alert_level(self, fire_prob, lead_time_class):
        """Determine alert level based on predictions"""
        if fire_prob > 0.8 and lead_time_class <= 1:
            return 'CRITICAL'
        elif fire_prob > 0.6 and lead_time_class <= 2:
            return 'HIGH'
        elif fire_prob > 0.4:
            return 'MEDIUM'
        elif fire_prob > 0.2:
            return 'LOW'
        else:
            return 'NORMAL'

# Model classes (simplified versions for production)
class QuickFireTransformer(nn.Module):
    def __init__(self, input_dim=3, seq_len=60, d_model=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*2, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.fire_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        
        self.lead_time_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        
        return {
            'fire_probability': self.fire_head(x),
            'lead_time_logits': self.lead_time_head(x)
        }

class QuickLSTMCNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, dropout=0.2)
        
        self.fire_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        
        self.lead_time_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        return {
            'fire_probability': self.fire_head(x),
            'lead_time_logits': self.lead_time_head(x)
        }

# SageMaker inference functions
def model_fn(model_dir):
    """Load model for SageMaker inference"""
    return ProductionFireDetector(model_dir)

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction"""
    return model.get_final_prediction(input_data)

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
        
        # Save inference code
        inference_path = os.path.join(local_dir, 'inference.py')
        with open(inference_path, 'w') as f:
            f.write(inference_code)
        
        print(f"  ✅ Created inference code: {inference_path}")
        
        # Create requirements.txt
        requirements = '''
torch>=1.9.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
pandas>=1.3.0
'''
        
        requirements_path = os.path.join(local_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            f.write(requirements)
        
        print(f"  ✅ Created requirements: {requirements_path}")
        
        return inference_path, requirements_path

print("✅ Production Deployment Manager defined!")