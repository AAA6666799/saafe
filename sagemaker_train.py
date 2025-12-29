#!/usr/bin/env python3
"""
SageMaker Training Script for Saafe Fire Detection Models
Trains both transformer and anti-hallucination models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
import argparse
from datetime import datetime

# Import model architectures
from saafe_mvp.models.transformer import SpatioTemporalTransformer, ModelConfig
from saafe_mvp.models.anti_hallucination import EnsembleFireDetector

def generate_synthetic_data(num_samples=50000):
    """Generate synthetic fire detection training data"""
    
    # Create synthetic sensor data (batch, seq_len, sensors, features)
    batch_size = num_samples
    seq_len = 60
    num_sensors = 4
    num_features = 4
    
    data = torch.zeros(batch_size, seq_len, num_sensors, num_features)
    labels = torch.zeros(batch_size, 3)  # normal, cooking, fire
    
    for i in range(batch_size):
        scenario = np.random.choice(['normal', 'cooking', 'fire'], p=[0.6, 0.3, 0.1])
        
        if scenario == 'normal':
            # Normal conditions
            data[i, :, :, 0] = torch.normal(22, 3, (seq_len, num_sensors))  # temp
            data[i, :, :, 1] = torch.normal(15, 5, (seq_len, num_sensors))  # pm25
            data[i, :, :, 2] = torch.normal(400, 50, (seq_len, num_sensors))  # co2
            data[i, :, :, 3] = torch.normal(40, 10, (seq_len, num_sensors))  # audio
            labels[i] = torch.tensor([1, 0, 0])
            
        elif scenario == 'cooking':
            # Cooking scenario
            data[i, :, :, 0] = torch.normal(28, 4, (seq_len, num_sensors))
            data[i, :, :, 1] = torch.normal(35, 10, (seq_len, num_sensors))
            data[i, :, :, 2] = torch.normal(600, 100, (seq_len, num_sensors))
            data[i, :, :, 3] = torch.normal(45, 8, (seq_len, num_sensors))
            labels[i] = torch.tensor([0, 1, 0])
            
        else:  # fire
            # Fire scenario
            data[i, :, :, 0] = torch.normal(45, 8, (seq_len, num_sensors))
            data[i, :, :, 1] = torch.normal(80, 20, (seq_len, num_sensors))
            data[i, :, :, 2] = torch.normal(800, 150, (seq_len, num_sensors))
            data[i, :, :, 3] = torch.normal(65, 15, (seq_len, num_sensors))
            labels[i] = torch.tensor([0, 0, 1])
    
    return data, labels

def train_transformer_model(args):
    """Train the main transformer model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create model
    config = ModelConfig(
        d_model=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.1
    )
    
    model = SpatioTemporalTransformer(config).to(device)
    
    # Generate training data
    print("Generating synthetic training data...")
    data, labels = generate_synthetic_data(args.num_samples)
    data, labels = data.to(device), labels.to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    batch_size = args.batch_size
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Convert to area format for transformer
            area_data = {
                'kitchen': batch_data[:, :, 0, :1],
                'electrical': batch_data[:, :, 1, :1], 
                'laundry_hvac': batch_data[:, :, 2, :2],
                'living_bedroom': batch_data[:, :, 3, :1],
                'basement_storage': batch_data[:, :, 0, :3]
            }
            
            optimizer.zero_grad()
            outputs = model(area_data)
            
            # Use lead_time_logits as class prediction
            loss = criterion(outputs['lead_time_logits'][:, :3], batch_labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/num_batches:.4f}")
    
    return model

def train_anti_hallucination_model(transformer_models):
    """Train anti-hallucination ensemble"""
    
    # Create ensemble with multiple transformer variants
    ensemble = EnsembleFireDetector(transformer_models, voting_strategy='conservative')
    
    # Simple validation - ensemble is rule-based, no training needed
    print("Anti-hallucination ensemble configured")
    
    return ensemble

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=50000)
    
    args = parser.parse_args()
    
    print("🔥 Starting Saafe Model Training")
    print(f"Configuration: {vars(args)}")
    
    # Train main transformer
    print("\n1. Training Transformer Model...")
    transformer = train_transformer_model(args)
    
    # Create ensemble for anti-hallucination
    print("\n2. Creating Anti-Hallucination Ensemble...")
    ensemble = train_anti_hallucination_model([transformer])
    
    # Save models
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save transformer
    torch.save({
        'model_state_dict': transformer.state_dict(),
        'config': transformer.config.__dict__,
        'timestamp': datetime.now().isoformat()
    }, os.path.join(model_dir, 'transformer_model.pth'))
    
    # Save training metrics
    metrics = {
        'training_completed': True,
        'epochs': args.epochs,
        'samples': args.num_samples,
        'model_parameters': sum(p.numel() for p in transformer.parameters()),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(model_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Training completed! Models saved to {model_dir}")

if __name__ == "__main__":
    main()