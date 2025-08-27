#!/usr/bin/env python3
"""
Quick AWS Training Script - Train models locally then upload to S3
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import boto3
import json
import os
from datetime import datetime

class SimpleFireTransformer(nn.Module):
    """Simplified transformer for quick training"""
    
    def __init__(self, input_dim=4, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # normal, cooking, fire
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(), 
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        
        # Global average pooling
        pooled = x.mean(dim=1)
        
        logits = self.classifier(pooled)
        risk_score = self.risk_head(pooled) * 100  # 0-100 scale
        
        return {
            'logits': logits,
            'risk_score': risk_score,
            'features': pooled
        }

def generate_training_data(num_samples=10000):
    """Generate quick training data"""
    
    data = torch.zeros(num_samples, 30, 4)  # 30 timesteps, 4 features
    labels = torch.zeros(num_samples, dtype=torch.long)
    risk_scores = torch.zeros(num_samples, 1)
    
    for i in range(num_samples):
        scenario = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  # normal, cooking, fire
        
        if scenario == 0:  # normal
            data[i, :, 0] = torch.normal(22, 2, (30,))  # temp
            data[i, :, 1] = torch.normal(15, 3, (30,))  # pm25
            data[i, :, 2] = torch.normal(400, 30, (30,))  # co2
            data[i, :, 3] = torch.normal(40, 5, (30,))  # audio
            risk_scores[i] = torch.tensor([np.random.uniform(0, 30)])
            
        elif scenario == 1:  # cooking
            data[i, :, 0] = torch.normal(28, 3, (30,))
            data[i, :, 1] = torch.normal(35, 8, (30,))
            data[i, :, 2] = torch.normal(600, 50, (30,))
            data[i, :, 3] = torch.normal(45, 6, (30,))
            risk_scores[i] = torch.tensor([np.random.uniform(30, 60)])
            
        else:  # fire
            data[i, :, 0] = torch.normal(50, 10, (30,))
            data[i, :, 1] = torch.normal(100, 25, (30,))
            data[i, :, 2] = torch.normal(1000, 100, (30,))
            data[i, :, 3] = torch.normal(70, 10, (30,))
            risk_scores[i] = torch.tensor([np.random.uniform(80, 100)])
        
        labels[i] = scenario
    
    return data, labels, risk_scores

def train_model():
    """Train the model quickly"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Create model
    model = SimpleFireTransformer().to(device)
    
    # Generate data
    print("Generating training data...")
    data, labels, risk_scores = generate_training_data(10000)
    data, labels, risk_scores = data.to(device), labels.to(device), risk_scores.to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    class_criterion = nn.CrossEntropyLoss()
    risk_criterion = nn.MSELoss()
    
    # Quick training
    model.train()
    batch_size = 64
    epochs = 20
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batch_risks = risk_scores[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            class_loss = class_criterion(outputs['logits'], batch_labels)
            risk_loss = risk_criterion(outputs['risk_score'], batch_risks)
            
            total_loss = class_loss + 0.5 * risk_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}")
    
    return model

def main():
    print("🔥 Quick AWS Training Pipeline")
    print("=" * 40)
    
    # Train model
    print("1. Training transformer model...")
    model = train_model()
    
    # Save locally
    print("\n2. Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'SimpleFireTransformer',
        'timestamp': datetime.now().isoformat(),
        'parameters': sum(p.numel() for p in model.parameters())
    }, 'models/transformer_model.pth')
    
    print("✅ Model trained and saved to models/transformer_model.pth")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()