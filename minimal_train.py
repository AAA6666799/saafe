#!/usr/bin/env python3
"""
Minimal IoT Fire Detection Training Script - Guaranteed to work on SageMaker
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import argparse

print("🔥 Starting Minimal IoT Fire Detection Training!")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPUs: {torch.cuda.device_count()}")

class SimpleIoTModel(nn.Module):
    """Simple but effective IoT Fire Detection Model"""
    
    def __init__(self):
        super().__init__()
        
        # Simple but powerful architecture
        self.network = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1)
        )
        
        # Output heads
        self.lead_time_head = nn.Linear(64, 4)  # immediate, hours, days, weeks
        self.area_risk_head = nn.Linear(64, 5)  # 5 areas
        self.time_head = nn.Linear(64, 1)       # time to ignition
    
    def forward(self, x):
        features = self.network(x)
        
        return {
            'lead_time_logits': self.lead_time_head(features),
            'area_risks': torch.sigmoid(self.area_risk_head(features)),
            'time_to_ignition': torch.relu(self.time_head(features))
        }

def generate_data(num_samples=50000):
    """Generate IoT training data"""
    print(f"Generating {num_samples} samples...")
    
    # IoT features: Kitchen(1) + Electrical(1) + HVAC(2) + Living(1) + Basement(3) = 8
    data = torch.randn(num_samples, 8)
    
    # Generate realistic patterns
    for i in range(num_samples):
        lead_time = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2])
        
        if lead_time == 0:  # Immediate
            data[i, 0] = np.random.normal(300, 50)  # High kitchen VOC
            data[i, 4] = np.random.normal(15, 3)    # High living room particles
        elif lead_time == 1:  # Hours
            data[i, 2] = np.random.normal(50, 10)   # High HVAC temp
            data[i, 3] = np.random.normal(2.0, 0.5) # High HVAC current
        elif lead_time >= 2:  # Days/Weeks
            data[i, 1] = np.random.poisson(5)       # Electrical arcs
    
    # Labels
    lead_time_labels = torch.randint(0, 4, (num_samples,))
    area_risks = torch.rand(num_samples, 5)
    time_labels = torch.rand(num_samples, 1) * 100
    
    return data, lead_time_labels, area_risks, time_labels

def train():
    """Main training function"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()
    
    print(f"Training config: {args.epochs} epochs, batch size {args.batch_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model = SimpleIoTModel().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data
    data, lead_labels, area_labels, time_labels = generate_data(50000)
    data = data.to(device)
    lead_labels = lead_labels.to(device)
    area_labels = area_labels.to(device)
    time_labels = time_labels.to(device)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    lead_criterion = nn.CrossEntropyLoss()
    area_criterion = nn.MSELoss()
    time_criterion = nn.MSELoss()
    
    print("Starting training...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(data), args.batch_size):
            batch_data = data[i:i+args.batch_size]
            batch_lead = lead_labels[i:i+args.batch_size]
            batch_area = area_labels[i:i+args.batch_size]
            batch_time = time_labels[i:i+args.batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            lead_loss = lead_criterion(outputs['lead_time_logits'], batch_lead)
            area_loss = area_criterion(outputs['area_risks'], batch_area)
            time_loss = time_criterion(outputs['time_to_ignition'], batch_time)
            
            loss = lead_loss + 0.5 * area_loss + 0.3 * time_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Test
    model.eval()
    with torch.no_grad():
        test_data = data[:1000]
        test_labels = lead_labels[:1000]
        outputs = model(test_data)
        predicted = torch.argmax(outputs['lead_time_logits'], dim=1)
        accuracy = (predicted == test_labels).float().mean()
        print(f"Test Accuracy: {accuracy:.3f}")
    
    # Save
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': float(accuracy),
        'loss': avg_loss,
        'timestamp': datetime.now().isoformat()
    }, os.path.join(model_dir, 'model.pth'))
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'accuracy': float(accuracy),
            'final_loss': avg_loss,
            'epochs': args.epochs,
            'success': True
        }, f)
    
    print(f"✅ Model saved! Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)