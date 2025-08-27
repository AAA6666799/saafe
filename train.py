#!/usr/bin/env python3
"""
Direct SageMaker Training Script - No dependencies
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

print("🔥 IoT Fire Detection Training - Direct Approach")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

class IoTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.lead_time = nn.Linear(32, 4)
        self.area_risk = nn.Linear(32, 5)
        self.time_pred = nn.Linear(32, 1)
    
    def forward(self, x):
        features = self.net(x)
        return {
            'lead_time': self.lead_time(features),
            'area_risk': torch.sigmoid(self.area_risk(features)),
            'time_pred': torch.relu(self.time_pred(features))
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = IoTModel().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate data
    data = torch.randn(10000, 8).to(device)
    lead_labels = torch.randint(0, 4, (10000,)).to(device)
    area_labels = torch.rand(10000, 5).to(device)
    time_labels = torch.rand(10000, 1).to(device) * 100
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print("Training...")
    for epoch in range(args.epochs):
        total_loss = 0
        for i in range(0, len(data), args.batch_size):
            batch_data = data[i:i+args.batch_size]
            batch_lead = lead_labels[i:i+args.batch_size]
            batch_area = area_labels[i:i+args.batch_size]
            batch_time = time_labels[i:i+args.batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            loss = (nn.CrossEntropyLoss()(outputs['lead_time'], batch_lead) +
                   nn.MSELoss()(outputs['area_risk'], batch_area) +
                   nn.MSELoss()(outputs['time_pred'], batch_time))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(data)*args.batch_size:.4f}")
    
    # Test
    with torch.no_grad():
        test_outputs = model(data[:1000])
        pred = torch.argmax(test_outputs['lead_time'], dim=1)
        acc = (pred == lead_labels[:1000]).float().mean()
        print(f"Accuracy: {acc:.3f}")
    
    # Save
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save({
        'model': model.state_dict(),
        'accuracy': float(acc)
    }, f"{model_dir}/model.pth")
    
    with open(f"{model_dir}/metrics.json", 'w') as f:
        json.dump({'accuracy': float(acc), 'success': True}, f)
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()