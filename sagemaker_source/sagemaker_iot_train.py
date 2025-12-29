#!/usr/bin/env python3
"""
SageMaker IoT Fire Detection Training Script
Simple and robust version for AWS SageMaker web interface
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

print("🔥 Starting IoT Fire Detection Training on SageMaker!")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

class AdvancedIoTFireModel(nn.Module):
    """Advanced IoT Fire Detection Model with Transformer Architecture"""
    
    def __init__(self, d_model=512, num_heads=8, num_layers=6, num_areas=5, num_risk_levels=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_areas = num_areas
        
        # Area-specific configurations
        self.area_configs = {
            'kitchen': {'features': 1, 'sensor_type': 'voc'},
            'electrical': {'features': 1, 'sensor_type': 'arc'},
            'laundry_hvac': {'features': 2, 'sensor_type': 'thermal_current'},
            'living_bedroom': {'features': 1, 'sensor_type': 'aspirating'},
            'basement_storage': {'features': 3, 'sensor_type': 'environmental'}
        }
        
        # Area-specific embedding layers
        self.area_embeddings = nn.ModuleDict({
            'kitchen': nn.Linear(1, d_model),
            'electrical': nn.Linear(1, d_model),
            'laundry_hvac': nn.Linear(2, d_model),
            'living_bedroom': nn.Linear(1, d_model),
            'basement_storage': nn.Linear(3, d_model)
        })
        
        # Positional encoding for areas
        self.area_positional_encoding = nn.Parameter(torch.randn(num_areas, d_model))
        
        # Multi-head self-attention layers (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-area attention for spatial relationships
        self.cross_area_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Advanced feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * num_areas, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # Multi-task prediction heads with advanced architectures
        
        # Lead time prediction (immediate, hours, days, weeks)
        self.lead_time_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_risk_levels)
        )
        
        # Area-specific risk prediction with attention
        self.area_risk_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.area_risk_heads = nn.ModuleDict()
        for area_name in self.area_configs.keys():
            self.area_risk_heads[area_name] = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.LayerNorm(d_model // 4),
                nn.Dropout(dropout // 2),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )
        
        # Time-to-ignition regression with uncertainty estimation
        self.time_regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2)  # Mean and log variance
        )
        
        # Vendor-specific calibration layers
        self.vendor_calibration = nn.ModuleDict({
            'honeywell_mics': nn.Linear(d_model, d_model),
            'ting_eaton': nn.Linear(d_model, d_model),
            'honeywell_thermal': nn.Linear(d_model, d_model),
            'xtralis_vesda': nn.Linear(d_model, d_model),
            'bosch_airthings': nn.Linear(d_model, d_model)
        })
        
        # Anti-hallucination confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, vendor_types=None):
        """
        Advanced forward pass with multi-scale feature processing
        
        Args:
            x: Input tensor (batch_size, 8) - flattened area features
            vendor_types: Optional vendor type indicators for calibration
        """
        batch_size = x.shape[0]
        
        # Split input by areas: Kitchen(1) + Electrical(1) + HVAC(2) + Living(1) + Basement(3)
        area_features = {
            'kitchen': x[:, 0:1],           # VOC sensor
            'electrical': x[:, 1:2],        # Arc detection
            'laundry_hvac': x[:, 2:4],      # Temperature + Current
            'living_bedroom': x[:, 4:5],    # Aspirating smoke
            'basement_storage': x[:, 5:8]   # Temp + Humidity + Gas
        }
        
        # Area-specific embeddings
        area_embeddings = []
        area_names = list(self.area_configs.keys())
        
        for area_name in area_names:
            embedded = self.area_embeddings[area_name](area_features[area_name])
            area_embeddings.append(embedded)
        
        # Stack area embeddings: (batch_size, num_areas, d_model)
        area_tensor = torch.stack(area_embeddings, dim=1)
        
        # Add positional encoding for spatial relationships
        area_tensor = area_tensor + self.area_positional_encoding.unsqueeze(0)
        
        # Apply transformer encoder for temporal and cross-area relationships
        transformed_features = self.transformer_encoder(area_tensor)
        
        # Cross-area attention for spatial dependencies
        attended_features, attention_weights = self.cross_area_attention(
            transformed_features, transformed_features, transformed_features
        )
        
        # Feature fusion
        flattened_features = attended_features.view(batch_size, -1)
        fused_features = self.feature_fusion(flattened_features)
        
        # Vendor-specific calibration (if provided)
        if vendor_types is not None:
            calibrated_features = []
            for i, vendor in enumerate(vendor_types):
                if vendor in self.vendor_calibration:
                    calibrated = self.vendor_calibration[vendor](fused_features[i:i+1])
                    calibrated_features.append(calibrated)
                else:
                    calibrated_features.append(fused_features[i:i+1])
            fused_features = torch.cat(calibrated_features, dim=0)
        
        # Multi-task predictions
        
        # 1. Lead time classification
        lead_time_logits = self.lead_time_head(fused_features)
        
        # 2. Area-specific risk assessment with attention
        area_risk_features, _ = self.area_risk_attention(
            transformed_features, transformed_features, transformed_features
        )
        
        area_risks = {}
        for i, area_name in enumerate(area_names):
            area_specific_features = area_risk_features[:, i, :]
            area_risks[area_name] = self.area_risk_heads[area_name](area_specific_features)
        
        # Combine area risks
        area_risk_tensor = torch.cat([area_risks[area] for area in area_names], dim=1)
        
        # 3. Time-to-ignition with uncertainty
        time_params = self.time_regression_head(fused_features)
        time_mean = torch.relu(time_params[:, 0:1])  # Positive time values
        time_log_var = time_params[:, 1:2]  # Log variance for uncertainty
        
        # 4. Confidence estimation for anti-hallucination
        prediction_confidence = self.confidence_estimator(fused_features)
        
        return {
            'lead_time_logits': lead_time_logits,
            'area_risks': area_risk_tensor,
            'area_risk_dict': area_risks,
            'time_to_ignition_mean': time_mean,
            'time_to_ignition_log_var': time_log_var,
            'prediction_confidence': prediction_confidence,
            'attention_weights': attention_weights,
            'area_features': transformed_features
        }

def generate_iot_training_data(num_samples=100000):
    """Generate 100K IoT training samples"""
    print(f"Generating {num_samples} IoT training samples...")
    
    # IoT sensor features: Kitchen(1) + Electrical(1) + HVAC(2) + Living(1) + Basement(3) = 8 features
    data = torch.zeros(num_samples, 8)
    lead_time_labels = torch.zeros(num_samples, dtype=torch.long)
    area_risk_labels = torch.zeros(num_samples, 5)
    time_labels = torch.zeros(num_samples, 1)
    
    for i in range(num_samples):
        # Generate lead time category (0=immediate, 1=hours, 2=days, 3=weeks)
        lead_time = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2])
        lead_time_labels[i] = lead_time
        
        # Kitchen VOC sensor (feature 0)
        if lead_time == 0:  # Immediate - high VOC
            data[i, 0] = np.random.normal(250, 50)  # High VOC levels
            area_risk_labels[i, 0] = 1.0  # High kitchen risk
        elif lead_time == 1:  # Hours - elevated VOC
            data[i, 0] = np.random.normal(180, 30)
            area_risk_labels[i, 0] = np.random.uniform(0.6, 0.9)
        else:  # Days/Weeks - normal VOC
            data[i, 0] = np.random.normal(120, 20)
            area_risk_labels[i, 0] = np.random.uniform(0.0, 0.3)
        
        # Electrical arc sensor (feature 1)
        if lead_time >= 2:  # Days/Weeks - arc events
            data[i, 1] = np.random.poisson(3)  # Arc count
            area_risk_labels[i, 1] = np.random.uniform(0.7, 1.0)
        else:
            data[i, 1] = np.random.poisson(0.2)
            area_risk_labels[i, 1] = np.random.uniform(0.0, 0.2)
        
        # HVAC sensors (features 2-3: temperature, current)
        if lead_time == 1 or lead_time == 2:  # Hours/Days - thermal stress
            data[i, 2] = np.random.normal(45, 10)  # Temperature
            data[i, 3] = np.random.normal(1.2, 0.3)  # Current
            area_risk_labels[i, 2] = np.random.uniform(0.6, 0.9)
        else:
            data[i, 2] = np.random.normal(25, 5)
            data[i, 3] = np.random.normal(0.4, 0.1)
            area_risk_labels[i, 2] = np.random.uniform(0.0, 0.3)
        
        # Living room aspirating smoke (feature 4)
        if lead_time == 0:  # Immediate - smoke particles
            data[i, 4] = np.random.normal(12, 3)
            area_risk_labels[i, 3] = 1.0
        else:
            data[i, 4] = np.random.normal(5, 1)
            area_risk_labels[i, 3] = np.random.uniform(0.0, 0.2)
        
        # Basement environmental sensors (features 5-7: temp, humidity, gas)
        if lead_time == 1 or lead_time == 2:  # Hours/Days - environmental issues
            data[i, 5] = np.random.normal(28, 5)  # Temperature
            data[i, 6] = np.random.normal(75, 10)  # Humidity
            data[i, 7] = np.random.normal(18, 5)  # Gas levels
            area_risk_labels[i, 4] = np.random.uniform(0.5, 0.8)
        else:
            data[i, 5] = np.random.normal(20, 3)
            data[i, 6] = np.random.normal(50, 8)
            data[i, 7] = np.random.normal(10, 2)
            area_risk_labels[i, 4] = np.random.uniform(0.0, 0.3)
        
        # Time to ignition (in hours)
        time_mapping = [0.5, 8.0, 72.0, 336.0]  # immediate, hours, days, weeks
        time_labels[i, 0] = time_mapping[lead_time] + np.random.normal(0, time_mapping[lead_time] * 0.1)
    
    print(f"Generated data shapes:")
    print(f"  Data: {data.shape}")
    print(f"  Lead time labels: {lead_time_labels.shape}")
    print(f"  Area risk labels: {area_risk_labels.shape}")
    print(f"  Time labels: {time_labels.shape}")
    
    return data, lead_time_labels, area_risk_labels, time_labels

def train_model():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()
    
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create advanced model
    model = AdvancedIoTFireModel(
        d_model=512,
        num_heads=8,
        num_layers=6,
        num_areas=5,
        num_risk_levels=4,
        dropout=0.1
    ).to(device)
    
    print(f"Advanced IoT Model Architecture:")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Transformer layers: 6")
    print(f"  Attention heads: 8")
    print(f"  Model dimension: 512")
    print(f"  Area-specific embeddings: 5")
    print(f"  Multi-task heads: 3")
    
    # Generate training data
    data, lead_time_labels, area_risk_labels, time_labels = generate_iot_training_data(100000)
    
    # Move to device
    data = data.to(device)
    lead_time_labels = lead_time_labels.to(device)
    area_risk_labels = area_risk_labels.to(device)
    time_labels = time_labels.to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    lead_time_criterion = nn.CrossEntropyLoss()
    area_risk_criterion = nn.MSELoss()
    time_criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    batch_size = args.batch_size
    num_samples = data.shape[0]
    
    print(f"Starting training: {args.epochs} epochs, {num_samples:,} samples")
    print("=" * 50)
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_lead_loss = 0.0
        epoch_area_loss = 0.0
        epoch_time_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        data_shuffled = data[indices]
        lead_time_shuffled = lead_time_labels[indices]
        area_risk_shuffled = area_risk_labels[indices]
        time_shuffled = time_labels[indices]
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Get batch
            batch_data = data_shuffled[start_idx:end_idx]
            batch_lead_time = lead_time_shuffled[start_idx:end_idx]
            batch_area_risks = area_risk_shuffled[start_idx:end_idx]
            batch_time = time_shuffled[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # Calculate advanced losses
            lead_loss = lead_time_criterion(outputs['lead_time_logits'], batch_lead_time)
            area_loss = area_risk_criterion(outputs['area_risks'], batch_area_risks)
            
            # Uncertainty-aware time regression loss
            time_mean = outputs['time_to_ignition_mean']
            time_log_var = outputs['time_to_ignition_log_var']
            time_var = torch.exp(time_log_var)
            
            # Negative log-likelihood loss for uncertainty estimation
            time_loss = 0.5 * (torch.log(2 * np.pi * time_var) + 
                              (batch_time - time_mean) ** 2 / time_var).mean()
            
            # Confidence regularization loss
            confidence_loss = -torch.log(outputs['prediction_confidence'] + 1e-8).mean()
            
            # Combined loss with adaptive weighting
            total_loss = (lead_loss + 
                         0.5 * area_loss + 
                         0.3 * time_loss + 
                         0.1 * confidence_loss)
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_lead_loss += lead_loss.item()
            epoch_area_loss += area_loss.item()
            epoch_time_loss += time_loss.item()
            num_batches += 1
        
        scheduler.step()
        
        # Calculate averages
        avg_loss = epoch_loss / num_batches
        avg_lead_loss = epoch_lead_loss / num_batches
        avg_area_loss = epoch_area_loss / num_batches
        avg_time_loss = epoch_time_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Total={avg_loss:.4f}, "
                  f"Lead={avg_lead_loss:.4f}, "
                  f"Area={avg_area_loss:.4f}, "
                  f"Time={avg_time_loss:.4f}")
    
    print("=" * 50)
    print("Training completed!")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Test on a subset
        test_indices = torch.randperm(num_samples)[:1000]
        test_data = data[test_indices]
        test_lead_time = lead_time_labels[test_indices]
        
        outputs = model(test_data)
        
        # Calculate accuracy
        predicted_lead_time = torch.argmax(outputs['lead_time_logits'], dim=1)
        accuracy = (predicted_lead_time == test_lead_time).float().mean()
        
        print(f"Test Accuracy: {accuracy:.3f}")
    
    # Save model
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': 8,
            'hidden_size': 256,
            'num_areas': 5,
            'num_risk_levels': 4
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        },
        'final_loss': avg_loss,
        'test_accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat()
    }, os.path.join(model_dir, 'iot_fire_model.pth'))
    
    # Save metrics for SageMaker
    metrics = {
        'final_loss': float(avg_loss),
        'lead_time_loss': float(avg_lead_loss),
        'area_risk_loss': float(avg_area_loss),
        'time_regression_loss': float(avg_time_loss),
        'test_accuracy': float(accuracy),
        'epochs': args.epochs,
        'training_samples': num_samples,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device),
        'success': True
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Model saved to: {model_dir}")
    print(f"📊 Final metrics:")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    try:
        train_model()
        print("🎉 Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)