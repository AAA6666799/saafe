#!/usr/bin/env python3
"""
PyTorch Training Script for SageMaker
Handles LSTM, GRU, Transformer, and other deep learning models.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class FireDetectionDataset(Dataset):
    """Dataset for fire detection."""
    
    def __init__(self, data, labels, scaler=None, sequence_length=30):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.data_scaled = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            self.data_scaled = self.scaler.transform(data)
        
        # Create sequences for temporal models
        self.sequences, self.sequence_labels = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        labels = []
        
        for i in range(len(self.data_scaled) - self.sequence_length + 1):
            sequences.append(self.data_scaled[i:i + self.sequence_length])
            labels.append(self.labels[i + self.sequence_length - 1])
        
        return np.array(sequences), np.array(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.sequence_labels[idx]])

class LSTMFireClassifier(nn.Module):
    """LSTM-based fire classifier."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        output = lstm_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

class GRUFireClassifier(nn.Module):
    """GRU-based fire classifier."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # GRU forward
        gru_out, _ = self.gru(x)
        
        # Take the last output
        output = gru_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

class TransformerFireClassifier(nn.Module):
    """Transformer-based fire classifier."""
    
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, num_classes=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 
                                                 dim_feedforward=d_model*4, 
                                                 dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def _create_positional_encoding(self, max_length, d_model):
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer forward
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

def train_model(model, train_loader, val_loader, epochs, device):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device).squeeze()
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_precision = precision_score(val_targets, val_preds, average='weighted')
        val_recall = recall_score(val_targets, val_preds, average='weighted')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.4f}')
        print(f'Validation F1: {val_f1:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')
        print('-' * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return {
        'accuracy': best_val_acc,
        'f1_score': val_f1,
        'precision': val_precision,
        'recall': val_recall
    }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', './data/val'))
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data - find CSV files in directory
    train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
    val_files = [f for f in os.listdir(args.validation) if f.endswith('.csv')]
    
    if not train_files or not val_files:
        raise FileNotFoundError("No CSV files found in training or validation directories")
    
    train_data = pd.read_csv(os.path.join(args.train, train_files[0]))
    val_data = pd.read_csv(os.path.join(args.validation, val_files[0]))
    
    # Prepare features and labels
    feature_cols = [col for col in train_data.columns if col != 'fire_detected']
    X_train = train_data[feature_cols].values
    y_train = train_data['fire_detected'].values
    X_val = val_data[feature_cols].values
    y_val = val_data['fire_detected'].values
    
    # Create datasets
    train_dataset = FireDetectionDataset(X_train, y_train)
    val_dataset = FireDetectionDataset(X_val, y_val, scaler=train_dataset.scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    input_size = X_train.shape[1]
    
    if 'lstm' in args.model_type.lower():
        model = LSTMFireClassifier(input_size)
    elif 'gru' in args.model_type.lower():
        model = GRUFireClassifier(input_size)
    elif 'transformer' in args.model_type.lower():
        model = TransformerFireClassifier(input_size)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = model.to(device)
    
    print(f'Training {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train model
    metrics = train_model(model, train_loader, val_loader, args.epochs, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': train_dataset.scaler,
        'metrics': metrics,
        'model_type': args.model_type
    }, os.path.join(args.model_dir, 'model.pth'))
    
    print(f'Model saved with metrics: {metrics}')

if __name__ == '__main__':
    main()