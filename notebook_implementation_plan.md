# Fire Detection AI - 5M Training Notebook Implementation Plan

## Notebook Structure

The notebook will be structured into the following sections:

1. **Setup and Configuration**
2. **Data Loading and Sampling**
3. **Data Preprocessing**
4. **Optimized Transformer Model**
5. **ML Ensemble Models**
6. **Training and Evaluation**
7. **Model Saving**
8. **Performance Analysis**

## Detailed Implementation Plan

### 1. Setup and Configuration

```python
# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import boto3
import sagemaker
import json
import os
import time
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import matplotlib.pyplot as plt
import warnings

# Optional libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# Configuration
DATASET_BUCKET = "processedd-synthetic-data"
DATASET_PREFIX = "cleaned-data/"
SAMPLE_SIZE_PER_AREA = 1000000  # 1M samples per area = 5M total
RANDOM_SEED = 42
EPOCHS = 50  # Reduced from 100
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 0.002
```

### 2. Data Loading and Sampling

```python
class OptimizedDataLoader:
    """Optimized data loader with sampling for 5M dataset"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.area_files = {
            'basement': 'basement_data_cleaned.csv',
            'laundry': 'laundry_data_cleaned.csv',
            'asd': 'asd_data_cleaned.csv',
            'voc': 'voc_data_cleaned.csv',
            'arc': 'arc_data_cleaned.csv'
        }
        self.area_to_idx = {area: idx for idx, area in enumerate(self.area_files.keys())}
    
    def load_area_data_sample(self, area_name, max_samples=SAMPLE_SIZE_PER_AREA):
        """Load and sample area data"""
        file_key = f"{DATASET_PREFIX}{self.area_files[area_name]}"
        
        logger.info(f"📥 Loading {area_name}: s3://{DATASET_BUCKET}/{file_key}")
        
        try:
            # Load data in chunks to avoid memory issues
            chunk_size = 100000  # 100K rows per chunk
            chunks = []
            
            # Get object size to estimate number of chunks
            response = self.s3_client.head_object(Bucket=DATASET_BUCKET, Key=file_key)
            file_size = response['ContentLength']
            estimated_rows = file_size // 200  # Rough estimate: 200 bytes per row
            
            logger.info(f"   📊 Estimated rows: {estimated_rows:,}")
            logger.info(f"   📊 Target sample: {max_samples:,} rows")
            
            # Calculate sampling ratio
            if estimated_rows > max_samples:
                sampling_ratio = max_samples / estimated_rows
                logger.info(f"   📊 Sampling ratio: {sampling_ratio:.2%}")
            else:
                sampling_ratio = 1.0
                logger.info(f"   📊 Using all available data")
            
            # Stream data from S3 in chunks and sample
            response = self.s3_client.get_object(Bucket=DATASET_BUCKET, Key=file_key)
            
            # Use pandas to read CSV in chunks
            chunk_iter = pd.read_csv(response['Body'], chunksize=chunk_size)
            
            total_rows = 0
            for i, chunk in enumerate(chunk_iter):
                # Sample from chunk based on ratio
                if sampling_ratio < 1.0:
                    chunk = chunk.sample(frac=sampling_ratio, random_state=RANDOM_SEED)
                
                chunks.append(chunk)
                total_rows += len(chunk)
                
                logger.info(f"   📊 Chunk {i+1}: {len(chunk):,} rows, Total: {total_rows:,}")
                
                # Stop if we have enough samples
                if total_rows >= max_samples:
                    break
            
            # Combine chunks
            df = pd.concat(chunks, ignore_index=True)
            
            # Final sampling if we have more than max_samples
            if len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=RANDOM_SEED).reset_index(drop=True)
            
            logger.info(f"   📊 Final sample: {len(df):,} rows")
            
            return self.preprocess_area_data(df, area_name)
            
        except Exception as e:
            logger.error(f"   ❌ Error loading {area_name}: {e}")
            return None, None
    
    # Rest of the methods remain similar to the original implementation
    # ...
```

### 3. Optimized Transformer Model

```python
class OptimizedFireTransformer(nn.Module):
    """Optimized transformer for multi-area fire detection"""
    
    def __init__(self, input_dim=6, seq_len=60, d_model=128, num_heads=4, 
                 num_layers=3, num_classes=3, num_areas=5, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.area_embedding = nn.Embedding(num_areas, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fire_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, num_classes)
        )
        
        self.risk_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, area_types):
        batch_size, seq_len, _ = x.shape
        
        x = self.input_proj(x)
        area_emb = self.area_embedding(area_types).unsqueeze(1).expand(-1, seq_len, -1)
        x = x + area_emb + self.pos_encoding[:seq_len].unsqueeze(0)
        
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global pooling
        
        return {
            'fire_logits': self.fire_classifier(x),
            'risk_score': self.risk_predictor(x) * 100.0
        }
```

### 4. Optimized ML Ensemble

```python
def train_optimized_ml_ensemble(X_train, y_train, X_val, y_val):
    """Train optimized ML ensemble models"""
    logger.info("📊 TRAINING OPTIMIZED ML ENSEMBLE")
    logger.info("=" * 30)
    
    # Feature engineering
    logger.info("🔧 Engineering features...")
    X_train_features = engineer_features(X_train)
    X_val_features = engineer_features(X_val)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled = scaler.transform(X_val_features)
    
    ml_models = {}
    ml_results = {}
    
    # Random Forest - reduced estimators
    logger.info("🌳 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Reduced from 200
        max_depth=15,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_val_acc = rf_model.score(X_val_scaled, y_val)
    ml_models['random_forest'] = rf_model
    ml_results['random_forest'] = rf_val_acc
    logger.info(f"   ✅ Random Forest Val Acc: {rf_val_acc:.4f}")
    
    # XGBoost (if available) - reduced estimators
    if XGB_AVAILABLE:
        logger.info("⚡ Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,  # Reduced from 300
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_SEED
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_val_acc = xgb_model.score(X_val_scaled, y_val)
        ml_models['xgboost'] = xgb_model
        ml_results['xgboost'] = xgb_val_acc
        logger.info(f"   ✅ XGBoost Val Acc: {xgb_val_acc:.4f}")
    
    # LightGBM (if available) - reduced estimators
    if LGB_AVAILABLE:
        logger.info("💡 Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,  # Reduced from 300
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_SEED,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        lgb_val_acc = lgb_model.score(X_val_scaled, y_val)
        ml_models['lightgbm'] = lgb_model
        ml_results['lightgbm'] = lgb_val_acc
        logger.info(f"   ✅ LightGBM Val Acc: {lgb_val_acc:.4f}")
    
    return ml_models, ml_results, scaler
```

### 5. Training with Early Stopping

```python
def train_transformer_with_early_stopping(X_train, y_train, areas_train, X_val, y_val, areas_val, device):
    """Train the optimized transformer model with early stopping"""
    logger.info("🤖 TRAINING OPTIMIZED TRANSFORMER")
    logger.info("=" * 40)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    areas_train_tensor = torch.LongTensor(areas_train).to(device)
    areas_val_tensor = torch.LongTensor(areas_val).to(device)
    
    # Create model
    model = OptimizedFireTransformer(
        input_dim=X_train.shape[2],
        seq_len=X_train.shape[1],
        d_model=128,  # Reduced from 256
        num_heads=4,  # Reduced from 8
        num_layers=3,  # Reduced from 6
        num_classes=len(np.unique(y_train)),
        num_areas=len(np.unique(areas_train))
    ).to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training for {EPOCHS} epochs with early stopping...")
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor, areas_train_tensor)
        loss = criterion(outputs['fire_logits'], y_train_tensor)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor, areas_val_tensor)
            val_preds = torch.argmax(val_outputs['fire_logits'], dim=1)
            val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            logger.info(f"Epoch {epoch:3d}: Loss={loss:.4f}, Val_Acc={val_acc:.4f}, Best={best_val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    logger.info(f"✅ Transformer training completed!")
    logger.info(f"   Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"   Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    
    return model, best_val_acc
```

### 6. Performance Analysis and Visualization

```python
def visualize_performance(ensemble_acc, training_time, memory_usage):
    """Visualize model performance metrics"""
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    accuracies = {
        '5M Model': ensemble_acc,
        '50M Model (Est.)': 0.975  # Estimated from documentation
    }
    
    ax1.bar(accuracies.keys(), accuracies.values(), color=['#3498db', '#2ecc71'])
    ax1.set_ylim([0.9, 1.0])
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    for i, v in enumerate(accuracies.values()):
        ax1.text(i, v-0.02, f"{v:.1%}", ha='center', fontweight='bold')
    
    # Training time comparison
    times = {
        '5M Model': training_time / 3600,  # Convert to hours
        '50M Model': 43  # From user input
    }
    
    ax2.bar(times.keys(), times.values(), color=['#3498db', '#e74c3c'])
    ax2.set_title('Training Time Comparison')
    ax2.set_ylabel('Hours')
    for i, v in enumerate(times.values()):
        ax2.text(i, v/2, f"{v:.1f}h", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()
    
    # Print summary
    speedup = 43 / (training_time / 3600)
    accuracy_diff = 0.975 - ensemble_acc
    
    print(f"🚀 Performance Summary:")
    print(f"   ⏱️ Training Speedup: {speedup:.1f}x faster")
    print(f"   📊 Accuracy Difference: {accuracy_diff:.1%} lower")
    print(f"   💰 Cost Savings: {(1 - 1/speedup):.1%}")
```

### 7. Main Function

```python
def main():
    """Main training function"""
    
    logger.info("🔥" * 80)
    logger.info("FIRE DETECTION AI - 5M DATASET OPTIMIZED TRAINING")
    logger.info("🔥" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    start_time = time.time()
    
    # Load data
    data_loader = OptimizedDataLoader()
    X, y, areas = data_loader.load_all_data_sample()
    
    # Split data
    X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
        X, y, areas, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
        X_train, y_train, areas_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    logger.info(f"📊 Data splits:")
    logger.info(f"   Training: {len(X_train):,}")
    logger.info(f"   Validation: {len(X_val):,}")
    logger.info(f"   Test: {len(X_test):,}")
    
    # Train transformer with early stopping
    transformer_model, transformer_acc = train_transformer_with_early_stopping(
        X_train, y_train, areas_train, X_val, y_val, areas_val, device
    )
    
    # Train ML ensemble
    ml_models, ml_results, scaler = train_optimized_ml_ensemble(X_train, y_train, X_val, y_val)
    
    # Final evaluation
    ensemble_acc = evaluate_ensemble(
        transformer_model, ml_models, scaler, X_test, y_test, areas_test, device
    )
    
    total_time = time.time() - start_time
    
    # Save models
    save_models_to_s3(transformer_model, ml_models, scaler, ensemble_acc)
    
    # Visualize performance
    memory_usage = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    visualize_performance(ensemble_acc, total_time, memory_usage)
    
    # Final summary
    logger.info("\n" + "🎉" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("🎉" * 80)
    logger.info(f"🏆 Final Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
    logger.info(f"⏱️ Total training time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    logger.info(f"📊 Total samples processed: {len(X):,}")
    logger.info(f"🚀 Models ready for production deployment!")
```

## Notebook Execution Instructions

1. Upload the notebook to a SageMaker instance (ml.g5.2xlarge recommended)
2. Install required dependencies:
   ```
   !pip install torch torchvision xgboost lightgbm scikit-learn matplotlib
   ```
3. Execute the notebook cells in order
4. Monitor training progress through the logs
5. Review the performance comparison visualization
6. Download the trained models from S3 for deployment

## Expected Results

- **Training Time**: 2-4 hours
- **Accuracy**: 94-96%
- **Memory Usage**: 4-8 GB
- **Cost**: ~$3-7 USD (vs $43-86 USD for full dataset)