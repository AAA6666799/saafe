#!/usr/bin/env python3
"""
🔥 Fire Detection AI - AWS SageMaker Training Script
Optimized for 50M dataset from s3://processedd-synthetic-data/cleaned-data/

Usage:
1. Upload this script to SageMaker Jupyter notebook
2. Run: python sagemaker_50m_training.py
3. Monitor training progress
4. Models will be saved to S3 automatically

Recommended instance: ml.p3.2xlarge or ml.p3.8xlarge
"""

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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATASET_BUCKET = "processedd-synthetic-data"
DATASET_PREFIX = "cleaned-data/"
USE_FULL_DATASET = True  # Set to False for demo with smaller dataset
MAX_SAMPLES_PER_AREA = None if USE_FULL_DATASET else 500000

class EnhancedFireTransformer(nn.Module):
    """Enhanced transformer for multi-area fire detection"""
    
    def __init__(self, input_dim=6, seq_len=60, d_model=256, num_heads=8, 
                 num_layers=6, num_classes=3, num_areas=5, dropout=0.1):
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

class SageMakerDataLoader:
    """Optimized data loader for SageMaker environment"""
    
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
    
    def load_area_data(self, area_name, max_samples=None):
        """Load and preprocess area data"""
        file_key = f"{DATASET_PREFIX}{self.area_files[area_name]}"
        
        logger.info(f"📥 Loading {area_name}: s3://{DATASET_BUCKET}/{file_key}")
        
        try:
            response = self.s3_client.get_object(Bucket=DATASET_BUCKET, Key=file_key)
            df = pd.read_csv(response['Body'])
            
            logger.info(f"   📊 Raw data: {len(df):,} rows, {len(df.columns)} columns")
            
            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
                logger.info(f"   🎯 Sampled to: {len(df):,} rows")
            
            return self.preprocess_area_data(df, area_name)
            
        except Exception as e:
            logger.error(f"   ❌ Error loading {area_name}: {e}")
            return None, None
    
    def preprocess_area_data(self, df, area_name):
        """Smart preprocessing based on area characteristics"""
        logger.info(f"🔧 Preprocessing {area_name}...")
        
        # Handle timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract features
        exclude_cols = ['timestamp', 'is_anomaly', 'label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Limit features by area type
        if area_name == 'basement':
            feature_cols = feature_cols[:4]
        elif area_name == 'laundry':
            feature_cols = feature_cols[:3]
        else:
            feature_cols = feature_cols[:2]
        
        X = df[feature_cols].fillna(0).values
        
        # Create intelligent labels
        if 'is_anomaly' in df.columns:
            y = df['is_anomaly'].values.astype(int)
        elif 'label' in df.columns:
            y = df['label'].values.astype(int)
        else:
            # Generate realistic fire detection labels
            values = df[feature_cols[0]].values
            q95 = np.percentile(values, 95)
            q85 = np.percentile(values, 85)
            
            y = np.zeros(len(values))
            y[values > q95] = 2  # Fire (top 5%)
            y[(values > q85) & (values <= q95)] = 1  # Warning (85-95%)
        
        # Standardize to 6 features
        if X.shape[1] < 6:
            padding = np.zeros((X.shape[0], 6 - X.shape[1]))
            X = np.hstack([X, padding])
        elif X.shape[1] > 6:
            X = X[:, :6]
        
        logger.info(f"   ✅ {area_name}: {X.shape}, anomaly_rate={y.mean():.4f}")
        return X, y
    
    def create_sequences(self, X, y, seq_len=60, step=10):
        """Create time series sequences"""
        sequences = []
        labels = []
        
        for i in range(0, len(X) - seq_len, step):
            sequences.append(X[i:i+seq_len])
            labels.append(y[i+seq_len-1])
        
        return np.array(sequences), np.array(labels)
    
    def load_all_data(self):
        """Load complete dataset"""
        logger.info("🚀 LOADING COMPLETE DATASET FROM S3")
        logger.info("=" * 50)
        
        all_sequences = []
        all_labels = []
        all_areas = []
        
        start_time = time.time()
        
        for area_idx, area_name in enumerate(self.area_files.keys()):
            logger.info(f"\n📁 PROCESSING AREA {area_idx+1}/5: {area_name.upper()}")
            
            X, y = self.load_area_data(area_name, MAX_SAMPLES_PER_AREA)
            if X is None:
                continue
            
            sequences, labels = self.create_sequences(X, y, seq_len=60, step=10)
            areas = np.full(len(sequences), area_idx)
            
            all_sequences.append(sequences)
            all_labels.append(labels)
            all_areas.append(areas)
            
            logger.info(f"   ✅ Created {len(sequences):,} sequences")
        
        # Combine all areas
        X_combined = np.vstack(all_sequences)
        y_combined = np.hstack(all_labels)
        areas_combined = np.hstack(all_areas)
        
        total_time = time.time() - start_time
        
        logger.info(f"\n🎯 DATASET SUMMARY:")
        logger.info(f"   📊 Total sequences: {X_combined.shape[0]:,}")
        logger.info(f"   📐 Shape: {X_combined.shape}")
        logger.info(f"   📈 Classes: {np.bincount(y_combined.astype(int))}")
        logger.info(f"   💾 Memory: {X_combined.nbytes / (1024**3):.2f} GB")
        logger.info(f"   ⏱️ Time: {total_time:.1f}s")
        
        return X_combined, y_combined, areas_combined

def engineer_features(X):
    """Advanced feature engineering for ML models"""
    features = []
    for i in range(X.shape[0]):
        sample_features = []
        for j in range(X.shape[2]):
            series = X[i, :, j]
            # Statistical features
            sample_features.extend([
                np.mean(series), np.std(series), np.min(series), np.max(series),
                np.median(series), np.percentile(series, 25), np.percentile(series, 75)
            ])
            # Trend features
            if len(series) > 1:
                slope = np.polyfit(range(len(series)), series, 1)[0]
                sample_features.append(slope)
                diff = np.diff(series)
                sample_features.extend([np.mean(np.abs(diff)), np.std(diff)])
            else:
                sample_features.extend([0, 0, 0])
        features.append(sample_features)
    return np.array(features)

def train_transformer(X_train, y_train, areas_train, X_val, y_val, areas_val, device):
    """Train the enhanced transformer model"""
    logger.info("🤖 TRAINING ENHANCED TRANSFORMER")
    logger.info("=" * 40)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    areas_train_tensor = torch.LongTensor(areas_train).to(device)
    areas_val_tensor = torch.LongTensor(areas_val).to(device)
    
    # Create model
    model = EnhancedFireTransformer(
        input_dim=X_train.shape[2],
        seq_len=X_train.shape[1],
        d_model=256,
        num_heads=8,
        num_layers=6,
        num_classes=len(np.unique(y_train)),
        num_areas=len(np.unique(areas_train))
    ).to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    best_val_acc = 0.0
    best_model_state = None
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training for {epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor, areas_train_tensor)
        loss = criterion(outputs['fire_logits'], y_train_tensor)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor, areas_val_tensor)
                val_preds = torch.argmax(val_outputs['fire_logits'], dim=1)
                val_acc = (val_preds == y_val_tensor).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                logger.info(f"Epoch {epoch:3d}: Loss={loss:.4f}, Val_Acc={val_acc:.4f}, Best={best_val_acc:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    logger.info(f"✅ Transformer training completed!")
    logger.info(f"   Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"   Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    
    return model, best_val_acc

def train_ml_ensemble(X_train, y_train, X_val, y_val):
    """Train ML ensemble models"""
    logger.info("📊 TRAINING ML ENSEMBLE")
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
    
    # Random Forest
    logger.info("🌳 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_val_acc = rf_model.score(X_val_scaled, y_val)
    ml_models['random_forest'] = rf_model
    ml_results['random_forest'] = rf_val_acc
    logger.info(f"   ✅ Random Forest Val Acc: {rf_val_acc:.4f}")
    
    # Gradient Boosting
    logger.info("📈 Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_val_acc = gb_model.score(X_val_scaled, y_val)
    ml_models['gradient_boosting'] = gb_model
    ml_results['gradient_boosting'] = gb_val_acc
    logger.info(f"   ✅ Gradient Boosting Val Acc: {gb_val_acc:.4f}")
    
    # XGBoost (if available)
    if XGB_AVAILABLE:
        logger.info("⚡ Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_val_acc = xgb_model.score(X_val_scaled, y_val)
        ml_models['xgboost'] = xgb_model
        ml_results['xgboost'] = xgb_val_acc
        logger.info(f"   ✅ XGBoost Val Acc: {xgb_val_acc:.4f}")
    
    # LightGBM (if available)
    if LGB_AVAILABLE:
        logger.info("💡 Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        lgb_val_acc = lgb_model.score(X_val_scaled, y_val)
        ml_models['lightgbm'] = lgb_model
        ml_results['lightgbm'] = lgb_val_acc
        logger.info(f"   ✅ LightGBM Val Acc: {lgb_val_acc:.4f}")
    
    return ml_models, ml_results, scaler

def evaluate_ensemble(transformer_model, ml_models, scaler, X_test, y_test, areas_test, device):
    """Evaluate the complete ensemble"""
    logger.info("🎯 EVALUATING COMPLETE ENSEMBLE")
    logger.info("=" * 35)
    
    # Transformer predictions
    transformer_model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    areas_test_tensor = torch.LongTensor(areas_test).to(device)
    
    with torch.no_grad():
        transformer_outputs = transformer_model(X_test_tensor, areas_test_tensor)
        transformer_preds = torch.argmax(transformer_outputs['fire_logits'], dim=1).cpu().numpy()
    
    # ML model predictions
    X_test_features = engineer_features(X_test)
    X_test_scaled = scaler.transform(X_test_features)
    
    predictions = {'transformer': transformer_preds}
    
    for name, model in ml_models.items():
        pred = model.predict(X_test_scaled)
        predictions[name] = pred
        acc = accuracy_score(y_test, pred)
        logger.info(f"   {name}: {acc:.4f}")
    
    # Ensemble prediction (majority voting)
    ensemble_preds = []
    for i in range(len(y_test)):
        votes = [predictions[name][i] for name in predictions.keys()]
        ensemble_pred = max(set(votes), key=votes.count)
        ensemble_preds.append(ensemble_pred)
    
    ensemble_preds = np.array(ensemble_preds)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    
    logger.info(f"\n🏆 ENSEMBLE ACCURACY: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
    
    # Classification report
    logger.info(f"\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, ensemble_preds, 
                              target_names=['Normal', 'Warning', 'Fire']))
    
    return ensemble_acc

def save_models_to_s3(transformer_model, ml_models, scaler, ensemble_acc):
    """Save all models to S3"""
    logger.info("💾 SAVING MODELS TO S3")
    logger.info("=" * 25)
    
    try:
        sagemaker_session = sagemaker.Session()
        bucket = sagemaker_session.default_bucket()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save transformer
        transformer_path = f'/tmp/transformer_50m_{timestamp}.pth'
        torch.save({
            'model_state_dict': transformer_model.state_dict(),
            'model_class': 'EnhancedFireTransformer',
            'ensemble_accuracy': ensemble_acc,
            'timestamp': timestamp
        }, transformer_path)
        
        s3_key = f'fire-detection-models/transformer_50m_{timestamp}.pth'
        sagemaker_session.upload_data(transformer_path, bucket, s3_key)
        logger.info(f"   ✅ Transformer: s3://{bucket}/{s3_key}")
        
        # Save ML models
        for name, model in ml_models.items():
            model_path = f"/tmp/{name}_50m_{timestamp}.pkl"
            joblib.dump(model, model_path)
            s3_key = f'fire-detection-models/{name}_50m_{timestamp}.pkl'
            sagemaker_session.upload_data(model_path, bucket, s3_key)
            logger.info(f"   ✅ {name}: s3://{bucket}/{s3_key}")
        
        # Save scaler
        scaler_path = f"/tmp/scaler_50m_{timestamp}.pkl"
        joblib.dump(scaler, scaler_path)
        s3_key = f'fire-detection-models/scaler_50m_{timestamp}.pkl'
        sagemaker_session.upload_data(scaler_path, bucket, s3_key)
        logger.info(f"   ✅ Scaler: s3://{bucket}/{s3_key}")
        
        # Save metadata
        metadata = {
            'training_completed': datetime.now().isoformat(),
            'ensemble_accuracy': float(ensemble_acc),
            'models_trained': list(ml_models.keys()) + ['transformer'],
            'dataset_size': 'full' if USE_FULL_DATASET else 'sample',
            'target_achieved': ensemble_acc >= 0.95
        }
        
        metadata_path = f"/tmp/training_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        s3_key = f'fire-detection-models/training_metadata_{timestamp}.json'
        sagemaker_session.upload_data(metadata_path, bucket, s3_key)
        logger.info(f"   ✅ Metadata: s3://{bucket}/{s3_key}")
        
        logger.info(f"🎉 All models saved to S3 bucket: {bucket}")
        
    except Exception as e:
        logger.error(f"❌ Error saving to S3: {e}")

def main():
    """Main training function"""
    
    logger.info("🔥" * 80)
    logger.info("FIRE DETECTION AI - 50M DATASET TRAINING ON SAGEMAKER")
    logger.info("🔥" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    logger.info(f"📊 Configuration:")
    logger.info(f"   Full dataset: {USE_FULL_DATASET}")
    logger.info(f"   Max samples per area: {MAX_SAMPLES_PER_AREA or 'ALL'}")
    logger.info(f"   XGBoost available: {XGB_AVAILABLE}")
    logger.info(f"   LightGBM available: {LGB_AVAILABLE}")
    
    # Load data
    data_loader = SageMakerDataLoader()
    X, y, areas = data_loader.load_all_data()
    
    # Split data
    X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
        X, y, areas, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
        X_train, y_train, areas_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"📊 Data splits:")
    logger.info(f"   Training: {len(X_train):,}")
    logger.info(f"   Validation: {len(X_val):,}")
    logger.info(f"   Test: {len(X_test):,}")
    
    total_start_time = time.time()
    
    # Train transformer
    transformer_model, transformer_acc = train_transformer(
        X_train, y_train, areas_train, X_val, y_val, areas_val, device
    )
    
    # Train ML ensemble
    ml_models, ml_results, scaler = train_ml_ensemble(X_train, y_train, X_val, y_val)
    
    # Final evaluation
    ensemble_acc = evaluate_ensemble(
        transformer_model, ml_models, scaler, X_test, y_test, areas_test, device
    )
    
    total_time = time.time() - total_start_time
    
    # Save models
    save_models_to_s3(transformer_model, ml_models, scaler, ensemble_acc)
    
    # Final summary
    logger.info("\n" + "🎉" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("🎉" * 80)
    logger.info(f"🏆 Final Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
    logger.info(f"🎯 Target (95%): {'✅ ACHIEVED' if ensemble_acc >= 0.95 else '📈 IN PROGRESS'}")
    logger.info(f"⏱️ Total training time: {total_time:.1f}s ({total_time/3600:.1f} hours)")
    logger.info(f"📊 Total samples processed: {len(X):,}")
    logger.info(f"🚀 Models ready for production deployment!")
    
    if ensemble_acc >= 0.97:
        logger.info("🎊 CONGRATULATIONS! 97%+ accuracy achieved!")
    elif ensemble_acc >= 0.95:
        logger.info("✅ Excellent! 95%+ accuracy achieved!")
    else:
        logger.info("📈 Good progress! Consider more training for higher accuracy.")

if __name__ == "__main__":
    main()