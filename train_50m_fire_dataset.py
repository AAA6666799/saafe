#!/usr/bin/env python3
"""
🔥 Fire Detection Training - 50M Dataset from S3
Custom training script for processedd-synthetic-data/cleaned-data/

Dataset Structure:
- basement_data_cleaned.csv (922.8 MB) - 6 columns
- laundry_data_cleaned.csv (756.2 MB) - 5 columns  
- asd_data_cleaned.csv (600.6 MB) - 4 columns
- voc_data_cleaned.csv (566.4 MB) - 4 columns
- arc_data_cleaned.csv (489.2 MB) - 4 columns

Total: ~19.9M samples, 3.26 GB
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import boto3
import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# Optional advanced libraries
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

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🔥" * 80)
print("FIRE DETECTION AI - 50M DATASET TRAINING")
print("🔥" * 80)
print(f"📊 Target: 19.9M samples from 5 area datasets")
print(f"🎯 Goal: 97-98% accuracy, <0.5% false positive rate")
print(f"⚡ XGBoost: {'✅' if XGB_AVAILABLE else '❌'}")
print(f"⚡ LightGBM: {'✅' if LGB_AVAILABLE else '❌'}")
print(f"⚡ CatBoost: {'✅' if CB_AVAILABLE else '❌'}")

class FireDetectionDataset(Dataset):
    """Custom dataset for fire detection training"""
    
    def __init__(self, sequences, labels, area_types, transform=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.area_types = torch.LongTensor(area_types)
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        area_type = self.area_types[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label, area_type

class EnhancedFireTransformer(nn.Module):
    """Enhanced transformer for multi-area fire detection"""
    
    def __init__(self, input_dim=6, seq_len=60, d_model=256, num_heads=8, 
                 num_layers=6, num_classes=3, num_areas=5, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.area_embedding = nn.Embedding(num_areas, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-head outputs
        self.fire_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.risk_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, area_types):
        batch_size, seq_len, _ = x.shape
        
        # Input projection and area embedding
        x = self.input_proj(x)
        area_emb = self.area_embedding(area_types).unsqueeze(1).expand(-1, seq_len, -1)
        x = x + area_emb + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return {
            'fire_logits': self.fire_classifier(x),
            'risk_score': self.risk_predictor(x) * 100.0,
            'confidence': self.confidence_estimator(x)
        }

class S3DataLoader:
    """Load and preprocess data from S3"""
    
    def __init__(self, bucket_name="processedd-synthetic-data", prefix="cleaned-data/"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
        
        # Area mappings based on dataset analysis
        self.area_files = {
            'basement': 'basement_data_cleaned.csv',
            'laundry': 'laundry_data_cleaned.csv', 
            'asd': 'asd_data_cleaned.csv',
            'voc': 'voc_data_cleaned.csv',
            'arc': 'arc_data_cleaned.csv'
        }
        
        self.area_to_idx = {area: idx for idx, area in enumerate(self.area_files.keys())}
        
    def load_area_data(self, area_name: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Load data for specific area"""
        file_key = f"{self.prefix}{self.area_files[area_name]}"
        
        logger.info(f"Loading {area_name} data from s3://{self.bucket_name}/{file_key}")
        
        try:
            # Stream data from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(response['Body'])
            
            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            
            logger.info(f"✅ Loaded {len(df):,} samples for {area_name}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading {area_name} data: {e}")
            raise
    
    def preprocess_area_data(self, df: pd.DataFrame, area_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for specific area"""
        
        # Handle different column structures based on our analysis
        if area_name == 'basement':
            # 6 columns - assume timestamp, value, is_anomaly + 3 other features
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Extract features (exclude timestamp and target)
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'is_anomaly']]
            if len(feature_cols) > 4:
                feature_cols = feature_cols[:4]  # Keep top 4 features
                
        elif area_name == 'laundry':
            # 5 columns
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'is_anomaly']]
            if len(feature_cols) > 3:
                feature_cols = feature_cols[:3]
                
        else:  # asd, voc, arc - 4 columns each
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'is_anomaly']]
            if len(feature_cols) > 2:
                feature_cols = feature_cols[:2]
        
        # Extract features and labels
        X = df[feature_cols].fillna(0).values
        
        # Create labels - assume 'is_anomaly' or similar column exists
        if 'is_anomaly' in df.columns:
            y = df['is_anomaly'].values.astype(int)
        elif 'label' in df.columns:
            y = df['label'].values.astype(int)
        else:
            # Generate synthetic labels based on value patterns
            value_col = 'value' if 'value' in df.columns else feature_cols[0]
            values = df[value_col].values
            
            # Create labels based on statistical thresholds
            q95 = np.percentile(values, 95)
            q80 = np.percentile(values, 80)
            
            y = np.zeros(len(values))
            y[values > q95] = 2  # Fire
            y[(values > q80) & (values <= q95)] = 1  # Warning
            # Rest remain 0 (Normal)
        
        # Ensure we have consistent feature dimensions (pad with zeros if needed)
        if X.shape[1] < 6:
            padding = np.zeros((X.shape[0], 6 - X.shape[1]))
            X = np.hstack([X, padding])
        elif X.shape[1] > 6:
            X = X[:, :6]
        
        logger.info(f"✅ Preprocessed {area_name}: X={X.shape}, y={y.shape}, anomaly_rate={y.mean():.4f}")
        return X, y
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int = 60, 
                        step: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series sequences from data"""
        sequences = []
        labels = []
        
        for i in range(0, len(X) - seq_len, step):
            seq = X[i:i+seq_len]
            label = y[i+seq_len-1]  # Use label at end of sequence
            
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def load_all_data(self, max_samples_per_area: Optional[int] = None, 
                     seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and combine all area data"""
        
        all_sequences = []
        all_labels = []
        all_areas = []
        
        for area_name in self.area_files.keys():
            try:
                # Load area data
                df = self.load_area_data(area_name, max_samples_per_area)
                
                # Preprocess
                X, y = self.preprocess_area_data(df, area_name)
                
                # Create sequences
                sequences, labels = self.create_sequences(X, y, seq_len)
                
                # Add area information
                area_idx = self.area_to_idx[area_name]
                areas = np.full(len(sequences), area_idx)
                
                all_sequences.append(sequences)
                all_labels.append(labels)
                all_areas.append(areas)
                
                logger.info(f"✅ Created {len(sequences):,} sequences for {area_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {area_name}: {e}")
                continue
        
        # Combine all areas
        if all_sequences:
            X_combined = np.vstack(all_sequences)
            y_combined = np.hstack(all_labels)
            areas_combined = np.hstack(all_areas)
            
            logger.info(f"🎯 COMBINED DATASET:")
            logger.info(f"   Sequences: {X_combined.shape}")
            logger.info(f"   Labels: {y_combined.shape}")
            logger.info(f"   Areas: {areas_combined.shape}")
            logger.info(f"   Class distribution: {np.bincount(y_combined)}")
            
            return X_combined, y_combined, areas_combined
        else:
            raise ValueError("No data could be loaded from any area")

class FireDetectionEnsemble:
    """Production fire detection ensemble"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        logger.info(f"🚀 Ensemble initialized on device: {self.device}")
    
    def _engineer_features(self, X: np.ndarray) -> np.ndarray:
        """Engineer features for ML models"""
        features = []
        
        for i in range(X.shape[0]):
            sample_features = []
            
            for j in range(X.shape[2]):  # For each feature dimension
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
                    
                    # Change features
                    diff = np.diff(series)
                    sample_features.extend([
                        np.mean(np.abs(diff)), np.std(diff)
                    ])
                else:
                    sample_features.extend([0, 0, 0])
            
            # Cross-feature interactions
            if X.shape[2] > 1:
                for j1 in range(X.shape[2]):
                    for j2 in range(j1+1, X.shape[2]):
                        corr = np.corrcoef(X[i, :, j1], X[i, :, j2])[0, 1]
                        sample_features.append(0 if np.isnan(corr) else corr)
            
            features.append(sample_features)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray, areas: np.ndarray, 
           validation_split: float = 0.2, epochs: int = 100):
        """Train the ensemble"""
        
        logger.info("🚀 TRAINING FIRE DETECTION ENSEMBLE")
        logger.info("=" * 60)
        
        # Split data
        X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
            X, y, areas, test_size=validation_split, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training: {len(X_train):,} samples")
        logger.info(f"📊 Validation: {len(X_val):,} samples")
        
        # 1. Train Transformer
        logger.info("\n🤖 Training Enhanced Transformer...")
        self._train_transformer(X_train, y_train, areas_train, X_val, y_val, areas_val, epochs)
        
        # 2. Train ML models
        logger.info("\n📊 Training ML Models...")
        self._train_ml_models(X_train, y_train, X_val, y_val)
        
        self.is_fitted = True
        
        # Final evaluation
        logger.info("\n🎯 FINAL ENSEMBLE EVALUATION")
        self._evaluate_ensemble(X_val, y_val, areas_val)
        
        return self
    
    def _train_transformer(self, X_train, y_train, areas_train, X_val, y_val, areas_val, epochs):
        """Train transformer model"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        areas_train_tensor = torch.LongTensor(areas_train).to(self.device)
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        areas_val_tensor = torch.LongTensor(areas_val).to(self.device)
        
        # Create model
        model = EnhancedFireTransformer(
            input_dim=X_train.shape[2],
            seq_len=X_train.shape[1],
            d_model=256,
            num_heads=8,
            num_layers=6,
            num_classes=len(np.unique(y_train)),
            num_areas=len(np.unique(areas_train))
        ).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_train_tensor, areas_train_tensor)
            
            loss = criterion(outputs['fire_logits'], y_train_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor, areas_val_tensor)
                    val_preds = torch.argmax(val_outputs['fire_logits'], dim=1)
                    val_acc = (val_preds == y_val_tensor).float().mean().item()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        self.models['transformer'] = model.state_dict()
                    
                    logger.info(f"Epoch {epoch:3d}: Loss={loss:.4f}, Val_Acc={val_acc:.4f}")
        
        # Load best model
        model.load_state_dict(self.models['transformer'])
        self.models['transformer_model'] = model
        
        logger.info(f"✅ Transformer trained - Best Val Acc: {best_val_acc:.4f}")
    
    def _train_ml_models(self, X_train, y_train, X_val, y_val):
        """Train ML models"""
        
        # Engineer features
        logger.info("🔧 Engineering features...")
        X_train_features = self._engineer_features(X_train)
        X_val_features = self._engineer_features(X_val)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train_features)
        X_val_scaled = self.scalers['standard'].transform(X_val_features)
        
        # Train models
        ml_models = {}
        
        # Random Forest
        logger.info("  🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_acc = rf_model.score(X_val_scaled, y_val)
        ml_models['random_forest'] = rf_model
        logger.info(f"     ✅ Random Forest Val Acc: {rf_acc:.4f}")
        
        # Gradient Boosting
        logger.info("  📈 Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_acc = gb_model.score(X_val_scaled, y_val)
        ml_models['gradient_boosting'] = gb_model
        logger.info(f"     ✅ Gradient Boosting Val Acc: {gb_acc:.4f}")
        
        # XGBoost (if available)
        if XGB_AVAILABLE:
            logger.info("  ⚡ Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1, 
                subsample=0.8, random_state=42
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_acc = xgb_model.score(X_val_scaled, y_val)
            ml_models['xgboost'] = xgb_model
            logger.info(f"     ✅ XGBoost Val Acc: {xgb_acc:.4f}")
        
        # LightGBM (if available)
        if LGB_AVAILABLE:
            logger.info("  💡 Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42, verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            lgb_acc = lgb_model.score(X_val_scaled, y_val)
            ml_models['lightgbm'] = lgb_model
            logger.info(f"     ✅ LightGBM Val Acc: {lgb_acc:.4f}")
        
        self.models.update(ml_models)
    
    def _evaluate_ensemble(self, X_val, y_val, areas_val):
        """Evaluate ensemble performance"""
        
        # Get predictions from all models
        transformer_model = self.models['transformer_model']
        transformer_model.eval()
        
        # Transformer predictions
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            areas_val_tensor = torch.LongTensor(areas_val).to(self.device)
            
            transformer_outputs = transformer_model(X_val_tensor, areas_val_tensor)
            transformer_preds = torch.argmax(transformer_outputs['fire_logits'], dim=1).cpu().numpy()
            transformer_probs = torch.softmax(transformer_outputs['fire_logits'], dim=1).cpu().numpy()
        
        # ML model predictions
        X_val_features = self._engineer_features(X_val)
        X_val_scaled = self.scalers['standard'].transform(X_val_features)
        
        predictions = {'transformer': transformer_preds}
        probabilities = {'transformer': transformer_probs}
        
        for name, model in self.models.items():
            if name not in ['transformer', 'transformer_model']:
                try:
                    pred = model.predict(X_val_scaled)
                    prob = model.predict_proba(X_val_scaled) if hasattr(model, 'predict_proba') else None
                    
                    predictions[name] = pred
                    if prob is not None:
                        probabilities[name] = prob
                        
                except Exception as e:
                    logger.warning(f"❌ Could not get predictions from {name}: {e}")
        
        # Ensemble prediction (majority voting)
        ensemble_preds = []
        for i in range(len(y_val)):
            votes = [predictions[name][i] for name in predictions.keys()]
            ensemble_pred = max(set(votes), key=votes.count)
            ensemble_preds.append(ensemble_pred)
        
        ensemble_preds = np.array(ensemble_preds)
        ensemble_acc = accuracy_score(y_val, ensemble_preds)
        
        # Individual accuracies
        individual_accs = {}
        for name, preds in predictions.items():
            acc = accuracy_score(y_val, preds)
            individual_accs[name] = acc
            logger.info(f"   {name}: {acc:.4f}")
        
        logger.info(f"\n🏆 ENSEMBLE ACCURACY: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
        
        # Classification report
        logger.info(f"\n📊 CLASSIFICATION REPORT:")
        print(classification_report(y_val, ensemble_preds, 
                                  target_names=['Normal', 'Warning', 'Fire']))
        
        return ensemble_acc, individual_accs
    
    def save_models(self, save_dir: str = "models_50m"):
        """Save all trained models"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save transformer
        if 'transformer_model' in self.models:
            torch.save({
                'model_state_dict': self.models['transformer_model'].state_dict(),
                'model_class': 'EnhancedFireTransformer',
                'timestamp': datetime.now().isoformat()
            }, os.path.join(save_dir, 'transformer_model.pth'))
        
        # Save ML models
        for name, model in self.models.items():
            if name not in ['transformer', 'transformer_model']:
                joblib.dump(model, os.path.join(save_dir, f'{name}_model.pkl'))
        
        # Save scalers
        joblib.dump(self.scalers, os.path.join(save_dir, 'scalers.pkl'))
        
        # Save metadata
        metadata = {
            'models_trained': list(self.models.keys()),
            'training_completed': datetime.now().isoformat(),
            'device': str(self.device),
            'ensemble_ready': True
        }
        
        with open(os.path.join(save_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ All models saved to: {save_dir}")

def main():
    """Main training function"""
    
    # Load configuration
    with open('fire_detection_50m_config.json', 'r') as f:
        config = json.load(f)
    
    logger.info(f"📋 Configuration loaded:")
    logger.info(f"   Dataset: {config['dataset']['estimated_samples']:,} samples")
    logger.info(f"   Size: {config['dataset']['total_size_gb']:.2f} GB")
    logger.info(f"   Batch size: {config['training']['batch_size']:,}")
    logger.info(f"   Epochs: {config['training']['epochs']}")
    
    # Initialize data loader
    data_loader = S3DataLoader()
    
    # Load data (limit samples for demonstration - remove limit for full training)
    logger.info("\n📥 Loading dataset from S3...")
    X, y, areas = data_loader.load_all_data(
        max_samples_per_area=500000,  # Limit per area for demo - remove for full dataset
        seq_len=60
    )
    
    # Initialize ensemble
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = FireDetectionEnsemble(device=device)
    
    # Train ensemble
    logger.info(f"\n🚀 Starting ensemble training...")
    start_time = time.time()
    
    ensemble.fit(
        X, y, areas,
        validation_split=config['training']['validation_split'],
        epochs=config['training']['epochs']
    )
    
    training_time = time.time() - start_time
    
    # Save models
    ensemble.save_models("models_50m_trained")
    # Final summary
    logger.info("\n" + "🎉" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("🎉" * 60)
    logger.info(f"⏱️ Total training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    logger.info(f"📊 Samples processed: {len(X):,}")
    logger.info(f"🎯 Models trained: {len(ensemble.models)}")
    logger.info(f"💾 Models saved to: models_50m_trained/")
    logger.info(f"🚀 Ready for deployment!")
    
    # Performance summary
    logger.info(f"\n📈 PERFORMANCE SUMMARY:")
    logger.info(f"   Target: 97-98% accuracy")
    logger.info(f"   Status: Training completed - check validation results above")
    logger.info(f"   Next: Deploy models to production endpoints")

if __name__ == "__main__":
    main()
    