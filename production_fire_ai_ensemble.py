#!/usr/bin/env python3
"""
🔥 Production-Ready Fire Detection AI Ensemble System
Combines Saafe MVP architecture with advanced ensemble techniques
Target: 97-98% accuracy with <0.5% false positive rate

Architecture:
- Tier 1: Enhanced Deep Learning Models (5 models)
- Tier 2: Specialist Algorithms (9 models) 
- Tier 3: Meta-Learning Systems (3 systems)
- Total: 17+ algorithms working together
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import asyncio
from pathlib import Path

# Core ML libraries
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    IsolationForest, ExtraTreesClassifier, StackingClassifier
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Advanced ML libraries (with fallbacks)
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

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🔥 PRODUCTION FIRE DETECTION AI ENSEMBLE")
print("=" * 50)
print(f"🎯 Target: 97-98% accuracy, <0.5% false positive rate")
print(f"🚀 17+ algorithms working together")
print(f"📊 XGBoost: {'✅' if XGB_AVAILABLE else '❌'}")
print(f"📊 LightGBM: {'✅' if LGB_AVAILABLE else '❌'}")
print(f"📊 CatBoost: {'✅' if CB_AVAILABLE else '❌'}")
print(f"📊 Prophet: {'✅' if PROPHET_AVAILABLE else '❌'}")

# ===============================================================================
# 🏗️ ENHANCED TIER 1: DEEP LEARNING MODELS
# ===============================================================================

@dataclass
class EnhancedModelConfig:
    """Enhanced configuration for production fire detection models."""
    # Core architecture
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_seq_length: int = 512
    dropout: float = 0.1
    
    # Fire detection specifics
    num_classes: int = 3  # normal, warning, fire
    num_areas: int = 5    # kitchen, electrical, laundry, living, basement
    feature_dim: int = 6  # temp, humidity, smoke, pressure, gas, wind
    
    # Lead time prediction
    num_risk_levels: int = 4  # immediate, hours, days, weeks
    
    # Enhanced features
    use_multi_scale_attention: bool = True
    use_uncertainty_quantification: bool = True
    use_advanced_positional_encoding: bool = True


class MultiScaleAttention(nn.Module):
    """Multi-scale attention for capturing patterns at different time scales."""
    
    def __init__(self, d_model: int, num_heads: int, scales: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in scales
        ])
        self.fusion = nn.Linear(d_model * len(scales), d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale attention."""
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []
        
        for scale, attention in zip(self.scales, self.attentions):
            if scale == 1:
                # Standard attention
                attended, _ = attention(x, x, x)
            else:
                # Downsampled attention
                if seq_len >= scale:
                    # Simple downsampling by taking every scale-th element
                    downsampled = x[:, ::scale, :]
                    attended_down, _ = attention(downsampled, downsampled, downsampled)
                    
                    # Upsample back to original length
                    attended = torch.zeros_like(x)
                    attended[:, ::scale, :] = attended_down
                    
                    # Interpolate missing values
                    for i in range(1, scale):
                        if i < seq_len:
                            attended[:, i::scale, :] = attended_down
                else:
                    attended = x
            
            scale_outputs.append(attended)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.fusion(fused)
        
        return output


class UncertaintyQuantificationLayer(nn.Module):
    """Layer for quantifying prediction uncertainty."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean_layer = nn.Linear(input_dim, output_dim)
        self.var_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and variance predictions."""
        mean = self.mean_layer(x)
        log_var = self.var_layer(x)
        var = torch.exp(log_var)
        return mean, var


class EnhancedSpatioTemporalTransformer(nn.Module):
    """Enhanced Spatio-Temporal Transformer with production features."""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.feature_dim, config.d_model)
        
        # Enhanced positional encoding
        if config.use_advanced_positional_encoding:
            self.pos_encoding = self._create_advanced_positional_encoding()
        else:
            self.pos_encoding = nn.Parameter(
                torch.randn(config.max_seq_length, config.d_model)
            )
        
        # Multi-scale attention layers
        self.attention_layers = nn.ModuleList([
            MultiScaleAttention(config.d_model, config.num_heads)
            if config.use_multi_scale_attention
            else nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
            for _ in range(config.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.d_model) for _ in range(config.num_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model * 4, config.d_model),
                nn.Dropout(config.dropout)
            ) for _ in range(config.num_layers)
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification heads with uncertainty
        if config.use_uncertainty_quantification:
            self.fire_classifier = UncertaintyQuantificationLayer(config.d_model, config.num_classes)
            self.lead_time_classifier = UncertaintyQuantificationLayer(config.d_model, config.num_risk_levels)
        else:
            self.fire_classifier = nn.Linear(config.d_model, config.num_classes)
            self.lead_time_classifier = nn.Linear(config.d_model, config.num_risk_levels)
        
        # Risk assessment head
        self.risk_assessment = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Area-specific risk heads
        self.area_risk_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 4),
                nn.ReLU(),
                nn.Linear(config.d_model // 4, 1),
                nn.Sigmoid()
            ) for _ in range(config.num_areas)
        ])
        
    def _create_advanced_positional_encoding(self) -> nn.Parameter:
        """Create advanced positional encoding with multiple frequencies."""
        max_len = self.config.max_seq_length
        d_model = self.config.d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Use multiple frequency scales
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add learnable components
        learnable_pe = torch.randn(max_len, d_model) * 0.1
        
        return nn.Parameter(pe + learnable_pe)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with uncertainty quantification."""
        batch_size, seq_len, num_areas, feature_dim = x.shape
        
        # Reshape for processing: (batch * areas, seq_len, features)
        x_reshaped = x.view(batch_size * num_areas, seq_len, feature_dim)
        
        # Input projection
        x_proj = self.input_projection(x_reshaped)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)
        x_proj = x_proj + pos_enc
        
        # Apply transformer layers
        hidden_states = x_proj
        
        for i, (attention, layer_norm, feed_forward) in enumerate(
            zip(self.attention_layers, self.layer_norms, self.feed_forwards)
        ):
            # Multi-head attention with residual connection
            if isinstance(attention, MultiScaleAttention):
                attended = attention(hidden_states)
            else:
                attended, _ = attention(hidden_states, hidden_states, hidden_states)
            
            hidden_states = layer_norm(hidden_states + attended)
            
            # Feed-forward with residual connection
            ff_output = feed_forward(hidden_states)
            hidden_states = layer_norm(hidden_states + ff_output)
        
        # Global pooling: (batch * areas, seq_len, d_model) -> (batch * areas, d_model)
        pooled = self.global_pool(hidden_states.transpose(1, 2)).squeeze(-1)
        
        # Reshape back: (batch, areas, d_model)
        pooled = pooled.view(batch_size, num_areas, self.config.d_model)
        
        # Area-specific risk assessment
        area_risks = []
        for i, area_head in enumerate(self.area_risk_heads):
            area_risk = area_head(pooled[:, i, :])  # (batch, 1)
            area_risks.append(area_risk)
        
        area_risks_tensor = torch.cat(area_risks, dim=1)  # (batch, num_areas)
        
        # Global feature representation (average across areas)
        global_features = pooled.mean(dim=1)  # (batch, d_model)
        
        # Generate predictions
        outputs = {'features': global_features, 'area_risks': area_risks_tensor}
        
        if self.config.use_uncertainty_quantification:
            # Predictions with uncertainty
            fire_mean, fire_var = self.fire_classifier(global_features)
            lead_time_mean, lead_time_var = self.lead_time_classifier(global_features)
            
            outputs.update({
                'fire_logits': fire_mean,
                'fire_uncertainty': fire_var,
                'lead_time_logits': lead_time_mean,
                'lead_time_uncertainty': lead_time_var
            })
        else:
            # Standard predictions
            outputs.update({
                'fire_logits': self.fire_classifier(global_features),
                'lead_time_logits': self.lead_time_classifier(global_features)
            })
        
        # Overall risk score
        outputs['risk_score'] = self.risk_assessment(global_features) * 100.0
        
        return outputs


class LSTMCNNHybrid(nn.Module):
    """LSTM-CNN Hybrid for sequential + local pattern detection."""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # CNN layers for local pattern detection
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, config.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU()
        )
        
        # LSTM layers for sequential dependencies
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model // 2,
            num_layers=3,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            config.d_model, config.num_heads, batch_first=True
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM-CNN hybrid."""
        batch_size, seq_len, num_areas, feature_dim = x.shape
        
        # Process each area separately then combine
        area_outputs = []
        
        for area in range(num_areas):
            area_data = x[:, :, area, :]  # (batch, seq_len, features)
            
            # CNN processing: (batch, features, seq_len)
            conv_input = area_data.transpose(1, 2)
            conv_output = self.conv_layers(conv_input)
            conv_output = conv_output.transpose(1, 2)  # Back to (batch, seq_len, d_model)
            
            # LSTM processing
            lstm_output, _ = self.lstm(conv_output)
            
            # Self-attention
            attended_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
            
            # Global pooling
            pooled = attended_output.mean(dim=1)  # (batch, d_model)
            area_outputs.append(pooled)
        
        # Combine area outputs
        combined = torch.stack(area_outputs, dim=1).mean(dim=1)  # (batch, d_model)
        
        # Classification
        output = self.classifier(combined)
        
        return output


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for sensor relationship modeling."""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # Node feature transformation
        self.node_encoder = nn.Linear(config.feature_dim, config.d_model)
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model) for _ in range(3)
        ])
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model * config.num_areas, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes)
        )
        
    def create_adjacency_matrix(self, batch_size: int) -> torch.Tensor:
        """Create adjacency matrix for sensor relationships."""
        # Simple fully connected graph for now
        adj = torch.ones(self.config.num_areas, self.config.num_areas)
        adj = adj / adj.sum(dim=1, keepdim=True)  # Normalize
        return adj.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph neural network."""
        batch_size, seq_len, num_areas, feature_dim = x.shape
        
        # Use latest timestep for simplicity
        latest_features = x[:, -1, :, :]  # (batch, num_areas, features)
        
        # Encode node features
        node_features = self.node_encoder(latest_features)  # (batch, num_areas, d_model)
        
        # Create adjacency matrix
        adj = self.create_adjacency_matrix(batch_size).to(x.device)
        
        # Apply graph convolutions
        for graph_conv in self.graph_convs:
            # Message passing: aggregate neighbor features
            aggregated = torch.bmm(adj, node_features)  # (batch, num_areas, d_model)
            
            # Transform and add residual connection
            transformed = graph_conv(aggregated)
            node_features = F.relu(transformed + node_features)
        
        # Global pooling
        graph_features = node_features.view(batch_size, -1)  # (batch, num_areas * d_model)
        
        # Classification
        output = self.classifier(graph_features)
        
        return output


class TemporalConvolutionalNetwork(nn.Module):
    """Temporal Convolutional Network for parallel processing."""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # TCN layers with increasing dilation
        self.tcn_layers = nn.ModuleList()
        channels = [config.feature_dim] + [config.d_model] * 4
        
        for i in range(len(channels) - 1):
            dilation = 2 ** i
            self.tcn_layers.append(
                self._make_tcn_block(
                    channels[i], channels[i + 1], 
                    kernel_size=3, dilation=dilation
                )
            )
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes)
        )
    
    def _make_tcn_block(self, in_channels: int, out_channels: int, 
                       kernel_size: int, dilation: int) -> nn.Module:
        """Create a TCN block with residual connection."""
        padding = (kernel_size - 1) * dilation
        
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                     dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                     dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN."""
        batch_size, seq_len, num_areas, feature_dim = x.shape
        
        # Process all areas together
        # Reshape: (batch * areas, features, seq_len)
        x_reshaped = x.view(batch_size * num_areas, seq_len, feature_dim)
        x_reshaped = x_reshaped.transpose(1, 2)
        
        # Apply TCN layers
        for tcn_layer in self.tcn_layers:
            x_reshaped = tcn_layer(x_reshaped)
        
        # Global pooling and reshape
        pooled = self.global_pool(x_reshaped)  # (batch * areas, d_model, 1)
        pooled = pooled.view(batch_size, num_areas, self.config.d_model)
        
        # Average across areas
        averaged = pooled.mean(dim=1)  # (batch, d_model)
        
        # Classification
        output = self.classifier(averaged)
        
        return output


class LSTMVariationalAutoencoder(nn.Module):
    """LSTM-VAE for unsupervised anomaly detection."""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.d_model // 4
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            config.feature_dim, config.d_model // 2, 
            batch_first=True, bidirectional=True
        )
        
        # Variational parameters
        self.fc_mu = nn.Linear(config.d_model, self.latent_dim)
        self.fc_logvar = nn.Linear(config.d_model, self.latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(self.latent_dim, config.d_model)
        self.decoder_lstm = nn.LSTM(
            config.d_model, config.d_model // 2, batch_first=True
        )
        self.decoder_output = nn.Linear(config.d_model // 2, config.feature_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_classes)
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through LSTM-VAE."""
        batch_size, seq_len, num_areas, feature_dim = x.shape
        
        # Average across areas for simplicity
        x_avg = x.mean(dim=2)  # (batch, seq_len, features)
        
        # Encode
        lstm_out, _ = self.encoder_lstm(x_avg)
        encoded = lstm_out[:, -1, :]  # Use last hidden state
        
        # Variational parameters
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode (for training)
        decoded_input = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        decoded_lstm, _ = self.decoder_lstm(decoded_input)
        reconstruction = self.decoder_output(decoded_lstm)
        
        # Classify
        classification = self.classifier(z)
        
        return {
            'classification': classification,
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

# ===============================================================================
# 🏗️ PRODUCTION ENSEMBLE SYSTEM
# ===============================================================================

class ProductionFireEnsemble:
    """
    Production-ready fire detection ensemble system.
    Combines enhanced deep learning models with specialist algorithms.
    """
    
    def __init__(self, config: EnhancedModelConfig = None, device: torch.device = None):
        """Initialize the production ensemble system."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or EnhancedModelConfig()
        
        # Initialize deep learning models (Tier 1)
        self.deep_models = self._initialize_deep_models()
        
        # Initialize specialist models (Tier 2)
        self.specialist_models = {}
        
        # Initialize meta-learning systems (Tier 3)
        self.meta_models = {}
        
        # Scalers for preprocessing
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'accuracy_history': [],
            'false_positive_rate': 0.0,
            'processing_times': []
        }
        
        self.is_fitted = False
        
        logger.info("🚀 Production Fire Ensemble initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Deep learning models: {len(self.deep_models)}")
    
    def _initialize_deep_models(self) -> Dict[str, nn.Module]:
        """Initialize all deep learning models."""
        models = {
            'enhanced_transformer': EnhancedSpatioTemporalTransformer(self.config),
            'lstm_cnn_hybrid': LSTMCNNHybrid(self.config),
            'graph_neural_network': GraphNeuralNetwork(self.config),
            'temporal_conv_network': TemporalConvolutionalNetwork(self.config),
            'lstm_vae': LSTMVariationalAutoencoder(self.config)
        }
        
        # Move models to device
        for name, model in models.items():
            models[name] = model.to(self.device)
            
        return models
    
    def _initialize_specialist_models(self):
        """Initialize specialist ML models."""
        if XGB_AVAILABLE:
            self.specialist_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        
        if LGB_AVAILABLE:
            self.specialist_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        
        if CB_AVAILABLE:
            self.specialist_models['catboost'] = cb.CatBoostClassifier(
                iterations=500,
                depth=8,
                learning_rate=0.01,
                random_seed=42,
                verbose=False
            )
        
        # Always available models
        self.specialist_models.update({
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.01, max_depth=6, random_state=42
            ),
            'isolation_forest': IsolationForest(
                n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )
        })
        
        logger.info(f"Initialized {len(self.specialist_models)} specialist models")
    
    def engineer_advanced_features(self, X: np.ndarray) -> np.ndarray:
        """Engineer advanced features for maximum performance."""
        if X.ndim != 3:
            return X
            
        batch_size, seq_len, num_features = X.shape
        features = []
        
        # Original features
        features.append(X.reshape(batch_size, -1))
        
        # Statistical features over time
        features.append(np.mean(X, axis=1))
        features.append(np.std(X, axis=1))
        features.append(np.max(X, axis=1))
        features.append(np.min(X, axis=1))
        features.append(np.median(