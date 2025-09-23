"""
Spatio-Temporal Transformer model for fire detection.

This module contains the complete implementation of the Spatio-Temporal Transformer
architecture extracted from the Colab notebook, adapted for standalone use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration class for the IoT-based Predictive Fire Detection Transformer model."""
    # Area-based sensor configuration
    areas: Dict[str, Dict[str, Any]] = None
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_seq_length: int = 512
    dropout: float = 0.1
    num_risk_levels: int = 4  # immediate, hours, days, weeks
    
    def __post_init__(self):
        if self.areas is None:
            self.areas = {
                'kitchen': {
                    'sensor_type': 'voc_ml',
                    'features': ['voc_level'],
                    'feature_dim': 1,
                    'lead_time_range': 'minutes_hours',
                    'vendor': 'honeywell_mics'
                },
                'electrical': {
                    'sensor_type': 'arc_detection',
                    'features': ['arc_count'],
                    'feature_dim': 1,
                    'lead_time_range': 'days_weeks',
                    'vendor': 'ting_eaton'
                },
                'laundry_hvac': {
                    'sensor_type': 'thermal_current',
                    'features': ['temperature', 'current'],
                    'feature_dim': 2,
                    'lead_time_range': 'hours_days',
                    'vendor': 'honeywell_thermal'
                },
                'living_bedroom': {
                    'sensor_type': 'aspirating_smoke',
                    'features': ['particle_level'],
                    'feature_dim': 1,
                    'lead_time_range': 'minutes_hours',
                    'vendor': 'xtralis_vesda'
                },
                'basement_storage': {
                    'sensor_type': 'environmental_iot',
                    'features': ['temperature', 'humidity', 'gas_levels'],
                    'feature_dim': 3,
                    'lead_time_range': 'hours_days',
                    'vendor': 'bosch_airthings'
                }
            }
        
        # Calculate total features across all areas
        self.total_feature_dim = sum(area['feature_dim'] for area in self.areas.values())
        self.num_areas = len(self.areas)


class SpatialAttentionLayer(nn.Module):
    """
    Spatial attention layer for capturing relationships between sensor locations.
    Uses multi-head attention to model how sensors at different locations influence each other.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the spatial attention layer.
        
        Args:
            d_model (int): Model dimension (hidden size)
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Spatial position encoding for different areas (5 areas)
        self.spatial_encoding = nn.Parameter(torch.randn(5, d_model))  # 5 areas
    
    def forward(self, x: torch.Tensor, spatial_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, num_areas, d_model)
            spatial_mask (torch.Tensor): Optional spatial attention mask
            
        Returns:
            torch.Tensor: Output with spatial attention applied
        """
        batch_size, seq_len, num_areas, d_model = x.shape
        
        # Add spatial position encoding
        x_with_pos = x + self.spatial_encoding.unsqueeze(0).unsqueeze(0)
        
        # Reshape for attention computation: (batch_size * seq_len, num_areas, d_model)
        x_reshaped = x_with_pos.view(batch_size * seq_len, num_areas, d_model)
        
        # Compute Q, K, V projections
        Q = self.query_projection(x_reshaped)  # (batch_size * seq_len, num_areas, d_model)
        K = self.key_projection(x_reshaped)
        V = self.value_projection(x_reshaped)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size * seq_len, num_areas, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * seq_len, num_areas, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size * seq_len, num_areas, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply spatial mask if provided
        if spatial_mask is not None:
            attention_scores = attention_scores.masked_fill(spatial_mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape back to original dimensions
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size * seq_len, num_areas, d_model
        )
        
        # Apply output projection
        output = self.output_projection(attended_values)
        
        # Reshape back to original tensor shape
        output = output.view(batch_size, seq_len, num_areas, d_model)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + x)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization purposes.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Attention weights
        """
        batch_size, seq_len, num_sensors, d_model = x.shape
        
        # Add spatial position encoding
        x_with_pos = x + self.spatial_encoding.unsqueeze(0).unsqueeze(0)
        x_reshaped = x_with_pos.view(batch_size * seq_len, num_sensors, d_model)
        
        # Compute Q, K projections
        Q = self.query_projection(x_reshaped)
        K = self.key_projection(x_reshaped)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size * seq_len, num_sensors, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * seq_len, num_sensors, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights.view(batch_size, seq_len, self.num_heads, num_sensors, num_sensors)


class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention layer for modeling temporal dependencies in sensor readings.
    Uses causal attention to capture how past sensor readings influence current predictions.
    """
    
    def __init__(self, d_model: int, num_heads: int, max_seq_length: int = 512, dropout: float = 0.1):
        """
        Initialize the temporal attention layer.
        
        Args:
            d_model (int): Model dimension (hidden size)
            num_heads (int): Number of attention heads
            max_seq_length (int): Maximum sequence length for positional encoding
            dropout (float): Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_length = max_seq_length
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Temporal positional encoding
        self.temporal_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Causal mask for temporal attention (prevents looking into the future)
        self.register_buffer('causal_mask', self._create_causal_mask(max_seq_length))
    
    def _create_positional_encoding(self, max_seq_length: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding for temporal positions.
        
        Args:
            max_seq_length (int): Maximum sequence length
            d_model (int): Model dimension
            
        Returns:
            torch.Tensor: Positional encoding tensor
        """
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """
        Create causal mask to prevent attention to future positions.
        
        Args:
            seq_length (int): Sequence length
            
        Returns:
            torch.Tensor: Causal mask tensor
        """
        mask = torch.tril(torch.ones(seq_length, seq_length))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, x: torch.Tensor, temporal_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of temporal attention.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, num_sensors, d_model)
            temporal_mask (torch.Tensor): Optional temporal attention mask
            
        Returns:
            torch.Tensor: Output with temporal attention applied
        """
        batch_size, seq_len, num_sensors, d_model = x.shape
        
        # Add temporal positional encoding
        pos_encoding = self.temporal_encoding[:, :seq_len, :].to(x.device)
        x_with_pos = x + pos_encoding.unsqueeze(2)  # Broadcast across sensors
        
        # Reshape for attention computation: (batch_size * num_sensors, seq_len, d_model)
        x_reshaped = x_with_pos.permute(0, 2, 1, 3).contiguous().view(
            batch_size * num_sensors, seq_len, d_model
        )
        
        # Compute Q, K, V projections
        Q = self.query_projection(x_reshaped)  # (batch_size * num_sensors, seq_len, d_model)
        K = self.key_projection(x_reshaped)
        V = self.value_projection(x_reshaped)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size * num_sensors, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * num_sensors, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size * num_sensors, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply causal mask (prevent looking into future)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)
        
        # Apply additional temporal mask if provided
        if temporal_mask is not None:
            attention_scores = attention_scores.masked_fill(temporal_mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape back to original dimensions
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size * num_sensors, seq_len, d_model
        )
        
        # Apply output projection
        output = self.output_projection(attended_values)
        
        # Reshape back to original tensor shape
        output = output.view(batch_size, num_sensors, seq_len, d_model).permute(0, 2, 1, 3)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + x)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization purposes.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Attention weights
        """
        batch_size, seq_len, num_sensors, d_model = x.shape
        
        # Add temporal positional encoding
        pos_encoding = self.temporal_encoding[:, :seq_len, :].to(x.device)
        x_with_pos = x + pos_encoding.unsqueeze(2)
        
        # Reshape for attention computation
        x_reshaped = x_with_pos.permute(0, 2, 1, 3).contiguous().view(
            batch_size * num_sensors, seq_len, d_model
        )
        
        # Compute Q, K projections
        Q = self.query_projection(x_reshaped)
        K = self.key_projection(x_reshaped)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size * num_sensors, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * num_sensors, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights.view(batch_size, num_sensors, self.num_heads, seq_len, seq_len)


class SpatioTemporalTransformerLayer(nn.Module):
    """
    Single Spatio-Temporal Transformer layer combining spatial and temporal attention.
    Applies spatial attention first, then temporal attention with residual connections.
    """
    
    def __init__(self, d_model: int, num_heads: int, max_seq_length: int = 512, dropout: float = 0.1):
        """
        Initialize a single Spatio-Temporal Transformer layer.
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            max_seq_length (int): Maximum sequence length
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # Spatial and temporal attention layers
        self.spatial_attention = SpatialAttentionLayer(d_model, num_heads, dropout)
        self.temporal_attention = TemporalAttentionLayer(d_model, num_heads, max_seq_length, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, 
                spatial_mask: torch.Tensor = None, 
                temporal_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the Spatio-Temporal Transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, num_sensors, d_model)
            spatial_mask (torch.Tensor): Optional spatial attention mask
            temporal_mask (torch.Tensor): Optional temporal attention mask
            
        Returns:
            Tuple[torch.Tensor, Dict]: Output tensor and attention weights
        """
        # Store input for residual connection
        residual = x
        
        # Apply spatial attention
        x_spatial = self.spatial_attention(x, spatial_mask)
        
        # Apply temporal attention
        x_temporal = self.temporal_attention(x_spatial, temporal_mask)
        
        # Apply feed-forward network with residual connection
        x_ff = self.feed_forward(x_temporal)
        output = self.layer_norm(x_ff + x_temporal)
        
        # Collect attention weights for visualization
        attention_weights = {
            'spatial': self.spatial_attention.get_attention_weights(x),
            'temporal': self.temporal_attention.get_attention_weights(x_spatial)
        }
        
        return output, attention_weights


class SpatioTemporalTransformer(nn.Module):
    """
    Main Spatio-Temporal Transformer model for fire detection.
    Combines spatial and temporal attention layers to process multi-sensor time-series data.
    """
    
    def __init__(self, config: ModelConfig = None):
        """
        Initialize the Spatio-Temporal Transformer model.
        
        Args:
            config (ModelConfig): Model configuration object
        """
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        self.num_areas = config.num_areas
        self.total_feature_dim = config.total_feature_dim
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.num_risk_levels = config.num_risk_levels
        
        # Area-specific input embedding layers
        self.area_embeddings = nn.ModuleDict()
        for area_name, area_config in config.areas.items():
            self.area_embeddings[area_name] = nn.Linear(area_config['feature_dim'], config.d_model)
        
        # Combined feature embedding for heterogeneous data
        self.feature_fusion = nn.Linear(config.d_model * config.num_areas, config.d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            SpatioTemporalTransformerLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                max_seq_length=config.max_seq_length,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Output layers
        self.global_pooling = nn.AdaptiveAvgPool2d((1, config.d_model))  # Pool over time and areas
        
        # Lead time prediction head (minutes, hours, days, weeks)
        self.lead_time_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_risk_levels)
        )
        
        # Area-specific risk assessment heads
        self.area_risk_heads = nn.ModuleDict()
        for area_name in config.areas.keys():
            self.area_risk_heads[area_name] = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 4),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 4, 1),
                nn.Sigmoid()  # 0-1 risk probability
            )
        
        # Time-to-ignition regression head (in hours)
        self.time_to_ignition = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.ReLU()  # Positive time values
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize model weights using Xavier/Glorot initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, area_data: Dict[str, torch.Tensor], 
                spatial_mask: torch.Tensor = None, 
                temporal_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the IoT-based Predictive Fire Detection Transformer.
        
        Args:
            area_data (Dict[str, torch.Tensor]): Dictionary of area-specific sensor data
                - Each key is area name, value is tensor (batch_size, seq_len, feature_dim)
            spatial_mask (torch.Tensor): Optional spatial attention mask
            temporal_mask (torch.Tensor): Optional temporal attention mask
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'lead_time_logits': Lead time classification (batch_size, num_risk_levels)
                - 'area_risks': Area-specific risk scores (batch_size, num_areas)
                - 'time_to_ignition': Predicted time to ignition in hours (batch_size, 1)
                - 'features': Final feature representations (batch_size, d_model)
        """
        batch_size = next(iter(area_data.values())).shape[0]
        seq_len = next(iter(area_data.values())).shape[1]
        
        # Process each area's data through its specific embedding
        area_embeddings = []
        for area_name in self.config.areas.keys():
            if area_name in area_data:
                area_tensor = area_data[area_name]  # (batch_size, seq_len, feature_dim)
                embedded = self.area_embeddings[area_name](area_tensor)  # (batch_size, seq_len, d_model)
                area_embeddings.append(embedded)
            else:
                # Handle missing area data with zeros
                feature_dim = self.config.areas[area_name]['feature_dim']
                zero_data = torch.zeros(batch_size, seq_len, feature_dim, device=next(iter(area_data.values())).device)
                embedded = self.area_embeddings[area_name](zero_data)
                area_embeddings.append(embedded)
        
        # Stack area embeddings: (batch_size, seq_len, num_areas, d_model)
        embedded = torch.stack(area_embeddings, dim=2)
        
        # Apply transformer layers
        hidden_states = embedded
        attention_weights = []
        
        for layer in self.transformer_layers:
            hidden_states, layer_attention = layer(
                hidden_states, 
                spatial_mask=spatial_mask, 
                temporal_mask=temporal_mask
            )
            attention_weights.append(layer_attention)
        
        # Global pooling to get fixed-size representation
        # Pool over both time and area dimensions
        pooled_features = self.global_pooling(hidden_states.view(batch_size, seq_len * self.num_areas, self.d_model))
        pooled_features = pooled_features.squeeze(1)  # (batch_size, d_model)
        
        # Lead time classification (immediate, hours, days, weeks)
        lead_time_logits = self.lead_time_classifier(pooled_features)
        
        # Area-specific risk assessment
        area_risks = {}
        for area_name in self.config.areas.keys():
            area_risks[area_name] = self.area_risk_heads[area_name](pooled_features)
        
        # Combine area risks into tensor
        area_risk_tensor = torch.cat([area_risks[area] for area in self.config.areas.keys()], dim=1)
        
        # Time-to-ignition prediction (in hours)
        time_to_ignition = self.time_to_ignition(pooled_features)
        
        return {
            'lead_time_logits': lead_time_logits,
            'area_risks': area_risk_tensor,
            'area_risk_dict': area_risks,
            'time_to_ignition': time_to_ignition,
            'features': pooled_features,
            'attention_weights': attention_weights
        }
    
    def predict_risk_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get only risk scores for inference.
        
        Args:
            x (torch.Tensor): Input sensor data
            
        Returns:
            torch.Tensor: Risk scores (0-100)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['risk_score']
    
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get class predictions.
        
        Args:
            x (torch.Tensor): Input sensor data
            
        Returns:
            torch.Tensor: Predicted class indices
        """
        with torch.no_grad():
            outputs = self.forward(x)
            return torch.argmax(outputs['logits'], dim=-1)
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from all layers for visualization.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of attention weights by layer
        """
        attention_weights = {}
        
        for i, layer in enumerate(self.transformer_layers):
            attention_weights[f'layer_{i}_spatial'] = layer.spatial_attention.get_attention_weights
            attention_weights[f'layer_{i}_temporal'] = layer.temporal_attention.get_attention_weights
        
        return attention_weights


def create_model(config: ModelConfig = None, device: torch.device = None) -> SpatioTemporalTransformer:
    """
    Factory function to create a Spatio-Temporal Transformer model.
    
    Args:
        config (ModelConfig): Model configuration
        device (torch.device): Device to place the model on
        
    Returns:
        SpatioTemporalTransformer: Initialized model
    """
    if config is None:
        config = ModelConfig()
    
    model = SpatioTemporalTransformer(config)
    
    if device is not None:
        model = model.to(device)
    
    return model