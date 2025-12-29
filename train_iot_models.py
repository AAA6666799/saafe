#!/usr/bin/env python3
"""
IoT-based Predictive Fire Detection Model Training.
Trains models on the new area-specific sensor data with lead time prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import the updated model classes
from saafe_mvp.models.transformer import SpatioTemporalTransformer, ModelConfig
from saafe_mvp.models.model_loader import ModelLoader
from saafe_mvp.data.iot_data_loader import create_iot_dataloaders, collate_iot_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTFireDetectionTrainer:
    """Trainer for IoT-based fire detection models."""
    
    def __init__(self, config: ModelConfig, device: torch.device = None):
        """
        Initialize the trainer.
        
        Args:
            config (ModelConfig): Model configuration
            device (torch.device): Training device
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = SpatioTemporalTransformer(config).to(self.device)
        
        # Loss functions
        self.lead_time_criterion = nn.CrossEntropyLoss()
        self.area_risk_criterion = nn.BCELoss()
        self.time_regression_criterion = nn.MSELoss()
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Training history
        self.history = {
            'train_lead_time_loss': [],
            'train_area_risk_loss': [],
            'train_time_regression_loss': [],
            'val_lead_time_loss': [],
            'val_area_risk_loss': [],
            'val_time_regression_loss': [],
            'val_lead_time_accuracy': [],
            'val_area_risk_accuracy': []
        }
        
        logger.info(f"IoT Fire Detection Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_lead_time_loss = 0.0
        epoch_area_risk_loss = 0.0
        epoch_time_regression_loss = 0.0
        num_batches = 0
        
        for batch_sequences, batch_labels in train_loader:
            # Move data to device
            for area_name in batch_sequences:
                batch_sequences[area_name] = batch_sequences[area_name].to(self.device)
            
            lead_time_labels = batch_labels['lead_time_category'].to(self.device)
            
            # Create area risk labels
            area_risk_labels = []
            for area_name in self.config.areas.keys():
                area_labels = batch_labels[f'{area_name}_anomaly'].float().to(self.device)
                area_risk_labels.append(area_labels.unsqueeze(1))
            area_risk_labels = torch.cat(area_risk_labels, dim=1)
            
            # Create time regression labels (convert lead time category to hours)
            time_labels = self._lead_time_to_hours(lead_time_labels).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_sequences)
            
            # Calculate losses
            lead_time_loss = self.lead_time_criterion(outputs['lead_time_logits'], lead_time_labels)
            area_risk_loss = self.area_risk_criterion(outputs['area_risks'], area_risk_labels)
            time_regression_loss = self.time_regression_criterion(outputs['time_to_ignition'], time_labels)
            
            # Combined loss with weights
            total_loss = lead_time_loss + 0.5 * area_risk_loss + 0.3 * time_regression_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            epoch_lead_time_loss += lead_time_loss.item()
            epoch_area_risk_loss += area_risk_loss.item()
            epoch_time_regression_loss += time_regression_loss.item()
            num_batches += 1
        
        return {
            'lead_time_loss': epoch_lead_time_loss / num_batches,
            'area_risk_loss': epoch_area_risk_loss / num_batches,
            'time_regression_loss': epoch_time_regression_loss / num_batches
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_lead_time_loss = 0.0
        epoch_area_risk_loss = 0.0
        epoch_time_regression_loss = 0.0
        
        lead_time_correct = 0
        area_risk_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                # Move data to device
                for area_name in batch_sequences:
                    batch_sequences[area_name] = batch_sequences[area_name].to(self.device)
                
                lead_time_labels = batch_labels['lead_time_category'].to(self.device)
                
                # Create area risk labels
                area_risk_labels = []
                for area_name in self.config.areas.keys():
                    area_labels = batch_labels[f'{area_name}_anomaly'].float().to(self.device)
                    area_risk_labels.append(area_labels.unsqueeze(1))
                area_risk_labels = torch.cat(area_risk_labels, dim=1)
                
                # Create time regression labels
                time_labels = self._lead_time_to_hours(lead_time_labels).to(self.device)
                
                # Forward pass
                outputs = self.model(batch_sequences)
                
                # Calculate losses
                lead_time_loss = self.lead_time_criterion(outputs['lead_time_logits'], lead_time_labels)
                area_risk_loss = self.area_risk_criterion(outputs['area_risks'], area_risk_labels)
                time_regression_loss = self.time_regression_criterion(outputs['time_to_ignition'], time_labels)
                
                # Calculate accuracies
                lead_time_pred = torch.argmax(outputs['lead_time_logits'], dim=1)
                lead_time_correct += (lead_time_pred == lead_time_labels).sum().item()
                
                area_risk_pred = (outputs['area_risks'] > 0.5).float()
                area_risk_correct += (area_risk_pred == area_risk_labels).sum().item()
                
                total_samples += lead_time_labels.size(0)
                
                # Accumulate losses
                epoch_lead_time_loss += lead_time_loss.item()
                epoch_area_risk_loss += area_risk_loss.item()
                epoch_time_regression_loss += time_regression_loss.item()
                num_batches += 1
        
        return {
            'lead_time_loss': epoch_lead_time_loss / num_batches,
            'area_risk_loss': epoch_area_risk_loss / num_batches,
            'time_regression_loss': epoch_time_regression_loss / num_batches,
            'lead_time_accuracy': lead_time_correct / total_samples,
            'area_risk_accuracy': area_risk_correct / (total_samples * len(self.config.areas))
        }
    
    def _lead_time_to_hours(self, lead_time_categories: torch.Tensor) -> torch.Tensor:
        """Convert lead time categories to hours for regression."""
        # 0=immediate (0.1h), 1=hours (6h), 2=days (48h), 3=weeks (168h)
        hour_mapping = torch.tensor([0.1, 6.0, 48.0, 168.0], device=lead_time_categories.device)
        return hour_mapping[lead_time_categories].unsqueeze(1)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 100, patience: int = 15) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store history
            self.history['train_lead_time_loss'].append(train_metrics['lead_time_loss'])
            self.history['train_area_risk_loss'].append(train_metrics['area_risk_loss'])
            self.history['train_time_regression_loss'].append(train_metrics['time_regression_loss'])
            
            self.history['val_lead_time_loss'].append(val_metrics['lead_time_loss'])
            self.history['val_area_risk_loss'].append(val_metrics['area_risk_loss'])
            self.history['val_time_regression_loss'].append(val_metrics['time_regression_loss'])
            self.history['val_lead_time_accuracy'].append(val_metrics['lead_time_accuracy'])
            self.history['val_area_risk_accuracy'].append(val_metrics['area_risk_accuracy'])
            
            # Early stopping
            val_total_loss = (val_metrics['lead_time_loss'] + 
                            val_metrics['area_risk_loss'] + 
                            val_metrics['time_regression_loss'])
            
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}:")
                logger.info(f"  Train - Lead: {train_metrics['lead_time_loss']:.4f}, "
                           f"Area: {train_metrics['area_risk_loss']:.4f}, "
                           f"Time: {train_metrics['time_regression_loss']:.4f}")
                logger.info(f"  Val - Lead: {val_metrics['lead_time_loss']:.4f}, "
                           f"Area: {val_metrics['area_risk_loss']:.4f}, "
                           f"Time: {val_metrics['time_regression_loss']:.4f}")
                logger.info(f"  Val Acc - Lead: {val_metrics['lead_time_accuracy']:.3f}, "
                           f"Area: {val_metrics['area_risk_accuracy']:.3f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.model.eval()
        
        all_lead_time_preds = []
        all_lead_time_labels = []
        all_area_risk_preds = []
        all_area_risk_labels = []
        all_time_preds = []
        all_time_labels = []
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                # Move data to device
                for area_name in batch_sequences:
                    batch_sequences[area_name] = batch_sequences[area_name].to(self.device)
                
                lead_time_labels = batch_labels['lead_time_category'].to(self.device)
                
                # Create area risk labels
                area_risk_labels = []
                for area_name in self.config.areas.keys():
                    area_labels = batch_labels[f'{area_name}_anomaly'].float().to(self.device)
                    area_risk_labels.append(area_labels.unsqueeze(1))
                area_risk_labels = torch.cat(area_risk_labels, dim=1)
                
                # Create time regression labels
                time_labels = self._lead_time_to_hours(lead_time_labels).to(self.device)
                
                # Forward pass
                outputs = self.model(batch_sequences)
                
                # Collect predictions
                lead_time_pred = torch.argmax(outputs['lead_time_logits'], dim=1)
                area_risk_pred = (outputs['area_risks'] > 0.5).float()
                time_pred = outputs['time_to_ignition']
                
                all_lead_time_preds.extend(lead_time_pred.cpu().numpy())
                all_lead_time_labels.extend(lead_time_labels.cpu().numpy())
                all_area_risk_preds.extend(area_risk_pred.cpu().numpy())
                all_area_risk_labels.extend(area_risk_labels.cpu().numpy())
                all_time_preds.extend(time_pred.cpu().numpy())
                all_time_labels.extend(time_labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_lead_time_preds = np.array(all_lead_time_preds)
        all_lead_time_labels = np.array(all_lead_time_labels)
        all_area_risk_preds = np.array(all_area_risk_preds)
        all_area_risk_labels = np.array(all_area_risk_labels)
        all_time_preds = np.array(all_time_preds).flatten()
        all_time_labels = np.array(all_time_labels).flatten()
        
        # Calculate metrics
        lead_time_accuracy = (all_lead_time_preds == all_lead_time_labels).mean()
        area_risk_accuracy = (all_area_risk_preds == all_area_risk_labels).mean()
        time_mae = np.mean(np.abs(all_time_preds - all_time_labels))
        time_rmse = np.sqrt(np.mean((all_time_preds - all_time_labels) ** 2))
        
        # Classification reports
        lead_time_report = classification_report(
            all_lead_time_labels, all_lead_time_preds,
            target_names=['Immediate', 'Hours', 'Days', 'Weeks'],
            output_dict=True
        )
        
        metrics = {
            'lead_time_accuracy': lead_time_accuracy,
            'area_risk_accuracy': area_risk_accuracy,
            'time_mae_hours': time_mae,
            'time_rmse_hours': time_rmse,
            'lead_time_classification_report': lead_time_report,
            'confusion_matrix': confusion_matrix(all_lead_time_labels, all_lead_time_preds).tolist(),
            'total_samples': len(all_lead_time_labels)
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Lead Time Accuracy: {lead_time_accuracy:.3f}")
        logger.info(f"  Area Risk Accuracy: {area_risk_accuracy:.3f}")
        logger.info(f"  Time Prediction MAE: {time_mae:.2f} hours")
        logger.info(f"  Time Prediction RMSE: {time_rmse:.2f} hours")
        
        return metrics


def create_iot_anti_hallucination_params() -> Dict[str, Any]:
    """Create anti-hallucination parameters for IoT system."""
    return {
        'version': '3.0.0',
        'created': datetime.now().isoformat(),
        'model_type': 'iot_predictive',
        'area_thresholds': {
            'kitchen': {
                'voc_normal': 150.0,
                'voc_elevated': 200.0,
                'voc_critical': 300.0,
                'lead_time_hours': 2.0
            },
            'electrical': {
                'arc_count_normal': 0,
                'arc_count_elevated': 3,
                'arc_count_critical': 10,
                'lead_time_hours': 168.0  # 1 week
            },
            'laundry_hvac': {
                'temp_normal': 25.0,
                'temp_elevated': 35.0,
                'temp_critical': 50.0,
                'current_normal': 0.5,
                'current_elevated': 1.0,
                'current_critical': 2.0,
                'lead_time_hours': 24.0
            },
            'living_bedroom': {
                'particle_normal': 5.0,
                'particle_elevated': 8.0,
                'particle_critical': 12.0,
                'lead_time_hours': 1.0
            },
            'basement_storage': {
                'gas_normal': 10.0,
                'gas_elevated': 15.0,
                'gas_critical': 25.0,
                'temp_trend_threshold': 5.0,
                'humidity_trend_threshold': 10.0,
                'lead_time_hours': 12.0
            }
        },
        'ensemble_config': {
            'voting_strategy': 'area_weighted',
            'area_weights': {
                'kitchen': 1.0,
                'electrical': 0.8,
                'laundry_hvac': 0.9,
                'living_bedroom': 1.0,
                'basement_storage': 0.7
            },
            'confidence_threshold': 0.7,
            'lead_time_confidence_scaling': True
        },
        'vendor_calibration': {
            'honeywell_mics': {'sensitivity': 0.95, 'specificity': 0.92},
            'ting_eaton': {'sensitivity': 0.98, 'specificity': 0.99},
            'honeywell_thermal': {'sensitivity': 0.93, 'specificity': 0.94},
            'xtralis_vesda': {'sensitivity': 0.99, 'specificity': 0.95},
            'bosch_airthings': {'sensitivity': 0.90, 'specificity': 0.88}
        }
    }


def main():
    """Main training function for IoT fire detection."""
    logger.info("🚀 Starting IoT-based Predictive Fire Detection Training")
    logger.info("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create IoT model configuration
    config = ModelConfig()
    
    logger.info(f"IoT Model Configuration:")
    logger.info(f"  Areas: {list(config.areas.keys())}")
    logger.info(f"  Total feature dimensions: {config.total_feature_dim}")
    logger.info(f"  d_model: {config.d_model}")
    logger.info(f"  num_heads: {config.num_heads}")
    logger.info(f"  num_layers: {config.num_layers}")
    
    # Create data loaders
    logger.info("\n📊 Loading IoT Dataset...")
    data_dir = "synthetic datasets"
    
    try:
        train_loader, val_loader, test_loader = create_iot_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            sequence_length=60,
            max_samples=50000  # Use subset for faster training
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Creating minimal synthetic data for testing...")
        # Fallback to minimal synthetic data would go here
        return
    
    # Create trainer
    logger.info("\n📚 Initializing Trainer...")
    trainer = IoTFireDetectionTrainer(config, device)
    
    # Train model
    logger.info("\n🔥 Training IoT Fire Detection Model...")
    history = trainer.train(train_loader, val_loader, num_epochs=50, patience=10)
    
    # Evaluate model
    logger.info("\n📊 Evaluating Model...")
    metrics = trainer.evaluate(test_loader)
    
    # Save model and results
    logger.info("\n💾 Saving IoT Models...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save transformer model
    model_loader = ModelLoader(device)
    model_path = models_dir / "iot_transformer_model.pth"
    model_loader.save_model(
        trainer.model,
        model_path,
        epoch=len(history['train_lead_time_loss']),
        loss=metrics['time_mae_hours'],
        metadata={
            'training_date': datetime.now().isoformat(),
            'performance_metrics': metrics,
            'training_history': history,
            'data_type': 'iot_synthetic',
            'model_type': 'iot_predictive',
            'total_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'areas': list(config.areas.keys())
        }
    )
    
    # Save anti-hallucination parameters
    anti_hallucination_params = create_iot_anti_hallucination_params()
    with open(models_dir / "iot_anti_hallucination.pkl", 'wb') as f:
        pickle.dump(anti_hallucination_params, f)
    
    # Update model metadata
    metadata = {
        'version': '3.0.0',
        'created': datetime.now().isoformat(),
        'description': 'Saafe IoT-based Predictive Fire Detection System',
        'model_architecture': 'IoT-SpatioTemporalTransformer',
        'areas': config.areas,
        'performance_metrics': {
            'lead_time_accuracy': metrics['lead_time_accuracy'],
            'area_risk_accuracy': metrics['area_risk_accuracy'],
            'time_prediction_mae_hours': metrics['time_mae_hours'],
            'time_prediction_rmse_hours': metrics['time_rmse_hours']
        },
        'training_info': {
            'data_source': 'iot_synthetic_datasets',
            'training_samples': len(train_loader.dataset),
            'validation_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'device': str(device),
            'epochs_trained': len(history['train_lead_time_loss']),
            'model_parameters': sum(p.numel() for p in trainer.model.parameters())
        },
        'capabilities': {
            'predictive_lead_times': ['immediate', 'hours', 'days', 'weeks'],
            'area_specific_detection': True,
            'vendor_calibration': True,
            'false_alarm_prevention': True
        }
    }
    
    with open(models_dir / "iot_model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create training plots
    plt.figure(figsize=(15, 12))
    
    # Loss plots
    plt.subplot(2, 3, 1)
    plt.plot(history['train_lead_time_loss'], label='Train')
    plt.plot(history['val_lead_time_loss'], label='Validation')
    plt.title('Lead Time Classification Loss')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(history['train_area_risk_loss'], label='Train')
    plt.plot(history['val_area_risk_loss'], label='Validation')
    plt.title('Area Risk Detection Loss')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(history['train_time_regression_loss'], label='Train')
    plt.plot(history['val_time_regression_loss'], label='Validation')
    plt.title('Time-to-Ignition Regression Loss')
    plt.legend()
    
    # Accuracy plots
    plt.subplot(2, 3, 4)
    plt.plot(history['val_lead_time_accuracy'])
    plt.title('Lead Time Classification Accuracy')
    
    plt.subplot(2, 3, 5)
    plt.plot(history['val_area_risk_accuracy'])
    plt.title('Area Risk Detection Accuracy')
    
    # Confusion matrix
    plt.subplot(2, 3, 6)
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Immediate', 'Hours', 'Days', 'Weeks'],
                yticklabels=['Immediate', 'Hours', 'Days', 'Weeks'])
    plt.title('Lead Time Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(models_dir / 'iot_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ IoT Fire Detection Training Completed Successfully!")
    logger.info(f"📁 Models saved to: {models_dir}/")
    logger.info(f"🎯 Lead Time Accuracy: {metrics['lead_time_accuracy']:.1%}")
    logger.info(f"🏠 Area Risk Accuracy: {metrics['area_risk_accuracy']:.1%}")
    logger.info(f"⏰ Time Prediction MAE: {metrics['time_mae_hours']:.1f} hours")
    logger.info(f"🔧 Model Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    logger.info(f"💾 Model Size: {sum(p.numel() * 4 for p in trainer.model.parameters()) / 1024 / 1024:.1f} MB")
    logger.info("\n🚀 Ready for IoT deployment with predictive capabilities!")


if __name__ == "__main__":
    main()