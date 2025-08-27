#!/usr/bin/env python3
"""
Working Model Training Script - Trains Existing Fire Detection Models

This script trains the working models we know exist and can be imported:
- Classification models (Binary, Multi-class, Deep Learning, Ensemble)
- Using the working AWS ensemble trainer as the foundation
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Add the current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class WorkingModelTrainer:
    """
    Trains the working fire detection models we know exist.
    """
    
    def __init__(self):
        """Initialize the trainer."""
        self.start_time = datetime.now()
        self.training_results = {
            'total_models': 0,
            'successful_models': 0,
            'failed_models': 0,
            'training_times': {},
            'model_metrics': {},
            'errors': []
        }
    
    def generate_synthetic_data(self, n_samples: int = 5000) -> Dict[str, Any]:
        """Generate synthetic training data for fire detection."""
        logger.info(f"Generating {n_samples:,} synthetic training samples...")
        
        np.random.seed(42)
        
        # Generate realistic fire detection features
        n_features = 18  # Standard number of features
        
        # Create feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Add realistic patterns for fire detection
        for i in range(n_samples):
            if i % 4 == 0:  # 25% fire cases
                # Electrical fire pattern
                X[i, 0:3] += np.random.uniform(2, 4, 3)  # High electrical readings
                X[i, 10:13] += np.random.uniform(1, 3, 3)  # High temperature
            elif i % 4 == 1:  # Chemical fire pattern  
                X[i, 4:7] += np.random.uniform(3, 5, 3)  # High gas concentrations
                X[i, 13:16] += np.random.uniform(2, 4, 3)  # Specific gas signatures
            elif i % 4 == 2:  # Smoldering fire pattern
                X[i, 7:10] += np.random.uniform(1, 2, 3)  # Gradual temperature rise
                X[i, 16:18] += np.random.uniform(0.5, 1.5, 2)  # Smoke particles
        
        # Generate labels
        y_binary = np.zeros(n_samples)  # Binary: fire vs no fire
        y_multiclass = np.zeros(n_samples)  # Multi-class: fire types
        
        for i in range(n_samples):
            if i % 4 == 0:  # Electrical fire
                y_binary[i] = 1
                y_multiclass[i] = 0
            elif i % 4 == 1:  # Chemical fire
                y_binary[i] = 1  
                y_multiclass[i] = 1
            elif i % 4 == 2:  # Smoldering fire
                y_binary[i] = 1
                y_multiclass[i] = 2
            else:  # No fire
                y_binary[i] = 0
                y_multiclass[i] = 3
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        logger.info(f"✅ Generated data: {X_df.shape[0]:,} samples, {X_df.shape[1]} features")
        logger.info(f"   Fire cases: {int(y_binary.sum()):,} ({y_binary.mean()*100:.1f}%)")
        logger.info(f"   Class distribution: {np.bincount(y_multiclass.astype(int))}")
        
        return {
            'features': X_df,
            'binary_labels': y_binary,
            'multiclass_labels': y_multiclass
        }
    
    def train_classification_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train classification models using the existing framework."""
        
        logger.info("\n🔥 TRAINING CLASSIFICATION MODELS")
        logger.info("=" * 50)
        
        results = {}
        
        # Extract data
        X = training_data['features']
        y_binary = training_data['binary_labels']
        y_multi = training_data['multiclass_labels']
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train_binary = y_binary[:split_idx]
        y_val_binary = y_binary[split_idx:]
        y_train_multi = y_multi[:split_idx]
        y_val_multi = y_multi[split_idx:]
        
        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Validation set: {len(X_val):,} samples")
        
        # Try to import and train models
        models_to_train = [
            {
                'name': 'binary_classifier',
                'module': 'ml.models.classification',
                'class': 'BinaryFireClassifier',
                'config': {
                    'algorithm': 'random_forest',
                    'n_estimators': 50,  # Smaller for faster training
                    'max_depth': 10,
                    'class_weight': 'balanced'
                },
                'labels': (y_train_binary, y_val_binary)
            },
            {
                'name': 'multi_class_classifier', 
                'module': 'ml.models.classification',
                'class': 'MultiClassFireClassifier',
                'config': {
                    'algorithm': 'random_forest',
                    'n_estimators': 50,
                    'max_depth': 10,
                    'class_weight': 'balanced'
                },
                'labels': (y_train_multi, y_val_multi)
            },
            {
                'name': 'deep_learning_classifier',
                'module': 'ml.models.classification', 
                'class': 'DeepLearningFireClassifier',
                'config': {
                    'hidden_layers': [32, 16],  # Smaller for faster training
                    'activation': 'relu',
                    'dropout_rate': 0.2,
                    'epochs': 20,  # Fewer epochs
                    'batch_size': 32
                },
                'labels': (y_train_multi, y_val_multi)
            }
        ]
        
        for model_info in models_to_train:
            model_start_time = time.time()
            model_name = model_info['name']
            
            try:
                logger.info(f"\n📚 Training {model_name}...")
                
                # Dynamic import
                module = __import__(model_info['module'], fromlist=[model_info['class']])
                model_class = getattr(module, model_info['class'])
                
                # Create model instance
                model = model_class(model_info['config'])
                
                # Train model
                y_train, y_val = model_info['labels']
                training_metrics = model.train(
                    X_train, y_train,
                    validation_data=(X_val, y_val)
                )
                
                # Calculate training time
                training_time = time.time() - model_start_time
                
                logger.info(f"   ✅ {model_name} trained in {training_time:.1f}s")
                logger.info(f"   📊 Metrics: {training_metrics}")
                
                # Save results
                results[model_name] = {
                    'status': 'success',
                    'metrics': training_metrics,
                    'training_time': training_time,
                    'model_instance': model
                }
                
                self.training_results['successful_models'] += 1
                self.training_results['training_times'][model_name] = training_time
                self.training_results['model_metrics'][model_name] = training_metrics
                
            except Exception as e:
                training_time = time.time() - model_start_time
                error_msg = f"Failed to train {model_name}: {str(e)}"
                
                logger.error(f"   ❌ {error_msg}")
                
                results[model_name] = {
                    'status': 'failed',
                    'error': error_msg,
                    'training_time': training_time
                }
                
                self.training_results['failed_models'] += 1
                self.training_results['errors'].append({
                    'model': model_name,
                    'error': error_msg
                })
        
        self.training_results['total_models'] += len(models_to_train)
        
        return results
    
    def train_ensemble_with_aws(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train models using the working AWS ensemble trainer."""
        
        logger.info("\n🚀 TRAINING WITH AWS ENSEMBLE SYSTEM")
        logger.info("=" * 50)
        
        try:
            # Use the working simple ensemble trainer
            from simple_ensemble_trainer import SimpleEnsembleTrainer
            
            # Create trainer
            config = {
                'data_bucket': 'fire-detection-training-691595239825',
                'role_arn': 'arn:aws:iam::691595239825:role/SageMakerExecutionRole',
                'instance_type': 'ml.m5.large',
                'models_to_train': ['lstm_classifier', 'gru_classifier']  # Working models
            }
            
            trainer = SimpleEnsembleTrainer(config)
            
            # Train models (this uses existing working code)
            logger.info("Starting AWS ensemble training...")
            ensemble_results = trainer.train_ensemble()
            
            logger.info(f"✅ AWS ensemble training completed")
            logger.info(f"📊 Results: {ensemble_results}")
            
            return {
                'aws_ensemble': {
                    'status': 'success',
                    'results': ensemble_results
                }
            }
            
        except Exception as e:
            logger.error(f"❌ AWS ensemble training failed: {str(e)}")
            return {
                'aws_ensemble': {
                    'status': 'failed',
                    'error': str(e)
                }
            }
    
    def train_all_working_models(self) -> Dict[str, Any]:
        """Train all models that we know work."""
        
        logger.info("🔥 COMPREHENSIVE WORKING MODEL TRAINING")
        logger.info("=" * 70)
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {}
        
        # Generate training data
        training_data = self.generate_synthetic_data(n_samples=3000)  # Smaller for faster training
        
        # Train local classification models
        classification_results = self.train_classification_models(training_data)
        all_results['classification'] = classification_results
        
        # Train AWS ensemble models (if available)
        aws_results = self.train_ensemble_with_aws(training_data)
        all_results['aws_ensemble'] = aws_results
        
        return all_results
    
    def generate_final_report(self, training_results: Dict[str, Any]) -> None:
        """Generate comprehensive training report."""
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("\n" + "🎉" * 70)
        logger.info("WORKING MODEL TRAINING COMPLETED!")
        logger.info("🎉" * 70)
        
        logger.info(f"\n📊 TRAINING SUMMARY:")
        logger.info(f"   ⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   📈 Total models attempted: {self.training_results['total_models']}")
        logger.info(f"   ✅ Successful: {self.training_results['successful_models']}")
        logger.info(f"   ❌ Failed: {self.training_results['failed_models']}")
        
        if self.training_results['total_models'] > 0:
            success_rate = (self.training_results['successful_models'] / self.training_results['total_models']) * 100
            logger.info(f"   🎯 Success rate: {success_rate:.1f}%")
        
        # Model-by-model breakdown
        logger.info(f"\n📋 MODEL RESULTS:")
        for category, category_results in training_results.items():
            logger.info(f"   {category.upper()}:")
            if isinstance(category_results, dict):
                for model_name, result in category_results.items():
                    status = result.get('status', 'unknown')
                    status_icon = "✅" if status == 'success' else "❌"
                    logger.info(f"     {status_icon} {model_name}: {status}")
                    
                    if status == 'success' and 'metrics' in result:
                        metrics = result['metrics']
                        if isinstance(metrics, dict) and 'accuracy' in metrics:
                            logger.info(f"        Accuracy: {metrics['accuracy']:.3f}")
        
        # Error summary
        if self.training_results['errors']:
            logger.info(f"\n🚨 ERRORS:")
            for error in self.training_results['errors']:
                logger.info(f"   ❌ {error['model']}: {error['error']}")
        
        # Next steps
        logger.info(f"\n🚀 NEXT STEPS:")
        if self.training_results['successful_models'] > 0:
            logger.info(f"   ✅ {self.training_results['successful_models']} models trained successfully")
            logger.info(f"   🔧 Consider deploying successful models to production")
            logger.info(f"   📈 Extend training to more model types as needed")
        else:
            logger.info(f"   ⚠️ No models trained successfully")
            logger.info(f"   🔧 Review import paths and dependencies")
            logger.info(f"   🔄 Fix issues and retry training")


def main():
    """Main function."""
    
    trainer = WorkingModelTrainer()
    
    try:
        # Train all working models
        training_results = trainer.train_all_working_models()
        
        # Generate report
        trainer.generate_final_report(training_results)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())