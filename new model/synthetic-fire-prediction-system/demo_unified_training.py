#!/usr/bin/env python3
"""
Demo script for the FLIR+SCD41 Unified Training Pipeline
This script demonstrates the key components without requiring all heavy dependencies.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def generate_synthetic_data(num_samples=10000):
    """Generate synthetic FLIR+SCD41 dataset with controlled noise"""
    print("🔄 Generating synthetic FLIR+SCD41 dataset...")
    
    # Generate FLIR features (15 features)
    np.random.seed(42)
    flir_features = np.random.normal(25, 10, (num_samples, 15))
    flir_features[:, 0] = np.clip(flir_features[:, 0], -40, 330)  # t_mean: -40 to 330°C
    flir_features[:, 2] = np.clip(flir_features[:, 2], -40, 330)  # t_max: -40 to 330°C
    flir_features[:, 4] = np.clip(flir_features[:, 4], 0, 100)    # t_hot_area_pct: 0-100%
    
    # Generate SCD41 features (3 features)
    scd41_features = np.random.normal(450, 100, (num_samples, 3))
    scd41_features[:, 0] = np.clip(scd41_features[:, 0], 400, 40000)  # gas_val: 400-40000 ppm
    
    print(f"✅ Generated {num_samples} samples")
    return flir_features, scd41_features

def create_dataset(flir_features, scd41_features):
    """Combine features and create labels with balanced distribution"""
    print("💾 Combining features and creating dataset...")
    
    # Combine all features (15 FLIR + 3 SCD41 = 18 features)
    all_features = np.concatenate([flir_features, scd41_features], axis=1)
    
    # Create labels (fire detected or not) with more realistic patterns
    # Fire probability based on multiple factors
    fire_probability = (
        (flir_features[:, 2] > 60).astype(int) * 0.3 +  # High max temperature
        (scd41_features[:, 0] > 1000).astype(int) * 0.3 +  # High CO2
        (flir_features[:, 4] > 10).astype(int) * 0.2 +  # Large hot area
        (flir_features[:, 12] > 50).astype(int) * 0.2  # High temperature proxy
    )
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, len(fire_probability))
    fire_probability = np.clip(fire_probability + noise, 0, 1)
    
    labels = np.random.binomial(1, fire_probability)
    
    # Create DataFrame
    feature_names = [
        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
        'gas_val', 'gas_delta', 'gas_vel'
    ]
    
    df = pd.DataFrame(all_features, columns=feature_names)
    df['fire_detected'] = labels
    
    print(f"✅ Dataset created with shape: {df.shape}")
    print(f"Fire samples: {sum(labels)} ({sum(labels)/len(labels)*100:.2f}%)")
    
    return df, feature_names

def split_dataset(df, feature_names, test_size=0.15, val_size=0.15):
    """Split dataset into train/validation/test sets with stratification"""
    print("📊 Splitting dataset into train/validation/test sets...")
    
    # Separate features and labels
    X = df.drop('fire_detected', axis=1).values
    y = df['fire_detected'].values
    
    # For demo purposes, we'll just show the split logic without actually splitting
    print(f"Dataset size: {X.shape[0]} samples")
    print(f"Features: {X.shape[1]}")
    print(f"Test size: {int(X.shape[0] * test_size)} samples")
    print(f"Validation size: {int(X.shape[0] * val_size)} samples")
    print(f"Training size: {X.shape[0] - int(X.shape[0] * test_size) - int(X.shape[0] * val_size)} samples")
    
    return X, y

def demonstrate_model_training():
    """Demonstrate model training concepts without actually training"""
    print("\n🚀 Model Training Demonstration")
    print("=" * 40)
    
    print("1. XGBoost Model with Regularization:")
    print("   - L1 regularization (reg_alpha=0.1)")
    print("   - L2 regularization (reg_lambda=1.0)")
    print("   - Early stopping (patience=10)")
    print("   - Reduced depth (max_depth=4)")
    print("   - Subsampling (subsample=0.8)")
    
    print("\n2. Neural Network with Regularization:")
    print("   - Dropout layers (rate=0.3)")
    print("   - Batch normalization")
    print("   - L2 regularization (weight_decay=1e-4)")
    print("   - Early stopping")
    print("   - Learning rate scheduling")
    
    print("\n3. Ensemble Weight Calculation:")
    print("   - Performance-based weighting")
    print("   - Exponential scaling of validation scores")
    print("   - Normalized weights sum to 1.0")

def demonstrate_diagnostics():
    """Demonstrate model diagnostics"""
    print("\n🔍 Model Diagnostics Demonstration")
    print("=" * 40)
    
    print("1. Underfitting Detection:")
    print("   - Poor performance on both training and validation sets")
    print("   - Solutions: Increase model complexity, add features, train longer")
    
    print("\n2. Overfitting Detection:")
    print("   - High performance on training set, poor on validation set")
    print("   - Large gap between training and validation accuracy")
    print("   - Solutions: Add regularization, reduce complexity, get more data")
    
    print("\n3. Learning Curves:")
    print("   - Visualize model performance vs. training set size")
    print("   - Diagnose bias and variance problems")
    
    print("\n4. Validation Curves:")
    print("   - Optimize hyperparameters")
    print("   - Identify optimal model complexity")

def save_demo_results(df, feature_names):
    """Save demo results"""
    print("\n💾 Saving Demo Results...")
    
    # Create data directory
    data_dir = os.path.join(project_root, 'data', 'flir_scd41')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save dataset
    dataset_path = os.path.join(data_dir, 'flir_scd41_dataset_demo.csv')
    df.to_csv(dataset_path, index=False)
    
    # Save feature names
    feature_info = {
        'feature_names': feature_names,
        'total_features': len(feature_names),
        'description': 'Demo dataset for FLIR+SCD41 fire detection system'
    }
    
    feature_path = os.path.join(data_dir, 'feature_info_demo.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save demo model info
    model_info = {
        'models': ['xgboost', 'neural_network'],
        'ensemble_method': 'performance_based_exponential_scaling',
        'demo_only': True,
        'note': 'This is a demo file. Actual model training requires full dependencies.'
    }
    
    model_info_path = os.path.join(data_dir, 'model_info_demo.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Demo dataset saved to {dataset_path}")
    print(f"✅ Feature information saved to {feature_path}")
    print(f"✅ Model information saved to {model_info_path}")

def main():
    """Main execution function"""
    print("🔥 FLIR+SCD41 Fire Detection System - Unified Training Demo")
    print("=" * 60)
    
    try:
        # Generate synthetic data
        flir_features, scd41_features = generate_synthetic_data(num_samples=10000)
        
        # Create dataset
        df, feature_names = create_dataset(flir_features, scd41_features)
        
        # Split dataset (demo only)
        X, y = split_dataset(df, feature_names)
        
        # Demonstrate model training
        demonstrate_model_training()
        
        # Demonstrate diagnostics
        demonstrate_diagnostics()
        
        # Save demo results
        save_demo_results(df, feature_names)
        
        print("\n🎉 Demo completed successfully!")
        print("📁 Check the data/flir_scd41/ directory for output files")
        print("\n📝 Note: This is a demonstration only.")
        print("   To run the full training pipeline, install all dependencies and run:")
        print("   python run_complete_training.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)