#!/usr/bin/env python3
"""
Local Training Demo for FLIR+SCD41 Fire Detection System

This script demonstrates the training process locally without AWS dependencies.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def generate_sample_data(n_samples=1000):
    """Generate sample FLIR+SCD41 data for demonstration."""
    print("🔄 Generating sample FLIR+SCD41 data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate FLIR features (15 features)
    flir_features = {
        't_mean': np.random.normal(25, 10, n_samples),
        't_std': np.random.normal(2.0, 1.0, n_samples),
        't_max': np.random.normal(30, 15, n_samples),
        't_p95': np.random.normal(28, 12, n_samples),
        't_hot_area_pct': np.random.uniform(0, 50, n_samples),
        't_hot_largest_blob_pct': np.random.uniform(0, 30, n_samples),
        't_grad_mean': np.random.normal(1.0, 0.5, n_samples),
        't_grad_std': np.random.normal(0.5, 0.2, n_samples),
        't_diff_mean': np.random.normal(0.5, 0.3, n_samples),
        't_diff_std': np.random.normal(0.2, 0.1, n_samples),
        'flow_mag_mean': np.random.normal(1.0, 0.5, n_samples),
        'flow_mag_std': np.random.normal(0.3, 0.2, n_samples),
        'tproxy_val': np.random.normal(25, 8, n_samples),
        'tproxy_delta': np.random.normal(2.0, 1.5, n_samples),
        'tproxy_vel': np.random.normal(0.5, 0.3, n_samples)
    }
    
    # Generate SCD41 features (3 features)
    scd41_features = {
        'gas_val': np.random.normal(500, 200, n_samples),
        'gas_delta': np.random.normal(20, 50, n_samples),
        'gas_vel': np.random.normal(5, 10, n_samples)
    }
    
    # Combine all features
    all_features = {**flir_features, **scd41_features}
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Generate labels based on feature values (simplified fire detection logic)
    # Higher temperature and CO2 levels indicate higher probability of fire
    fire_score = (
        np.clip((df['t_max'] - 20) / 100, 0, 1) * 0.5 +
        np.clip((df['gas_val'] - 400) / 2000, 0, 1) * 0.3 +
        np.clip(df['t_hot_area_pct'] / 50, 0, 1) * 0.2
    )
    
    # Add some noise to make it more realistic
    fire_score = np.clip(fire_score + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    # Convert to binary labels
    df['fire_detected'] = (fire_score > 0.5).astype(int)
    
    print(f"✅ Generated {n_samples} samples")
    print(f"Fire samples: {df['fire_detected'].sum()} ({df['fire_detected'].mean()*100:.1f}%)")
    
    return df

def train_models(df):
    """Train fire detection models."""
    print("\n🚀 Training fire detection models...")
    
    # Separate features and labels
    X = df.drop('fire_detected', axis=1)
    y = df['fire_detected']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    return rf_model, X_train.columns.tolist(), metrics

def save_model(model, feature_names, metrics, model_path="flir_scd41_model.joblib"):
    """Save the trained model and metadata."""
    print(f"\n💾 Saving model to {model_path}...")
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'metrics': metrics,
        'created_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, model_path)
    print("✅ Model saved successfully!")
    
    # Also save feature names to a JSON file for easy reference
    feature_info = {
        'feature_names': feature_names,
        'model_metrics': metrics,
        'created_at': datetime.now().isoformat()
    }
    
    with open("feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("✅ Feature information saved to feature_info.json")

def demonstrate_inference(model, feature_names):
    """Demonstrate how to use the trained model for inference."""
    print("\n🔍 Demonstrating model inference...")
    
    # Create sample test data
    test_data = {
        'Normal Conditions': {
            't_mean': 22.5, 't_std': 1.2, 't_max': 25.1, 't_p95': 24.8,
            't_hot_area_pct': 0.5, 't_hot_largest_blob_pct': 0.3,
            't_grad_mean': 0.1, 't_grad_std': 0.05, 't_diff_mean': 0.2,
            't_diff_std': 0.1, 'flow_mag_mean': 0.3, 'flow_mag_std': 0.1,
            'tproxy_val': 23.0, 'tproxy_delta': 0.5, 'tproxy_vel': 0.1,
            'gas_val': 410.0, 'gas_delta': 5.0, 'gas_vel': 1.0
        },
        'Potential Fire': {
            't_mean': 45.2, 't_std': 8.7, 't_max': 78.5, 't_p95': 72.1,
            't_hot_area_pct': 25.3, 't_hot_largest_blob_pct': 18.7,
            't_grad_mean': 3.2, 't_grad_std': 1.8, 't_diff_mean': 2.9,
            't_diff_std': 1.5, 'flow_mag_mean': 4.2, 'flow_mag_std': 2.1,
            'tproxy_val': 52.0, 'tproxy_delta': 15.0, 'tproxy_vel': 3.2,
            'gas_val': 850.0, 'gas_delta': 120.0, 'gas_vel': 8.5
        }
    }
    
    for scenario_name, data in test_data.items():
        # Ensure features are in the correct order
        feature_values = [data[feature] for feature in feature_names]
        
        # Make prediction
        prediction_proba = model.predict_proba([feature_values])[0]
        prediction = model.predict([feature_values])[0]
        
        print(f"\n{scenario_name}:")
        print(f"  Fire Probability: {prediction_proba[1]:.4f} ({prediction_proba[1]*100:.2f}%)")
        print(f"  Prediction: {'🔥 FIRE DETECTED' if prediction == 1 else '✅ NO FIRE'}")

def main():
    """Main function to run the local training demo."""
    print("🔥 FLIR+SCD41 Fire Detection System - Local Training Demo")
    print("=" * 60)
    
    try:
        # Generate sample data
        df = generate_sample_data(2000)
        
        # Train models
        model, feature_names, metrics = train_models(df)
        
        # Save model
        save_model(model, feature_names, metrics)
        
        # Demonstrate inference
        demonstrate_inference(model, feature_names)
        
        print("\n" + "=" * 60)
        print("🎉 LOCAL TRAINING DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Sample data generated")
        print("✅ Model trained and evaluated")
        print("✅ Model saved to flir_scd41_model.joblib")
        print("✅ Inference demonstration completed")
        print("\n📁 Output files:")
        print("  - flir_scd41_model.joblib (trained model)")
        print("  - feature_info.json (model metadata)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during training demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    from datetime import datetime
    sys.exit(main())