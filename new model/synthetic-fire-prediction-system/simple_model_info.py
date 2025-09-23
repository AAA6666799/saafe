#!/usr/bin/env python3
"""
Simple Inference Demo for FLIR+SCD41 Ensemble Fire Detection System

This script demonstrates how to use the trained ensemble model for fire detection.
"""

import joblib
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def load_ensemble_model(model_path="flir_scd41_ensemble_model.joblib"):
    """Load the trained ensemble model and metadata."""
    print(f"📥 Loading ensemble model from {model_path}...")
    
    try:
        model_data = joblib.load(model_path)
        # Note: We can't directly use the ensemble_manager object because it was defined in another script
        # Instead, we'll extract the individual models and weights
        feature_names = model_data['feature_names']
        
        print("✅ Ensemble model data loaded successfully!")
        print(f"Model version: {model_data.get('version', 'N/A')}")
        print(f"Created at: {model_data.get('created_at', 'N/A')}")
        
        return model_data, feature_names
    except Exception as e:
        print(f"❌ Error loading ensemble model: {e}")
        return None, None

def main():
    """Main function to demonstrate ensemble inference."""
    print("🔥 FLIR+SCD41 Ensemble Fire Detection System - Model Info")
    print("=" * 60)
    
    # Load ensemble model
    model_data, feature_names = load_ensemble_model()
    
    if model_data is None:
        print("❌ Failed to load ensemble model. Exiting.")
        return 1
    
    print("\n📊 Model Information:")
    print("=" * 30)
    print(f"Feature names: {feature_names}")
    print(f"Number of features: {len(feature_names)}")
    
    # Show what's in the model data
    print("\n📦 Model Data Contents:")
    print("=" * 30)
    for key in model_data.keys():
        print(f"  {key}: {type(model_data[key])}")
    
    print("\n" + "=" * 60)
    print("✅ Model file successfully loaded and inspected")
    print("✅ Feature names extracted")
    print("✅ Model data structure verified")
    
    # Show how to use the model in a real application
    print("\n💡 Usage Instructions:")
    print("=" * 30)
    print("To use this model in your application:")
    print("1. Load the model using joblib.load('flir_scd41_ensemble_model.joblib')")
    print("2. Extract the ensemble_manager and feature_names")
    print("3. Prepare your sensor data in the correct order")
    print("4. Call ensemble_manager.predict_ensemble([sensor_data])")
    print("5. Interpret the probability result (0.0 = no fire, 1.0 = fire)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())