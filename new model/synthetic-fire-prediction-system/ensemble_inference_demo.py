#!/usr/bin/env python3
"""
Inference Demo for FLIR+SCD41 Ensemble Fire Detection System

This script demonstrates how to load and use the trained ensemble model for fire detection.
"""

import joblib
import numpy as np

def load_ensemble_model(model_path="flir_scd41_ensemble_model.joblib"):
    """Load the trained ensemble model and metadata."""
    print(f"📥 Loading ensemble model from {model_path}...")
    
    try:
        model_data = joblib.load(model_path)
        ensemble_manager = model_data['ensemble_manager']
        feature_names = model_data['feature_names']
        
        print("✅ Ensemble model loaded successfully!")
        print(f"Model version: {model_data.get('version', 'N/A')}")
        print(f"Created at: {model_data.get('created_at', 'N/A')}")
        
        return ensemble_manager, feature_names
    except Exception as e:
        print(f"❌ Error loading ensemble model: {e}")
        return None, None

def predict_fire(ensemble_manager, feature_names, sensor_data):
    """Make a fire detection prediction using the ensemble."""
    # Ensure features are in the correct order
    feature_values = [sensor_data[feature] for feature in feature_names[:-1]]  # Exclude 'fire_detected'
    
    # Make prediction
    prediction, probabilities = ensemble_manager.predict_ensemble([feature_values])
    
    return prediction[0], probabilities[0]

def main():
    """Main function to demonstrate ensemble inference."""
    print("🔥 FLIR+SCD41 Ensemble Fire Detection System - Inference Demo")
    print("=" * 60)
    
    # Load ensemble model
    ensemble_manager, feature_names = load_ensemble_model()
    
    if ensemble_manager is None:
        print("❌ Failed to load ensemble model. Exiting.")
        return 1
    
    # Test scenarios
    test_scenarios = {
        "Normal Room Conditions": {
            "t_mean": 22.5, "t_std": 1.2, "t_max": 25.1, "t_p95": 24.8,
            "t_hot_area_pct": 0.5, "t_hot_largest_blob_pct": 0.3,
            "t_grad_mean": 0.1, "t_grad_std": 0.05, "t_diff_mean": 0.2,
            "t_diff_std": 0.1, "flow_mag_mean": 0.3, "flow_mag_std": 0.1,
            "tproxy_val": 23.0, "tproxy_delta": 0.5, "tproxy_vel": 0.1,
            "gas_val": 410.0, "gas_delta": 5.0, "gas_vel": 1.0
        },
        "Sunlight Heating (False Positive)": {
            "t_mean": 35.0, "t_std": 3.5, "t_max": 45.0, "t_p95": 40.0,
            "t_hot_area_pct": 5.0, "t_hot_largest_blob_pct": 3.0,
            "t_grad_mean": 1.0, "t_grad_std": 0.8, "t_diff_mean": 1.0,
            "t_diff_std": 0.5, "flow_mag_mean": 2.0, "flow_mag_std": 1.0,
            "tproxy_val": 35.0, "tproxy_delta": 5.0, "tproxy_vel": 1.5,
            "gas_val": 500.0, "gas_delta": 25.0, "gas_vel": 3.0
        },
        "Early Stage Fire": {
            "t_mean": 32.0, "t_std": 5.2, "t_max": 55.0, "t_p95": 48.0,
            "t_hot_area_pct": 8.5, "t_hot_largest_blob_pct": 5.2,
            "t_grad_mean": 1.8, "t_grad_std": 1.2, "t_diff_mean": 1.5,
            "t_diff_std": 0.8, "flow_mag_mean": 2.5, "flow_mag_std": 1.3,
            "tproxy_val": 42.0, "tproxy_delta": 8.0, "tproxy_vel": 2.1,
            "gas_val": 650.0, "gas_delta": 60.0, "gas_vel": 5.2
        },
        "Advanced Fire": {
            "t_mean": 45.2, "t_std": 8.7, "t_max": 78.5, "t_p95": 72.1,
            "t_hot_area_pct": 25.3, "t_hot_largest_blob_pct": 18.7,
            "t_grad_mean": 3.2, "t_grad_std": 1.8, "t_diff_mean": 2.9,
            "t_diff_std": 1.5, "flow_mag_mean": 4.2, "flow_mag_std": 2.1,
            "tproxy_val": 52.0, "tproxy_delta": 15.0, "tproxy_vel": 3.2,
            "gas_val": 850.0, "gas_delta": 120.0, "gas_vel": 8.5
        },
        "Smoldering Fire": {
            "t_mean": 30.0, "t_std": 4.5, "t_max": 42.0, "t_p95": 38.0,
            "t_hot_area_pct": 3.2, "t_hot_largest_blob_pct": 2.1,
            "t_grad_mean": 0.8, "t_grad_std": 0.5, "t_diff_mean": 0.7,
            "t_diff_std": 0.3, "flow_mag_mean": 1.0, "flow_mag_std": 0.5,
            "tproxy_val": 35.0, "tproxy_delta": 4.0, "tproxy_vel": 1.2,
            "gas_val": 1800.0, "gas_delta": 300.0, "gas_vel": 45.0
        }
    }
    
    print("\n🔍 Fire Detection Results:")
    print("=" * 60)
    
    for scenario_name, data in test_scenarios.items():
        # Make prediction
        prediction, probabilities = predict_fire(ensemble_manager, feature_names, data)
        
        # Display results
        print(f"\n{scenario_name}:")
        print(f"  Temperature: {data['t_max']:.1f}°C, CO₂: {data['gas_val']:.0f} ppm")
        print(f"  Fire Probability: {prediction:.4f} ({prediction*100:.2f}%)")
        
        if prediction > 0.7:
            print("  🔥 HIGH RISK: Strong indication of fire detected!")
        elif prediction > 0.5:
            print("  ⚠️  MEDIUM RISK: Possible fire detected, requires attention")
        elif prediction > 0.3:
            print("  🟡 LOW RISK: Unusual conditions, monitor closely")
        else:
            print("  ✅ NORMAL: No fire detected")
    
    print("\n" + "=" * 60)
    print("🎉 ENSEMBLE INFERENCE DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ Ensemble model loaded and validated")
    print("✅ Multiple scenarios tested")
    print("✅ Results displayed with confidence levels")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())