#!/usr/bin/env python3
"""
Execute the key components of the FLIR+SCD41 Unified Training Notebook
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
    print(f"FLIR features shape: {flir_features.shape}")
    print(f"SCD41 features shape: {scd41_features.shape}")
    
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

def main():
    """Main execution function"""
    print("🔥 FLIR+SCD41 Fire Detection System - Unified Training Execution")
    print("=" * 60)
    
    try:
        # Generate synthetic data
        flir_features, scd41_features = generate_synthetic_data(num_samples=10000)
        
        # Create dataset
        df, feature_names = create_dataset(flir_features, scd41_features)
        
        # Save dataset to disk
        data_dir = os.path.join(project_root, 'data', 'flir_scd41')
        os.makedirs(data_dir, exist_ok=True)
        
        dataset_path = os.path.join(data_dir, 'flir_scd41_dataset.csv')
        df.to_csv(dataset_path, index=False)
        
        print(f"✅ Dataset saved to {dataset_path}")
        
        # Save feature names
        feature_info = {
            'feature_names': feature_names,
            'total_features': len(feature_names)
        }
        
        feature_path = os.path.join(data_dir, 'feature_info.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"✅ Feature information saved to {feature_path}")
        print("\n🎉 Basic execution completed successfully!")
        print(f"📁 Output files are in: {data_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)