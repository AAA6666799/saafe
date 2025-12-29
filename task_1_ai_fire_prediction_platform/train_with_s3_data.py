"""
Training script for the Synthetic Fire Prediction System using real S3 data
"""

import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import boto3
import csv
from io import StringIO

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from ai_fire_prediction_platform.core.config import ConfigurationManager
from ai_fire_prediction_platform.hardware.abstraction import S3HardwareInterface
from ai_fire_prediction_platform.feature_engineering.fusion import FeatureFusionEngine
from ai_fire_prediction_platform.models.ensemble import EnsembleModel


def fetch_s3_data(num_samples=100):
    """Fetch real data from S3 for training"""
    print(f"Fetching {num_samples} samples from S3...")
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name='us-east-1')
    bucket_name = 'data-collector-of-first-device'
    
    # Get recent files
    thermal_files = []
    gas_files = []
    
    try:
        # Get thermal data files
        thermal_response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='thermal-data/',
            MaxKeys=num_samples//2
        )
        
        if 'Contents' in thermal_response:
            thermal_files = sorted(thermal_response['Contents'], 
                                 key=lambda x: x['LastModified'], reverse=True)[:num_samples//2]
        
        # Get gas data files
        gas_response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='gas-data/',
            MaxKeys=num_samples//2
        )
        
        if 'Contents' in gas_response:
            gas_files = sorted(gas_response['Contents'], 
                             key=lambda x: x['LastModified'], reverse=True)[:num_samples//2]
        
        print(f"Found {len(thermal_files)} thermal files and {len(gas_files)} gas files")
        return thermal_files[:num_samples//2], gas_files[:num_samples//2]
        
    except Exception as e:
        print(f"Error fetching S3 data: {e}")
        return [], []


def create_training_labels(features_list):
    """Create training labels based on feature values (simplified approach)"""
    labels_list = []
    
    for features in features_list:
        # Simple heuristic: if there are hotspots or elevated gas levels, it's a fire
        # This is just for demonstration - in reality, you'd have actual labels
        thermal_features = features[:8]  # First 8 are thermal features
        gas_features = features[8:14]    # Next 6 are gas features (3 concentrations + 3 anomalies)
        
        # Hotspot detection (more than 10 hotspots or high intensity)
        hotspot_count = thermal_features[6] if len(thermal_features) > 6 else 0
        hotspot_intensity = thermal_features[7] if len(thermal_features) > 7 else 0
        
        # Gas anomaly detection (anomaly score > 1.5)
        gas_anomalies = gas_features[3:6] if len(gas_features) >= 6 else [0, 0, 0]
        # Handle case where gas_anomalies might be numpy arrays
        if hasattr(gas_anomalies, '__len__') and len(gas_anomalies) > 0:
            if isinstance(gas_anomalies[0], np.ndarray):
                max_gas_anomaly = float(np.max(gas_anomalies))
            else:
                max_gas_anomaly = float(max(gas_anomalies))
        else:
            max_gas_anomaly = 0.0
        
        # Label as fire (1) if either condition is met
        if hotspot_count > 5 or hotspot_intensity > 30 or max_gas_anomaly > 1.5:
            labels_list.append(1)  # Fire
        else:
            labels_list.append(0)  # No fire
    
    return np.array(labels_list)


def train_with_s3_data(num_samples=200):
    """Train the model using real S3 data"""
    print("Training model with real S3 data...")
    print("=" * 50)
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        s3_interface = S3HardwareInterface({
            's3_bucket': 'data-collector-of-first-device',
            'thermal_prefix': 'thermal-data/',
            'gas_prefix': 'gas-data/'
        })
        fusion_engine = FeatureFusionEngine(config_manager.synthetic_data_config.__dict__)
        
        if not s3_interface.is_connected():
            print("❌ Failed to connect to S3")
            return False
        
        # Collect sensor data and extract features
        print("Collecting sensor data and extracting features...")
        features_list = []
        samples_collected = 0
        
        # Collect data samples
        while samples_collected < num_samples:
            sensor_data = s3_interface.get_sensor_data()
            if sensor_data:
                # Extract features
                feature_vector = fusion_engine.extract_features(sensor_data)
                
                # Combine all features
                all_features = []
                if feature_vector.thermal_features is not None:
                    all_features.append(feature_vector.thermal_features)
                if feature_vector.gas_features is not None:
                    all_features.append(feature_vector.gas_features)
                if feature_vector.environmental_features is not None:
                    all_features.append(feature_vector.environmental_features)
                if feature_vector.fusion_features is not None:
                    all_features.append(feature_vector.fusion_features)
                
                if all_features:
                    combined_features = np.concatenate(all_features)
                    features_list.append(combined_features)
                    samples_collected += 1
                    
                    if samples_collected % 20 == 0:
                        print(f"   Collected {samples_collected}/{num_samples} samples")
            
            # Small delay to avoid overwhelming the S3 API
            import time
            time.sleep(0.5)
        
        if not features_list:
            print("❌ No features were extracted")
            return False
        
        # Convert to numpy arrays
        features = np.array(features_list)
        print(f"✅ Collected {len(features)} samples with {features.shape[1]} features each")
        
        # Create labels (this is a simplified approach - in reality you'd have actual labels)
        print("Creating training labels...")
        labels = create_training_labels(features_list)
        print(f"Label distribution: {np.bincount(labels)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Initialize and train model
        print("Training ensemble model...")
        model = EnsembleModel(config_manager.model_config.__dict__)
        model.train(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred_proba = []
        for sample in X_test:
            prob, _ = model.predict(sample)
            y_pred_proba.append(prob)
        
        # Convert probabilities to binary predictions
        y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_proba]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save model
        model_path = "trained_fire_model_s3.pkl"
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        print("✅ Model saved successfully!")
        
        print("\n🎉 Training with S3 data completed successfully!")
        print("Model performance:")
        for metric, value in zip(['Accuracy', 'Precision', 'Recall', 'F1 Score'], 
                                [accuracy, precision, recall, f1]):
            print(f"  {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during S3 training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training function"""
    print("Synthetic Fire Prediction System - S3 Data Training")
    print("=" * 55)
    
    if train_with_s3_data(num_samples=100):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())