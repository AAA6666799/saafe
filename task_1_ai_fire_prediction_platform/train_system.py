"""
Training script for the Synthetic Fire Prediction System
"""

import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from synthetic_fire_system.core.config import ConfigurationManager
from synthetic_fire_system.hardware.mock_interface import MockHardwareInterface
from synthetic_fire_system.feature_engineering.fusion import FeatureFusionEngine
from synthetic_fire_system.models.ensemble import EnsembleModel


def generate_training_data(num_samples=1000):
    """Generate synthetic training data"""
    print(f"Generating {num_samples} training samples...")
    
    # Initialize components
    config_manager = ConfigurationManager()
    hardware = MockHardwareInterface()
    fusion_engine = FeatureFusionEngine(config_manager.synthetic_data_config.__dict__)
    
    # Collect sensor data and extract features
    features_list = []
    labels_list = []
    
    for i in range(num_samples):
        # Get sensor data
        sensor_data = hardware.get_sensor_data()
        
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
            
            # Generate label (simplified - in reality, this would come from ground truth)
            # Higher probability of fire if there are hotspots or elevated gas levels
            has_hotspot = np.max(sensor_data.thermal_frame) > 40.0 if sensor_data.thermal_frame is not None else False
            high_gas = any(conc > 100.0 for conc in sensor_data.gas_readings.values()) if sensor_data.gas_readings else False
            label = 1 if (has_hotspot or high_gas) and np.random.random() > 0.3 else 0
            labels_list.append(label)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"   Generated {i + 1}/{num_samples} samples")
    
    if not features_list:
        raise ValueError("No features were extracted")
    
    # Convert to numpy arrays
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"Generated {len(features)} samples with {features.shape[1]} features each")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return features, labels


def train_and_evaluate_model(features, labels):
    """Train and evaluate the ensemble model"""
    print("Training and evaluating model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize model
    config_manager = ConfigurationManager()
    model = EnsembleModel(config_manager.model_config.__dict__)
    
    # Train model
    print("Training model...")
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
    
    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """Main training function"""
    print("Synthetic Fire Prediction System - Training Script")
    print("=" * 60)
    
    try:
        # Generate training data
        features, labels = generate_training_data(num_samples=1000)
        
        # Train and evaluate model
        model, metrics = train_and_evaluate_model(features, labels)
        
        # Save model
        model_path = "trained_fire_model.pkl"
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        print("Model saved successfully!")
        
        print("\nTraining completed successfully!")
        print("Model performance:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())