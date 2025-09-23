#!/usr/bin/env python3
"""
Test script for improved FLIR+SCD41 training components
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_early_stopping():
    """Test early stopping functionality"""
    print("Testing Early Stopping...")
    
    # Import EarlyStopping from our improved training script
    try:
        from scripts.improved_flir_scd41_training import EarlyStopping
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Simulate validation loss decreasing then increasing
        val_losses = [1.0, 0.9, 0.8, 0.85, 0.9, 0.95]
        
        for epoch, loss in enumerate(val_losses):
            early_stopping(loss)
            print(f"Epoch {epoch+1}: Loss = {loss:.3f}, Early stop = {early_stopping.early_stop}")
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print("✅ Early Stopping test passed")
        return True
    except Exception as e:
        print(f"❌ Early Stopping test failed: {e}")
        return False

def test_regularized_nn():
    """Test regularized neural network"""
    print("\nTesting Regularized Neural Network...")
    
    try:
        from scripts.improved_flir_scd41_training import ImprovedNN
        
        # Create model
        model = ImprovedNN(input_size=10, hidden_sizes=[32, 16], dropout_rate=0.3)
        
        # Test forward pass
        x = torch.randn(5, 10)
        output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model layers: {len(list(model.network))}")
        
        print("✅ Regularized Neural Network test passed")
        return True
    except Exception as e:
        print(f"❌ Regularized Neural Network test failed: {e}")
        return False

def test_data_augmentation():
    """Test data augmentation functionality"""
    print("\nTesting Data Augmentation...")
    
    try:
        from scripts.improved_flir_scd41_training import augment_data
        
        # Create sample data
        X = np.random.randn(100, 18)
        y = np.random.randint(0, 2, 100)
        
        # Augment data
        X_aug, y_aug = augment_data(X, y, factor=0.2)
        
        print(f"Original samples: {X.shape[0]}")
        print(f"Augmented samples: {X_aug.shape[0]}")
        print(f"Label samples: {y_aug.shape[0]}")
        
        # Check that augmentation worked correctly
        assert X_aug.shape[0] == 120, f"Expected 120 samples, got {X_aug.shape[0]}"
        assert y_aug.shape[0] == 120, f"Expected 120 labels, got {y_aug.shape[0]}"
        
        print("✅ Data Augmentation test passed")
        return True
    except Exception as e:
        print(f"❌ Data Augmentation test failed: {e}")
        return False

def test_model_diagnostics():
    """Test model diagnostics functionality"""
    print("\nTesting Model Diagnostics...")
    
    try:
        from scripts.model_diagnostics import diagnose_model_learning, detect_overfitting
        
        # Create sample data
        X_train = np.random.randn(100, 18)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 18)
        y_val = np.random.randint(0, 2, 20)
        
        # Test data quality check
        data_info = diagnose_model_learning(X_train, y_train, X_val, y_val)
        
        # Test overfitting detection
        train_metrics = {'accuracy': 0.95}
        val_metrics = {'accuracy': 0.85}
        overfitting = detect_overfitting(train_metrics, val_metrics)
        
        print("✅ Model Diagnostics test passed")
        return True
    except Exception as e:
        print(f"❌ Model Diagnostics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Improved FLIR+SCD41 Training Components")
    print("="*50)
    
    tests = [
        test_early_stopping,
        test_regularized_nn,
        test_data_augmentation,
        test_model_diagnostics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved training components are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()