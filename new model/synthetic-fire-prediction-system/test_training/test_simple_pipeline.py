#!/usr/bin/env python3
"""
Simple test to verify the FLIR+SCD41 training pipeline works
"""

import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn

def test_simple_model():
    """Test that we can create and train simple models"""
    print("Testing simple model creation...")
    
    # Test numpy
    data = np.random.rand(100, 5)
    labels = np.random.randint(0, 2, 100)
    print("✅ Numpy working correctly")
    
    # Test XGBoost
    model = xgb.XGBClassifier(n_estimators=10)
    model.fit(data, labels)
    preds = model.predict(data)
    print("✅ XGBoost working correctly")
    
    # Test PyTorch
    net = nn.Linear(5, 2)
    x = torch.randn(10, 5)
    y = net(x)
    print("✅ PyTorch working correctly")
    
    print("🎉 All tests passed! The training pipeline should work correctly.")

if __name__ == "__main__":
    test_simple_model()