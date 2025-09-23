#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - SageMaker Inference Script for Ensemble
This script handles model loading and inference for deployed ensemble models.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

def model_fn(model_dir):
    """Load model and ensemble weights."""
    print("Loading model...")
    
    # Load models
    models = {}
    model_files = {
        'random_forest': 'random_forest_model.joblib',
        'gradient_boosting': 'gradient_boosting_model.joblib',
        'logistic_regression': 'logistic_regression_model.joblib'
    }
    
    for name, filename in model_files.items():
        try:
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                print(f"Loaded {name} model")
        except Exception as e:
            print(f"Error loading {name} model: {e}")
    
    # Load ensemble weights
    weights = {}
    try:
        weights_path = os.path.join(model_dir, "ensemble_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                weights_data = json.load(f)
            weights = dict(zip(weights_data['models'], weights_data['weights']))
            print("Loaded ensemble weights")
    except Exception as e:
        print(f"Error loading ensemble weights: {e}")
        # Default to equal weights if loading fails
        weights = {name: 1.0/len(models) for name in models.keys()}
    
    return {
        'models': models,
        'weights': weights
    }

def input_fn(request_body, request_content_type):
    """
    Parse input data for inference.
    """
    print(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Extract features from the input
        if "features" in input_data:
            features = input_data["features"]
            # Convert to DataFrame
            df = pd.DataFrame([features])
        else:
            # Assume it's a direct feature array
            df = pd.DataFrame([input_data])
        return df
    elif request_content_type == 'text/csv':
        # Handle CSV input
        lines = request_body.strip().split('\n')
        data = []
        for line in lines:
            values = [float(x) for x in line.split(',')]
            data.append(values)
        return np.array(data)
    else:
        # Default to CSV
        lines = request_body.strip().split('\n')
        data = []
        for line in lines:
            values = [float(x) for x in line.split(',')]
            data.append(values)
        return np.array(data)

def predict_fn(input_data, model_dict):
    """
    Make predictions using the loaded ensemble model.
    """
    print("Making predictions")
    
    models = model_dict['models']
    weights = model_dict['weights']
    
    # Convert to DataFrame if it's a numpy array
    if isinstance(input_data, np.ndarray):
        feature_names = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel',
            'gas_val', 'gas_delta', 'gas_vel'
        ]
        if input_data.shape[1] == len(feature_names):
            input_data = pd.DataFrame(input_data, columns=feature_names)
        else:
            # Reshape if it's a single sample
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            input_data = pd.DataFrame(input_data)
    
    # Get predictions from each model
    probabilities = {}
    
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(input_data)[:, 1]
            else:
                probs = model.decision_function(input_data)
                # Normalize to [0, 1] range
                probs = (probs - probs.min()) / (probs.max() - probs.min())
            probabilities[name] = probs
        except Exception as e:
            print(f"Error predicting with {name}: {e}")
            probabilities[name] = np.zeros(len(input_data))
    
    # Calculate weighted ensemble predictions
    ensemble_probs = np.zeros(len(input_data))
    total_weight = 0
    for name, weight in weights.items():
        if name in probabilities:
            ensemble_probs += weight * probabilities[name]
            total_weight += weight
    
    # Normalize if weights don't sum to 1
    if total_weight > 0:
        ensemble_probs /= total_weight
    
    # Convert to binary predictions
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    return {
        "predictions": ensemble_preds.tolist(),
        "probabilities": ensemble_probs.tolist()
    }

def output_fn(prediction, accept):
    """
    Format the prediction output.
    """
    print(f"Formatting output with content type: {accept}")
    
    if accept == 'application/json':
        return json.dumps(prediction)
    elif accept == 'text/csv':
        # Convert to CSV format
        if isinstance(prediction, dict) and 'probabilities' in prediction:
            return ','.join([str(x) for x in prediction['probabilities']])
        else:
            return json.dumps(prediction)
    else:
        # Default to JSON
        return json.dumps(prediction)