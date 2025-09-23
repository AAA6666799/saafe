#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - SageMaker Inference Script
This script is compatible with SageMaker's inference environment.
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

def input_fn(request_body, content_type='text/csv'):
    """Parse input data."""
    if content_type == 'text/csv':
        # Convert CSV string to numpy array
        lines = request_body.strip().split('\n')
        data = []
        for line in lines:
            values = [float(x) for x in line.split(',')]
            data.append(values)
        return np.array(data)
    elif content_type == 'application/json':
        # Parse JSON
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    """Make predictions using ensemble."""
    models = model_dict['models']
    weights = model_dict['weights']
    
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)
    
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
    
    return ensemble_probs

def output_fn(prediction, accept='application/json'):
    """Format prediction output."""
    if accept == 'application/json':
        return json.dumps(prediction.tolist())
    elif accept == 'text/csv':
        return ','.join([str(x) for x in prediction])
    else:
        raise ValueError(f"Unsupported accept type: {accept}")