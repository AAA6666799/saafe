#!/usr/bin/env python3
"""
Ensemble debug serving script - compatible with SageMaker's serving environment
"""

import json
import os
import numpy as np
import joblib

def model_fn(model_dir):
    """Load ensemble models"""
    print("Loading ensemble models...")
    
    try:
        models = {}
        
        # Load all models
        for model_file in os.listdir(model_dir):
            if model_file.endswith('_model.joblib'):
                model_name = model_file.replace('_model.joblib', '')
                model_path = os.path.join(model_dir, model_file)
                models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model")
        
        print(f"Loaded {len(models)} models")
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def input_fn(request_body, request_content_type):
    """Parse input data"""
    print(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, models):
    """Make ensemble predictions"""
    print("Making ensemble predictions...")
    
    try:
        # Extract features from input (matching the features used in training)
        features = np.array([
            input_data['t_mean'],        # Temperature mean
            input_data['t_std'],         # Temperature std
            input_data['t_max'],         # Temperature max
            input_data['gas_val'],       # Gas value (CO2 proxy)
            input_data['tproxy_val'],    # Temperature proxy value
            input_data['flow_mag_mean'], # Flow magnitude mean
            input_data['flow_mag_std']   # Flow magnitude std
        ]).reshape(1, -1)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features)[0][1]  # Probability of positive class
            else:
                prob = model.decision_function(features)[0]
                # Normalize to [0, 1] range
                prob = (prob - prob.min()) / (prob.max() - prob.min())
            
            pred = int(prob > 0.5)
            predictions[name] = pred
            probabilities[name] = float(prob)
        
        # Simple ensemble: average the probabilities
        ensemble_prob = np.mean(list(probabilities.values()))
        ensemble_prediction = int(ensemble_prob > 0.5)
        
        result = {
            'ensemble_prediction': ensemble_prediction,
            'ensemble_probability': float(ensemble_prob),
            'individual_predictions': predictions,
            'individual_probabilities': probabilities
        }
        
        print(f"Ensemble prediction: {ensemble_prediction} (probability: {ensemble_prob:.4f})")
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def output_fn(prediction, accept):
    """Format output"""
    print(f"Formatting output with content type: {accept}")
    
    if accept == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")