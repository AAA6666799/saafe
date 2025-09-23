#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - SageMaker Serve Script for 100K Samples Ensemble
This script is compatible with SageMaker's serving environment and makes predictions using the trained ensemble.
"""

import json
import os
import numpy as np
import joblib

def model_fn(model_dir):
    """Load ensemble models and weights"""
    print("Loading ensemble models and weights...")
    
    try:
        models = {}
        
        # Load all models
        for model_file in os.listdir(model_dir):
            if model_file.endswith('_model.joblib'):
                model_name = model_file.replace('_model.joblib', '')
                model_path = os.path.join(model_dir, model_file)
                models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model")
        
        # Load ensemble weights
        weights_path = os.path.join(model_dir, "ensemble_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                weights_data = json.load(f)
            models['ensemble_weights'] = weights_data
            print("Loaded ensemble weights")
        else:
            # Default to equal weights if no weights file found
            model_count = len([k for k in models.keys() if k != 'ensemble_weights'])
            models['ensemble_weights'] = {
                'weights': [1.0/model_count] * model_count
            }
            print("Using equal weights for ensemble")
        
        print(f"Loaded {len([k for k in models.keys() if k != 'ensemble_weights'])} models")
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
        # Extract features from input
        features = np.array([
            input_data['t_mean'],           # Temperature mean
            input_data['t_std'],            # Temperature std
            input_data['t_max'],            # Temperature max
            input_data['t_p95'],            # Temperature 95th percentile
            input_data['t_hot_area_pct'],   # Hot area percentage
            input_data['t_hot_largest_blob_pct'],  # Largest hot blob percentage
            input_data['t_grad_mean'],      # Temperature gradient mean
            input_data['t_grad_std'],       # Temperature gradient std
            input_data['t_diff_mean'],      # Temperature difference mean
            input_data['t_diff_std'],       # Temperature difference std
            input_data['flow_mag_mean'],    # Flow magnitude mean
            input_data['flow_mag_std'],     # Flow magnitude std
            input_data['tproxy_val'],       # Temperature proxy value
            input_data['tproxy_delta'],     # Temperature proxy delta
            input_data['tproxy_vel'],       # Temperature proxy velocity
            input_data['gas_val'],          # Gas value
            input_data['gas_delta'],        # Gas delta
            input_data['gas_vel']           # Gas velocity
        ]).reshape(1, -1)
        
        # Get ensemble weights
        if 'ensemble_weights' in models:
            weights_data = models['ensemble_weights']
            weights = weights_data['weights']
        else:
            # Default to equal weights
            model_count = len([k for k in models.keys() if k != 'ensemble_weights'])
            weights = [1.0/model_count] * model_count
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        weighted_probs = []
        
        # Filter out ensemble_weights from models dict
        prediction_models = {k: v for k, v in models.items() if k != 'ensemble_weights'}
        
        for i, (name, model) in enumerate(prediction_models.items()):
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features)[0][1]  # Probability of positive class
            else:
                prob = model.decision_function(features)[0]
                # Normalize to [0, 1] range
                prob = (prob - prob.min()) / (prob.max() - prob.min())
            
            pred = int(prob > 0.5)
            predictions[name] = pred
            probabilities[name] = float(prob)
            weighted_probs.append(weights[i] * prob)
        
        # Calculate weighted ensemble prediction
        ensemble_prob = sum(weighted_probs)
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