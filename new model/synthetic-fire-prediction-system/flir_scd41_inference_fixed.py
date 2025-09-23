#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - SageMaker Inference Script (Fixed)
This script handles model loading and inference for deployed models.
"""

import json
import os
import pandas as pd
import joblib
import numpy as np

def model_fn(model_dir):
    """
    Load the trained model from the model directory.
    """
    print(f"Loading model from {model_dir}")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Model loaded successfully")
    return model

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
        else:
            features = input_data
            
        # Convert to DataFrame
        df = pd.DataFrame([features])
        return df
    else:
        # Handle CSV input
        df = pd.read_csv(request_body)
        return df

def predict_fn(input_data, model):
    """
    Make predictions using the loaded model.
    """
    print("Making predictions")
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Get prediction probabilities if available
    try:
        probabilities = model.predict_proba(input_data)
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
    except:
        return {
            "predictions": predictions.tolist()
        }

def output_fn(prediction, content_type):
    """
    Format the prediction output.
    """
    print(f"Formatting output with content type: {content_type}")
    
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        # Default to JSON
        return json.dumps(prediction)