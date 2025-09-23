#!/usr/bin/env python3
"""
Minimal inference script for testing
"""

import json
import os

def model_fn(model_dir):
    """Load model"""
    print("Loading minimal model")
    return "minimal_model"

def input_fn(request_body, request_content_type):
    """Parse input data"""
    print(f"Processing input with content type: {request_content_type}")
    return request_body

def predict_fn(input_data, model):
    """Make predictions"""
    print("Making predictions")
    return {"prediction": "minimal"}

def output_fn(prediction, accept):
    """Format output"""
    print(f"Formatting output with content type: {accept}")
    return json.dumps(prediction)