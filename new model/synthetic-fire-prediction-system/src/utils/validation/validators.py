"""
Validation utilities for the synthetic fire prediction system.

This module provides validation functions for various data types and configurations.
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import pandas as pd


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """
    Validate that a configuration dictionary contains all required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
        
    Returns:
        List of missing keys (empty if all required keys are present)
    """
    missing_keys = []
    
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    return missing_keys


def validate_nested_config(config: Dict[str, Any], required_structure: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Validate that a configuration dictionary contains all required nested keys.
    
    Args:
        config: Configuration dictionary to validate
        required_structure: Dictionary mapping section names to lists of required keys
        
    Returns:
        Dictionary mapping section names to lists of missing keys
    """
    missing_keys = {}
    
    for section, keys in required_structure.items():
        if section not in config:
            missing_keys[section] = keys
            continue
        
        section_missing_keys = validate_config(config[section], keys)
        if section_missing_keys:
            missing_keys[section] = section_missing_keys
    
    return missing_keys


def validate_file_path(file_path: str, must_exist: bool = True, file_type: Optional[str] = None) -> List[str]:
    """
    Validate a file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        file_type: Optional file type to check (e.g., '.json', '.yaml')
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not file_path:
        errors.append("File path cannot be empty")
        return errors
    
    if must_exist and not os.path.exists(file_path):
        errors.append(f"File does not exist: {file_path}")
    
    if file_type and not file_path.endswith(file_type):
        errors.append(f"File must have {file_type} extension: {file_path}")
    
    return errors


def validate_directory_path(dir_path: str, must_exist: bool = True, create_if_missing: bool = False) -> List[str]:
    """
    Validate a directory path.
    
    Args:
        dir_path: Path to validate
        must_exist: Whether the directory must exist
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not dir_path:
        errors.append("Directory path cannot be empty")
        return errors
    
    if not os.path.exists(dir_path):
        if must_exist and not create_if_missing:
            errors.append(f"Directory does not exist: {dir_path}")
        elif create_if_missing:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                errors.append(f"Failed to create directory {dir_path}: {str(e)}")
    elif os.path.exists(dir_path) and not os.path.isdir(dir_path):
        errors.append(f"Path exists but is not a directory: {dir_path}")
    
    return errors


def validate_json_file(file_path: str, schema: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Validate a JSON file.
    
    Args:
        file_path: Path to the JSON file
        schema: Optional schema to validate against
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = validate_file_path(file_path, must_exist=True, file_type='.json')
    if errors:
        return errors
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse JSON file {file_path}: {str(e)}")
        return errors
    
    if schema:
        # Basic schema validation
        for key, value_type in schema.items():
            if key not in data:
                errors.append(f"Missing required key in JSON: {key}")
            elif not isinstance(data[key], value_type):
                errors.append(f"Invalid type for key {key}: expected {value_type.__name__}, got {type(data[key]).__name__}")
    
    return errors


def validate_yaml_file(file_path: str, schema: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Validate a YAML file.
    
    Args:
        file_path: Path to the YAML file
        schema: Optional schema to validate against
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = validate_file_path(file_path, must_exist=True, file_type='.yaml')
    if not errors:
        errors = validate_file_path(file_path, must_exist=True, file_type='.yml')
    if errors:
        return errors
    
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Failed to parse YAML file {file_path}: {str(e)}")
        return errors
    
    if schema:
        # Basic schema validation
        for key, value_type in schema.items():
            if key not in data:
                errors.append(f"Missing required key in YAML: {key}")
            elif not isinstance(data[key], value_type):
                errors.append(f"Invalid type for key {key}: expected {value_type.__name__}, got {type(data[key]).__name__}")
    
    return errors


def validate_numeric_range(value: Union[int, float], min_value: Optional[Union[int, float]] = None, 
                         max_value: Optional[Union[int, float]] = None) -> List[str]:
    """
    Validate that a numeric value is within a specified range.
    
    Args:
        value: Value to validate
        min_value: Optional minimum value
        max_value: Optional maximum value
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(value, (int, float)):
        errors.append(f"Value must be numeric: {value}")
        return errors
    
    if min_value is not None and value < min_value:
        errors.append(f"Value {value} is less than minimum {min_value}")
    
    if max_value is not None and value > max_value:
        errors.append(f"Value {value} is greater than maximum {max_value}")
    
    return errors


def validate_string_length(value: str, min_length: Optional[int] = None, 
                         max_length: Optional[int] = None) -> List[str]:
    """
    Validate that a string is within a specified length range.
    
    Args:
        value: String to validate
        min_length: Optional minimum length
        max_length: Optional maximum length
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(value, str):
        errors.append(f"Value must be a string: {value}")
        return errors
    
    if min_length is not None and len(value) < min_length:
        errors.append(f"String length {len(value)} is less than minimum {min_length}")
    
    if max_length is not None and len(value) > max_length:
        errors.append(f"String length {len(value)} is greater than maximum {max_length}")
    
    return errors


def validate_array_shape(array: np.ndarray, expected_shape: Optional[tuple] = None, 
                       expected_dims: Optional[int] = None) -> List[str]:
    """
    Validate the shape of a numpy array.
    
    Args:
        array: Array to validate
        expected_shape: Optional expected shape
        expected_dims: Optional expected number of dimensions
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(array, np.ndarray):
        errors.append(f"Value must be a numpy array: {type(array)}")
        return errors
    
    if expected_shape is not None and array.shape != expected_shape:
        errors.append(f"Array shape {array.shape} does not match expected shape {expected_shape}")
    
    if expected_dims is not None and array.ndim != expected_dims:
        errors.append(f"Array has {array.ndim} dimensions, expected {expected_dims}")
    
    return errors


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None,
                     column_types: Optional[Dict[str, type]] = None) -> List[str]:
    """
    Validate a pandas DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: Optional list of required column names
        column_types: Optional dictionary mapping column names to expected types
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(df, pd.DataFrame):
        errors.append(f"Value must be a pandas DataFrame: {type(df)}")
        return errors
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"DataFrame is missing required columns: {missing_columns}")
    
    if column_types:
        for col, expected_type in column_types.items():
            if col not in df.columns:
                continue  # Skip columns that don't exist (already reported above)
            
            # Check if column has the expected type
            if not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                errors.append(f"Column {col} has type {df[col].dtype}, expected {expected_type}")
    
    return errors


def validate_with_custom_function(value: Any, validation_func: Callable[[Any], bool], 
                                error_message: str) -> List[str]:
    """
    Validate a value using a custom validation function.
    
    Args:
        value: Value to validate
        validation_func: Function that returns True if value is valid, False otherwise
        error_message: Error message to return if validation fails
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    try:
        if not validation_func(value):
            errors.append(error_message)
    except Exception as e:
        errors.append(f"Validation function raised an exception: {str(e)}")
    
    return errors