"""
Example script demonstrating how to use the ModelRegistry.

This script shows how to:
1. Initialize the ModelRegistry
2. Create model instances
3. Train models
4. Save models to the registry
5. Load models from the registry
6. Use models for prediction
7. Get model metadata and information
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.registry import ModelRegistry
from src.ml.models.identification import ElectricalFireIdentifier, ChemicalFireIdentifier
from src.ml.models.progression import FireGrowthPredictor
from src.ml.models.confidence import UncertaintyEstimator


def generate_sample_data(n_samples=1000):
    """
    Generate sample data for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    np.random.seed(42)
    
    # Features
    X = pd.DataFrame({
        # Electrical features
        'voltage_fluctuation': np.random.uniform(0, 1, n_samples),
        'electrical_noise_level': np.random.uniform(0, 1, n_samples),
        'circuit_load': np.random.uniform(0, 1, n_samples),
        'equipment_status': np.random.uniform(0, 1, n_samples),
        
        # Chemical features
        'toxic_gas_level': np.random.uniform(0, 1, n_samples),
        'chemical_reaction_rate': np.random.uniform(0, 1, n_samples),
        'ph_level': np.random.uniform(3, 11, n_samples),
        'oxidizer_concentration': np.random.uniform(0, 1, n_samples),
        
        # Environmental features
        'temperature': np.random.uniform(20, 100, n_samples),
        'humidity': np.random.uniform(0, 100, n_samples),
        'wind_speed': np.random.uniform(0, 30, n_samples),
        'wind_direction': np.random.uniform(0, 360, n_samples)
    })
    
    # Create labels for different fire types
    electrical_factors = (
        (X['voltage_fluctuation'] > 0.7).astype(int) +
        (X['electrical_noise_level'] > 0.7).astype(int) +
        (X['circuit_load'] > 0.8).astype(int) +
        (X['equipment_status'] < 0.3).astype(int)
    )
    
    chemical_factors = (
        (X['toxic_gas_level'] > 0.7).astype(int) +
        (X['chemical_reaction_rate'] > 0.7).astype(int) +
        (X['ph_level'] < 5).astype(int) +
        (X['oxidizer_concentration'] > 0.7).astype(int)
    )
    
    # Binary labels (1 for fire, 0 for no fire)
    y_electrical = (electrical_factors >= 2).astype(int)
    y_chemical = (chemical_factors >= 2).astype(int)
    
    # Fire size (for progression prediction)
    base_size = np.random.uniform(1, 10, n_samples)
    electrical_multiplier = 1 + 2 * y_electrical
    chemical_multiplier = 1 + 3 * y_chemical
    temperature_factor = (X['temperature'] - 20) / 80  # Normalize to 0-1
    
    y_size = base_size * electrical_multiplier * chemical_multiplier * (1 + temperature_factor)
    
    # Split into train and test sets
    train_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
    test_idx = np.array([i for i in range(n_samples) if i not in train_idx])
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    
    y_train_electrical = y_electrical[train_idx]
    y_test_electrical = y_electrical[test_idx]
    
    y_train_chemical = y_chemical[train_idx]
    y_test_chemical = y_chemical[test_idx]
    
    y_train_size = y_size[train_idx]
    y_test_size = y_size[test_idx]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_electrical': y_train_electrical,
        'y_test_electrical': y_test_electrical,
        'y_train_chemical': y_train_chemical,
        'y_test_chemical': y_test_chemical,
        'y_train_size': y_train_size,
        'y_test_size': y_test_size
    }


def main():
    """
    Main function demonstrating the ModelRegistry usage.
    """
    print("Fire Prediction System - Model Registry Example")
    print("=" * 50)
    
    # Create registry directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('metadata', exist_ok=True)
    
    # Initialize the ModelRegistry
    registry_config = {
        'registry_dir': 'models',
        'metadata_dir': 'metadata',
        'version_file': 'versions.json',
        'use_aws': False
    }
    
    registry = ModelRegistry(registry_config)
    
    # List available models
    print("\nAvailable models in the registry:")
    for model_type in ['classification', 'identification', 'progression', 'confidence']:
        models = registry.list_models(model_type)
        print(f"- {model_type.capitalize()} models: {', '.join(models)}")
    
    # Generate sample data
    print("\nGenerating sample data...")
    data = generate_sample_data(1000)
    
    # Create and train an electrical fire identifier
    print("\nCreating and training ElectricalFireIdentifier...")
    electrical_config = {
        'algorithm': 'random_forest',
        'n_estimators': 50,
        'feature_scaling': True
    }
    
    electrical_model = registry.create_model('electrical_fire_identifier', electrical_config)
    
    # Train the model
    electrical_metrics = electrical_model.train(
        data['X_train'], 
        data['y_train_electrical'],
        validation_data=(data['X_test'], data['y_test_electrical'])
    )
    
    print(f"Training metrics: {electrical_metrics}")
    
    # Save the model to the registry
    print("\nSaving ElectricalFireIdentifier to the registry...")
    electrical_path = registry.save_model(
        electrical_model, 
        'electrical_fire_identifier',
        metadata={'accuracy': electrical_metrics['accuracy']}
    )
    
    print(f"Model saved to: {electrical_path}")
    
    # Create and train a chemical fire identifier
    print("\nCreating and training ChemicalFireIdentifier...")
    chemical_config = {
        'algorithm': 'svm',
        'feature_scaling': True
    }
    
    chemical_model = registry.create_model('chemical_fire_identifier', chemical_config)
    
    # Train the model
    chemical_metrics = chemical_model.train(
        data['X_train'], 
        data['y_train_chemical'],
        validation_data=(data['X_test'], data['y_test_chemical'])
    )
    
    print(f"Training metrics: {chemical_metrics}")
    
    # Save the model to the registry
    print("\nSaving ChemicalFireIdentifier to the registry...")
    chemical_path = registry.save_model(
        chemical_model, 
        'chemical_fire_identifier',
        metadata={'accuracy': chemical_metrics['accuracy']}
    )
    
    print(f"Model saved to: {chemical_path}")
    
    # Create and train a fire growth predictor
    print("\nCreating and training FireGrowthPredictor...")
    growth_config = {
        'algorithm': 'gradient_boosting',
        'n_estimators': 50,
        'feature_scaling': True,
        'output_features': ['size']
    }
    
    growth_model = registry.create_model('fire_growth_predictor', growth_config)
    
    # Prepare data for regression
    y_train_df = pd.DataFrame({'size': data['y_train_size']})
    y_test_df = pd.DataFrame({'size': data['y_test_size']})
    
    # Train the model
    growth_metrics = growth_model.train(
        data['X_train'], 
        y_train_df,
        validation_data=(data['X_test'], y_test_df)
    )
    
    print(f"Training metrics: {growth_metrics}")
    
    # Save the model to the registry
    print("\nSaving FireGrowthPredictor to the registry...")
    growth_path = registry.save_model(
        growth_model, 
        'fire_growth_predictor',
        metadata={'r2': growth_metrics['size']['r2']}
    )
    
    print(f"Model saved to: {growth_path}")
    
    # Create and train an uncertainty estimator
    print("\nCreating and training UncertaintyEstimator...")
    uncertainty_config = {
        'method': 'bootstrap',
        'n_bootstrap_samples': 20,  # Small number for example
        'feature_scaling': True
    }
    
    uncertainty_model = registry.create_model('uncertainty_estimator', uncertainty_config)
    
    # Train the model
    uncertainty_metrics = uncertainty_model.train(
        data['X_train'], 
        data['y_train_size'],
        validation_data=(data['X_test'], data['y_test_size'])
    )
    
    print(f"Training metrics: {uncertainty_metrics}")
    
    # Save the model to the registry
    print("\nSaving UncertaintyEstimator to the registry...")
    uncertainty_path = registry.save_model(
        uncertainty_model, 
        'uncertainty_estimator',
        metadata={'mean_uncertainty': uncertainty_metrics['mean_uncertainty']}
    )
    
    print(f"Model saved to: {uncertainty_path}")
    
    # Load models from the registry
    print("\nLoading models from the registry...")
    
    loaded_electrical = registry.load_model('electrical_fire_identifier')
    loaded_chemical = registry.load_model('chemical_fire_identifier')
    loaded_growth = registry.load_model('fire_growth_predictor')
    loaded_uncertainty = registry.load_model('uncertainty_estimator')
    
    print("Models loaded successfully")
    
    # Use models for prediction
    print("\nMaking predictions with loaded models...")
    
    # Create a sample for prediction
    sample = pd.DataFrame({
        'voltage_fluctuation': [0.9],
        'electrical_noise_level': [0.8],
        'circuit_load': [0.9],
        'equipment_status': [0.2],
        'toxic_gas_level': [0.3],
        'chemical_reaction_rate': [0.4],
        'ph_level': [7.0],
        'oxidizer_concentration': [0.3],
        'temperature': [80],
        'humidity': [30],
        'wind_speed': [5],
        'wind_direction': [180]
    })
    
    # Make predictions
    electrical_pred = loaded_electrical.predict(sample)
    chemical_pred = loaded_chemical.predict(sample)
    
    # For growth prediction, we need to reshape the output
    growth_pred = loaded_growth.predict(sample)
    size_pred = growth_pred['size'][0] if isinstance(growth_pred, dict) else growth_pred[0]
    
    # Get uncertainty estimate
    uncertainty = loaded_uncertainty.estimate_uncertainty(sample, np.array([size_pred]))
    
    print(f"Electrical fire prediction: {bool(electrical_pred[0])}")
    print(f"Chemical fire prediction: {bool(chemical_pred[0])}")
    print(f"Fire size prediction: {size_pred:.2f} square meters")
    print(f"Uncertainty estimate: ±{uncertainty[0]:.2f}")
    
    # Get model characteristics
    print("\nIdentifying fire characteristics...")
    electrical_chars = loaded_electrical.identify_characteristics(sample)
    chemical_chars = loaded_chemical.identify_characteristics(sample)
    
    print("\nElectrical fire characteristics:")
    for char, value in electrical_chars.items():
        if isinstance(value, (int, float)):
            print(f"- {char}: {value:.2f}")
        else:
            print(f"- {char}: {value}")
    
    print("\nChemical fire characteristics:")
    for char, value in chemical_chars.items():
        if isinstance(value, (int, float)):
            print(f"- {char}: {value:.2f}")
        else:
            print(f"- {char}: {value}")
    
    # Get model metadata
    print("\nGetting model metadata...")
    electrical_metadata = registry.get_model_metadata('electrical_fire_identifier')
    
    print(f"Electrical fire identifier metadata:")
    print(f"- Model type: {electrical_metadata['model_type']}")
    print(f"- Version: {electrical_metadata['version']}")
    print(f"- Timestamp: {electrical_metadata['timestamp']}")
    
    # Get registry summary
    print("\nModel registry summary:")
    summary = registry.get_model_registry_summary()
    
    print(f"Total models: {summary['total_models']}")
    print(f"Models by type:")
    for model_type, count in summary['models_by_type'].items():
        print(f"- {model_type}: {count}")
    
    print(f"Version counts:")
    for model_name, count in summary['version_counts'].items():
        print(f"- {model_name}: {count}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()