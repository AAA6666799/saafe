#!/usr/bin/env python3
"""
Test script to verify Phase 3 components without import errors.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test that all Phase 3 components can be imported without errors."""
    print("🔍 Testing Phase 3 imports...")
    
    # Test main implementation
    try:
        import phase3_complete_implementation
        print("✅ phase3_complete_implementation imported successfully")
    except Exception as e:
        print(f"❌ Failed to import phase3_complete_implementation: {e}")
        return False
    
    # Test training scripts
    try:
        import src.ml.training.train_sklearn_model
        print("✅ train_sklearn_model imported successfully")
    except Exception as e:
        print(f"❌ Failed to import train_sklearn_model: {e}")
        return False
    
    try:
        import src.ml.training.train_xgboost_model
        print("✅ train_xgboost_model imported successfully")
    except Exception as e:
        print(f"❌ Failed to import train_xgboost_model: {e}")
        return False
    
    # Test data generation
    try:
        from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
        print("✅ SyntheticDataGenerator imported successfully")
    except Exception as e:
        print(f"❌ Failed to import SyntheticDataGenerator: {e}")
        return False
    
    # Test model training pipeline
    try:
        from src.ml.training.training_pipeline import ModelTrainingPipeline
        print("✅ ModelTrainingPipeline imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ModelTrainingPipeline: {e}")
        return False
    
    return True

def test_component_initialization():
    """Test that components can be initialized without errors."""
    print("\n🔍 Testing component initialization...")
    
    try:
        # Test synthetic data generator
        from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
        config = {
            'thermal': {'base_temperature': 22.0},
            'gas': {'base_concentration': 0.1},
            'environmental': {'temperature_variance': 3.0}
        }
        generator = SyntheticDataGenerator(config)
        print("✅ SyntheticDataGenerator initialized successfully")
        
        # Test training pipeline
        from src.ml.training.training_pipeline import ModelTrainingPipeline
        pipeline_config = {
            'test_size': 0.2,
            'validation_size': 0.1,
            'output_dir': '/tmp/test_models'
        }
        pipeline = ModelTrainingPipeline(pipeline_config)
        print("✅ ModelTrainingPipeline initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_scripts():
    """Test that training scripts have the correct structure."""
    print("\n🔍 Testing training script structure...")
    
    try:
        # Check that training scripts have main functions
        import src.ml.training.train_sklearn_model
        import src.ml.training.train_xgboost_model
        
        # Check for main function
        if hasattr(src.ml.training.train_sklearn_model, 'main'):
            print("✅ train_sklearn_model has main function")
        else:
            print("❌ train_sklearn_model missing main function")
            return False
            
        if hasattr(src.ml.training.train_xgboost_model, 'main'):
            print("✅ train_xgboost_model has main function")
        else:
            print("❌ train_xgboost_model missing main function")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Phase 3 Components")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        return 1
    
    # Test component initialization
    if not test_component_initialization():
        return 1
    
    # Test training scripts
    if not test_training_scripts():
        return 1
    
    print("\n" + "=" * 40)
    print("✅ All Phase 3 components passed!")
    print("🎉 No import errors or configuration errors detected.")
    return 0

if __name__ == "__main__":
    sys.exit(main())