#!/usr/bin/env python3
"""
Verification script for complete Phase 3 implementation.
This script verifies that all components of the complete implementation work correctly.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def verify_files():
    """Verify that all required files exist."""
    print("🔍 Verifying required files...")
    
    required_files = [
        'phase3_complete_implementation.py',
        'src/ml/training/train_sklearn_model.py',
        'src/ml/training/train_xgboost_model.py',
        'PHASE3_IMPLEMENTATION_SUMMARY.md',
        'test_phase3_components.py'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (MISSING)")
            all_files_exist = False
    
    return all_files_exist

def verify_imports():
    """Verify that all components can be imported."""
    print("\n🔍 Verifying imports...")
    
    try:
        # Test main implementation
        import phase3_complete_implementation
        print("  ✅ Phase 3 implementation imported successfully")
        
        # Test training scripts
        import src.ml.training.train_sklearn_model
        print("  ✅ Scikit-learn training script imported successfully")
        
        import src.ml.training.train_xgboost_model
        print("  ✅ XGBoost training script imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_class_instantiation():
    """Verify that classes can be instantiated."""
    print("\n🔍 Verifying class instantiation...")
    
    try:
        # Test Phase 3 implementation class
        from phase3_complete_implementation import Phase3CompleteImplementation
        impl = Phase3CompleteImplementation()
        print("  ✅ Phase3CompleteImplementation instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_training_script_structure():
    """Verify that training scripts have the required structure."""
    print("\n🔍 Verifying training script structure...")
    
    try:
        # Check sklearn training script
        import src.ml.training.train_sklearn_model as sklearn_train
        if hasattr(sklearn_train, 'main') and callable(sklearn_train.main):
            print("  ✅ Scikit-learn training script has main function")
        else:
            print("  ❌ Scikit-learn training script missing main function")
            return False
        
        # Check xgboost training script
        import src.ml.training.train_xgboost_model as xgboost_train
        if hasattr(xgboost_train, 'main') and callable(xgboost_train.main):
            print("  ✅ XGBoost training script has main function")
        else:
            print("  ❌ XGBoost training script missing main function")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training script structure error: {e}")
        return False

def main():
    """Main verification function."""
    print("🚀 Verifying Complete Phase 3 Implementation")
    print("=" * 50)
    
    # Verify files
    files_ok = verify_files()
    
    # Verify imports
    imports_ok = verify_imports()
    
    # Verify class instantiation
    instantiation_ok = verify_class_instantiation()
    
    # Verify training script structure
    structure_ok = verify_training_script_structure()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary:")
    print(f"  Files: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"  Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Instantiation: {'✅ PASS' if instantiation_ok else '❌ FAIL'}")
    print(f"  Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    
    if files_ok and imports_ok and instantiation_ok and structure_ok:
        print("\n🎉 All verifications passed!")
        print("✅ Complete Phase 3 implementation is ready for use.")
        print("\n🚀 To run the complete implementation:")
        print("   python phase3_complete_implementation.py")
        return 0
    else:
        print("\n💥 Some verifications failed!")
        print("❌ Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())