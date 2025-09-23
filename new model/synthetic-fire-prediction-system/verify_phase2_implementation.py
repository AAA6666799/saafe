#!/usr/bin/env python3
"""
Verification script for Phase 2 implementation.
This script verifies that all Phase 2 components are correctly implemented
and can be imported without errors.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def verify_files_exist():
    """Verify that all required files for Phase 2 exist."""
    required_files = [
        'src/ml/ensemble/simple_ensemble_manager.py',
        'phase2_ensemble_implementation.py',
        'PHASE2_IMPLEMENTATION_SUMMARY.md',
        'test_simple_ensemble.py',
        'test_phase2_basic.py'
    ]
    
    print("🔍 Verifying Phase 2 files...")
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
    """Verify that all Phase 2 components can be imported."""
    print("\n🔍 Verifying imports...")
    
    try:
        # Test ensemble manager import
        from src.ml.ensemble.simple_ensemble_manager import SimpleEnsembleManager
        print("  ✅ SimpleEnsembleManager imported successfully")
        
        # Test Phase 2 implementation import
        import phase2_ensemble_implementation
        print("  ✅ Phase 2 implementation imported successfully")
        
        # Test ensemble system creation
        ensemble_system = phase2_ensemble_implementation.FLIRSCD41EnsembleSystem()
        print("  ✅ FLIRSCD41EnsembleSystem created successfully")
        
        # Test system info
        info = ensemble_system.get_system_info()
        print(f"  ✅ System info retrieved (Version: {info.get('system_version', 'Unknown')})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_ensemble_functionality():
    """Verify basic ensemble functionality."""
    print("\n🔍 Verifying ensemble functionality...")
    
    try:
        import phase2_ensemble_implementation
        
        # Create ensemble system
        ensemble_system = phase2_ensemble_implementation.FLIRSCD41EnsembleSystem()
        
        # Initialize with mock models
        ensemble_system.create_mock_models()
        
        # Test system info
        info = ensemble_system.get_system_info()
        print(f"  ✅ System initialized with {info.get('total_models', 0)} models")
        
        # Test prediction (should fail gracefully before training)
        try:
            result = ensemble_system.predict_with_ensemble(
                {'t_mean': 25.0, 't_max': 45.0, 't_hot_area_pct': 5.0},
                {'gas_val': 450.0, 'gas_vel': 5.0}
            )
            print("  ⚠️  Prediction succeeded (unexpected - should require training)")
            return False
        except ValueError as e:
            print("  ✅ Prediction correctly failed before training")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function."""
    print("🚀 Verifying Phase 2 Implementation")
    print("=" * 50)
    
    # Verify files
    files_ok = verify_files_exist()
    
    # Verify imports
    imports_ok = verify_imports()
    
    # Verify functionality
    functionality_ok = verify_ensemble_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary:")
    print(f"  Files: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"  Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    
    if files_ok and imports_ok and functionality_ok:
        print("\n🎉 All Phase 2 verifications passed!")
        print("✅ Phase 2 implementation is ready for use.")
        return 0
    else:
        print("\n💥 Some verifications failed!")
        print("❌ Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())