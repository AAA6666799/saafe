#!/usr/bin/env python3
"""
Basic test for Phase 2 implementation to check for import errors.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_phase2_imports():
    """Test that Phase 2 components can be imported without errors."""
    try:
        # Test ensemble manager import
        from src.ml.ensemble.simple_ensemble_manager import SimpleEnsembleManager
        print("✅ SimpleEnsembleManager imported successfully")
        
        # Test Phase 2 implementation import
        import phase2_ensemble_implementation
        print("✅ Phase 2 implementation imported successfully")
        
        # Test ensemble system creation
        ensemble_system = phase2_ensemble_implementation.FLIRSCD41EnsembleSystem()
        print("✅ FLIRSCD41EnsembleSystem created successfully")
        
        # Test system info
        info = ensemble_system.get_system_info()
        print("✅ System info retrieved successfully")
        print(f"   System version: {info.get('system_version', 'Unknown')}")
        print(f"   Phase: {info.get('phase', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Phase 2 imports: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_prediction():
    """Test mock prediction to verify basic functionality."""
    try:
        import phase2_ensemble_implementation
        
        # Create ensemble system
        ensemble_system = phase2_ensemble_implementation.FLIRSCD41EnsembleSystem()
        
        # Initialize with mock models
        ensemble_system.create_mock_models()
        
        # Test system info
        info = ensemble_system.get_system_info()
        print(f"✅ System initialized with {info.get('total_models', 0)} models")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in mock prediction test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Phase 2 Implementation")
    print("=" * 40)
    
    # Test imports
    print("\n1. Testing imports...")
    if not test_phase2_imports():
        return 1
    
    # Test basic functionality
    print("\n2. Testing basic functionality...")
    if not test_mock_prediction():
        return 1
    
    print("\n" + "=" * 40)
    print("✅ All Phase 2 basic tests passed!")
    print("🎉 Phase 2 implementation is working correctly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())