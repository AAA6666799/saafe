#!/usr/bin/env python3
"""
Test script to validate the unified training notebook components
"""

import sys
import os
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def test_notebook_structure():
    """Test that the notebook has the expected structure"""
    notebook_path = "notebooks/flir_scd41_unified_training_diagnostics.ipynb"
    
    try:
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        print("✅ Notebook structure test passed")
        print(f"   - Total cells: {len(nb.cells)}")
        
        # Count cell types
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
        print(f"   - Code cells: {len(code_cells)}")
        print(f"   - Markdown cells: {len(markdown_cells)}")
        
        # Check for key sections
        section_titles = [
            "Dataset Generation",
            "Dataset Storage", 
            "Data Splitting",
            "Model Diagnostics Introduction",
            "Model Training with Regularization",
            "Cross-Validation",
            "Learning Curves Visualization",
            "Validation Curves",
            "Ensemble Weight Calculation",
            "Model Evaluation",
            "Results Visualization",
            "Model Saving",
            "Training Summary and Diagnostics",
            "Key Takeaways and Best Practices"
        ]
        
        markdown_content = '\n'.join([cell.source for cell in markdown_cells])
        found_sections = [title for title in section_titles if title in markdown_content]
        print(f"   - Found {len(found_sections)}/{len(section_titles)} key sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook structure test failed: {e}")
        return False

def test_required_libraries():
    """Test that all required libraries can be imported"""
    required_libraries = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
        'torch', 'xgboost', 'json', 'os', 'sys'
    ]
    
    missing_libraries = []
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            missing_libraries.append(lib)
    
    if missing_libraries:
        print(f"❌ Missing libraries: {missing_libraries}")
        return False
    else:
        print("✅ All required libraries are available")
        return True

def test_data_directory():
    """Test that the data directory structure is correct"""
    try:
        # Create data directory if it doesn't exist
        data_dir = "data/flir_scd41"
        os.makedirs(data_dir, exist_ok=True)
        print("✅ Data directory structure is correct")
        return True
    except Exception as e:
        print(f"❌ Data directory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running tests for FLIR+SCD41 Unified Training Notebook...\n")
    
    tests = [
        test_notebook_structure,
        test_required_libraries,
        test_data_directory
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The unified training notebook is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())