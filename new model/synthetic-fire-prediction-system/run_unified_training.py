#!/usr/bin/env python3
"""
Script to execute the FLIR+SCD41 Unified Training Notebook
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn',
        'torch', 'xgboost', 'jupyter', 'nbformat'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print("  pip install -r requirements_unified_notebook.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def setup_environment():
    """Setup the required directory structure"""
    try:
        # Create data directory
        data_dir = os.path.join("data", "flir_scd41")
        os.makedirs(data_dir, exist_ok=True)
        print(f"✅ Data directory created: {data_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create data directory: {e}")
        return False

def execute_notebook():
    """Execute the unified training notebook"""
    notebook_path = "notebooks/flir_scd41_unified_training_diagnostics.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    try:
        # Execute notebook using jupyter
        cmd = [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--output", "flir_scd41_unified_training_diagnostics_executed.ipynb",
            notebook_path
        ]
        
        print("🚀 Executing unified training notebook...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Notebook executed successfully")
            print(f"📝 Output saved to: flir_scd41_unified_training_diagnostics_executed.ipynb")
            return True
        else:
            print("❌ Notebook execution failed")
            print(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Jupyter not found. Please install it using: pip install jupyter")
        return False
    except Exception as e:
        print(f"❌ Failed to execute notebook: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Execute FLIR+SCD41 Unified Training Notebook")
    parser.add_argument("--check-only", action="store_true", 
                        help="Only check dependencies and environment, don't execute")
    args = parser.parse_args()
    
    print("🔥 FLIR+SCD41 Fire Detection System - Unified Training Execution")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # If check-only mode, exit here
    if args.check_only:
        print("✅ Environment check completed successfully")
        return
    
    # Execute notebook
    if execute_notebook():
        print("\n🎉 Unified training notebook execution completed!")
        print("📁 Check the data/flir_scd41/ directory for output files")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()