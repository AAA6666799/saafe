#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files for AWS deployment.
Removes Kiro-specific files, development artifacts, and temporary files.
"""

import os
import shutil
import glob
from pathlib import Path

def remove_directory(path):
    """Safely remove directory if it exists"""
    if os.path.exists(path):
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"✅ Removed directory: {path}")
        except Exception as e:
            print(f"⚠️  Error removing {path}: {e}")
    else:
        print(f"⚠️  Directory not found: {path}")

def remove_file(path):
    """Safely remove file if it exists"""
    if os.path.exists(path):
        os.remove(path)
        print(f"✅ Removed file: {path}")
    else:
        print(f"⚠️  File not found: {path}")

def remove_files_by_pattern(pattern):
    """Remove files matching a pattern"""
    files = glob.glob(pattern)
    for file in files:
        remove_file(file)

def main():
    print("🧹 Starting cleanup for AWS deployment...")
    print("=" * 60)
    
    # Remove Kiro-specific directories
    print("\n📁 Removing Kiro-specific directories...")
    remove_directory(".kiro")
    
    # Remove development and build artifacts
    print("\n🔨 Removing development artifacts...")
    remove_directory(".ipynb_checkpoints")
    remove_directory(".pytest_cache")
    remove_directory(".vscode")
    remove_directory("htmlcov")
    remove_directory("build")
    remove_directory("dist")
    remove_directory("prototype")
    remove_directory("temp")
    
    # Remove virtual environments
    print("\n🐍 Removing virtual environments...")
    remove_directory("safeguard_env")
    remove_directory("fire_detection_env")
    
    # Remove test and demo directories
    print("\n🧪 Removing test and demo files...")
    remove_directory("tests")
    remove_directory("demo day")
    remove_directory("Dataset")
    remove_directory("Fire Dataset by regions")
    remove_directory("scripts")
    
    # Remove logs and exports
    print("\n📊 Removing logs and exports...")
    remove_directory("logs")
    remove_directory("exports")
    remove_directory("data")
    
    # Remove documentation (keep essential ones)
    print("\n📚 Cleaning up documentation...")
    remove_directory("docs/archive")
    remove_directory("docs/presentations")
    remove_directory("docs/specs")
    remove_directory("docs/visualizations")
    
    # Remove deployment artifacts (we'll create new ones)
    print("\n🚀 Removing old deployment files...")
    remove_directory("deployment")
    
    # Remove hidden macOS files
    print("\n🍎 Removing macOS hidden files...")
    remove_files_by_pattern("._*")
    
    # Remove development and test files
    print("\n🗑️  Removing development files...")
    development_files = [
        "CODEBASE_FIX_PROGRESS_REPORT.md",
        "CODEBASE_HEALTH_REPORT.md",
        "CODEBASE_ORGANIZATION_COMPLETE.md",
        "CODEBASE_REORGANIZATION_PLAN.md",
        "IMPORT_FIXES_COMPLETE_REPORT.md",
        "SETUP_COMPLETE.md",
        "analyze_fire_data.py",
        "run_ceo_presentation.py",
        "safeguard_colab_demo.ipynb",
        "start_safeguard.sh",
        "download.png",
        ".flake8",
        "pyproject.toml",
        "local_requirements.txt",
        "notification_requirements.txt"
    ]
    
    for file in development_files:
        remove_file(file)
    
    # Remove test result files
    print("\n📋 Removing test result files...")
    test_result_patterns = [
        "*_results_*.json",
        "*_test.log",
        "test_*.txt",
        "demo_scenario_results.json",
        "structure_validation_results.json",
        "system_integration_*.json"
    ]
    
    for pattern in test_result_patterns:
        remove_files_by_pattern(pattern)
    
    # Clean up assets directory
    print("\n🎨 Cleaning up assets...")
    if os.path.exists("assets"):
        # Keep the directory but remove placeholder files
        remove_file("assets/icon_info.txt")
    
    # Remove __pycache__ directories recursively
    print("\n🐍 Removing Python cache files...")
    for root, dirs, files in os.walk("."):
        for dir in dirs:
            if dir == "__pycache__":
                remove_directory(os.path.join(root, dir))
    
    # Remove .pyc files
    remove_files_by_pattern("**/*.pyc")
    remove_files_by_pattern("**/*.pyo")
    
    print("\n" + "=" * 60)
    print("✅ Cleanup completed!")
    print("\n📦 Files remaining for AWS deployment:")
    
    # Show remaining structure
    essential_files = [
        "app.py",
        "main.py", 
        "requirements.txt",
        "README.md",
        "LICENSE",
        "INSTALLATION_GUIDE.md",
        "saafe_mvp/",
        "models/",
        "config/",
        "docs/USER_MANUAL.md",
        "docs/TECHNICAL_DOCUMENTATION.md",
        "docs/TROUBLESHOOTING_GUIDE.md"
    ]
    
    print("\n🎯 Essential files for deployment:")
    for item in essential_files:
        if os.path.exists(item):
            print(f"  ✅ {item}")
        else:
            print(f"  ❌ {item} (missing)")
    
    print(f"\n📊 Current directory size:")
    total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk('.')
                    for filename in filenames)
    print(f"  Total: {total_size / (1024*1024):.1f} MB")
    
    print("\n🚀 Ready for AWS deployment!")
    print("Next steps:")
    print("1. Review remaining files")
    print("2. Update requirements.txt if needed")
    print("3. Test application locally: streamlit run app.py")
    print("4. Follow AWS deployment guide")

if __name__ == "__main__":
    main()