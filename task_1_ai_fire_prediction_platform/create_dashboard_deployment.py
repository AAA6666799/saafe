#!/usr/bin/env python3
"""
Script to create a clean deployment package for the dashboard
"""

import os
import zipfile
import shutil

def create_dashboard_deployment():
    """Create a deployment package specifically for the dashboard"""
    
    # Define the output zip file name
    output_file = 'saafe-fire-dashboard.zip'
    
    # Remove existing zip file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Create temporary directory for deployment files
    temp_dir = 'temp_deployment'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Files to include in the deployment
    files_to_include = [
        'dashboard.py',
        'dashboard_requirements.txt',
        'run_dashboard.py',
        'Dockerfile',
        'deploy_dashboard.sh',
        '__init__.py'
    ]
    
    # Copy files to temporary directory
    for file in files_to_include:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(temp_dir, file))
            print(f"Copied: {file}")
        else:
            print(f"Warning: {file} not found")
    
    # Copy directories
    dirs_to_include = [
        'ai_fire_prediction_platform'
    ]
    
    for dir_name in dirs_to_include:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, os.path.join(temp_dir, dir_name))
            print(f"Copied directory: {dir_name}")
        else:
            print(f"Warning: {dir_name} not found")
    
    # Create the zip file
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                archive_path = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, archive_path)
                print(f"Added to zip: {archive_path}")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ Dashboard deployment package created: {output_file}")

if __name__ == '__main__':
    create_dashboard_deployment()