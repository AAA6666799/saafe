#!/usr/bin/env python3
"""
Script to create a clean deployment package without macOS hidden files
"""

import os
import zipfile
import fnmatch

def should_exclude(file_path):
    """Determine if a file should be excluded from the deployment package"""
    # Get the filename
    filename = os.path.basename(file_path)
    
    # Exclude macOS hidden files
    if filename.startswith('._') or filename.startswith('.DS_Store'):
        return True
    
    # Exclude other common hidden files
    if filename.startswith('.') and not filename.startswith(('.', '..')):
        return True
        
    return False

def create_clean_zip(output_filename, source_dir):
    """Create a zip file excluding hidden macOS files"""
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Remove hidden directories from dirs list so they won't be processed
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, source_dir)
                
                # Skip hidden files
                if not should_exclude(file_path):
                    zipf.write(file_path, relative_path)
                    print(f"Added: {relative_path}")
                else:
                    print(f"Excluded: {relative_path}")

def main():
    # Files and directories to include in the deployment
    deployment_files = [
        'saafe_aws_dashboard.py',
        'dashboard_requirements.txt',
        'Dockerfile',
        'Dockerfile.dashboard',
        'deploy_dashboard.sh',
        'deploy_dashboard_simple.sh',
        'deploy_to_ec2.sh',
        'eb-config.json',
        'eb-config-simple.json',
        'eb-config-new.json',
        'AWS_DASHBOARD_DEPLOYMENT_GUIDE.md',
        'README.md'
    ]
    
    # Create clean deployment package
    output_file = 'saafe-dashboard-clean.zip'
    
    # Remove existing zip file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Create new clean zip file
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in deployment_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added: {file}")
            else:
                print(f"Warning: {file} not found")
    
    print(f"\n✅ Clean deployment package created: {output_file}")

if __name__ == '__main__':
    main()