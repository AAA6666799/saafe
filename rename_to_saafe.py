#!/usr/bin/env python3
"""
Script to rename everything from "Saafe" to "Saafe"
- Renames files and directories
- Updates content in all files
"""

import os
import re
import shutil
from pathlib import Path

def rename_files_and_directories():
    """Rename files and directories containing 'Saafe' to 'Saafe'"""
    
    print("🔄 Renaming files and directories...")
    
    # Get all files and directories
    all_paths = []
    for root, dirs, files in os.walk('.'):
        # Add directories
        for d in dirs:
            full_path = os.path.join(root, d)
            if 'Saafe' in d or 'saafe' in d:
                all_paths.append(full_path)
        
        # Add files
        for f in files:
            full_path = os.path.join(root, f)
            if 'Saafe' in f or 'saafe' in f:
                all_paths.append(full_path)
    
    # Sort by depth (deepest first) to avoid issues with nested renames
    all_paths.sort(key=lambda x: x.count(os.sep), reverse=True)
    
    renamed_count = 0
    for old_path in all_paths:
        if os.path.exists(old_path):
            # Create new path with Saafe -> Saafe and saafe -> saafe
            new_path = old_path.replace('Saafe', 'Saafe').replace('saafe', 'saafe')
            
            if old_path != new_path:
                try:
                    # Ensure parent directory exists
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    
                    # Rename
                    shutil.move(old_path, new_path)
                    print(f"  ✅ Renamed: {old_path} → {new_path}")
                    renamed_count += 1
                except Exception as e:
                    print(f"  ❌ Error renaming {old_path}: {e}")
    
    print(f"📁 Renamed {renamed_count} files/directories")
    return renamed_count

def update_file_contents():
    """Update content in all text files"""
    
    print("\n📝 Updating file contents...")
    
    # File extensions to process
    text_extensions = {
        '.py', '.md', '.txt', '.json', '.yaml', '.yml', '.sh', '.bat',
        '.html', '.css', '.js', '.dockerfile', '.cfg', '.ini', '.toml'
    }
    
    # Patterns to replace
    replacements = [
        (r'\bSafeguard\b', 'Saafe'),
        (r'\bsafeguard\b', 'saafe'),
        (r'\bSAFEGUARD\b', 'SAAFE'),
        # Handle specific cases
        (r'saafe-mvp', 'saafe-mvp'),
        (r'Saafe MVP', 'Saafe MVP'),
        (r'saafe_mvp', 'saafe_mvp'),
        # Handle URLs and identifiers
        (r'com\.saafe', 'com.saafe'),
        (r'saafe-ai\.com', 'saafe-ai.com'),
        (r'SaafeExecutionRole', 'SaafeExecutionRole'),
        (r'SaafeSensorPolicy', 'SaafeSensorPolicy'),
        (r'SaafeDataProcessing', 'SaafeDataProcessing'),
        (r'SaafePredictions', 'SaafePredictions'),
        (r'SaafeDevices', 'SaafeDevices'),
        (r'SaafeAlerts', 'SaafeAlerts'),
        (r'saafe-cluster', 'saafe-cluster'),
        (r'saafe-service', 'saafe-service'),
        (r'saafe-alb', 'saafe-alb'),
        (r'saafe-targets', 'saafe-targets'),
        (r'saafe-ecs-sg', 'saafe-ecs-sg'),
        (r'saafe-endpoint', 'saafe-endpoint'),
        (r'saafe-models', 'saafe-models'),
        (r'saafe-sensor-stream', 'saafe-sensor-stream'),
        (r'saafe-scaling-policy', 'saafe-scaling-policy'),
        (r'/ecs/saafe-mvp', '/ecs/saafe-mvp'),
        (r'SaafeHighRiskScore', 'SaafeHighRiskScore'),
        (r'topic/saafe/sensors', 'topic/saafe/sensors'),
    ]
    
    updated_files = 0
    total_replacements = 0
    
    # Walk through all files
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.vscode'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Only process text files
            if file_ext in text_extensions or file_ext == '':
                try:
                    # Read file
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    original_content = content
                    file_replacements = 0
                    
                    # Apply all replacements
                    for pattern, replacement in replacements:
                        new_content = re.sub(pattern, replacement, content)
                        if new_content != content:
                            file_replacements += len(re.findall(pattern, content))
                            content = new_content
                    
                    # Write back if changed
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        print(f"  ✅ Updated {file_path} ({file_replacements} replacements)")
                        updated_files += 1
                        total_replacements += file_replacements
                
                except Exception as e:
                    print(f"  ❌ Error processing {file_path}: {e}")
    
    print(f"📝 Updated {updated_files} files with {total_replacements} total replacements")
    return updated_files, total_replacements

def main():
    """Main function"""
    print("🔄 Starting Saafe → Saafe rename process...")
    print("=" * 60)
    
    # Step 1: Update file contents first (before renaming files)
    files_updated, total_replacements = update_file_contents()
    
    # Step 2: Rename files and directories
    files_renamed = rename_files_and_directories()
    
    print("\n" + "=" * 60)
    print("✅ Rename process completed!")
    print(f"📊 Summary:")
    print(f"  • Files with content updated: {files_updated}")
    print(f"  • Total text replacements: {total_replacements}")
    print(f"  • Files/directories renamed: {files_renamed}")
    
    print(f"\n🎯 Key changes made:")
    print(f"  • Saafe → Saafe")
    print(f"  • saafe → saafe")
    print(f"  • saafe_mvp → saafe_mvp")
    print(f"  • saafe-mvp → saafe-mvp")
    print(f"  • All AWS resource names updated")
    print(f"  • All file and directory names updated")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Test the application: streamlit run app.py")
    print(f"  2. Update any remaining references manually")
    print(f"  3. Commit changes to version control")
    print(f"  4. Proceed with AWS deployment")

if __name__ == "__main__":
    main()