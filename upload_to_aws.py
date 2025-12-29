#!/usr/bin/env python3
"""
Upload cleaned codebase to AWS S3
"""

import boto3
import zipfile
import os
from datetime import datetime
import json

def create_codebase_archive():
    """Create a zip archive of the cleaned codebase"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"saafe_codebase_{timestamp}.zip"
    
    print(f"📦 Creating archive: {archive_name}")
    
    # Files and directories to include
    include_patterns = [
        "app.py",
        "main.py", 
        "requirements.txt",
        "README.md",
        "LICENSE",
        "INSTALLATION_GUIDE.md",
        "saafe_mvp/",
        "models/",
        "config/",
        "docs/",
        "assets/",
        ".gitignore"
    ]
    
    # Files to exclude
    exclude_patterns = [
        "__pycache__",
        ".pyc",
        ".pyo",
        ".DS_Store",
        "._*",
        ".kiro",
        "saafe_env/",
        "fire_detection_env/"
    ]
    
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip excluded files
                if any(pattern in file_path for pattern in exclude_patterns):
                    continue
                
                # Only include files matching our patterns
                relative_path = os.path.relpath(file_path, '.')
                if any(relative_path.startswith(pattern.rstrip('/')) for pattern in include_patterns):
                    zipf.write(file_path, relative_path)
                    print(f"  ✅ Added: {relative_path}")
    
    # Get archive size
    archive_size = os.path.getsize(archive_name)
    print(f"📊 Archive created: {archive_size / (1024*1024):.1f} MB")
    
    return archive_name

def upload_to_s3(archive_name, bucket_name=None):
    """Upload archive to S3"""
    
    s3_client = boto3.client('s3')
    
    # If no bucket specified, create one
    if not bucket_name:
        timestamp = datetime.now().strftime("%Y%m%d")
        bucket_name = f"saafe-codebase-{timestamp}"
        
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"✅ Created S3 bucket: {bucket_name}")
        except Exception as e:
            if "BucketAlreadyExists" in str(e):
                bucket_name = f"saafe-codebase-{timestamp}-{os.urandom(4).hex()}"
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"✅ Created S3 bucket: {bucket_name}")
            else:
                print(f"❌ Error creating bucket: {e}")
                return None
    
    # Upload archive
    try:
        print(f"⬆️  Uploading to S3...")
        s3_client.upload_file(
            archive_name, 
            bucket_name, 
            f"codebase/{archive_name}",
            ExtraArgs={
                'Metadata': {
                    'project': 'saafe-fire-detection',
                    'version': '1.0',
                    'upload-date': datetime.now().isoformat()
                }
            }
        )
        
        s3_url = f"s3://{bucket_name}/codebase/{archive_name}"
        print(f"✅ Upload successful!")
        print(f"📍 S3 Location: {s3_url}")
        
        return s3_url
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def upload_to_codecommit(repo_name="saafe-fire-detection"):
    """Alternative: Upload to AWS CodeCommit"""
    
    try:
        codecommit = boto3.client('codecommit')
        
        # Check if repo exists, create if not
        try:
            codecommit.get_repository(repositoryName=repo_name)
            print(f"✅ Using existing CodeCommit repo: {repo_name}")
        except codecommit.exceptions.RepositoryDoesNotExistException:
            codecommit.create_repository(
                repositoryName=repo_name,
                repositoryDescription="Saafe Fire Detection AI System"
            )
            print(f"✅ Created CodeCommit repo: {repo_name}")
        
        # Get clone URL
        repo_info = codecommit.get_repository(repositoryName=repo_name)
        clone_url = repo_info['repositoryMetadata']['cloneUrlHttp']
        
        print(f"📍 CodeCommit URL: {clone_url}")
        print(f"🔧 To push code:")
        print(f"   git remote add aws {clone_url}")
        print(f"   git push aws main")
        
        return clone_url
        
    except Exception as e:
        print(f"❌ CodeCommit setup failed: {e}")
        return None

def main():
    print("🚀 Uploading Saafe codebase to AWS...")
    print("=" * 60)
    
    # Create archive
    archive_name = create_codebase_archive()
    
    print(f"\n📤 Choose upload method:")
    print("1. S3 (recommended for backup/storage)")
    print("2. CodeCommit (recommended for version control)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    results = {}
    
    if choice in ['1', '3']:
        print(f"\n📦 Uploading to S3...")
        bucket_name = input("Enter S3 bucket name (or press Enter for auto-generated): ").strip()
        if not bucket_name:
            bucket_name = None
        
        s3_url = upload_to_s3(archive_name, bucket_name)
        results['s3'] = s3_url
    
    if choice in ['2', '3']:
        print(f"\n📝 Setting up CodeCommit...")
        repo_name = input("Enter CodeCommit repo name (or press Enter for 'saafe-fire-detection'): ").strip()
        if not repo_name:
            repo_name = "saafe-fire-detection"
        
        codecommit_url = upload_to_codecommit(repo_name)
        results['codecommit'] = codecommit_url
    
    # Save results
    with open('aws_upload_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'archive_name': archive_name,
            'archive_size_mb': os.path.getsize(archive_name) / (1024*1024),
            'results': results
        }, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("✅ Upload completed!")
    print(f"📋 Results saved to: aws_upload_results.json")
    
    # Clean up local archive
    cleanup = input(f"\nDelete local archive {archive_name}? (y/N): ").strip().lower()
    if cleanup == 'y':
        os.remove(archive_name)
        print(f"🗑️  Deleted local archive")

if __name__ == "__main__":
    main()