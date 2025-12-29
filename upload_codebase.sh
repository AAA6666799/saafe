#!/bin/bash

# Simple script to upload Saafe codebase to AWS
# This script provides multiple options for uploading your codebase

echo "🚀 Saafe Codebase Upload to AWS"
echo "================================"

# Find the archive
ARCHIVE=$(ls saafe_codebase_*.zip | head -1)

if [ -z "$ARCHIVE" ]; then
    echo "❌ No archive found. Creating one now..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVE="saafe_codebase_${TIMESTAMP}.zip"
    
    zip -r "$ARCHIVE" . \
        -x "*.pyc" "*__pycache__*" "*.DS_Store" "*._*" \
        ".kiro/*" "saafe_env/*" "fire_detection_env/*" \
        "*.log" "*_results_*.json"
    
    echo "✅ Created archive: $ARCHIVE"
fi

ARCHIVE_SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "📦 Archive: $ARCHIVE ($ARCHIVE_SIZE)"

echo ""
echo "📤 Upload Options:"
echo "1. AWS S3 (using AWS CLI)"
echo "2. AWS CodeCommit (using git)"
echo "3. Manual upload instructions"
echo "4. Exit"

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "📦 Uploading to S3..."
        
        # Check if AWS CLI is available
        if ! command -v aws &> /dev/null; then
            echo "❌ AWS CLI not found. Please install it first:"
            echo "   brew install awscli"
            echo "   aws configure"
            exit 1
        fi
        
        # Check if AWS is configured
        if ! aws sts get-caller-identity &> /dev/null; then
            echo "❌ AWS CLI not configured. Run: aws configure"
            exit 1
        fi
        
        # Get bucket name
        read -p "Enter S3 bucket name (or press Enter for auto-generated): " BUCKET_NAME
        
        if [ -z "$BUCKET_NAME" ]; then
            BUCKET_NAME="saafe-codebase-$(date +%Y%m%d)"
        fi
        
        # Create bucket if it doesn't exist
        aws s3 mb "s3://$BUCKET_NAME" 2>/dev/null || echo "Bucket may already exist"
        
        # Upload archive
        aws s3 cp "$ARCHIVE" "s3://$BUCKET_NAME/codebase/$ARCHIVE"
        
        if [ $? -eq 0 ]; then
            echo "✅ Upload successful!"
            echo "📍 S3 Location: s3://$BUCKET_NAME/codebase/$ARCHIVE"
            
            # Generate presigned URL for download
            echo "🔗 Generating download URL (valid for 7 days)..."
            aws s3 presign "s3://$BUCKET_NAME/codebase/$ARCHIVE" --expires-in 604800
        else
            echo "❌ Upload failed"
        fi
        ;;
        
    2)
        echo ""
        echo "📝 Setting up CodeCommit..."
        
        if ! command -v aws &> /dev/null; then
            echo "❌ AWS CLI not found. Please install it first:"
            echo "   brew install awscli"
            exit 1
        fi
        
        read -p "Enter CodeCommit repository name (default: saafe-fire-detection): " REPO_NAME
        REPO_NAME=${REPO_NAME:-saafe-fire-detection}
        
        # Create repository
        aws codecommit create-repository --repository-name "$REPO_NAME" --repository-description "Saafe Fire Detection AI System" 2>/dev/null || echo "Repository may already exist"
        
        # Get clone URL
        CLONE_URL=$(aws codecommit get-repository --repository-name "$REPO_NAME" --query 'repositoryMetadata.cloneUrlHttp' --output text)
        
        echo "✅ CodeCommit repository ready!"
        echo "📍 Repository: $REPO_NAME"
        echo "🔗 Clone URL: $CLONE_URL"
        echo ""
        echo "🔧 To push your code:"
        echo "   git init"
        echo "   git add ."
        echo "   git commit -m 'Initial commit: Saafe Fire Detection System'"
        echo "   git remote add aws $CLONE_URL"
        echo "   git push aws main"
        ;;
        
    3)
        echo ""
        echo "📋 Manual Upload Instructions"
        echo "============================="
        echo ""
        echo "Option A: AWS S3 Console"
        echo "1. Go to https://console.aws.amazon.com/s3/"
        echo "2. Create a new bucket (e.g., 'saafe-codebase-$(date +%Y%m%d)')"
        echo "3. Upload the file: $ARCHIVE"
        echo ""
        echo "Option B: AWS CodeCommit Console"
        echo "1. Go to https://console.aws.amazon.com/codesuite/codecommit/"
        echo "2. Create a new repository (e.g., 'saafe-fire-detection')"
        echo "3. Follow the instructions to push your code"
        echo ""
        echo "Option C: GitHub/GitLab"
        echo "1. Create a new repository on GitHub or GitLab"
        echo "2. Extract the archive: unzip $ARCHIVE"
        echo "3. Push to your repository"
        echo ""
        echo "📦 Your archive is ready: $ARCHIVE ($ARCHIVE_SIZE)"
        ;;
        
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
        
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "✅ Process completed!"
echo "📋 Archive details:"
echo "   File: $ARCHIVE"
echo "   Size: $ARCHIVE_SIZE"
echo "   Contents: Saafe Fire Detection System (cleaned, no Kiro files)"