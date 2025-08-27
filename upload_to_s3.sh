#!/bin/bash
# Script to upload Fire Detection AI notebooks to an S3 bucket

# Default values
BUCKET_NAME=""
PREFIX="fire-detection-notebooks"

# Display help message
show_help() {
    echo "Usage: $0 -b <bucket_name> [-p <prefix>]"
    echo ""
    echo "Options:"
    echo "  -b, --bucket    S3 bucket name (required)"
    echo "  -p, --prefix    S3 prefix/folder name (default: fire-detection-notebooks)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -b my-sagemaker-bucket -p fire-detection"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -b|--bucket)
            BUCKET_NAME="$2"
            shift
            shift
            ;;
        -p|--prefix)
            PREFIX="$2"
            shift
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if bucket name is provided
if [ -z "$BUCKET_NAME" ]; then
    echo "Error: S3 bucket name is required"
    show_help
fi

# Check if required packages are installed
check_dependencies() {
    echo "Checking dependencies..."
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi
    
    # Check for pip
    if ! command -v pip3 &> /dev/null; then
        echo "pip3 is not installed. Please install pip3 and try again."
        exit 1
    fi
    
    # Check for boto3
    if ! python3 -c "import boto3" &> /dev/null; then
        echo "Installing boto3..."
        pip3 install boto3
    fi
    
    # Check for tqdm
    if ! python3 -c "import tqdm" &> /dev/null; then
        echo "Installing tqdm..."
        pip3 install tqdm
    fi
    
    # Check for AWS CLI
    if ! command -v aws &> /dev/null; then
        echo "AWS CLI is not installed. It's recommended for AWS credential management."
        echo "Would you like to install it? (y/n)"
        read -r install_aws
        if [[ $install_aws == "y" ]]; then
            pip3 install awscli
        fi
    fi
    
    echo "All dependencies are installed."
}

# Check AWS credentials
check_aws_credentials() {
    echo "Checking AWS credentials..."
    
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "AWS credentials not configured or invalid."
        echo "Please run 'aws configure' to set up your AWS credentials."
        exit 1
    fi
    
    echo "AWS credentials are valid."
}

# Main function
main() {
    echo "========================================"
    echo "Fire Detection AI Notebook Upload Script"
    echo "========================================"
    echo ""
    echo "This script will upload Fire Detection AI training notebooks to S3."
    echo "Target: s3://$BUCKET_NAME/$PREFIX/"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Check AWS credentials
    check_aws_credentials
    
    # Run the Python upload script
    echo ""
    echo "Starting upload process..."
    python3 upload_notebooks_to_s3.py --bucket "$BUCKET_NAME" --prefix "$PREFIX"
    
    # Check if upload was successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Upload completed successfully!"
        echo ""
        echo "Your notebooks are now available at:"
        echo "s3://$BUCKET_NAME/$PREFIX/"
        echo ""
        echo "Next steps:"
        echo "1. Launch a SageMaker ml.p3.16xlarge instance"
        echo "2. Use download_notebooks_from_s3.py to download the notebooks"
        echo "3. Follow the instructions in sagemaker_p3_16xlarge_guide.md"
        echo "========================================"
    else
        echo ""
        echo "========================================"
        echo "Upload failed. Please check the error messages above."
        echo "========================================"
    fi
}

# Execute main function
main