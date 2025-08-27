#!/bin/bash
# Script to download Fire Detection AI notebooks from an S3 bucket to a SageMaker instance

# Default values
BUCKET_NAME=""
PREFIX="fire-detection-notebooks"
OUTPUT_DIR="fire_detection"

# Display help message
show_help() {
    echo "Usage: $0 -b <bucket_name> [-p <prefix>] [-o <output_dir>]"
    echo ""
    echo "Options:"
    echo "  -b, --bucket    S3 bucket name (required)"
    echo "  -p, --prefix    S3 prefix/folder name (default: fire-detection-notebooks)"
    echo "  -o, --output    Local output directory (default: fire_detection)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -b my-sagemaker-bucket -p fire-detection -o notebooks"
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
        -o|--output)
            OUTPUT_DIR="$2"
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
    
    echo "All dependencies are installed."
}

# Check AWS credentials
check_aws_credentials() {
    echo "Checking AWS credentials..."
    
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "AWS credentials not configured or invalid."
        echo "SageMaker instances should have credentials via IAM role."
        echo "If you're not running this on SageMaker, please run 'aws configure'."
    else
        echo "AWS credentials are valid."
    fi
}

# Setup environment for training
setup_environment() {
    echo "Setting up environment for training..."
    
    # Create conda environment if it doesn't exist
    if ! conda info --envs | grep -q "fire-detection"; then
        echo "Creating conda environment 'fire-detection'..."
        conda create -n fire-detection python=3.8 -y
        
        # Activate environment and install packages
        source activate fire-detection
        
        # Check if requirements file exists
        if [ -f "$OUTPUT_DIR/supporting_files/requirements_gpu.txt" ]; then
            echo "Installing packages from requirements_gpu.txt..."
            pip install -r "$OUTPUT_DIR/supporting_files/requirements_gpu.txt"
        else
            echo "Installing common packages for deep learning..."
            pip install torch torchvision torchaudio
            pip install pandas numpy scikit-learn matplotlib seaborn
            pip install xgboost lightgbm
            pip install boto3 sagemaker
        fi
        
        # Register kernel for Jupyter
        echo "Registering Jupyter kernel..."
        python -m ipykernel install --user --name fire-detection --display-name "Fire Detection"
        
        echo "Environment setup complete."
    else
        echo "Environment 'fire-detection' already exists."
    fi
}

# Create GPU setup script
create_gpu_setup_script() {
    echo "Creating GPU setup script..."
    
    mkdir -p "$OUTPUT_DIR"
    cat > "$OUTPUT_DIR/gpu_setup.py" << 'EOF'
import torch
import os

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set environment variables for optimal performance
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # All 8 GPUs on p3.16xlarge
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_P2P_DISABLE'] = '1'  # May help with some multi-GPU issues

print("\nGPU setup complete. You can now run your training notebooks.")
print("Remember to use torch.nn.DataParallel(model) for multi-GPU training.")
EOF

    echo "GPU setup script created at $OUTPUT_DIR/gpu_setup.py"
}

# Main function
main() {
    echo "=============================================="
    echo "Fire Detection AI Notebook Download Script"
    echo "=============================================="
    echo ""
    echo "This script will download Fire Detection AI training notebooks from S3."
    echo "Source: s3://$BUCKET_NAME/$PREFIX/"
    echo "Destination: $OUTPUT_DIR/"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Check AWS credentials
    check_aws_credentials
    
    # Run the Python download script
    echo ""
    echo "Starting download process..."
    python3 download_notebooks_from_s3.py --bucket "$BUCKET_NAME" --prefix "$PREFIX" --output-dir "$OUTPUT_DIR"
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        # Create GPU setup script
        create_gpu_setup_script
        
        # Ask if user wants to set up the environment
        echo ""
        echo "Would you like to set up a conda environment for training? (y/n)"
        read -r setup_env
        if [[ $setup_env == "y" ]]; then
            setup_environment
        fi
        
        echo ""
        echo "=============================================="
        echo "Download completed successfully!"
        echo ""
        echo "Your notebooks are now available in: $OUTPUT_DIR/"
        echo ""
        echo "Next steps:"
        echo "1. Run the GPU setup script: python $OUTPUT_DIR/gpu_setup.py"
        echo "2. Open the notebooks in JupyterLab"
        echo "3. Select the 'Fire Detection' kernel"
        echo "4. Run the notebooks"
        echo "=============================================="
    else
        echo ""
        echo "=============================================="
        echo "Download failed. Please check the error messages above."
        echo "=============================================="
    fi
}

# Execute main function
main