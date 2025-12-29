#!/bin/bash
# Script to set up a conda environment for Fire Detection AI training on SageMaker

echo "Setting up conda environment for Fire Detection AI training..."

# Create a new conda environment
conda create -n fire-detection python=3.8 -y

# Activate the environment
source activate fire-detection

# Install PyTorch with CUDA support
# This installs PyTorch 1.12.1 with CUDA 11.3 support
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other required packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install "numexpr>=2.8.4"  # Update numexpr to required version
pip install xgboost lightgbm
pip install boto3 sagemaker

# Register the conda environment as a kernel
python -m ipykernel install --user --name fire-detection --display-name "Fire Detection"

echo "Verifying installations..."

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify numexpr version
python -c "import numexpr; print(f'Numexpr version: {numexpr.__version__}')"

echo ""
echo "========================================"
echo "Setup complete!"
echo "To use this environment:"
echo "1. Restart the kernel in your notebook"
echo "2. Select the 'Fire Detection' kernel"
echo "3. Continue with your notebook"
echo "========================================"