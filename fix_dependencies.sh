#!/bin/bash
# Fix numexpr and pandas compatibility issues in SageMaker

echo "🔧 Fixing SageMaker Dependencies..."

# Update conda and pip
conda update conda -y
pip install --upgrade pip

# Fix numexpr issue
pip uninstall numexpr -y
pip install numexpr==2.8.4

# Upgrade pandas to compatible version
pip install pandas==2.0.3

# Install missing dependencies
pip install tqdm==4.65.0
conda install -c conda-forge xgboost lightgbm -y

# Install additional performance packages
pip install pyarrow fastparquet

echo "✅ Dependencies fixed!"
echo "🚀 You can now run your training script without errors"