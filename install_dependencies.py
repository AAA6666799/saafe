# Run this cell to install the required dependencies for the Fire Detection AI notebook

# Install PyTorch and related packages
!pip install torch torchvision torchaudio

# Install other required packages
!pip install pandas numpy scikit-learn matplotlib seaborn
!pip install xgboost lightgbm
!pip install boto3 sagemaker
!pip install numexpr>=2.8.4  # Update numexpr to required version

# Verify installations
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("PyTorch not installed correctly")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError:
    print("Pandas not installed correctly")

try:
    import numexpr
    print(f"Numexpr version: {numexpr.__version__}")
except ImportError:
    print("Numexpr not installed correctly")

print("\nAfter running this cell, restart the kernel and then continue with the notebook.")