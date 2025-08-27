# Fix for "X_train is not defined" error
# Add this code before initializing the model

# Check if data was loaded properly
import os
import numpy as np

# Option 1: Try to load saved numpy files if they exist
if os.path.exists('X_train.npy') and os.path.exists('y_train.npy'):
    print("Loading data from saved numpy files...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    if os.path.exists('areas_train.npy'):
        areas_train = np.load('areas_train.npy')
        areas_val = np.load('areas_val.npy')
        areas_test = np.load('areas_test.npy')
    else:
        # Create dummy area values if not available
        areas_train = np.zeros(len(y_train), dtype=int)
        areas_val = np.zeros(len(y_val), dtype=int)
        areas_test = np.zeros(len(y_test), dtype=int)
    
    print(f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}")

# Option 2: Create synthetic data if files don't exist
else:
    print("Creating synthetic data for testing...")
    # Define dimensions
    n_samples = 10000
    n_timesteps = 60
    n_features = 6
    n_classes = 3
    
    # Create synthetic data
    X = np.random.randn(n_samples, n_timesteps, n_features).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    areas = np.random.choice(range(5), size=n_samples)
    
    # Split into train/val/test
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
        X, y, areas, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
        X_train, y_train, areas_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Created synthetic data: X_train={X_train.shape}, y_train={y_train.shape}")

# Print data summary
print("\nData Summary:")
print(f"  Train: {X_train.shape}, {np.bincount(y_train.astype(int))}")
print(f"  Validation: {X_val.shape}, {np.bincount(y_val.astype(int))}")
print(f"  Test: {X_test.shape}, {np.bincount(y_test.astype(int))}")