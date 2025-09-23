# 🔥 FLIR+SCD41 Fire Detection System - Unified Training Notebook

## Overview

This document describes the unified training notebook that combines all diagnostics and fixes for the FLIR+SCD41 fire detection system.

## Notebook: `flir_scd41_unified_training_diagnostics.ipynb`

This comprehensive notebook provides a complete end-to-end training pipeline for the FLIR+SCD41 fire detection system with built-in diagnostics to prevent common machine learning problems:

### Features

1. **Dataset Generation**
   - Synthetic data generation for FLIR Lepton 3.5 thermal camera (15 features)
   - Synthetic data generation for Sensirion SCD41 CO₂ sensor (3 features)
   - Total: 18 features for fire detection

2. **Data Management**
   - Dataset storage and persistence
   - Stratified data splitting (train/validation/test sets)
   - Data quality diagnostics

3. **Model Training with Regularization**
   - XGBoost model with L1/L2 regularization and early stopping
   - Neural Network with dropout, batch normalization, and L2 regularization
   - Data augmentation techniques

4. **Diagnostics for Learning Issues**
   - Detection of underfitting (model not learning)
   - Detection of overfitting (remembering patterns instead of learning)
   - Learning curves visualization
   - Validation curves for hyperparameter optimization
   - Cross-validation for robust performance estimates

5. **Ensemble Weight Calculation**
   - Performance-based weighting using exponential scaling
   - Validation score-based ensemble optimization

6. **Model Evaluation**
   - Comprehensive test set evaluation
   - Multiple metrics (accuracy, F1, precision, recall, AUC)
   - Confusion matrix visualization
   - Performance comparison plots

7. **Model Persistence**
   - Saving trained models to disk
   - Saving ensemble weights
   - Saving model metadata and performance metrics

### Common ML Problems Addressed

- **Underfitting** (model not learning): Poor performance on both training and validation sets
- **Overfitting** (remembering patterns): High performance on training set, poor performance on validation/test sets

### Key Diagnostic Techniques

1. **Training vs. Validation Performance Comparison**
   - Large gap indicates overfitting
   - Poor performance on both indicates underfitting

2. **Learning Curves**
   - Converge to low scores = underfitting
   - Large gap = overfitting
   - Converge to high scores with small gap = good fit

3. **Validation Curves**
   - Find optimal hyperparameters
   - Identify when increasing complexity hurts performance

### Prevention Techniques

1. **Regularization**
   - L1/L2 regularization
   - Dropout layers
   - Early stopping

2. **Model Architecture**
   - Reduced model complexity
   - Appropriate depth/width

3. **Data Techniques**
   - Data augmentation
   - Cross-validation
   - More training data

## Usage

To run the notebook:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook in Jupyter:
   ```bash
   jupyter notebook notebooks/flir_scd41_unified_training_diagnostics.ipynb
   ```

3. Execute all cells sequentially

## Output Files

The notebook generates the following files in `data/flir_scd41/`:
- `flir_scd41_dataset.csv` - Complete synthetic dataset
- `train.csv` - Training set
- `val.csv` - Validation set
- `test.csv` - Test set
- `flir_scd41_xgboost_model_improved.json` - Trained XGBoost model
- `best_improved_nn_model.pth` - Trained Neural Network model
- `ensemble_weights_improved.json` - Ensemble weights based on validation performance
- `model_info_improved.json` - Model metadata and performance metrics
- `xgboost_learning_curves.png` - Learning curves visualization
- `validation_curves_max_depth.png` - Validation curves for max_depth parameter

## Best Practices Implemented

### Detecting Learning Issues
1. Compare training and validation performance
2. Use learning curves for bias/variance diagnosis
3. Use validation curves for hyperparameter optimization

### Preventing Overfitting
1. Regularization (L1/L2, dropout, early stopping)
2. Appropriate model architecture
3. Data augmentation and cross-validation

### Ensuring Proper Learning
1. Data quality checks (leakage, variance, balance)
2. Proper model configuration (learning rate, training time)
3. Comprehensive evaluation (separate test set, multiple metrics)

This notebook represents the culmination of all diagnostic and fix implementations for the FLIR+SCD41 fire detection system, providing a robust, well-documented training pipeline that addresses common machine learning problems.