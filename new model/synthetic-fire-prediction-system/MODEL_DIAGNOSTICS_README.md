# 🔍 FLIR+SCD41 Model Diagnostics and Fixes

This document explains how to diagnose and fix common machine learning problems in the FLIR+SCD41 fire detection system.

## 📋 Overview

Common machine learning problems and their solutions:

### 1. Model Not Learning (Underfitting)
- **Symptoms**: Poor performance on both training and validation sets
- **Causes**: 
  - Insufficient model capacity
  - Poor data quality
  - Inappropriate learning rate
  - Feature scaling issues

### 2. Overfitting (Remembering Patterns)
- **Symptoms**: High performance on training set, poor performance on validation/test sets
- **Causes**:
  - Model too complex for the data
  - Insufficient regularization
  - Too little training data
  - Data leakage

## 📁 Project Structure

```
synthetic-fire-prediction-system/
├── notebooks/
│   └── model_diagnostics_and_fixes.ipynb     # Diagnostics demonstration
├── scripts/
│   ├── model_diagnostics.py                   # Diagnostic tools
│   └── improved_flir_scd41_training.py        # Training with fixes
└── data/
    └── flir_scd41/                           # Generated diagnostics plots
```

## 🚀 How to Use the Diagnostics

### 1. Using the Jupyter Notebook

The notebook `notebooks/model_diagnostics_and_fixes.ipynb` demonstrates:
- How to detect underfitting and overfitting
- Learning curves visualization
- Validation curves for hyperparameter tuning
- Examples of regularized models

To run the notebook:
```bash
jupyter notebook notebooks/model_diagnostics_and_fixes.ipynb
```

### 2. Using the Diagnostic Script

The script `scripts/model_diagnostics.py` provides functions to:
- Check data quality
- Generate learning curves
- Generate validation curves
- Detect overfitting
- Suggest solutions

Example usage:
```python
from scripts.model_diagnostics import run_comprehensive_diagnostics

# After training your models
overfitting_detected = run_comprehensive_diagnostics(
    X_train, y_train, X_val, y_val,
    xgb_train_metrics, xgb_val_metrics
)
```

### 3. Using the Improved Training Script

The script `scripts/improved_flir_scd41_training.py` implements:
- Early stopping
- Cross-validation
- Regularization
- Data augmentation
- Proper train/validation/test splits

To run the improved training:
```bash
cd /path/to/synthetic-fire-prediction-system
python scripts/improved_flir_scd41_training.py
```

## 🔧 Diagnostic Techniques

### 1. Performance Gap Analysis
Compare training and validation performance:
```python
# Calculate gap
gap = train_accuracy - val_accuracy

if gap > 0.1:
    print("⚠️  Possible overfitting")
elif train_accuracy < 0.7 and val_accuracy < 0.7:
    print("🤔 Possible underfitting")
else:
    print("✅ Good fit")
```

### 2. Learning Curves
Visualize how performance changes with more data:
- High bias: Both curves converge to low scores
- High variance: Large gap between curves
- Good fit: Both curves converge to high scores with small gap

### 3. Validation Curves
Find optimal hyperparameters by varying one parameter:
- Shows the trade-off between bias and variance
- Identifies when increasing complexity hurts performance

## 🛠️ Solutions for Common Problems

### For Underfitting (Model Not Learning)
1. **Increase model complexity**:
   - More trees in XGBoost
   - Deeper neural networks
   - More layers/neurons

2. **Feature engineering**:
   - Add more relevant features
   - Create interaction features
   - Polynomial features

3. **Training adjustments**:
   - Increase learning rate
   - Train for more epochs
   - Try different optimizers

### For Overfitting (Remembering Patterns)
1. **Regularization**:
   ```python
   # XGBoost with regularization
   model = xgb.XGBClassifier(
       reg_alpha=0.1,    # L1 regularization
       reg_lambda=1.0,   # L2 regularization
       max_depth=4,      # Limit depth
       subsample=0.8     # Subsampling
   )
   
   # Neural Network with dropout
   class RegularizedNN(nn.Module):
       def __init__(self, dropout_rate=0.3):
           super().__init__()
           self.dropout = nn.Dropout(dropout_rate)
   ```

2. **Early Stopping**:
   ```python
   # XGBoost early stopping
   model.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)],
             early_stopping_rounds=10)
   
   # Custom early stopping for PyTorch
   class EarlyStopping:
       def __init__(self, patience=7):
           self.patience = patience
           self.counter = 0
           self.best_loss = None
   ```

3. **Data Augmentation**:
   ```python
   def augment_data(X, y, factor=0.1):
       """Add noise to training data"""
       n_samples = X.shape[0]
       n_augment = int(n_samples * factor)
       indices = np.random.choice(n_samples, n_augment)
       noise = np.random.normal(0, 0.01, (n_augment, X.shape[1]))
       X_augmented = X[indices] + noise
       return np.vstack([X, X_augmented]), np.hstack([y, y[indices]])
   ```

## 📊 Monitoring Training

### 1. Track Metrics
Monitor multiple metrics during training:
- Accuracy
- F1 Score
- Precision and Recall
- AUC-ROC

### 2. Visualize Progress
Plot training and validation curves:
```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()
```

### 3. Cross-Validation
Use cross-validation for more robust performance estimates:
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

## 🎯 Best Practices

### 1. Data Management
- Ensure no data leakage between train/validation/test sets
- Check for class imbalance
- Verify feature quality and variance

### 2. Model Selection
- Start simple and increase complexity gradually
- Use validation curves to find optimal hyperparameters
- Compare multiple algorithms

### 3. Evaluation
- Always use a separate test set
- Use multiple metrics for comprehensive evaluation
- Consider domain-specific requirements (e.g., prefer high recall for fire detection)

### 4. Regularization
- Use appropriate regularization for your model type
- Tune regularization parameters
- Combine multiple regularization techniques

## 📈 Expected Outcomes

A well-trained model should show:
- Training accuracy: 85-95%
- Validation accuracy: 80-90%
- Test accuracy: 80-90%
- Small gap between training and validation (< 10%)
- Consistent performance across cross-validation folds

## 🚨 Troubleshooting

### Common Issues and Solutions

1. **Poor Performance on All Sets**
   - Solution: Increase model complexity, check data quality

2. **Large Training-Validation Gap**
   - Solution: Add regularization, reduce model complexity

3. **Inconsistent Cross-Validation Scores**
   - Solution: Get more data, check for data quality issues

4. **Slow Training**
   - Solution: Reduce model complexity, use early stopping

### Getting Help

If you encounter persistent issues:
1. Check data quality and preprocessing
2. Verify model implementation
3. Try different algorithms
4. Consult domain experts for feature engineering

## 📚 Additional Resources

- [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [PyTorch Regularization](https://pytorch.org/docs/stable/nn.html#dropout-layers)
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/modules/model_selection.html)
- [Learning Curves Explanation](https://en.wikipedia.org/wiki/Learning_curve_(machine_learning))

## 🤝 Contributing

Feel free to contribute improvements to the diagnostic tools:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request