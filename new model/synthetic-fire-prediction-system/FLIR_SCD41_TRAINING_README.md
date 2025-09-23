# 🔥 FLIR+SCD41 Fire Detection System - Training Pipeline

This document explains how to use the complete end-to-end training pipeline for the FLIR+SCD41 fire detection system.

## 📋 Overview

The training pipeline includes:
1. Dataset generation
2. Data storage
3. Data splitting
4. Model training (XGBoost and Neural Network)
5. Ensemble weight calculation
6. Model evaluation

## 📁 Project Structure

```
synthetic-fire-prediction-system/
├── notebooks/
│   └── flir_scd41_complete_training_pipeline.ipynb  # Jupyter notebook version
├── scripts/
│   ├── flir_scd41_training_pipeline_simple.py       # Python script version
│   └── flir_scd41_training_pipeline.py              # Full training pipeline
├── test_training/
│   ├── test_flir_scd41_training_simple.py           # Simple test script
│   └── test_flir_scd41_training.py                  # Comprehensive test script
└── data/
    └── flir_scd41/                                  # Generated data and models
```

## 🚀 How to Use the Training Pipeline

### Prerequisites

Make sure you have installed all required dependencies:
```bash
pip install -r requirements.txt
```

### Option 1: Using the Jupyter Notebook (Recommended for Exploration)

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `notebooks/flir_scd41_complete_training_pipeline.ipynb`

3. Run all cells to execute the complete training pipeline

### Option 2: Using the Python Script

Run the training pipeline directly from the command line:
```bash
cd /path/to/synthetic-fire-prediction-system
python scripts/flir_scd41_training_pipeline_simple.py
```

### Option 3: Using the Full Training Pipeline

For the complete training pipeline with all features:
```bash
cd /path/to/synthetic-fire-prediction-system
python scripts/flir_scd41_training_pipeline.py
```

## 🧪 Testing the Pipeline

Run the test script to verify the pipeline works correctly:
```bash
cd /path/to/synthetic-fire-prediction-system
python test_training/test_flir_scd41_training_simple.py
```

## 📊 Output Files

After running the training pipeline, the following files will be generated in `data/flir_scd41/`:

- `flir_scd41_dataset.csv` - Complete synthetic dataset
- `train.csv` - Training data split
- `val.csv` - Validation data split
- `test.csv` - Test data split
- `flir_scd41_xgboost_model.json` - Trained XGBoost model
- `best_nn_model.pth` - Trained Neural Network model
- `ensemble_weights.json` - Ensemble weights for model combination
- `model_info.json` - Model information and performance metrics

## 🧠 Model Architecture

### XGBoost Classifier
- Uses 100 estimators with max depth of 6
- Learning rate of 0.3
- Subsample ratio of 0.8
- Column subsample ratio of 0.8

### Neural Network
- 3-layer fully connected network
- Hidden layers with 64 neurons each
- ReLU activation functions
- Dropout for regularization (20%)
- Trained for 30 epochs with Adam optimizer

### Ensemble Method
- Performance-based weighting using exponential scaling
- Weights calculated based on validation accuracy
- Final prediction is weighted average of both models

## 📈 Expected Performance

The models typically achieve the following performance on the test set:
- Accuracy: 85-95%
- F1-Score: 80-90%
- Precision: 80-95%
- Recall: 75-90%

Performance may vary based on the synthetic data generation parameters.

## 🛠️ Customization

### Adjusting Dataset Size
Modify the `num_samples` parameter in the data generation function to create larger or smaller datasets.

### Changing Model Parameters
Update the model parameters in the training functions to experiment with different configurations.

### Modifying Ensemble Weights
The ensemble weights are calculated automatically based on validation performance, but you can modify the `calculate_performance_weights` function to use different weighting schemes.

## 🚨 Troubleshooting

### Common Issues

1. **Module not found errors**: Make sure you're running the scripts from the project root directory.

2. **CUDA out of memory**: Reduce the batch size in the neural network training function.

3. **Poor model performance**: Increase the dataset size or adjust model hyperparameters.

### Getting Help

If you encounter any issues, please check:
1. All dependencies are installed correctly
2. You're running the scripts from the correct directory
3. Your Python environment has the required packages

## 📚 Additional Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

## 🤝 Contributing

Feel free to contribute improvements to the training pipeline:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request