# 🚀 AWS SageMaker 50M Dataset Training Guide

## Quick Start (5 minutes to launch)

### 1. **Launch SageMaker Notebook Instance**
```bash
# Go to AWS Console -> SageMaker -> Notebook instances -> Create notebook instance
# Instance name: fire-detection-50m-training
# Instance type: ml.p3.2xlarge (Tesla V100 GPU) - $3.06/hour
# Or: ml.p3.8xlarge (4x V100) for 4x faster - $12.24/hour
# Volume size: 100 GB (for large dataset)
# IAM role: Create new role with S3 access to processedd-synthetic-data bucket
```

### 2. **Upload Training Script**
- Open Jupyter Lab in your SageMaker instance
- Upload `sagemaker_50m_training.py` to the notebook
- Or create new file and copy the script content

### 3. **Install Dependencies & Run**
```python
# In a Jupyter cell or terminal:
!pip install torch torchvision xgboost lightgbm catboost scikit-learn

# Then run the training:
!python sagemaker_50m_training.py
```

---

## 📊 **Training Configuration Options**

### Option 1: Full 50M Dataset (Recommended)
```python
# In sagemaker_50m_training.py, line 31:
USE_FULL_DATASET = True
MAX_SAMPLES_PER_AREA = None

# Expected results:
# - Training time: 2-4 hours on ml.p3.2xlarge
# - Memory usage: ~8-15 GB
# - Target accuracy: 97-98%
# - Cost: $6-12 USD
```

### Option 2: Demo with Subset
```python
# For testing/demo purposes:
USE_FULL_DATASET = False
MAX_SAMPLES_PER_AREA = 500000  # 500K per area = ~2.5M total samples

# Expected results:
# - Training time: 30-60 minutes
# - Memory usage: ~4-8 GB  
# - Target accuracy: 94-96%
# - Cost: $1.50-3 USD
```

---

## 💰 **Cost & Performance Estimates**

| Instance Type | GPU | Memory | Cost/Hour | Full Dataset Time | Total Cost |
|---------------|-----|--------|-----------|-------------------|------------|
| **ml.p3.2xlarge** ⭐ | 1x V100 | 61 GB | $3.06 | 2-4 hours | **$6-12** |
| **ml.p3.8xlarge** 🚀 | 4x V100 | 244 GB | $12.24 | 1-2 hours | **$12-24** |
| **ml.m5.2xlarge** 💰 | CPU only | 32 GB | $0.46 | 8-12 hours | **$4-6** |

⭐ **Recommended**: `ml.p3.2xlarge` for best balance of cost/performance

---

## 🎯 **What the Training Does**

### 1. **Data Loading & Processing**
- Loads all 5 area datasets from S3 (`processedd-synthetic-data/cleaned-data/`)
- Intelligent preprocessing based on each area's characteristics
- Creates time-series sequences (60 timesteps each)
- **Result**: ~19.9M sequences ready for training

### 2. **Enhanced Transformer Training**
- **Architecture**: Multi-area aware transformer with positional encoding
- **Features**: Area embeddings, attention mechanisms, risk prediction
- **Training**: 100 epochs with learning rate scheduling
- **Target**: 95%+ accuracy on fire detection

### 3. **ML Ensemble Training**  
- **Random Forest**: 200 trees, optimized for fire patterns
- **Gradient Boosting**: 200 estimators with regularization
- **XGBoost**: 300 estimators (if available)
- **LightGBM**: 300 estimators (if available)

### 4. **Ensemble Integration**
- Combines transformer + ML models via majority voting
- **Target**: 97-98% accuracy with <0.5% false positive rate

### 5. **Automatic Model Saving**
- All models saved to S3 automatically
- Includes metadata with performance metrics
- Ready for production deployment

---

## 📈 **Monitoring Training Progress**

### Real-time Logs
The script provides detailed logging:
```
🔥 FIRE DETECTION AI - 50M DATASET TRAINING ON SAGEMAKER
📥 Loading basement: s3://processedd-synthetic-data/cleaned-data/basement_data_cleaned.csv
✅ basement: (500000, 6), anomaly_rate=0.0907
🤖 TRAINING ENHANCED TRANSFORMER
Epoch  10: Loss=0.1234, Val_Acc=0.9456, Best=0.9456
📊 TRAINING ML ENSEMBLE
✅ Random Forest Val Acc: 0.9234
✅ XGBoost Val Acc: 0.9345
🏆 ENSEMBLE ACCURACY: 0.9678 (96.78%)
✅ Target (95%): ✅ ACHIEVED
```

### AWS CloudWatch Monitoring
- GPU utilization: Should be 80-95% during training
- Memory usage: Monitor for out-of-memory errors
- Network I/O: High during S3 data loading phase

---

## 🔧 **Troubleshooting**

### Common Issues & Solutions

#### 1. **Out of Memory Error**
```python
# Reduce batch size or use CPU fallback
# In the script, modify:
USE_FULL_DATASET = False
MAX_SAMPLES_PER_AREA = 250000  # Reduce dataset size
```

#### 2. **S3 Access Denied**
```bash
# Ensure IAM role has S3 access to processedd-synthetic-data bucket
# Add policy: AmazonS3ReadOnlyAccess
```

#### 3. **Package Installation Errors**
```python
# Use conda instead of pip:
!conda install pytorch torchvision -c pytorch -y
!pip install xgboost lightgbm catboost scikit-learn
```

#### 4. **Training Too Slow**
```python
# Options to speed up:
# 1. Use ml.p3.8xlarge (4x faster)
# 2. Reduce epochs from 100 to 50
# 3. Use smaller model: d_model=128 instead of 256
```

---

## 📤 **After Training Completes**

### 1. **Download Models**
Models are automatically saved to S3. To download:
```python
import boto3
import sagemaker

session = sagemaker.Session()
bucket = session.default_bucket()

# List all trained models
!aws s3 ls s3://{bucket}/fire-detection-models/ --recursive
```

### 2. **Deploy for Inference**
```python
# Create SageMaker endpoint for real-time inference
# Use the trained models for production fire detection
```

### 3. **Performance Validation**
Expected final results:
- **Ensemble Accuracy**: 96-98%
- **Individual Models**: 92-96% each
- **False Positive Rate**: <0.5%
- **Training Time**: 1-4 hours depending on instance
- **Ready for Production**: ✅

---

## 🚀 **Next Steps After Training**

1. **✅ Validate Performance**: Check if 97%+ accuracy achieved
2. **🚀 Deploy Models**: Create SageMaker endpoints for inference  
3. **📱 Integration**: Connect to your existing fire detection system
4. **📊 Monitoring**: Set up CloudWatch alarms for model performance
5. **🔄 Continuous Learning**: Retrain with new data periodically

---

## 📞 **Support**

- **AWS Support**: For SageMaker instance or billing issues
- **Script Issues**: Check CloudWatch logs for detailed error messages
- **Performance Tuning**: Adjust hyperparameters in the script
- **Cost Optimization**: Use Spot instances for 70% cost reduction

---

**🎯 Goal**: Deploy production-ready fire detection AI with 97-98% accuracy  
**⏰ Timeline**: 2-4 hours total training time  
**💰 Budget**: $6-24 USD depending on instance choice  
**🏆 Outcome**: Enterprise-grade fire detection system ready for deployment