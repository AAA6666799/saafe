# 🚀 Saafe AWS Training Guide

Train your Saafe fire detection models using AWS cloud infrastructure for production-grade results.

## 🎯 Quick Start (5 minutes)

```bash
# 1. Setup AWS training environment
./setup_aws_training.sh

# 2. Start training with recommended settings
./start_training.sh
```

**That's it!** Your models will be trained on AWS and ready for deployment.

---

## 📊 Training Options & Costs

| Option | Instance | Cost/Hour | Training Time | Total Cost | Best For |
|--------|----------|-----------|---------------|------------|----------|
| **SageMaker GPU** ⭐ | ml.p3.2xlarge | $3.06 | 1.5 hours | **$4.59** | Production models |
| **SageMaker CPU** 💰 | ml.m5.2xlarge | $0.46 | 3.0 hours | **$1.38** | Budget training |
| **EC2 Spot GPU** 🔥 | p3.2xlarge | $0.31 | 1.5 hours | **$0.47** | Advanced users |
| **AWS Batch** 📦 | c5.2xlarge | $0.34 | 2.5 hours | **$0.85** | Multiple experiments |
| **ECS Fargate** ☁️ | Serverless | $0.20 | 4.0 hours | **$0.80** | Serverless approach |

⭐ **Recommended**: SageMaker GPU for best balance of speed and ease of use.

---

## 🛠️ Setup Requirements

### Prerequisites
- AWS CLI installed and configured
- Python 3.8+ with pip
- AWS account with appropriate permissions

### Quick Setup
```bash
# Install AWS CLI (if not installed)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region

# Run setup script
./setup_aws_training.sh
```

---

## 🚀 Training Methods

### Method 1: Quick Start (Recommended)
```bash
# Automated training with best practices
python aws_training_pipeline.py
```

### Method 2: Interactive Setup
```bash
# Choose your training options interactively
python aws_training_options.py
```

### Method 3: Custom Configuration
```python
from aws_training_pipeline import AWSTrainingPipeline

# Initialize pipeline
pipeline = AWSTrainingPipeline(region='us-east-1')

# Custom training configuration
config = {
    'instance_type': 'ml.p3.2xlarge',
    'instance_count': 1,
    'max_runtime_hours': 2,
    'hyperparameters': {
        'epochs': 150,
        'batch-size': 64,
        'learning-rate': 0.001,
        'num-samples': 20000
    }
}

# Launch training
job_name = pipeline.launch_training_job(**config)
```

---

## 📈 What Gets Trained

### 🔥 Fire Detection Transformer
- **Architecture**: Spatio-Temporal Transformer
- **Parameters**: ~7.2M parameters
- **Input**: Multi-sensor time-series data (temperature, PM2.5, CO₂, audio)
- **Output**: Fire risk score (0-100) + classification (normal/cooking/fire)

### 🛡️ Anti-Hallucination System
- **Ensemble Voting**: Multiple model consensus
- **Cooking Detection**: Prevents false alarms during cooking
- **Fire Signature Validation**: Multi-modal fire confirmation
- **Confidence Calibration**: Risk score adjustment

### 📊 Training Data
- **Synthetic Dataset**: 20,000 realistic scenarios
- **Normal Scenarios**: 60% (12,000 samples)
- **Cooking Scenarios**: 25% (5,000 samples)
- **Fire Scenarios**: 15% (3,000 samples)

---

## 🔍 Monitoring Training

### AWS Console
- **SageMaker**: https://console.aws.amazon.com/sagemaker/
- **CloudWatch**: https://console.aws.amazon.com/cloudwatch/
- **Billing**: https://console.aws.amazon.com/billing/

### Command Line
```bash
# List training jobs
aws sagemaker list-training-jobs --status-equals InProgress

# Get job details
aws sagemaker describe-training-job --training-job-name <job-name>

# View logs
aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name <job-name>/algo-1-<timestamp>
```

### Python Monitoring
```python
from aws_training_pipeline import AWSTrainingPipeline

pipeline = AWSTrainingPipeline()
status = pipeline.monitor_training_job('your-job-name')
print(f"Status: {status['TrainingJobStatus']}")
```

---

## 📥 Getting Your Trained Models

### Automatic Download
```python
# Download after training completes
pipeline = AWSTrainingPipeline()
success = pipeline.download_trained_model('your-job-name', 'models/')
```

### Manual Download
```bash
# Get model S3 location
aws sagemaker describe-training-job --training-job-name <job-name> \
    --query 'ModelArtifacts.S3ModelArtifacts'

# Download model
aws s3 cp s3://bucket/path/model.tar.gz ./models/
tar -xzf ./models/model.tar.gz -C ./models/
```

### Model Files
After training, you'll have:
```
models/
├── transformer_model.pth      # Trained PyTorch model
├── anti_hallucination.pkl     # Anti-hallucination parameters
├── model_metadata.json        # Model information and metrics
└── training_results.png       # Training plots and metrics
```

---

## 🎯 Performance Expectations

### Model Performance
- **Classification Accuracy**: 95-98%
- **Fire Detection Rate**: 95-99%
- **False Alarm Rate**: <0.1%
- **Risk Score MAE**: <5.0

### Training Metrics
- **Training Time**: 1.5-4 hours (depending on instance)
- **Model Size**: ~28.7 MB
- **Inference Speed**: <50ms per prediction
- **Memory Usage**: <2GB for inference

---

## 💰 Cost Optimization

### Budget-Friendly Options
1. **Use Spot Instances**: Save up to 90%
2. **Choose CPU Training**: $1.38 vs $4.59
3. **Set Budget Alerts**: Automatic cost monitoring
4. **Use Smaller Models**: Reduce training time

### Cost Monitoring
```python
# Set up budget alert
pipeline = AWSTrainingPipeline()
pipeline.create_training_budget_alert(budget_limit=10.0)
```

### Estimated Costs by Use Case
- **Demo/Testing**: $0.47 - $1.38
- **Development**: $1.38 - $4.59
- **Production**: $4.59 - $10.00

---

## 🔧 Advanced Configuration

### Custom Hyperparameters
```python
hyperparameters = {
    'epochs': 200,              # Training epochs
    'batch-size': 64,           # Batch size (larger for GPU)
    'learning-rate': 0.0005,    # Learning rate
    'num-samples': 25000,       # Training samples
    'd-model': 512,             # Model dimension
    'num-heads': 16,            # Attention heads
    'num-layers': 8,            # Transformer layers
    'dropout': 0.15             # Dropout rate
}
```

### Multi-Instance Training
```python
# Distributed training across multiple instances
config = {
    'instance_type': 'ml.p3.2xlarge',
    'instance_count': 2,        # Use 2 instances
    'distribution': {
        'smdistributed': {
            'dataparallel': {
                'enabled': True
            }
        }
    }
}
```

### Custom Training Data
```python
# Use your own training data
training_input = TrainingInput(
    s3_data='s3://your-bucket/training-data/',
    content_type='application/json'
)

estimator.fit({'train': training_input})
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. "ResourceLimitExceeded" Error
```bash
# Check service quotas
aws service-quotas get-service-quota \
    --service-code sagemaker \
    --quota-code L-1194F27D

# Request quota increase if needed
aws service-quotas request-service-quota-increase \
    --service-code sagemaker \
    --quota-code L-1194F27D \
    --desired-value 5
```

#### 2. "AccessDenied" Error
```bash
# Check IAM permissions
aws iam get-role --role-name SaafeTrainingRole

# Attach missing policies
aws iam attach-role-policy \
    --role-name SaafeTrainingRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

#### 3. Training Job Fails
```bash
# Check logs
aws logs describe-log-streams \
    --log-group-name /aws/sagemaker/TrainingJobs

# View specific error
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name <job-name>/algo-1-<timestamp>
```

#### 4. High Costs
```bash
# Stop running jobs
aws sagemaker stop-training-job --training-job-name <job-name>

# Use spot instances
python aws_training_options.py  # Select EC2 Spot option
```

### Getting Help
- **AWS Support**: https://console.aws.amazon.com/support/
- **SageMaker Docs**: https://docs.aws.amazon.com/sagemaker/
- **GitHub Issues**: Create an issue in the repository

---

## 🎉 Success Checklist

After training completion, verify:

- [ ] ✅ Training job completed successfully
- [ ] ✅ Model files downloaded to `models/` directory
- [ ] ✅ Model metadata shows good performance metrics
- [ ] ✅ Anti-hallucination parameters calibrated
- [ ] ✅ Total cost within expected range
- [ ] ✅ Models ready for AWS deployment

---

## 🚀 Next Steps

Once training is complete:

1. **Test Models Locally**
   ```bash
   python -c "from saafe_mvp.models.model_loader import load_model; model = load_model('models/transformer_model.pth'); print('Model loaded successfully!')"
   ```

2. **Deploy to AWS**
   ```bash
   python aws_training_pipeline.py --create-endpoint
   ```

3. **Run Full Application**
   ```bash
   streamlit run app.py
   ```

4. **Production Deployment**
   ```bash
   # Follow AWS deployment guide
   python setup_codeartifact.py
   # Deploy to ECS/Fargate
   ```

---

## 📞 Support

- **Technical Issues**: Create GitHub issue
- **AWS Billing**: Check AWS Billing Console
- **Training Questions**: Review logs and metrics
- **Performance Issues**: Try different instance types

**Happy Training! 🚀🔥**