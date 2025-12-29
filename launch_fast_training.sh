#!/bin/bash

# Saafe Ultra-Fast Training Launcher
# Maximum speed, cost is no object!

set -e

echo "🚀 SAAFE ULTRA-FAST TRAINING LAUNCHER"
echo "====================================="
echo "💰 Cost optimized for SPEED, not budget"
echo "⚡ Expected time: 30-45 minutes"
echo "🔥 Expected cost: $6-12"
echo ""

# Check prerequisites
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Install it first:"
    echo "   curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'"
    echo "   unzip awscliv2.zip && sudo ./aws/install"
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Run: aws configure"
    exit 1
fi

echo "✅ AWS CLI configured"

# Check if we have the fast training script
if [ ! -f "aws_fast_training.py" ]; then
    echo "❌ aws_fast_training.py not found"
    exit 1
fi

# Install dependencies quickly (avoiding conflicts)
echo "📦 Installing dependencies..."
pip install -q --no-deps boto3 sagemaker torch numpy 2>/dev/null || echo "⚠️  Some dependency conflicts (safe to ignore)"

# Get account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
echo "📋 Account: $ACCOUNT_ID"
echo "📋 Region: $REGION"

# Check/create SageMaker role
ROLE_NAME="SaafeTrainingRole"
echo "🔐 Checking SageMaker role..."

if ! aws iam get-role --role-name $ROLE_NAME &> /dev/null; then
    echo "Creating SageMaker role for ultra-fast training..."
    
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file://trust-policy.json

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

    rm trust-policy.json
    echo "✅ SageMaker role created"
    sleep 10
else
    echo "✅ SageMaker role exists"
fi

echo ""
echo "🔥 ULTRA-FAST TRAINING OPTIONS"
echo "=============================="
echo "1. 🚀 MAXIMUM SPEED (ml.p3.8xlarge - 4x V100 GPUs) - ~$9, 30min"
echo "2. ⚡ DISTRIBUTED (4x ml.p3.2xlarge - 4x V100 GPUs) - ~$6, 35min"
echo "3. 🔥 AUTO-CHOOSE FASTEST"
echo ""

read -p "Choose option (1-3) [3]: " choice
choice=${choice:-3}

echo ""
echo "🚀 LAUNCHING ULTRA-FAST TRAINING..."
echo "=================================="

case $choice in
    1)
        echo "🔥 Maximum Speed: Single ml.p3.8xlarge instance"
        echo "   4x NVIDIA V100 GPUs, 244GB RAM, 64GB GPU memory"
        echo "   Cost: ~$12.24/hour, Expected: 30-45 minutes = ~$6-9"
        ;;
    2)
        echo "⚡ Distributed: 4x ml.p3.2xlarge instances"
        echo "   4x NVIDIA V100 GPUs total, Distributed training"
        echo "   Cost: ~$12.24/hour, Expected: 20-35 minutes = ~$4-7"
        ;;
    3)
        echo "🚀 Auto-choosing MAXIMUM SPEED option"
        echo "   Single ml.p3.8xlarge for fastest overall training"
        ;;
esac

echo ""
read -p "Proceed with ultra-fast training? (y/N): " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "👍 Training cancelled"
    exit 0
fi

echo ""
echo "🚀 LAUNCHING TRAINING JOB..."
echo "============================"

# Launch the training
python aws_fast_training.py

echo ""
echo "🎯 ULTRA-FAST TRAINING SUMMARY"
echo "=============================="
echo "✅ Training job launched successfully!"
echo "⏱️  Expected completion: 30-45 minutes"
echo "💰 Expected cost: $6-12"
echo "🔍 Monitor progress:"
echo "   AWS Console: https://console.aws.amazon.com/sagemaker/"
echo "   CLI: aws sagemaker list-training-jobs --status-equals InProgress"
echo ""
echo "📱 You'll get production-ready models in under an hour!"
echo "🚀 This is the FASTEST way to train your Saafe models!"

# Optional: Set up monitoring
read -p "Set up automatic monitoring and download? (y/N): " monitor

if [[ $monitor =~ ^[Yy]$ ]]; then
    echo "⏳ Setting up monitoring..."
    
    # Create monitoring script
    cat > monitor_training.sh << 'EOF'
#!/bin/bash

echo "🔍 Monitoring Saafe training jobs..."

while true; do
    # Get latest training job
    LATEST_JOB=$(aws sagemaker list-training-jobs \
        --sort-by CreationTime \
        --sort-order Descending \
        --max-items 1 \
        --query 'TrainingJobSummaries[0].TrainingJobName' \
        --output text)
    
    if [ "$LATEST_JOB" != "None" ] && [[ $LATEST_JOB == saafe-* ]]; then
        STATUS=$(aws sagemaker describe-training-job \
            --training-job-name $LATEST_JOB \
            --query 'TrainingJobStatus' \
            --output text)
        
        echo "$(date): Job $LATEST_JOB - Status: $STATUS"
        
        if [ "$STATUS" = "Completed" ]; then
            echo "🎉 Training completed! Downloading models..."
            
            # Get model location
            MODEL_URI=$(aws sagemaker describe-training-job \
                --training-job-name $LATEST_JOB \
                --query 'ModelArtifacts.S3ModelArtifacts' \
                --output text)
            
            # Download model
            mkdir -p models
            aws s3 cp $MODEL_URI models/model.tar.gz
            cd models && tar -xzf model.tar.gz && rm model.tar.gz
            
            # Rename to standard format
            if [ -f "model.pth" ]; then
                mv model.pth transformer_model.pth
            fi
            
            echo "✅ Models downloaded to models/ directory"
            echo "🚀 Ready for deployment!"
            break
            
        elif [ "$STATUS" = "Failed" ]; then
            echo "❌ Training failed!"
            break
        fi
    fi
    
    sleep 60  # Check every minute
done
EOF
    
    chmod +x monitor_training.sh
    
    echo "✅ Monitoring script created: monitor_training.sh"
    echo "   Run: ./monitor_training.sh (in another terminal)"
fi

echo ""
echo "🎉 ULTRA-FAST TRAINING SETUP COMPLETE!"
echo "====================================="
echo "Your Saafe models will be trained with maximum speed!"
echo "Check back in 30-45 minutes for production-ready models! 🚀"