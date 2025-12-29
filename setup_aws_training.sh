#!/bin/bash

# Saafe AWS Training Setup Script
# This script sets up everything needed for AWS-based model training

set -e

echo "🚀 Saafe AWS Training Setup"
echo "=========================="

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first:"
    echo "   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run:"
    echo "   aws configure"
    exit 1
fi

echo "✅ AWS CLI configured"

# Get AWS account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
echo "📋 Account ID: $ACCOUNT_ID"
echo "📋 Region: $REGION"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install boto3 sagemaker torch numpy scikit-learn matplotlib seaborn

# Check if required files exist
if [ ! -f "aws_training_pipeline.py" ]; then
    echo "❌ aws_training_pipeline.py not found"
    exit 1
fi

if [ ! -f "aws_training_options.py" ]; then
    echo "❌ aws_training_options.py not found"
    exit 1
fi

echo "✅ Training scripts found"

# Create IAM role for SageMaker (if it doesn't exist)
echo "🔐 Setting up IAM role for SageMaker..."

ROLE_NAME="SaafeTrainingRole"
ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME"

if aws iam get-role --role-name $ROLE_NAME &> /dev/null; then
    echo "✅ SageMaker role already exists: $ROLE_ARN"
else
    echo "Creating SageMaker execution role..."
    
    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create role
    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file://trust-policy.json \
        --description "SageMaker execution role for Saafe training"

    # Attach policies
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

    # Clean up
    rm trust-policy.json

    echo "✅ SageMaker role created: $ROLE_ARN"
    echo "⏳ Waiting 10 seconds for role to propagate..."
    sleep 10
fi

# Check service limits
echo "📊 Checking service limits..."

# Check SageMaker limits
SAGEMAKER_LIMITS=$(aws service-quotas get-service-quota \
    --service-code sagemaker \
    --quota-code L-1194F27D \
    --query 'Quota.Value' \
    --output text 2>/dev/null || echo "0")

if [ "$SAGEMAKER_LIMITS" -gt 0 ]; then
    echo "✅ SageMaker training instances available: $SAGEMAKER_LIMITS"
else
    echo "⚠️  Could not check SageMaker limits. You may need to request quota increases."
fi

# Estimate costs
echo ""
echo "💰 ESTIMATED TRAINING COSTS"
echo "=========================="
echo "SageMaker GPU (ml.p3.2xlarge): ~$4.59 (1.5 hours)"
echo "SageMaker CPU (ml.m5.2xlarge): ~$1.38 (3 hours)"
echo "EC2 Spot GPU (p3.2xlarge):     ~$0.47 (1.5 hours)"
echo ""

# Show training options
echo "🎯 TRAINING OPTIONS"
echo "=================="
echo "1. Quick Start (Recommended): python aws_training_pipeline.py"
echo "2. Interactive Setup:         python aws_training_options.py"
echo "3. View All Options:          python aws_training_options.py"
echo ""

# Create a simple launcher script
cat > start_training.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Saafe AWS Training"
echo "=============================="

# Check if user wants to see options first
read -p "View training options first? (y/N): " show_options

if [[ $show_options =~ ^[Yy]$ ]]; then
    python aws_training_options.py
    echo ""
fi

# Confirm training start
read -p "Start training with recommended settings? (y/N): " start_training

if [[ $start_training =~ ^[Yy]$ ]]; then
    echo "🚀 Launching training job..."
    python aws_training_pipeline.py
else
    echo "👍 Training cancelled. Run this script again when ready."
fi
EOF

chmod +x start_training.sh

echo "✅ Setup completed successfully!"
echo ""
echo "🚀 NEXT STEPS"
echo "============"
echo "1. Quick start:      ./start_training.sh"
echo "2. View options:     python aws_training_options.py"
echo "3. Direct training:  python aws_training_pipeline.py"
echo ""
echo "📊 Monitor training: https://console.aws.amazon.com/sagemaker/"
echo "💰 Monitor costs:    https://console.aws.amazon.com/billing/"
echo ""
echo "🎯 Ready to train your Saafe models on AWS!"