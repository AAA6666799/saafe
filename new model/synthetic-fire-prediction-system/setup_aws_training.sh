#!/bin/bash

# AWS Fire Detection Training Setup Script
# This script helps you set up your AWS environment for training

echo "🔥 AWS Fire Detection Training Setup"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if AWS CLI is installed
echo -e "${BLUE}Checking AWS CLI installation...${NC}"
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install AWS CLI first.${NC}"
    echo "Install from: https://aws.amazon.com/cli/"
    exit 1
else
    echo -e "${GREEN}✅ AWS CLI found${NC}"
fi

# Check AWS credentials
echo -e "${BLUE}Checking AWS credentials...${NC}"
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region)
    echo -e "${GREEN}✅ AWS credentials configured${NC}"
    echo -e "Account ID: ${ACCOUNT_ID}"
    echo -e "Region: ${REGION}"
else
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    echo "Please run: aws configure"
    exit 1
fi

# Check if Python is installed
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
else
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ ${PYTHON_VERSION} found${NC}"
fi

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
if pip3 install -i https://pypi.org/simple/ --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements-aws.txt; then
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

# Prompt for S3 bucket name
echo -e "${BLUE}Setting up S3 bucket for training data...${NC}"
read -p "Enter your S3 bucket name for training data (or press Enter for default): " BUCKET_NAME

if [ -z "$BUCKET_NAME" ]; then
    BUCKET_NAME="fire-detection-training-${ACCOUNT_ID}"
fi

echo "Using bucket: $BUCKET_NAME"

# Create S3 bucket if it doesn't exist
if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
    echo -e "${YELLOW}Creating S3 bucket: $BUCKET_NAME${NC}"
    if [ "$REGION" == "us-east-1" ]; then
        aws s3 mb "s3://$BUCKET_NAME"
    else
        aws s3 mb "s3://$BUCKET_NAME" --region "$REGION"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ S3 bucket created successfully${NC}"
    else
        echo -e "${RED}❌ Failed to create S3 bucket${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ S3 bucket already exists${NC}"
fi

# Check/Create SageMaker execution role
echo -e "${BLUE}Checking SageMaker execution role...${NC}"
ROLE_NAME="SageMakerExecutionRole"

if aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
    echo -e "${GREEN}✅ SageMaker execution role exists${NC}"
else
    echo -e "${YELLOW}Creating SageMaker execution role...${NC}"
    
    # Create trust policy
    cat > /tmp/trust-policy.json << EOF
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

    # Create the role
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file:///tmp/trust-policy.json
    
    # Attach policies
    aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
    aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    
    echo -e "${GREEN}✅ SageMaker execution role created${NC}"
    
    # Clean up
    rm /tmp/trust-policy.json
fi

# Create training command
echo -e "${BLUE}Creating training command...${NC}"

cat > start_training.sh << EOF
#!/bin/bash
# AWS Fire Detection Training Start Script

echo "🔥 Starting AWS Fire Detection Training"
echo "======================================"

python3 aws_ensemble_trainer.py \\
    --config config/base_config.yaml \\
    --data-bucket $BUCKET_NAME

echo "Training completed!"
EOF

chmod +x start_training.sh

# Create dry-run command
cat > validate_setup.sh << EOF
#!/bin/bash
# AWS Setup Validation Script

echo "🔍 Validating AWS Setup"
echo "======================"

python3 aws_ensemble_trainer.py \\
    --config config/base_config.yaml \\
    --data-bucket $BUCKET_NAME \\
    --dry-run

echo "Validation completed!"
EOF

chmod +x validate_setup.sh

# Display final instructions
echo ""
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Validate your setup:"
echo -e "   ${YELLOW}./validate_setup.sh${NC}"
echo ""
echo "2. Start training:"
echo -e "   ${YELLOW}./start_training.sh${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "• AWS Account: $ACCOUNT_ID"
echo "• Region: $REGION"
echo "• S3 Bucket: $BUCKET_NAME"
echo "• SageMaker Role: arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo ""
echo -e "${YELLOW}Note: Training 25 models will take several hours and incur AWS costs.${NC}"
echo -e "${YELLOW}Monitor your costs in the AWS Console.${NC}"
echo ""
echo -e "${GREEN}Happy training! 🚀${NC}"