#!/bin/bash
# Saafe MVP - AWS CodeBuild Deployment Script
# Deploys using AWS CodeCommit + CodeBuild + ECS

set -e

# Configuration
PROJECT_NAME="saafe-mvp"
AWS_REGION="eu-west-1"
AWS_ACCOUNT_ID="691595239825"
CODECOMMIT_REPO="${PROJECT_NAME}-repo"
CODEBUILD_PROJECT="${PROJECT_NAME}-build"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Saafe MVP - CodeBuild Deployment${NC}"
echo "=========================================="

# Step 1: Create CodeCommit repository
echo -e "\n${YELLOW}📝 Setting up CodeCommit repository...${NC}"
if ! aws codecommit get-repository --repository-name ${CODECOMMIT_REPO} > /dev/null 2>&1; then
    aws codecommit create-repository \
        --repository-name ${CODECOMMIT_REPO} \
        --repository-description "Saafe Fire Detection MVP"
    echo -e "${GREEN}✅ CodeCommit repository created${NC}"
else
    echo -e "${GREEN}✅ CodeCommit repository already exists${NC}"
fi

# Get repository URL
REPO_URL=$(aws codecommit get-repository --repository-name ${CODECOMMIT_REPO} --query 'repositoryMetadata.cloneUrlHttp' --output text)
echo "Repository URL: ${REPO_URL}"

# Step 2: Create ECR repository
echo -e "\n${YELLOW}📦 Setting up ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names ${PROJECT_NAME} > /dev/null 2>&1; then
    aws ecr create-repository \
        --repository-name ${PROJECT_NAME} \
        --image-scanning-configuration scanOnPush=true
    echo -e "${GREEN}✅ ECR repository created${NC}"
else
    echo -e "${GREEN}✅ ECR repository already exists${NC}"
fi

# Step 3: Create IAM role for CodeBuild
echo -e "\n${YELLOW}🔑 Setting up IAM roles...${NC}"
CODEBUILD_ROLE_NAME="SaafeCodeBuildRole"

if ! aws iam get-role --role-name ${CODEBUILD_ROLE_NAME} > /dev/null 2>&1; then
    # Create trust policy for CodeBuild
    cat > codebuild-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "codebuild.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create the role
    aws iam create-role \
        --role-name ${CODEBUILD_ROLE_NAME} \
        --assume-role-policy-document file://codebuild-trust-policy.json

    # Create and attach policy
    cat > codebuild-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:GetAuthorizationToken",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "codecommit:GitPull",
        "codeartifact:GetAuthorizationToken",
        "codeartifact:GetRepositoryEndpoint",
        "codeartifact:ReadFromRepository",
        "sts:GetServiceBearerToken"
      ],
      "Resource": "*"
    }
  ]
}
EOF

    aws iam put-role-policy \
        --role-name ${CODEBUILD_ROLE_NAME} \
        --policy-name SaafeCodeBuildPolicy \
        --policy-document file://codebuild-policy.json

    rm codebuild-trust-policy.json codebuild-policy.json
    echo -e "${GREEN}✅ CodeBuild IAM role created${NC}"
else
    echo -e "${GREEN}✅ CodeBuild IAM role already exists${NC}"
fi

# Step 4: Create CodeBuild project
echo -e "\n${YELLOW}🏗️  Setting up CodeBuild project...${NC}"
if ! aws codebuild batch-get-projects --names ${CODEBUILD_PROJECT} > /dev/null 2>&1; then
    cat > codebuild-project.json << EOF
{
  "name": "${CODEBUILD_PROJECT}",
  "description": "Build project for Saafe MVP",
  "source": {
    "type": "CODECOMMIT",
    "location": "${REPO_URL}",
    "buildspec": "buildspec.yml"
  },
  "artifacts": {
    "type": "NO_ARTIFACTS"
  },
  "environment": {
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/amazonlinux2-x86_64-standard:3.0",
    "computeType": "BUILD_GENERAL1_MEDIUM",
    "privilegedMode": true,
    "environmentVariables": [
      {
        "name": "AWS_DEFAULT_REGION",
        "value": "${AWS_REGION}"
      },
      {
        "name": "AWS_ACCOUNT_ID",
        "value": "${AWS_ACCOUNT_ID}"
      },
      {
        "name": "IMAGE_REPO_NAME",
        "value": "${PROJECT_NAME}"
      }
    ]
  },
  "serviceRole": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${CODEBUILD_ROLE_NAME}"
}
EOF

    aws codebuild create-project --cli-input-json file://codebuild-project.json
    rm codebuild-project.json
    echo -e "${GREEN}✅ CodeBuild project created${NC}"
else
    echo -e "${GREEN}✅ CodeBuild project already exists${NC}"
fi

# Step 5: Initialize git repository and push code
echo -e "\n${YELLOW}📤 Pushing code to CodeCommit...${NC}"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "Initial commit - Saafe MVP"
fi

# Add CodeCommit remote if not exists
if ! git remote get-url codecommit > /dev/null 2>&1; then
    git remote add codecommit ${REPO_URL}
fi

# Push to CodeCommit
git push codecommit main 2>/dev/null || git push codecommit master

echo -e "${GREEN}✅ Code pushed to CodeCommit${NC}"

# Step 6: Start build
echo -e "\n${YELLOW}🔨 Starting CodeBuild...${NC}"
BUILD_ID=$(aws codebuild start-build --project-name ${CODEBUILD_PROJECT} --query 'build.id' --output text)
echo "Build ID: ${BUILD_ID}"

echo -e "\n${BLUE}📊 Monitor build progress:${NC}"
echo "https://console.aws.amazon.com/codesuite/codebuild/projects/${CODEBUILD_PROJECT}/build/${BUILD_ID}/"

echo -e "\n${GREEN}🎉 Deployment initiated!${NC}"
echo "=========================================="
echo "Next steps:"
echo "1. Monitor the build in AWS Console"
echo "2. Once build completes, deploy to ECS using the built image"
echo "3. Access your application via the ECS service endpoint"
echo "=========================================="