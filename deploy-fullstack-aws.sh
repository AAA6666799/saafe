#!/bin/bash

# SAAFE Dashboard Full-Stack AWS Deployment
# Deploys both React frontend and Node.js backend to AWS Elastic Beanstalk

set -e

echo "🚀 SAAFE Dashboard Full-Stack AWS Deployment"
echo "============================================="

# Configuration
APP_NAME="saafe-dashboard-$(date +%s)"
ENV_NAME="${APP_NAME}-production"
REGION="us-east-1"
INSTANCE_TYPE="t3.small"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Installing...${NC}"
    curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
    sudo installer -pkg AWSCLIV2.pkg -target /
    rm AWSCLIV2.pkg
fi

# Check if AWS is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not configured. Please run:${NC}"
    echo "aws configure"
    exit 1
fi

# Check if EB CLI is installed
if ! command -v eb &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Elastic Beanstalk CLI...${NC}"
    pip3 install awsebcli --upgrade --user
    export PATH=$PATH:~/.local/bin
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites ready${NC}"

# Build the frontend
echo -e "${YELLOW}📦 Building React frontend...${NC}"
cd saafe-lovable
npm install
npm run build
cd ..

# Install backend dependencies
echo -e "${YELLOW}📦 Installing backend dependencies...${NC}"
cd saafe-lovable/backend
npm install
cd ../..

echo -e "${GREEN}✅ Build completed${NC}"

# Create .ebextensions directory for EB configuration
echo -e "${YELLOW}⚙️ Creating Elastic Beanstalk configuration...${NC}"
mkdir -p .ebextensions

# Create EB configuration for Node.js
cat > .ebextensions/01_nodecommand.config << 'EOF'
option_settings:
  aws:elasticbeanstalk:container:nodejs:
    NodeCommand: "npm start"
  aws:elasticbeanstalk:application:environment:
    NODE_ENV: production
    PORT: 8081
  aws:elasticbeanstalk:environment:proxy:staticfiles:
    /static: dist
  aws:autoscaling:launchconfiguration:
    InstanceType: t3.small
  aws:elasticbeanstalk:healthreporting:system:
    SystemType: enhanced
EOF

# Create package.json for EB deployment
cat > package.json << 'EOF'
{
  "name": "saafe-dashboard-fullstack",
  "version": "1.0.0",
  "description": "SAAFE Fire Detection Dashboard - Full Stack",
  "main": "server.js",
  "scripts": {
    "start": "node saafe-lovable/backend/server.js",
    "postinstall": "cd saafe-lovable && npm install && cd backend && npm install"
  },
  "engines": {
    "node": "20.x",
    "npm": "10.x"
  },
  "dependencies": {
    "aws-sdk": "^2.1692.0",
    "cors": "^2.8.5",
    "express": "^4.21.1"
  },
  "keywords": [
    "fire-detection",
    "saafe",
    "aws",
    "react",
    "nodejs"
  ],
  "author": "SAAFE Team",
  "license": "ISC"
}
EOF

# Create .ebignore to exclude unnecessary files
cat > .ebignore << 'EOF'
node_modules/
.git/
*.log
.DS_Store
.env
.vscode/
*.zip
AWSCLIV2.pkg
*.ipynb
*.png
*.pkl*
*.md
docs/
tests/
deployment/
monitoring/
task_1_*/
saafe_mvp/
*.py
*.sh
*.json
!package.json
!saafe-lovable/
.git_backup/
*.pem
requirements*.txt
Dockerfile*
docker-compose*.yml
*.yaml
*.yml
!.ebextensions/
EOF

# Initialize EB application
echo -e "${YELLOW}🚀 Initializing Elastic Beanstalk application...${NC}"
eb init $APP_NAME --region $REGION --platform "node.js-20"

# Create environment and deploy
echo -e "${YELLOW}🌍 Creating production environment and deploying...${NC}"
eb create $ENV_NAME --instance-type $INSTANCE_TYPE --cname $APP_NAME

# Wait for deployment to complete
echo -e "${YELLOW}⏳ Waiting for deployment to complete...${NC}"
eb status

# Get the application URL
APP_URL=$(eb status | grep "CNAME" | awk '{print $2}')

# Configure environment variables for AWS access
echo -e "${YELLOW}🔧 Configuring environment variables...${NC}"
eb setenv NODE_ENV=production AWS_REGION=$REGION

echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "================================="
echo -e "${GREEN}✅ Application Name:${NC} $APP_NAME"
echo -e "${GREEN}✅ Environment:${NC} $ENV_NAME"
echo -e "${GREEN}✅ URL:${NC} http://$APP_URL"
echo -e "${GREEN}✅ HTTPS URL:${NC} https://$APP_URL"
echo -e "${GREEN}✅ API Endpoint:${NC} https://$APP_URL/api/fire-detection-data"
echo ""
echo -e "${BLUE}🌍 Your SAAFE dashboard is now live and accessible worldwide!${NC}"
echo ""
echo -e "${YELLOW}💡 Management Commands:${NC}"
echo "   - View logs: eb logs"
echo "   - Deploy updates: eb deploy"
echo "   - Open in browser: eb open"
echo "   - Check status: eb status"
echo "   - Terminate: eb terminate $ENV_NAME"
echo ""
echo -e "${YELLOW}📊 Monitoring:${NC}"
echo "   - AWS Console: https://console.aws.amazon.com/elasticbeanstalk/"
echo "   - CloudWatch Logs: Available in AWS Console"
echo "   - Health Dashboard: Available in EB Console"
echo ""
echo -e "${GREEN}🔥 Your fire detection system is now protecting users globally!${NC}"