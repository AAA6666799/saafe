#!/bin/bash

# SAAFE Dashboard AWS Amplify Deployment (Alternative Method)
# This is the easiest deployment option with built-in CI/CD

set -e

echo "🚀 SAAFE Dashboard AWS Amplify Deployment"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install it first${NC}"
    exit 1
fi

# Check if Amplify CLI is installed
if ! command -v amplify &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Amplify CLI...${NC}"
    npm install -g @aws-amplify/cli
fi

echo -e "${GREEN}✅ Prerequisites ready${NC}"

# Initialize Amplify project
echo -e "${YELLOW}🔧 Initializing Amplify project...${NC}"
cd saafe-lovable

# Create amplify.yml for build configuration
cat > amplify.yml << EOF
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - npm ci
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: dist
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
EOF

echo -e "${GREEN}✅ Amplify configuration created${NC}"
echo ""
echo "🎯 Next Steps:"
echo "1. Go to AWS Amplify Console: https://console.aws.amazon.com/amplify/"
echo "2. Click 'New app' > 'Host web app'"
echo "3. Choose 'Deploy without Git provider'"
echo "4. Upload the saafe-lovable folder as a ZIP"
echo "5. AWS will automatically build and deploy your dashboard"
echo ""
echo "🌍 Your dashboard will be available at: https://[random-id].amplifyapp.com"
echo "💡 You can also connect a Git repository for automatic deployments"

cd ..