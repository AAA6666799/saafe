#!/bin/bash

# Development Environment Setup Script for FLIR+SCD41 Fire Detection System
# This script sets up the complete development environment

echo "🔥 Setting up Development Environment for FLIR+SCD41 Fire Detection System"
echo "========================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: requirements.txt not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $PYTHON_VERSION == *"Python 3."* ]]; then
    echo -e "${GREEN}✅ $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.7 or higher.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Pip upgraded${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: Failed to upgrade pip${NC}"
fi

# Install requirements
echo -e "${BLUE}Installing Python requirements...${NC}"
pip install --index-url https://pypi.org/simple/ -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python requirements installed${NC}"
else
    echo -e "${RED}❌ Failed to install Python requirements${NC}"
    exit 1
fi

# Install additional development requirements
echo -e "${BLUE}Installing additional development requirements...${NC}"
pip install --index-url https://pypi.org/simple/ opencv-python-headless==4.10.0.84 structlog==24.4.0 prometheus-client==0.20.0 paho-mqtt==1.6.1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Additional development requirements installed${NC}"
else
    echo -e "${RED}❌ Failed to install additional development requirements${NC}"
    exit 1
fi

# Check AWS CLI
echo -e "${BLUE}Checking AWS CLI...${NC}"
if command -v aws &> /dev/null; then
    echo -e "${GREEN}✅ AWS CLI found${NC}"
    aws --version
else
    echo -e "${YELLOW}⚠️  AWS CLI not found. You can install it from https://aws.amazon.com/cli/${NC}"
fi

# Check Docker
echo -e "${BLUE}Checking Docker...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker found${NC}"
    docker --version
else
    echo -e "${YELLOW}⚠️  Docker not found. You can install it from https://www.docker.com/${NC}"
fi

# Create a simple test script to verify installation
cat > verify_installation.py << 'EOF'
import sys
import importlib

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ {package_name} import failed: {e}")
        return False

def main():
    print("🔍 Verifying package installations...")
    
    required_packages = [
        ("numpy", None),
        ("pandas", None),
        ("scipy", None),
        ("scikit-learn", "sklearn"),
        ("matplotlib", None),
        ("seaborn", None),
        ("pyyaml", "yaml"),
        ("boto3", None),
        ("botocore", None),
        ("torch", None),
        ("torchvision", None),
        ("cv2", "cv2"),
        ("sagemaker", None),
        ("pytest", None),
        ("tqdm", None),
        ("joblib", None),
        ("click", None),
        ("python-dotenv", "dotenv"),
        ("jupyter", None),
        ("structlog", None),
        ("prometheus_client", "prometheus_client"),
        ("paho.mqtt.client", "paho.mqtt.client")
    ]
    
    failed_packages = []
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            failed_packages.append(package_name)
    
    if failed_packages:
        print(f"\n❌ Failed to import {len(failed_packages)} packages: {', '.join(failed_packages)}")
        return 1
    else:
        print("\n🎉 All packages imported successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

# Run verification
echo -e "${BLUE}Verifying installation...${NC}"
python verify_installation.py
VERIFICATION_RESULT=$?

# Clean up
rm -f verify_installation.py

if [ $VERIFICATION_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Development environment setup completed successfully!${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run the verification script: python scripts/verify_development_environment.py"
    echo "3. Start developing!"
else
    echo -e "${RED}❌ Development environment setup failed${NC}"
    echo "Please check the error messages above and try again."
    exit 1
fi