#!/usr/bin/env python3
"""
Development Environment Verification Script for FLIR+SCD41 Fire Detection System.

This script verifies that all required components for development are properly installed
and configured.
"""

import sys
import os
import subprocess
import importlib
import json
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    version_info = sys.version_info
    logger.info(f"Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    # Check if Python 3.7 or higher
    if version_info.major >= 3 and version_info.minor >= 7:
        logger.info("✅ Python version meets requirements")
        return True
    else:
        logger.error("❌ Python version too old. Required: 3.7+")
        return False

def check_package_installed(package_name: str, import_name: str = None) -> bool:
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        logger.info(f"✅ {package_name} is installed")
        return True
    except ImportError:
        logger.error(f"❌ {package_name} is not installed")
        return False

def check_aws_cli() -> bool:
    """Check if AWS CLI is installed and configured."""
    try:
        # Check if AWS CLI is installed
        result = subprocess.run(['aws', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✅ AWS CLI installed: {result.stdout.strip()}")
        else:
            logger.error("❌ AWS CLI not installed")
            return False
        
        # Check if AWS credentials are configured
        result = subprocess.run(['aws', 'sts', 'get-caller-identity'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            identity = json.loads(result.stdout)
            logger.info(f"✅ AWS credentials configured for account: {identity.get('Account', 'Unknown')}")
            return True
        else:
            logger.warning("⚠️  AWS credentials not configured (this is OK for local development)")
            return True  # Not required for all development tasks
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("⚠️  AWS CLI not found (this is OK for local development)")
        return True  # Not required for all development tasks

def check_docker() -> bool:
    """Check if Docker is installed and running."""
    try:
        # Check if Docker is installed
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✅ Docker installed: {result.stdout.strip()}")
        else:
            logger.warning("⚠️  Docker not installed (this is OK for local development)")
            return True  # Not required for all development tasks
        
        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ Docker daemon is running")
            return True
        else:
            logger.warning("⚠️  Docker daemon not running (this is OK for local development)")
            return True  # Not required for all development tasks
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("⚠️  Docker not found (this is OK for local development)")
        return True  # Not required for all development tasks

def check_required_packages() -> Dict[str, bool]:
    """Check all required Python packages."""
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pyyaml': 'yaml',
        'boto3': 'boto3',
        'botocore': 'botocore',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'opencv-python': 'cv2',
        'sagemaker': 'sagemaker',
        'pytest': 'pytest',
        'tqdm': 'tqdm',
        'joblib': 'joblib',
        'click': 'click',
        'python-dotenv': 'dotenv',
        'jupyter': 'jupyter',
        'structlog': 'structlog',
        'prometheus-client': 'prometheus_client',
        'paho-mqtt': 'paho.mqtt.client'
    }
    
    results = {}
    for package_name, import_name in required_packages.items():
        results[package_name] = check_package_installed(package_name, import_name)
    
    return results

def check_s3_access() -> bool:
    """Check if S3 access is configured."""
    try:
        import boto3
        s3 = boto3.client('s3')
        # Try to list buckets (this will fail if credentials are invalid)
        s3.list_buckets()
        logger.info("✅ S3 access configured")
        return True
    except Exception as e:
        logger.warning(f"⚠️  S3 access not configured: {str(e)}")
        return True  # Not required for all development tasks

def check_sagemaker_role() -> bool:
    """Check if SageMaker execution role exists."""
    try:
        import boto3
        iam = boto3.client('iam')
        # Try to get the SageMaker execution role
        iam.get_role(RoleName='SageMakerExecutionRole')
        logger.info("✅ SageMaker execution role exists")
        return True
    except Exception as e:
        logger.warning(f"⚠️  SageMaker execution role not found: {str(e)}")
        return True  # Not required for all development tasks

def main() -> int:
    """Main verification function."""
    logger.info("🔍 Verifying Development Environment for FLIR+SCD41 Fire Detection System")
    logger.info("=" * 80)
    
    # Track overall success
    all_checks_passed = True
    
    # Check Python version
    logger.info("\n1. Checking Python version...")
    if not check_python_version():
        all_checks_passed = False
    
    # Check AWS CLI
    logger.info("\n2. Checking AWS CLI...")
    if not check_aws_cli():
        all_checks_passed = False
    
    # Check Docker
    logger.info("\n3. Checking Docker...")
    if not check_docker():
        all_checks_passed = False
    
    # Check required packages
    logger.info("\n4. Checking required Python packages...")
    package_results = check_required_packages()
    failed_packages = [pkg for pkg, success in package_results.items() if not success]
    if failed_packages:
        logger.error(f"❌ Failed packages: {', '.join(failed_packages)}")
        all_checks_passed = False
    else:
        logger.info("✅ All required packages installed")
    
    # Check S3 access
    logger.info("\n5. Checking S3 access...")
    if not check_s3_access():
        all_checks_passed = False
    
    # Check SageMaker role
    logger.info("\n6. Checking SageMaker execution role...")
    if not check_sagemaker_role():
        all_checks_passed = False
    
    # Final summary
    logger.info("\n" + "=" * 80)
    if all_checks_passed:
        logger.info("🎉 All development environment checks passed!")
        logger.info("✅ You're ready to develop the FLIR+SCD41 Fire Detection System")
        return 0
    else:
        logger.error("❌ Some development environment checks failed")
        logger.error("🔧 Please install the missing components and try again")
        return 1

if __name__ == "__main__":
    sys.exit(main())